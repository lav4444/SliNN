
from __future__ import annotations
import sys
from pathlib import Path

import torch

STUDENT_DIR = "/home/tomi/code/dipl/custom_models/student_2_m"
PURE_KD_DIR = "/home/tomi/code/dipl/custom_models/student_2_m/pure_KD"
EXP1_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment"
for _p in (PURE_KD_DIR, STUDENT_DIR, EXP1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch.utils.data
import train_kd as TK                 # noqa: E402
import pruning_lib as PL              # noqa: E402


def make_calib_batches(n_batches, device):
    ds = TK.KDTrainDataset(TK.TRAIN_IMG_DIR, TK.TEACHER_SOFT_DIR)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=TK.BATCH_SIZE, shuffle=True, num_workers=TK.NUM_WORKERS,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(TK.SEED))
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


def make_kd_loss_fn(device):
    def loss_fn(model, batch):
        imgs = batch["image"].to(device, non_blocking=True)
        tb = batch["teacher_boxes"].to(device, non_blocking=True)
        tp = batch["teacher_probs"].to(device, non_blocking=True)
        raw = model(imgs)
        loss, _, _ = TK.kd_loss(raw, tb, tp, model.anchor_xy, model.anchor_stride)
        return loss, imgs.size(0)
    return loss_fn


@torch.no_grad()
def flops_per_filter(model, device, imgsz=640):
    entries, by_name = PL.build_plan(model)
    out_hw = {}
    handles = []
    for e in entries:
        handles.append(e["conv"].register_forward_hook(
            lambda m, i, o, nm=e["name"]: out_hw.__setitem__(nm, (o.shape[2], o.shape[3]))))
    was = model.training
    model.eval()
    model(torch.randn(1, 3, imgsz, imgsz, device=device))
    for h in handles:
        h.remove()
    model.train(was)

    cost = {}
    for e in entries:
        if not e["prune_out"]:
            continue
        L = e["name"]; conv = e["conv"]; kh, kw = conv.kernel_size
        oh, ow = out_hw[L]
        own = 2 * conv.in_channels * kh * kw * oh * ow
        down = 0
        for C in entries:
            if L in C["in_sources"]:
                mult = C["in_sources"].count(L)
                cc = C["conv"]; ckh, ckw = cc.kernel_size; coh, cow = out_hw[C["name"]]
                down += mult * 2 * cc.out_channels * ckh * ckw * coh * cow
        cost[L] = own + down
    return cost


def layer_benefit(model, criterion, calib, loss_fn, device):
    entries, _ = PL.build_plan(model)
    imp, overhead = PL.compute_importance(model, entries, criterion, loss_fn, calib, device)
    benefit = {n: float(imp[n].float().mean()) for n in imp}
    return benefit, imp, overhead


def allocate(benefit, cost, cycle_flops):
    score = {L: benefit[L] / cost[L] for L in benefit
             if cost.get(L, 0) > 0 and benefit[L] > 0}
    tot_benefit = sum(benefit[L] for L in score)
    if tot_benefit <= 0:
        return {}
    s = cycle_flops / tot_benefit
    plan = {L: max(0, int(round(s * score[L]))) for L in score}
    return {L: k for L, k in plan.items() if k > 0}


def grow_to_budget(model, criterion, calib, loss_fn, target_total_flops, device, max_iter=12):
    import growing_lib as G
    benefit, imp, overhead = layer_benefit(model, criterion, calib, loss_fn, device)
    init_fn = make_init_fn(imp, criterion)
    added = {}
    for _ in range(max_iter):
        cur = PL.count_flops(model, device)
        if cur >= 0.98 * target_total_flops:
            break
        remaining = target_total_flops - cur
        cost = flops_per_filter(model, device)
        plan = allocate(benefit, cost, 0.6 * remaining)
        if not plan:
            break
        G.grow_many(model, plan, init_fn)
        for n, k in plan.items():
            added[n] = added.get(n, 0) + k
    return added, overhead


def make_init_fn(per_filter_imp, criterion):
    def init_fn(model, name, k):
        _, by_name = PL.build_plan(model)
        W = by_name[name]["conv"].weight.detach()
        O = W.shape[0]
        sc = per_filter_imp[name].float()
        n_imp = min(len(sc), O)
        order = torch.argsort(sc[:n_imp], descending=True)
        idx = [int(order[i % len(order)]) for i in range(k)]
        clones = W[idx].clone()
        scale = clones.abs().mean().clamp(min=1e-6)
        if criterion == "hessian":
            noise = torch.randn_like(clones) * 0.10 * scale
        else:
            noise = torch.randn_like(clones) * 0.02 * scale
        return clones + noise
    return init_fn


def _selfcheck():
    import growing_lib as G
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(EXP1_DIR) / "pruned_models" / "taylor.pt"

    loss_fn = make_kd_loss_fn(device)
    for crit, n_calib in [("magnitude", 0), ("taylor", 4)]:
        model = G.load_pruned_model(ckpt, device)
        f0 = PL.count_flops(model, device)
        if crit == "magnitude":
            top = sorted(flops_per_filter(model, device).items(), key=lambda kv: -kv[1])[:3]
            print("cost/filter (GFLOPs) top-3:", [(n, round(c/1e9, 4)) for n, c in top])
        calib = make_calib_batches(n_calib, device) if n_calib else []
        target = f0 * (1.0 + 2.0 / 3.0)
        x = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            out0 = model(x).clone()
        added, _ = grow_to_budget(model, crit, calib, loss_fn, target, device)
        f1 = PL.count_flops(model, device)
        with torch.no_grad():
            out1 = model(x)
        diff = (out1 - out0).abs().max().item()
        print(f"[{crit:9s}] slojeva={len(added)} sum_k={sum(added.values()):3d} | "
              f"GFLOPs {f0/1e9:.3f}->{f1/1e9:.3f} (cilj {target/1e9:.3f}, "
              f"odstup {(f1-target)/target*100:+.1f}%) | fwd diff={diff:.2e}")
    print("CRITERIA SELFCHECK OK")


if __name__ == "__main__":
    _selfcheck()
