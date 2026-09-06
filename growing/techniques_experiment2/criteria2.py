
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXP2_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment2"
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

import torch.utils.data
import common                          # noqa: E402
import train_baseline as TB            # noqa: E402
import pruning_lib2 as PL              # noqa: E402
from model_cnn import INPUT_SIZE       # noqa: E402


def make_calib_batches(n_batches, device):
    loader = common.make_loader("train", TB.BATCH_SIZE, shuffle=True, num_workers=TB.NUM_WORKERS)
    out = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        out.append(b)
    return out


def make_bce_loss_fn(device):
    crit = nn.BCEWithLogitsLoss()
    def loss_fn(model, batch):
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = crit(model(x), y)
        return loss, x.size(0)
    return loss_fn


@torch.no_grad()
def flops_per_filter(model, device, imgsz=INPUT_SIZE):
    entries, by_name = PL.build_plan(model)
    out_hw = {}
    handles = []
    for e in entries:
        if e["kind"] == "conv":
            handles.append(e["module"].register_forward_hook(
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
        L = e["name"]; m = e["module"]
        if e["kind"] == "conv":
            kh, kw = m.kernel_size; oh, ow = out_hw[L]
            own = 2 * m.in_channels * kh * kw * oh * ow
        else:
            own = 2 * m.in_features
        down = 0
        for C in entries:
            for src, block in C["in_sources"]:
                if src != L:
                    continue
                cm = C["module"]
                if C["kind"] == "conv":
                    ckh, ckw = cm.kernel_size; coh, cow = out_hw[C["name"]]
                    down += block * 2 * cm.out_channels * ckh * ckw * coh * cow
                else:
                    down += block * 2 * cm.out_features
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


def make_init_fn(per_unit_imp, criterion):
    def init_fn(model, name, k):
        _, by_name = PL.build_plan(model)
        W = by_name[name]["module"].weight.detach()
        O = W.shape[0]
        sc = per_unit_imp[name].float()
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


def grow_to_budget(model, criterion, calib, loss_fn, target_total_flops, device,
                   max_iter=20, max_grow_frac=0.5):
    import growing_lib2 as G
    benefit, imp, overhead = layer_benefit(model, criterion, calib, loss_fn, device)
    init_fn = make_init_fn(imp, criterion)
    _, by_name0 = PL.build_plan(model)
    w0 = G.current_widths(by_name0)
    cap = {L: max(1, int(round(max_grow_frac * w0[L]))) for L in w0}
    added = {}
    for _ in range(max_iter):
        cur = PL.count_flops(model, device, INPUT_SIZE)
        if cur >= 0.98 * target_total_flops:
            break
        remaining = target_total_flops - cur
        cost = flops_per_filter(model, device)
        avail = {L: b for L, b in benefit.items() if cap.get(L, 0) - added.get(L, 0) > 0}
        if not avail:
            break
        plan = allocate(avail, cost, 0.6 * remaining)
        if not plan:
            break
        plan = {L: min(k, cap.get(L, 0) - added.get(L, 0)) for L, k in plan.items()}
        plan = {L: k for L, k in plan.items() if k > 0}
        if not plan:
            break
        G.grow_many(model, plan, init_fn)
        for n, k in plan.items():
            added[n] = added.get(n, 0) + k
    return added, overhead


def _selfcheck():
    import growing_lib2 as G
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(EXP2_DIR) / "pruned_models" / "taylor.pt"

    loss_fn = make_bce_loss_fn(device)
    for crit, n_calib in [("magnitude", 0), ("taylor", 4)]:
        model = G.load_pruned_model(ckpt, device)
        f0 = PL.count_flops(model, device, INPUT_SIZE)
        if crit == "magnitude":
            top = sorted(flops_per_filter(model, device).items(), key=lambda kv: -kv[1])[:4]
            print("cost/unit (MFLOPs) top-4:", [(n, round(c / 1e6, 3)) for n, c in top])
        calib = make_calib_batches(n_calib, device) if n_calib else []
        target = f0 * 1.5
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        with torch.no_grad():
            out0 = model(x).clone()
        added, _ = grow_to_budget(model, crit, calib, loss_fn, target, device)
        f1 = PL.count_flops(model, device, INPUT_SIZE)
        with torch.no_grad():
            out1 = model(x)
        diff = (out1 - out0).abs().max().item()
        print(f"[{crit:9s}] slojeva={len(added)} sum_k={sum(added.values()):3d} | "
              f"GFLOPs {f0/1e9:.4f}->{f1/1e9:.4f} (cilj {target/1e9:.4f}, "
              f"odstup {(f1-target)/target*100:+.1f}%) | dodano={added} | fwd diff={diff:.2e}")
    print("CRITERIA SELFCHECK OK")


if __name__ == "__main__":
    _selfcheck()
