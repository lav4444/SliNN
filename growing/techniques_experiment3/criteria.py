"""
criteria.py — 4 GROWING kriterija + cost model + benefit/FLOP alokacija.

Simetricno s exp1 pruning kriterijima (reuse pruning_lib.compute_importance):
    magnitude  -> "net2net" kontrola: kloniraj najjace (|W|) filtere (data-free)
    gradient   -> GradMax-ish: kloniraj filtere s najvecim |g|
    taylor     -> kloniraj filtere s najvecom (g*W)^2 vaznosti
    hessian    -> "split": kloniraj+perturbiraj filtere s najvecom zakrivljenoscu

GDJE rasti (alokacija): benefit(L)=prosjecna vaznost filtera sloja (po kriteriju);
score(L)=benefit/cost_per_filter; k_L ~ score (vise filtera tamo gdje je najveca
korist po FLOP-u), skalirano da ukupni dodani FLOPs = budzet ciklusa.

KAKO (init): novi filteri = klonovi TOP-vaznih postojecih filtera (+ mali sum;
hessian: + perturbacija). Function-preserving osigurava grow_layer (downstream=0).
"""

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


# --------------------------------------------------------------------------- #
# KD calib + loss (reuse train_kd) — za gradijentne benefite
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Cost model: GFLOPs po DODANOM filteru sloja L (vlastiti conv + downstream rast)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def flops_per_filter(model, device, imgsz=640):
    """Vrati dict name->FLOPs koje doda 1 novi izlazni filter sloja L:
    own (jedan izlazni kanal L-a) + downstream (jedan ulazni kanal svakom
    potrosacu; SPPF -> x4 jer L se pojavljuje 4x u in_sources)."""
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


# --------------------------------------------------------------------------- #
# Benefit (po kriteriju) + alokacija (benefit/FLOP do budzeta ciklusa)
# --------------------------------------------------------------------------- #
def layer_benefit(model, criterion, calib, loss_fn, device):
    """Vrati (benefit: name->scalar, per_filter_imp: name->Tensor[O], overhead)."""
    entries, _ = PL.build_plan(model)
    imp, overhead = PL.compute_importance(model, entries, criterion, loss_fn, calib, device)
    benefit = {n: float(imp[n].float().mean()) for n in imp}     # prosjecna vaznost filtera
    return benefit, imp, overhead


def allocate(benefit, cost, cycle_flops):
    """k_L ~ score(L)=benefit/cost, skalirano da sum_L k_L*cost_L ~ cycle_flops."""
    score = {L: benefit[L] / cost[L] for L in benefit
             if cost.get(L, 0) > 0 and benefit[L] > 0}
    tot_benefit = sum(benefit[L] for L in score)
    if tot_benefit <= 0:
        return {}
    s = cycle_flops / tot_benefit
    plan = {L: max(0, int(round(s * score[L]))) for L in score}
    return {L: k for L, k in plan.items() if k > 0}


# --------------------------------------------------------------------------- #
# Init novih filtera: kloniraj TOP-vazne postojece (+ sum; hessian: perturbacija)
# --------------------------------------------------------------------------- #
def grow_to_budget(model, criterion, calib, loss_fn, target_total_flops, device, max_iter=12):
    """Rasti model dok count_flops ne dosegne target_total_flops. benefit/imp se
    racunaju JEDNOM (signal kriterija za ovaj ciklus); cost se RE-mjeri svaku
    iteraciju (compounding) i alocira se 0.6*preostalo da se ne preskoci budzet."""
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


def grow_one_shot(model, criterion, calib, loss_fn, target_total_flops, device):
    """ONE-SHOT grow (za exp3): benefit/imp + cost se racunaju JEDNOM, alokacija
    JEDNOM, grow JEDNOM — BEZ iterativnog re-mjerenja (za razliku od grow_to_budget).
    Posljedica: staticki cost/filter podcjenjuje stvarni trosak (compounding kroz
    spregnute slojeve), pa realni GFLOPs obicno PREMASE cilj. To je svojstvo
    one-shot pristupa i upravo ono sto exp3 testira nasuprot iterativnom exp1."""
    import growing_lib as G
    benefit, imp, overhead = layer_benefit(model, criterion, calib, loss_fn, device)
    init_fn = make_init_fn(imp, criterion)
    cur = PL.count_flops(model, device)
    budget = max(0.0, target_total_flops - cur)
    cost = flops_per_filter(model, device)
    plan = allocate(benefit, cost, budget)        # CIJELI budzet odjednom (bez 0.6, bez petlje)
    added = {}
    if plan:
        G.grow_many(model, plan, init_fn)
        added = dict(plan)
    return added, overhead


def make_init_fn(per_filter_imp, criterion):
    def init_fn(model, name, k):
        _, by_name = PL.build_plan(model)
        W = by_name[name]["conv"].weight.detach()        # [O, in, kh, kw] (trenutni)
        O = W.shape[0]
        sc = per_filter_imp[name].float()
        n_imp = min(len(sc), O)
        order = torch.argsort(sc[:n_imp], descending=True)   # najvazniji prvi
        idx = [int(order[i % len(order)]) for i in range(k)]
        clones = W[idx].clone()
        scale = clones.abs().mean().clamp(min=1e-6)
        if criterion == "hessian":
            noise = torch.randn_like(clones) * 0.10 * scale      # "split": jaca perturbacija
        else:
            noise = torch.randn_like(clones) * 0.02 * scale
        return clones + noise
    return init_fn


# --------------------------------------------------------------------------- #
# Self-check: cost + allocate + grow-to-budget (magnitude data-free + taylor)
# --------------------------------------------------------------------------- #
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
        target = f0 * (1.0 + 2.0 / 3.0)         # +0.667x (1. ciklus od 3 do 3x)
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
