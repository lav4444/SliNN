"""
criteria2.py — 4 GROWING kriterija + cost model + benefit/FLOP alokacija (SchoolCNN).

Simetricno s exp2 pruning kriterijima (reuse pruning_lib2.compute_importance):
    magnitude  -> "net2net" kontrola: kloniraj najjace (|W|) jedinice (data-free)
    gradient   -> GradMax-ish: kloniraj jedinice s najvecim |g|
    taylor     -> kloniraj jedinice s najvecom (g*W)^2 vaznosti
    hessian    -> "split": kloniraj+perturbiraj jedinice s najvecom zakrivljenoscu

GDJE rasti (alokacija): benefit(L)=prosjecna vaznost jedinice sloja (po kriteriju);
score(L)=benefit/cost_per_unit; k_L ~ score, skalirano da ukupni dodani FLOPs =
budzet ciklusa. Cost model uzima u obzir FLATTEN coupling (conv5 -> fc1: 1 kanal
= +flatten_hw ulaza u fc1), pa rast conv5 ima visok cost (kao i u stvarnosti).

KAKO (init): nove jedinice = klonovi TOP-vaznih postojecih (+ mali sum; hessian:
+ perturbacija). Function-preserving osigurava grow_layer (downstream=0).
"""

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


# --------------------------------------------------------------------------- #
# Calib batchevi + BCE loss (multi-label) — za gradijentne benefite
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Cost model: GFLOPs po DODANOJ jedinici sloja L (vlastiti + downstream rast)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def flops_per_filter(model, device, imgsz=INPUT_SIZE):
    """Vrati dict name->FLOPs koje doda 1 nova izlazna jedinica sloja L:
    own (jedna izlazna jedinica L-a) + downstream (block ulaznih jedinica svakom
    potrosacu; conv5->fc1 block=flatten_hw -> skupo)."""
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


# --------------------------------------------------------------------------- #
# Benefit (po kriteriju) + alokacija (benefit/FLOP do budzeta ciklusa)
# --------------------------------------------------------------------------- #
def layer_benefit(model, criterion, calib, loss_fn, device):
    """Vrati (benefit: name->scalar, per_unit_imp: name->Tensor[O], overhead)."""
    entries, _ = PL.build_plan(model)
    imp, overhead = PL.compute_importance(model, entries, criterion, loss_fn, calib, device)
    benefit = {n: float(imp[n].float().mean()) for n in imp}     # prosjecna vaznost jedinice
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
# Init novih jedinica: kloniraj TOP-vazne postojece (+ sum; hessian: perturbacija)
# --------------------------------------------------------------------------- #
def make_init_fn(per_unit_imp, criterion):
    def init_fn(model, name, k):
        _, by_name = PL.build_plan(model)
        W = by_name[name]["module"].weight.detach()      # conv [O,in,kh,kw] | lin [O,in]
        O = W.shape[0]
        sc = per_unit_imp[name].float()
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


def grow_to_budget(model, criterion, calib, loss_fn, target_total_flops, device,
                   max_iter=20, max_grow_frac=0.5):
    """Rasti model dok count_flops ne dosegne target_total_flops. benefit/imp se
    racunaju JEDNOM (signal kriterija za ovaj ciklus); cost se RE-mjeri svaku
    iteraciju (compounding) i alocira se 0.6*preostalo da se ne preskoci budzet.

    PER-LAYER CAP (max_grow_frac * trenutna sirina po iteraciji): nuzno za SchoolCNN
    jer su dense slojevi (fc1/fc2) FLOP-jeftini -> benefit/FLOP bi im inace alocirao
    desetke tisuca neurona (eksplozija params + compounding overshoot kroz flatten).
    Cap drzi rast razumnim i tjera FLOP-budzet u conv slojeve (gdje FLOPs zive),
    a kriterij i dalje bira raspodjelu unutar capova. Re-mjerenje konvergira na cilj."""
    import growing_lib2 as G
    benefit, imp, overhead = layer_benefit(model, criterion, calib, loss_fn, device)
    init_fn = make_init_fn(imp, criterion)
    _, by_name0 = PL.build_plan(model)
    w0 = G.current_widths(by_name0)                       # sirine na pocetku ciklusa
    cap = {L: max(1, int(round(max_grow_frac * w0[L]))) for L in w0}   # kumulativni headroom
    added = {}
    for _ in range(max_iter):
        cur = PL.count_flops(model, device, INPUT_SIZE)
        if cur >= 0.98 * target_total_flops:
            break
        remaining = target_total_flops - cur
        cost = flops_per_filter(model, device)
        # alociraj SAMO medu slojevima koji jos imaju headroom (capnuti se izbace ->
        # njihov benefit-udio budzeta se preraspodijeli na conv umjesto da se izgubi)
        avail = {L: b for L, b in benefit.items() if cap.get(L, 0) - added.get(L, 0) > 0}
        if not avail:
            break
        plan = allocate(avail, cost, 0.6 * remaining)
        if not plan:
            break
        # kumulativni cap po sloju: rast ovog ciklusa <= max_grow_frac * w0[L]
        plan = {L: min(k, cap.get(L, 0) - added.get(L, 0)) for L, k in plan.items()}
        plan = {L: k for L, k in plan.items() if k > 0}
        if not plan:
            break
        G.grow_many(model, plan, init_fn)
        for n, k in plan.items():
            added[n] = added.get(n, 0) + k
    return added, overhead


# --------------------------------------------------------------------------- #
# Self-check: cost + allocate + grow-to-budget (magnitude data-free + taylor)
# --------------------------------------------------------------------------- #
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
