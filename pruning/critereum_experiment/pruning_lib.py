"""
pruning_lib.py — jezgra za STRUCTURED pruning (cijeli neuron / conv filter).

Ovo je RUCNO pisan structured pruner (bez gotovih toolova) specijaliziran za
`StudentYOLO` (PAN neck + SPPF + concat/lateral veze). Radi STVARNO micanje
kanala (rebuild manjih Conv2d/BatchNorm2d), ne maskiranje — dakle params i
FLOPs stvarno padnu.

Podrzava i `nn.Linear` (neuron = izlazna znacajka) i `nn.Conv2d` (filter =
izlazni kanal) na razini surgery-ja; StudentYOLO je cisto konvolucijski pa se
ovdje efektivno rezu conv filteri, ali API je generican radi ponovne upotrebe.

Sadrzi:
  * build_plan(model)              - rucni dependency graf StudentYOLO
  * compute_importance(...)        - 4 kriterija + mjerenje overhead-a
  * select_kept_indices(...)       - globalni izbor do ciljanog budzeta params
  * apply_pruning(model, plan, ..) - surgery (rebuild Conv/BN, slice ulaza)
  * count_params / count_flops     - prije/poslije
  * recalibrate_bn(...)            - osvjezi BN statistike nakon rezanja

VAZNO o kriterijima (da budu genuino razliciti i usporedivi) — svi rade u
prostoru tezina, po IZLAZNOM kanalu c se zbroji preko (in, kh, kw):
    magnitude :  sum |W_c|
    gradient  :  sum |g_c|                 (g = prosjecni gradijent)
    taylor    :  ( sum g_c * W_c )^2        (Taylor 1. red: procjena dL)
    hessian   :  0.5 * sum (g2_c * W_c^2)   (OBD; dijag. Hessiana ~ empirijski
                                             Fisher g2 = prosjek g^2)
Napomena: 'hessian' je APROKSIMACIJA dijagonale Hessiana empirijskim Fisherom,
ne egzaktni Hessian (egzaktni je preskup). To je standardno i citabilno (OBD).
"""

from __future__ import annotations

import time
import copy

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Dependency graf (rucno za StudentYOLO)
# --------------------------------------------------------------------------- #
# Svaki "entry" opisuje jedan Conv2d:
#   name        : jedinstveno ime
#   conv        : nn.Conv2d
#   bn          : pripadni nn.BatchNorm2d ili None
#   replace     : fn(new_conv, new_bn) -> postavi nove module na mjesto
#   in_sources  : redoslijed izvora ULAZNIH kanala (imena drugih entryja ili
#                 'INPUT'); concat -> vise izvora; SPPF conv2 -> 4x isti izvor
#   prune_out   : smije li se rezati IZLAZ (False za finalne head conv-ove)
#   orig_out    : originalni broj izlaznih kanala
#   has_bias    : ima li conv bias
def build_plan(model):
    """Vraca listu entryja (dict) koji u potpunosti opisuju ovisnosti kanala."""
    entries = []
    by_name = {}

    def add_cba(name, cba, in_sources, prune_out=True):
        conv, bn = cba.conv, cba.bn

        def replace(new_conv, new_bn, _cba=cba):
            _cba.conv = new_conv
            _cba.bn = new_bn

        e = dict(name=name, conv=conv, bn=bn, replace=replace,
                 in_sources=list(in_sources), prune_out=prune_out,
                 orig_out=conv.out_channels, has_bias=conv.bias is not None)
        entries.append(e); by_name[name] = e

    def add_final(name, seq, idx, in_sources):
        conv = seq[idx]

        def replace(new_conv, _new_bn, _seq=seq, _idx=idx):
            _seq[_idx] = new_conv

        e = dict(name=name, conv=conv, bn=None, replace=replace,
                 in_sources=list(in_sources), prune_out=False,
                 orig_out=conv.out_channels, has_bias=conv.bias is not None)
        entries.append(e); by_name[name] = e

    # ---- Backbone ----
    add_cba("stem",     model.stem,     ["INPUT"])
    add_cba("dark2.0",  model.dark2[0], ["stem"])
    add_cba("dark2.1",  model.dark2[1], ["dark2.0"])
    add_cba("dark3.0",  model.dark3[0], ["dark2.1"])
    add_cba("dark3.1",  model.dark3[1], ["dark3.0"])
    add_cba("dark3.2",  model.dark3[2], ["dark3.1"])
    add_cba("dark3.3",  model.dark3[3], ["dark3.2"])        # p3
    add_cba("dark4.0",  model.dark4[0], ["dark3.3"])
    add_cba("dark4.1",  model.dark4[1], ["dark4.0"])
    add_cba("dark4.2",  model.dark4[2], ["dark4.1"])
    add_cba("dark4.3",  model.dark4[3], ["dark4.2"])        # p4
    add_cba("dark5.0",  model.dark5[0], ["dark4.3"])
    add_cba("dark5.1",  model.dark5[1], ["dark5.0"])        # p5 (pre-sppf)

    # ---- SPPF ----  conv2 ulaz = cat([x, y1, y2, y3]) = 4x conv1 izlaz
    add_cba("sppf.conv1", model.sppf.conv1, ["dark5.1"])
    add_cba("sppf.conv2", model.sppf.conv2, ["sppf.conv1"] * 4)   # sppf izlaz

    # ---- PAN neck laterali ----
    add_cba("lat_p3", model.lat_p3, ["dark3.3"])
    add_cba("lat_p4", model.lat_p4, ["dark4.3"])
    add_cba("lat_p5", model.lat_p5, ["sppf.conv2"])

    # ---- Top-down (redoslijed concata mora pratiti forward) ----
    add_cba("fuse_td_p4", model.fuse_td_p4, ["lat_p4", "lat_p5"])        # cat[l4, up(l5)]
    add_cba("fuse_td_p3", model.fuse_td_p3, ["lat_p3", "fuse_td_p4"])    # cat[l3, up(td_p4)]

    # ---- Bottom-up ----
    add_cba("down_p3",    model.down_p3,    ["fuse_td_p3"])
    add_cba("fuse_bu_p4", model.fuse_bu_p4, ["fuse_td_p4", "down_p3"])   # cat[td_p4, down_p3]
    add_cba("down_p4",    model.down_p4,    ["fuse_bu_p4"])
    add_cba("fuse_bu_p5", model.fuse_bu_p5, ["lat_p5", "down_p4"])       # cat[l5, down_p4]

    # ---- Heads (head.0 = ConvBNAct rez; head.1 = finalni 1x1, izlaz nedirnut) ----
    add_cba("head_p3.0", model.head_p3[0], ["fuse_td_p3"])
    add_final("head_p3.1", model.head_p3, 1, ["head_p3.0"])
    add_cba("head_p4.0", model.head_p4[0], ["fuse_bu_p4"])
    add_final("head_p4.1", model.head_p4, 1, ["head_p4.0"])
    add_cba("head_p5.0", model.head_p5[0], ["fuse_bu_p5"])
    add_final("head_p5.1", model.head_p5, 1, ["head_p5.0"])

    return entries, by_name


def _orig_out_of(src, by_name):
    return 3 if src == "INPUT" else by_name[src]["orig_out"]


# --------------------------------------------------------------------------- #
# Brojanje parametara iz "keep" stanja (za ciljanje budzeta)
# --------------------------------------------------------------------------- #
def params_from_counts(entries, by_name, counts):
    """counts: dict name->trenutni broj IZLAZNIH kanala (samo prune_out slojevi).
    Vraca ukupan broj parametara (conv + bn + bias) za dano stanje."""
    total = 0
    for e in entries:
        name = e["name"]
        out_eff = counts[name] if e["prune_out"] else e["orig_out"]
        in_eff = 0
        for src in e["in_sources"]:
            in_eff += 3 if src == "INPUT" else counts[src]
        kh, kw = e["conv"].kernel_size
        total += out_eff * in_eff * kh * kw          # conv weight
        if e["bn"] is not None:
            total += 2 * out_eff                     # bn weight+bias
        if e["has_bias"]:
            total += out_eff                         # conv bias (head finals)
    return total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Vaznost kanala (4 kriterija) + mjerenje overhead-a
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _reduce_magnitude(W):
    # W: (O, I, kh, kw) -> (O,)
    return W.abs().flatten(1).sum(1)


def compute_importance(model, entries, criterion, loss_fn, calib_batches, device):
    """Vrati (importance: dict name->Tensor[O], overhead: dict).

    criterion in {'magnitude','gradient','taylor','hessian'}.
    Za sve osim 'magnitude' radi forward+backward na `calib_batches`
    (lista batcheva) racunajuci prosjecni gradijent g i prosjek g^2.
    overhead = {'time_s', 'time_ms_per_image', 'peak_mem_mb', 'n_images',
                'needs_backward'}.
    """
    prunable = [e for e in entries if e["prune_out"]]
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    importance = {}
    n_images = 0

    if criterion == "magnitude":
        # data-free: samo iz tezina
        for e in prunable:
            importance[e["name"]] = _reduce_magnitude(e["conv"].weight.detach()).cpu()
        needs_backward = False
    else:
        model.eval()                       # eval: BN koristi running stats (ne kvarimo ih)
        grad_sum = {e["name"]: torch.zeros_like(e["conv"].weight) for e in prunable}
        grad_sq  = {e["name"]: torch.zeros_like(e["conv"].weight) for e in prunable}

        n_batches = 0
        for batch in calib_batches:
            for p in model.parameters():
                p.grad = None
            loss, bs = loss_fn(model, batch)        # fp32, bez autocasta
            loss.backward()
            for e in prunable:
                g = e["conv"].weight.grad
                if g is not None:
                    grad_sum[e["name"]] += g.detach()
                    grad_sq[e["name"]]  += g.detach() ** 2
            n_batches += 1
            n_images += bs

        for e in prunable:
            name = e["name"]
            W = e["conv"].weight.detach()
            g = grad_sum[name] / max(n_batches, 1)        # prosjecni gradijent
            g2 = grad_sq[name] / max(n_batches, 1)        # prosjek g^2 (Fisher diag)
            if criterion == "gradient":
                imp = g.abs().flatten(1).sum(1)
            elif criterion == "taylor":
                imp = (g * W).flatten(1).sum(1) ** 2
            elif criterion == "hessian":
                imp = 0.5 * (g2 * W ** 2).flatten(1).sum(1)
            else:
                raise ValueError(f"Nepoznat kriterij: {criterion}")
            importance[name] = imp.detach().cpu()
        # ocisti gradijente
        for p in model.parameters():
            p.grad = None
        needs_backward = True

    if is_cuda:
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if is_cuda else float("nan")

    overhead = {
        "time_s": dt,
        "n_images": n_images,
        "time_ms_per_image": (dt * 1e3 / n_images) if n_images > 0 else 0.0,
        "peak_mem_mb": peak_mb,
        "needs_backward": needs_backward,
    }
    return importance, overhead


# --------------------------------------------------------------------------- #
# Izbor kanala za zadrzati (globalni rang, do ciljanog budzeta parametara)
# --------------------------------------------------------------------------- #
def select_kept_indices(entries, by_name, importance, keep_param_frac,
                        min_keep=2, alloc="global"):
    """Vrati (kept_idx: dict name->LongTensor sortiranih indeksa, info: dict).

    alloc='global': svi kanali u jednu listu, per-layer L2 normalizacija
    vaznosti (radi usporedivosti medu slojevima), pa pohlepno mice najmanje
    vazne dok params ne padnu na keep_param_frac * baseline. Postuje min_keep.
    """
    prunable = [e for e in entries if e["prune_out"]]
    counts = {e["name"]: e["orig_out"] for e in prunable}
    baseline = params_from_counts(entries, by_name, counts)
    target = keep_param_frac * baseline

    if alloc != "global":
        raise NotImplementedError("Trenutno je implementiran samo alloc='global'.")

    # globalni rang s per-layer L2 normalizacijom
    ranked = []   # (norm_val, name, ch)
    for e in prunable:
        imp = importance[e["name"]].float()
        norm = imp / (imp.norm() + 1e-12)
        for ch in range(e["orig_out"]):
            ranked.append((float(norm[ch]), e["name"], ch))
    ranked.sort(key=lambda t: t[0])      # najmanje vazni prvi

    removed = {e["name"]: set() for e in prunable}
    cur = params_from_counts(entries, by_name, counts)
    for _val, name, ch in ranked:
        if cur <= target:
            break
        if counts[name] <= min_keep:
            continue
        removed[name].add(ch)
        counts[name] -= 1
        cur = params_from_counts(entries, by_name, counts)

    kept_idx = {}
    for e in prunable:
        name = e["name"]
        keep = [i for i in range(e["orig_out"]) if i not in removed[name]]
        kept_idx[name] = torch.tensor(keep, dtype=torch.long)

    info = {
        "baseline_params": baseline,
        "target_params": int(target),
        "achieved_params": cur,
        "achieved_keep_frac": cur / baseline,
        "per_layer_kept": {n: len(kept_idx[n]) for n in kept_idx},
        "per_layer_orig": {e["name"]: e["orig_out"] for e in prunable},
    }
    return kept_idx, info


# --------------------------------------------------------------------------- #
# Surgery: stvarno micanje kanala (rebuild Conv2d/BatchNorm2d)
# --------------------------------------------------------------------------- #
def _gather_in_idx(entry, by_name, kept_idx):
    """Globalni ulazni indeksi (nakon rezanja) za dani conv, slozeni po
    redoslijedu njegovih in_sources (concat-aware)."""
    idx_parts = []
    offset = 0
    for src in entry["in_sources"]:
        if src == "INPUT":
            keep_local = torch.arange(3)
            orig = 3
        else:
            keep_local = kept_idx[src]                     # zadrzani izlazi izvora
            orig = by_name[src]["orig_out"]
        idx_parts.append(keep_local + offset)
        offset += orig
    return torch.cat(idx_parts)


def apply_pruning(model, entries, by_name, kept_idx):
    """In-place izgradnja manjih Conv2d/BatchNorm2d prema kept_idx."""
    for e in entries:
        name = e["name"]
        conv = e["conv"]

        # izlazni indeksi
        if e["prune_out"]:
            out_idx = kept_idx[name]
        else:
            out_idx = torch.arange(e["orig_out"])

        in_idx = _gather_in_idx(e, by_name, kept_idx)

        new_conv = nn.Conv2d(
            in_channels=len(in_idx), out_channels=len(out_idx),
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation,
            groups=1, bias=e["has_bias"],
        ).to(conv.weight.device, conv.weight.dtype)

        with torch.no_grad():
            w = conv.weight.detach()[out_idx][:, in_idx]
            new_conv.weight.copy_(w)
            if e["has_bias"]:
                new_conv.bias.copy_(conv.bias.detach()[out_idx])

        new_bn = None
        if e["bn"] is not None:
            bn = e["bn"]
            new_bn = nn.BatchNorm2d(len(out_idx), eps=bn.eps, momentum=bn.momentum,
                                    affine=bn.affine,
                                    track_running_stats=bn.track_running_stats
                                    ).to(bn.weight.device, bn.weight.dtype)
            with torch.no_grad():
                new_bn.weight.copy_(bn.weight.detach()[out_idx])
                new_bn.bias.copy_(bn.bias.detach()[out_idx])
                if bn.track_running_stats:
                    new_bn.running_mean.copy_(bn.running_mean.detach()[out_idx])
                    new_bn.running_var.copy_(bn.running_var.detach()[out_idx])
                    new_bn.num_batches_tracked.copy_(bn.num_batches_tracked.detach())

        e["replace"](new_conv, new_bn)
    return model


# --------------------------------------------------------------------------- #
# FLOPs (forward hooks) + BN rekalibracija
# --------------------------------------------------------------------------- #
@torch.no_grad()
def count_flops(model, device, imgsz=640):
    """Ukupni FLOPs (2*MAC) za jedan forward na (1,3,imgsz,imgsz)."""
    total = [0]
    handles = []

    def hook(m, inp, out):
        oc = out.shape[1]
        oh, ow = out.shape[2], out.shape[3]
        ic = m.in_channels
        kh, kw = m.kernel_size
        total[0] += 2 * oc * ic * kh * kw * oh * ow // m.groups

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    model(x)
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return total[0]


@torch.no_grad()
def recalibrate_bn(model, calib_batches, get_imgs, device, reset=True):
    """Osvjezi BN running statistike nakon rezanja (bez ucenja tezina).

    Prolazi `calib_batches` u train() modu pa BN akumulira nove statistike.
    get_imgs(batch) -> tensor slika na uredaju. reset=True nuldira stare stats."""
    if reset:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None            # kumulativni prosjek (stabilnije)
    model.train()
    for batch in calib_batches:
        imgs = get_imgs(batch).to(device, non_blocking=True)
        model(imgs)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Self-check (samo oblici; pokrece se s `python pruning_lib.py`)
# --------------------------------------------------------------------------- #
def _selfcheck():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("/home/tomi/code/dipl/custom_models/student_2_m")))
    from model_arch import StudentYOLO

    device = torch.device("cpu")
    model = StudentYOLO(num_classes=6).to(device).eval()

    # sanity: rekonstrukcija param-brojanja mora pasati pravom broju
    entries, by_name = build_plan(model)
    counts = {e["name"]: e["orig_out"] for e in entries if e["prune_out"]}
    calc = params_from_counts(entries, by_name, counts)
    real = count_params(model)
    print(f"params_from_counts={calc:,}  real={real:,}  match={calc == real}")
    assert calc == real, "Brojanje parametara iz grafa NE odgovara stvarnom modelu!"

    # lazna 'magnitude' vaznost -> 80% rez -> surgery -> forward shape
    imp = {e["name"]: _reduce_magnitude(e["conv"].weight.detach())
           for e in entries if e["prune_out"]}
    kept, info = select_kept_indices(entries, by_name, imp, keep_param_frac=0.20)
    apply_pruning(model, entries, by_name, kept)

    pruned_params = count_params(model)
    print(f"baseline={info['baseline_params']:,}  target(20%)={info['target_params']:,}  "
          f"achieved={pruned_params:,}  keep={pruned_params/info['baseline_params']*100:.1f}%")

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        raw = model(x)
        boxes, probs = model.decode(raw)
    print(f"raw out {tuple(raw.shape)} (ocek. (2,10,8400)) | "
          f"boxes {tuple(boxes.shape)} probs {tuple(probs.shape)}")
    assert raw.shape == (2, 10, 8400), "Pogresan oblik izlaza nakon rezanja!"
    gf = count_flops(model, device)
    print(f"pruned GFLOPs={gf/1e9:.3f}")
    print("SELF-CHECK OK")


if __name__ == "__main__":
    _selfcheck()
