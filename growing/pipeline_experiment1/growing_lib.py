"""
growing_lib.py — jezgra za NETWORK GROWING (dodavanje conv filtera) na StudentYOLO.

Inverz pruninga: kad prosirimo IZLAZ sloja L za k kanala, moramo konzistentno
prosiriti ULAZ svih potrosaca L-a (concat-aware: SPPF conv2 = 4x conv1; PAN fuse).

FUNCTION-PRESERVING: novi filteri sloja L dobiju zadani init, a NOVE ulazne
kolone potrosaca su 0 -> izlaz mreze se NE mijenja dok trening ne pocne koristiti
nove kanale. (validira se u _selfcheck: forward identican prije/poslije.)

Reuse: build_plan / count_params / count_flops / recalibrate_bn iz pruning_lib.
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

STUDENT_DIR = "/home/tomi/code/dipl/custom_models/student_2_m"
EXP1_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment"
for _p in (STUDENT_DIR, EXP1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_arch import StudentYOLO          # noqa: E402
import pruning_lib                          # noqa: E402


# --------------------------------------------------------------------------- #
# Ucitavanje pruned modela (rekonstrukcija + load_state_dict)
# --------------------------------------------------------------------------- #
def load_pruned_model(ckpt_path, device, num_classes=6, input_size=640):
    model = StudentYOLO(num_classes=num_classes, input_size=input_size).to(device)
    entries, by_name = pruning_lib.build_plan(model)
    ckpt = torch.load(ckpt_path, map_location=device)
    kept = {k: torch.tensor(v, dtype=torch.long) for k, v in ckpt["kept"].items()}
    pruning_lib.apply_pruning(model, entries, by_name, kept)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Pomocno: trenutne sirine + potrosaci (inverz in_sources)
# --------------------------------------------------------------------------- #
def current_widths(by_name):
    """name -> trenutni broj IZLAZNIH kanala (za sve entryje)."""
    return {n: e["conv"].out_channels for n, e in by_name.items()}


def growable_layers(entries):
    """Imena slojeva kojima SMIJEMO rasti izlaz (prune_out=True; ne head-finali)."""
    return [e["name"] for e in entries if e["prune_out"]]


def consumers_of(entries, name):
    """Lista entryja koji koriste `name` kao (dio) ulaza."""
    return [e for e in entries if name in e["in_sources"]]


# --------------------------------------------------------------------------- #
# Surgery: prosiri IZLAZ sloja `name` za k (function-preserving)
# --------------------------------------------------------------------------- #
def _new_consumer_weight(conv, in_sources, widths, grown_name, k, device, dtype):
    """Rebuild ulaznih kanala potrosaca: umetni k NUL-kolona na kraj svakog
    segmenta koji pripada `grown_name` (concat-aware; SPPF -> 4 segmenta)."""
    old_w = conv.weight.detach()                 # [out, in_old, kh, kw]
    out_c, _, kh, kw = old_w.shape
    # segmenti po redoslijedu in_sources
    new_in = 0
    segs = []                                    # (old_width, n_new)
    for src in in_sources:
        ow = 3 if src == "INPUT" else widths[src]
        n_new = k if src == grown_name else 0
        segs.append((ow, n_new)); new_in += ow + n_new

    new_w = torch.zeros(out_c, new_in, kh, kw, device=device, dtype=dtype)
    old_off = new_off = 0
    for ow, n_new in segs:
        new_w[:, new_off:new_off + ow] = old_w[:, old_off:old_off + ow]
        old_off += ow
        new_off += ow + n_new                    # n_new NUL-kolona ostaju 0
    return new_w


@torch.no_grad()
def grow_layer(model, name, k, init_filters=None):
    """Prosiri IZLAZ sloja `name` za k kanala (function-preserving).
    init_filters: tensor [k, in_L, kh, kw] (ulazne tezine novih filtera). Ako None,
    mali random (≈ ne utjece jer su downstream kolone 0; bitno tek za trening)."""
    was_training = model.training          # novi moduli moraju naslijediti mod (BN!)
    entries, by_name = pruning_lib.build_plan(model)
    widths = current_widths(by_name)
    E = by_name[name]
    assert E["prune_out"], f"{name} nije growable (prune_out=False)"
    conv = E["conv"]; bn = E["bn"]
    dev, dt = conv.weight.device, conv.weight.dtype
    old_out, in_c = conv.out_channels, conv.in_channels
    kh, kw = conv.kernel_size

    # --- 1) novi conv sloja L: out += k (ulaz nepromijenjen) ---
    if init_filters is None:
        init_filters = torch.randn(k, in_c, kh, kw, device=dev, dtype=dt) * 1e-3
    init_filters = init_filters.to(dev, dt)
    new_conv = nn.Conv2d(in_c, old_out + k, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=1, bias=conv.bias is not None
                         ).to(dev, dt)
    new_conv.weight[:old_out].copy_(conv.weight.detach())
    new_conv.weight[old_out:].copy_(init_filters)
    if conv.bias is not None:
        new_conv.bias[:old_out].copy_(conv.bias.detach())
        new_conv.bias[old_out:].zero_()

    # --- 2) novi BN: out += k (novi kanali: gamma=1, beta=0, rm=0, rv=1) ---
    new_bn = None
    if bn is not None:
        new_bn = nn.BatchNorm2d(old_out + k, eps=bn.eps, momentum=bn.momentum,
                                affine=bn.affine, track_running_stats=bn.track_running_stats
                                ).to(dev, dt)
        new_bn.weight[:old_out].copy_(bn.weight.detach()); new_bn.weight[old_out:].fill_(1.0)
        new_bn.bias[:old_out].copy_(bn.bias.detach());     new_bn.bias[old_out:].zero_()
        if bn.track_running_stats:
            new_bn.running_mean[:old_out].copy_(bn.running_mean.detach()); new_bn.running_mean[old_out:].zero_()
            new_bn.running_var[:old_out].copy_(bn.running_var.detach());   new_bn.running_var[old_out:].fill_(1.0)
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked.detach())
    E["replace"](new_conv, new_bn)

    # --- 3) potrosaci: umetni k NUL-kolona (function-preserving) ---
    for C in consumers_of(entries, name):
        if C["name"] == name:
            continue
        cconv = C["conv"]
        new_w = _new_consumer_weight(cconv, C["in_sources"], widths, name, k, dev, dt)
        nc = nn.Conv2d(new_w.shape[1], cconv.out_channels, kernel_size=cconv.kernel_size,
                       stride=cconv.stride, padding=cconv.padding, dilation=cconv.dilation,
                       groups=1, bias=cconv.bias is not None).to(dev, dt)
        nc.weight.copy_(new_w)
        if cconv.bias is not None:
            nc.bias.copy_(cconv.bias.detach())
        C["replace"](nc, C["bn"])      # BN potrosaca nepromijenjen (out isti)

    model.train(was_training)          # vrati mod (rekurzivno na sve, ukljucujuci nove module)
    return model


@torch.no_grad()
def grow_many(model, plan_k, init_fn=None):
    """plan_k: dict name->k (koliko filtera dodati svakom sloju). init_fn(model,name,k)
    -> tensor [k,in,kh,kw] ili None. Primjenjuje se redom (svaki grow_layer rebuilda plan)."""
    for name, k in plan_k.items():
        if k <= 0:
            continue
        init = init_fn(model, name, k) if init_fn is not None else None
        grow_layer(model, name, k, init)
    return model


# --------------------------------------------------------------------------- #
# Self-check: function-preservation + oblici + params/flops rastu
# --------------------------------------------------------------------------- #
def _selfcheck():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(EXP1_DIR) / "pruned_models" / "taylor.pt"
    model = load_pruned_model(ckpt, device)
    p0 = pruning_lib.count_params(model); f0 = pruning_lib.count_flops(model, device)
    x = torch.randn(2, 3, 640, 640, device=device)
    with torch.no_grad():
        out0 = model(x).clone()
    print(f"start: params={p0:,} GFLOPs={f0/1e9:.3f}")

    # rasti razne tipove: backbone, SPPF conv1 (x4 coupling), PAN fuse (concat), head.0
    for name, k in [("dark3.3", 8), ("sppf.conv1", 6), ("fuse_td_p4", 5),
                    ("lat_p5", 4), ("head_p3.0", 4), ("stem", 3)]:
        grow_layer(model, name, k)
        with torch.no_grad():
            out1 = model(x)
        diff = (out1 - out0).abs().max().item()
        print(f"  grow {name:12s} +{k:2d} -> out diff(max)={diff:.2e}  "
              f"{'OK' if diff < 1e-3 else 'FAIL!!'}")
        assert diff < 1e-3, f"NIJE function-preserving za {name} (diff={diff})"

    p1 = pruning_lib.count_params(model); f1 = pruning_lib.count_flops(model, device)
    with torch.no_grad():
        raw = model(x); boxes, probs = model.decode(raw)
    print(f"end:   params={p1:,} ({p1/p0*100:.1f}%) GFLOPs={f1/1e9:.3f} ({f1/f0*100:.1f}%)")
    print(f"forward OK raw={tuple(raw.shape)} boxes={tuple(boxes.shape)} probs={tuple(probs.shape)}")
    assert raw.shape == (2, 10, 8400)
    assert p1 > p0 and f1 > f0
    print("SELFCHECK OK (function-preserving + oblici + rast)")


if __name__ == "__main__":
    _selfcheck()
