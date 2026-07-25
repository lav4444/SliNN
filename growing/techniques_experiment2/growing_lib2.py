"""
growing_lib2.py — jezgra za NETWORK GROWING (conv FILTERI + dense NEURONI) na SchoolCNN.

Inverz exp2 pruninga (pruning_lib2). SchoolCNN je cisti sekvencijalni lanac:
    conv1 -> ... -> conv5 -> [FLATTEN] -> fc1 -> fc2 -> fc3 (izlaz, ne raste)

Kad prosirimo IZLAZ sloja L za k jedinica, moramo konzistentno prosiriti ULAZ
njegovih potrosaca. Jedina netrivijalna veza je FLATTEN: 1 dodani conv5 kanal ->
+flatten_hw (=H*W) ulaznih znacajki u fc1 (channel-major; novi kanal se dodaje na
KRAJ pa njegov blok ide na kraj flatten-vektora -> umetanje nul-bloka na kraj).

FUNCTION-PRESERVING: nove jedinice sloja L dobiju zadani init, a NOVE ulazne
kolone potrosaca su 0 -> izlaz mreze se NE mijenja dok trening ne pocne koristiti
nove jedinice. (validira se u _selfcheck: forward identican prije/poslije.)

Reuse: build_plan / count_params / count_flops / recalibrate_bn iz pruning_lib2.
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

EXP2_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment2"
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

from model_cnn import SchoolCNN, INPUT_SIZE      # noqa: E402
import pruning_lib2                              # noqa: E402


# --------------------------------------------------------------------------- #
# Ucitavanje pruned modela (rekonstrukcija + load_state_dict)
# --------------------------------------------------------------------------- #
def load_pruned_model(ckpt_path, device, num_classes=6, input_size=INPUT_SIZE):
    model = SchoolCNN(num_classes=num_classes, input_size=input_size).to(device)
    entries, by_name = pruning_lib2.build_plan(model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    kept = {k: torch.tensor(v, dtype=torch.long) for k, v in ckpt["kept"].items()}
    pruning_lib2.apply_pruning(model, entries, by_name, kept)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Pomocno: trenutne sirine + potrosaci
# --------------------------------------------------------------------------- #
def current_widths(by_name):
    """name -> trenutni broj IZLAZNIH jedinica (conv: kanali, linear: neuroni)."""
    out = {}
    for n, e in by_name.items():
        m = e["module"]
        out[n] = m.out_channels if e["kind"] == "conv" else m.out_features
    return out


def growable_layers(entries):
    """Imena slojeva kojima SMIJEMO rasti izlaz (prune_out=True; ne fc3 izlaz)."""
    return [e["name"] for e in entries if e["prune_out"]]


def consumers_of(entries, name):
    """Lista entryja koji koriste `name` kao (dio) ulaza."""
    return [e for e in entries if any(src == name for src, _ in e["in_sources"])]


# --------------------------------------------------------------------------- #
# Surgery: prosiri IZLAZ sloja `name` za k (function-preserving)
# --------------------------------------------------------------------------- #
def _new_consumer_weight(module, in_sources, widths, grown_name, k, kind, device, dtype):
    """Rebuild ulaznih jedinica potrosaca: umetni k*block NUL-kolona na kraj
    segmenta koji pripada `grown_name`. block=1 za conv->conv / fc->fc, a za
    conv5->fc1 block=flatten_hw (channel-major flatten)."""
    old_w = module.weight.detach()                     # conv:[out,in,kh,kw] | lin:[out,in]
    out_c = old_w.shape[0]
    segs = []                                          # (old_seg_width, n_new)
    new_in = 0
    for src, block in in_sources:
        ow = (3 if src == "INPUT" else widths[src]) * block
        n_new = k * block if src == grown_name else 0
        segs.append((ow, n_new)); new_in += ow + n_new

    if kind == "conv":
        kh, kw = old_w.shape[2], old_w.shape[3]
        new_w = torch.zeros(out_c, new_in, kh, kw, device=device, dtype=dtype)
    else:
        new_w = torch.zeros(out_c, new_in, device=device, dtype=dtype)

    old_off = new_off = 0
    for ow, n_new in segs:
        new_w[:, new_off:new_off + ow] = old_w[:, old_off:old_off + ow]
        old_off += ow
        new_off += ow + n_new                          # n_new NUL-kolona ostaju 0
    return new_w


@torch.no_grad()
def grow_layer(model, name, k, init_filters=None):
    """Prosiri IZLAZ sloja `name` za k jedinica (function-preserving).
    init_filters: tensor oblika kao tezine novih jedinica (conv [k,in,kh,kw] ili
    linear [k,in]). Ako None, mali random (≈ ne utjece jer su downstream kolone 0)."""
    was_training = model.training          # novi moduli moraju naslijediti mod (BN!)
    entries, by_name = pruning_lib2.build_plan(model)
    widths = current_widths(by_name)
    E = by_name[name]
    assert E["prune_out"], f"{name} nije growable (prune_out=False)"
    m = E["module"]
    dev, dt = m.weight.device, m.weight.dtype

    # --- 1) prosiri sam sloj L: out += k (ulaz nepromijenjen) ---
    if E["kind"] == "conv":
        old_out, in_c = m.out_channels, m.in_channels
        kh, kw = m.kernel_size
        if init_filters is None:
            init_filters = torch.randn(k, in_c, kh, kw, device=dev, dtype=dt) * 1e-3
        init_filters = init_filters.to(dev, dt)
        new_m = nn.Conv2d(in_c, old_out + k, kernel_size=m.kernel_size, stride=m.stride,
                          padding=m.padding, dilation=m.dilation, groups=1,
                          bias=m.bias is not None).to(dev, dt)
        new_m.weight[:old_out].copy_(m.weight.detach())
        new_m.weight[old_out:].copy_(init_filters)
        if m.bias is not None:
            new_m.bias[:old_out].copy_(m.bias.detach()); new_m.bias[old_out:].zero_()

        new_bn = None
        bn = E["bn"]
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
        E["replace"](new_m, new_bn)
    else:  # linear
        old_out, in_f = m.out_features, m.in_features
        if init_filters is None:
            init_filters = torch.randn(k, in_f, device=dev, dtype=dt) * 1e-3
        init_filters = init_filters.to(dev, dt)
        new_m = nn.Linear(in_f, old_out + k, bias=m.bias is not None).to(dev, dt)
        new_m.weight[:old_out].copy_(m.weight.detach())
        new_m.weight[old_out:].copy_(init_filters)
        if m.bias is not None:
            new_m.bias[:old_out].copy_(m.bias.detach()); new_m.bias[old_out:].zero_()
        E["replace"](new_m, None)

    # --- 2) potrosaci: umetni k*block NUL-kolona (function-preserving) ---
    for C in consumers_of(entries, name):
        if C["name"] == name:
            continue
        cm = C["module"]
        new_w = _new_consumer_weight(cm, C["in_sources"], widths, name, k, C["kind"], dev, dt)
        if C["kind"] == "conv":
            nc = nn.Conv2d(new_w.shape[1], cm.out_channels, kernel_size=cm.kernel_size,
                           stride=cm.stride, padding=cm.padding, dilation=cm.dilation,
                           groups=1, bias=cm.bias is not None).to(dev, dt)
        else:
            nc = nn.Linear(new_w.shape[1], cm.out_features, bias=cm.bias is not None).to(dev, dt)
        nc.weight.copy_(new_w)
        if cm.bias is not None:
            nc.bias.copy_(cm.bias.detach())
        C["replace"](nc, C["bn"])      # BN potrosaca nepromijenjen (out isti); linear ignorira

    model.train(was_training)          # vrati mod (rekurzivno, ukljucujuci nove module)
    return model


@torch.no_grad()
def grow_many(model, plan_k, init_fn=None):
    """plan_k: dict name->k. init_fn(model,name,k)->tensor ili None. Primjenjuje
    se redom (svaki grow_layer rebuilda plan)."""
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
    ckpt = Path(EXP2_DIR) / "pruned_models" / "taylor.pt"
    model = load_pruned_model(ckpt, device)
    p0 = pruning_lib2.count_params(model); f0 = pruning_lib2.count_flops(model, device, INPUT_SIZE)
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.no_grad():
        out0 = model(x).clone()
    print(f"start: params={p0:,} GFLOPs={f0/1e9:.4f}")

    # rasti razne tipove: rani conv, conv5 (flatten coupling!), fc1, fc2
    for name, k in [("conv1", 4), ("conv3", 6), ("conv5", 5), ("fc1", 8), ("fc2", 4)]:
        grow_layer(model, name, k)
        with torch.no_grad():
            out1 = model(x)
        diff = (out1 - out0).abs().max().item()
        print(f"  grow {name:6s} +{k:2d} -> out diff(max)={diff:.2e}  "
              f"{'OK' if diff < 1e-3 else 'FAIL!!'}")
        assert diff < 1e-3, f"NIJE function-preserving za {name} (diff={diff})"

    p1 = pruning_lib2.count_params(model); f1 = pruning_lib2.count_flops(model, device, INPUT_SIZE)
    with torch.no_grad():
        y = model(x)
    print(f"end:   params={p1:,} ({p1/p0*100:.1f}%) GFLOPs={f1/1e9:.4f} ({f1/f0*100:.1f}%)")
    print(f"forward OK out={tuple(y.shape)}")
    assert y.shape == (2, 6)
    assert p1 > p0 and f1 > f0
    print("SELFCHECK OK (function-preserving + oblici + rast)")


if __name__ == "__main__":
    _selfcheck()
