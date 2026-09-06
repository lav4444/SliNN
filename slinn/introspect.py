
import math
import sys

import torch
import torch.nn as nn


def unfreeze_bn(module):
    from torchvision.ops.misc import FrozenBatchNorm2d
    for name, child in module.named_children():
        if isinstance(child, FrozenBatchNorm2d):
            bn = nn.BatchNorm2d(child.weight.shape[0], eps=float(getattr(child, "eps", 1e-5)))
            with torch.no_grad():
                bn.weight.copy_(child.weight); bn.bias.copy_(child.bias)
                bn.running_mean.copy_(child.running_mean); bn.running_var.copy_(child.running_var)
            setattr(module, name, bn)
        else:
            unfreeze_bn(child)
    return module


def load_eager(path, device, code_dirs=None):
    import os as _os
    for d in (code_dirs or []):
        if d not in sys.path:
            sys.path.insert(0, d)
    _own = _os.path.dirname(_os.path.abspath(str(path)))
    if _own not in sys.path and any(f.endswith(".py") for f in _os.listdir(_own) or []):
        sys.path.append(_own)
    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().to(device).float()
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().to(device).float()
    raise SystemExit(f"FORMAT NIJE PODRZAN: {type(obj).__name__} (treba cijeli eager modul).")


def load_any(spec, device, code_dirs=None):
    return load_eager(spec, device, code_dirs)


from classify import weighted_leaves                          # noqa: E402,F401


def layer_table(model, adapter, device):
    leaves = weighted_leaves(model)
    by_id = {id(m): (name, tn, w) for name, m, tn, w in leaves}
    rec, handles = [], []

    def mk(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and o:
                o = o[0]
            name, tn, w = by_id[id(mod)]
            ishape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            oshape = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() >= 3:
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:
                    out_f, in_f = w.shape
                    flops = 2 * out_f * in_f * (o.numel() // out_f if out_f else 0)
            rec.append({"name": name, "type": tn,
                        "role": "neuron" if w.dim() == 2 else "filter",
                        "units": int(w.shape[0]),
                        "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                        "gflops": flops / 1e9, "in": ishape, "out": oshape})
        return hook

    for name, m, tn, w in leaves:
        handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        adapter.forward(model, [torch.rand(3, 640, 640, device=device)])
    for h in handles:
        h.remove()
    return rec


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def gflops_total(model, adapter, device):
    return sum(r["gflops"] for r in layer_table(model, adapter, device))


ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Hardswish,
             nn.Hardsigmoid, nn.ELU, nn.Mish)


def activation_stats(model, adapter, device, loader, max_images=None, eps=1e-6, weak=0.01):
    leaf_id = {id(m): nm for nm, m, _, _ in weighted_leaves(model)}
    stats, handles = {}, []
    st = {"last": None}

    def leaf_hook(m, i, o):
        st["last"] = leaf_id[id(m)]

    def act_hook(m, i, o):
        leaf = st["last"]
        if leaf is None or not isinstance(o, torch.Tensor) or o.dim() not in (2, 4):
            return
        dims = (0, 2, 3) if o.dim() == 4 else (0,)
        nel = o.shape[0] * (o.shape[2] * o.shape[3] if o.dim() == 4 else 1)
        pos = (o > eps).sum(dim=dims).float(); mx = o.amax(dim=dims)
        s = stats.get(leaf)
        if s is None:
            stats[leaf] = {"pos": pos.clone(), "nel": nel, "max": mx.clone()}
        elif s["pos"].numel() == pos.numel():
            s["pos"] += pos; s["nel"] += nel; s["max"] = torch.maximum(s["max"], mx)

    for nm, m in model.named_modules():
        if id(m) in leaf_id:
            handles.append(m.register_forward_hook(leaf_hook))
        elif isinstance(m, ACT_TYPES):
            handles.append(m.register_forward_hook(act_hook))
    model.eval()
    nimg = 0
    for imgs, _ in loader:
        imgs = [im.to(device) for im in imgs]
        adapter.forward(model, imgs)
        nimg += len(imgs)
        if max_images and nimg >= max_images:
            break
    for h in handles:
        h.remove()

    out = {}
    for leaf, s in stats.items():
        afrac = s["pos"] / max(s["nel"], 1)
        dead_mask = s["max"] <= eps
        near_mask = (afrac < weak) & (s["max"] > eps)
        out[leaf] = {"dead": int(dead_mask.sum()), "near": int(near_mask.sum()), "C": s["pos"].numel(),
                     "dead_idx": dead_mask.nonzero(as_tuple=False).flatten().tolist(),
                     "near_idx": near_mask.nonzero(as_tuple=False).flatten().tolist()}
    return out, nimg
