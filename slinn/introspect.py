"""slinn/introspect.py — genericka introspekcija modela (preseljeno iz morphology/analysis.py).

TASK-AGNOSTICNO: layer-tablica, census aktivnosti, brojanje parametara, GFLOPs, eager load.
Detekcijski adapteri/decode/mAP NISU ovdje -- oni su plug (slinn/plugins/detection/).
Preseljeno BEZ IZMJENA osim sto `load_any` gubi "fasterrcnn" string-precac (zoo pogodnost).
"""

import math
import sys

import torch
import torch.nn as nn

# NAMJERNO bez `import config`: ovaj modul ne treba NIJEDNU konstantu — cista introspekcija.


def unfreeze_bn(module):
    """FrozenBatchNorm2d -> nn.BatchNorm2d (kopiraj statistike). Kopirano iz common3."""
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
    """Ucitaj CIJELI eager modul (torch.save(model)); hvata i ultralytics dict['model']. Iz analyze.py.

    Full-eager checkpoint nosi samo tezine i REFERENCU na klasu — modul koji je definira mora biti
    uvoziv, inace unpickle padne s `ModuleNotFoundError` ([[save-models-full-eager]]). `code_dirs` je
    izricit odgovor na to, ali mapa u kojoj .pt lezi je toliko cest odgovor da se dodaje SAMA:
    zoo konvencija je `baseline_models/<ime>/model.pt` uz `<ime>/model_*.py` ([[zoo-build-conventions]]).
    To nije znanje o modelu — to je konvencija putanje, vrijedi za bilo koji eager checkpoint.

    Redoslijed je bitan: izricit `code_dirs` ide na POCETAK sys.patha (korisnik ga je trazio),
    a auto-mapa na KRAJ. Zoo mape sadrze i generickа imena (`data.py`, `build.py`) pa bi na pocetku
    mogle zasjeniti module jezgre."""
    import os as _os
    for d in (code_dirs or []):
        if d not in sys.path:
            sys.path.insert(0, d)
    _own = _os.path.dirname(_os.path.abspath(str(path)))          # mapa samog .pt-a
    if _own not in sys.path and any(f.endswith(".py") for f in _os.listdir(_own) or []):
        sys.path.append(_own)                                     # KRAJ: samo fallback, ne zasjenjuje jezgru
    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().to(device).float()
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().to(device).float()
    raise SystemExit(f"FORMAT NIJE PODRZAN: {type(obj).__name__} (treba cijeli eager modul).")


def load_any(spec, device, code_dirs=None):
    """Putanja .pt -> load_eager. Vrati model (adapter se bira posebno).

    (Morphology je ovdje imao string-precac spec=='fasterrcnn' -> build_fasterrcnn(); to je bila
    pogodnost za testni zoo i jedina detekcijska referenca u genericnom loaderu -> izbacena.)"""
    return load_eager(spec, device, code_dirs)


# weighted_leaves: JEDNA definicija za cijeli slinn, u classify.py (dim 2/3/4 — vidi Conv1d).
# Morphologyjeva verzija (dim 2/4) je OBRISANA: bez dim-3 je citav 1D lanac (M5/audio/sekvence)
# nevidljiv grafu ovisnosti -> nema tapova ni terminala -> jezgra nije agnosticna.
# (Do 6.4 je isto vrijedilo u izvodenju, ali preko monkeypatcha u engine.install_sizing_shims;
#  sada je eksplicitno. Re-export drzi `A.weighted_leaves` pozivna mjesta nepromijenjenima.)
from classify import weighted_leaves                          # noqa: E402,F401


def layer_table(model, adapter, device):
    """Forward s hookovima -> per-layer zapis (role/units/params/GFLOPs/in-out shape),
    u FORWARD redoslijedu. Generalizira: conv (dim>=3) = filteri, linear (dim==2) = neuroni."""
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
                if w.dim() >= 3:                                  # conv (1d/2d/3d)
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:                                # linear
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
        adapter.forward(model, [torch.rand(3, 640, 640, device=device)])   # adapter rjesava resize/batch
    for h in handles:
        h.remove()
    return rec


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def gflops_total(model, adapter, device):
    """Suma GFLOPs po svim weighted slojevima (per-layer iz layer_table)."""
    return sum(r["gflops"] for r in layer_table(model, adapter, device))


# --------------------------------------------------------------------------- #
# Blok 3: aktivnost (dead / near-dead PO LOKACIJI) — post-aktivacijski, cijeli skup
# --------------------------------------------------------------------------- #
ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Hardswish,
             nn.Hardsigmoid, nn.ELU, nn.Mish)


def activation_stats(model, adapter, device, loader, max_images=None, eps=1e-6, weak=0.01):
    """Post-aktivacijska aktivnost PO LOKACIJI preko skupa. dead = max<=eps (nikad ne opali);
    near-dead = opali na <1% lokacija. Statistike se kljuce po PRETHODNOM conv/linear (forward redoslijed,
    st['last']); time radi i kad je aktivacija DIJELJEN modul (ultralytics: jedna SiLU za sve Conv-ove)."""
    leaf_id = {id(m): nm for nm, m, _, _ in weighted_leaves(model)}
    stats, handles = {}, []
    st = {"last": None}

    def leaf_hook(m, i, o):
        st["last"] = leaf_id[id(m)]                          # prati zadnji conv/linear (svaki forward)

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
        elif s["pos"].numel() == pos.numel():                # ista jedinica -> akumuliraj
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
