
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except Exception:
    cv2 = None

sys.path.insert(0, str(Path(__file__).parent))
import analyze as A

ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Hardswish,
             nn.Hardsigmoid, nn.ELU, nn.Mish, nn.Tanh, nn.Sigmoid)


def image_files(data_dir):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in exts)


def _load_one(p, input_size):
    bgr = cv2.imread(str(p))
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (input_size, input_size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def iter_batches(files, input_size, bs):
    buf = []
    for p in files:
        t = _load_one(p, input_size)
        if t is None:
            continue
        buf.append(t)
        if len(buf) == bs:
            yield torch.stack(buf); buf = []
    if buf:
        yield torch.stack(buf)


@torch.no_grad()
def activation_stats(model, files, input_size, device, bs=16, eps=1e-6, weak_frac=0.01):
    stats = {}
    handles = []

    def mk(nm):
        def hook(mod, inp, out):
            o = out
            if not isinstance(o, torch.Tensor):
                return
            if o.dim() == 4:
                C = o.shape[1]
                mx = o.amax(dim=(0, 2, 3)).detach()
                sm = o.sum(dim=(0, 2, 3)).detach()
                pos = (o > eps).sum(dim=(0, 2, 3)).detach().float()
                nel = o.shape[0] * o.shape[2] * o.shape[3]
            elif o.dim() == 2:
                C = o.shape[1]
                mx = o.amax(dim=0).detach()
                sm = o.sum(dim=0).detach()
                pos = (o > eps).sum(dim=0).detach().float()
                nel = o.shape[0]
            else:
                return
            key = nm if (nm not in stats or stats[nm]["C"] == C) else f"{nm}#{C}"
            s = stats.get(key)
            if s is None:
                stats[key] = {"type": type(mod).__name__, "C": C, "max": mx,
                              "sum": sm, "pos": pos, "nel": nel}
            else:
                s["max"] = torch.maximum(s["max"], mx)
                s["sum"] += sm; s["pos"] += pos; s["nel"] += nel
        return hook

    for nm, m in model.named_modules():
        if isinstance(m, ACT_TYPES):
            handles.append(m.register_forward_hook(mk(nm)))
    model.eval()
    n_imgs = 0
    for batch in iter_batches(files, input_size, bs):
        model(batch.to(device))
        n_imgs += batch.shape[0]
    for h in handles:
        h.remove()

    layers = []
    tot_c = tot_dead = tot_weak = 0
    for nm, s in stats.items():
        mx = s["max"]; mean = (s["sum"] / max(s["nel"], 1)); afrac = s["pos"] / max(s["nel"], 1)
        dead = int((mx <= eps).sum())
        weak = int(((afrac < weak_frac) & (mx > eps)).sum())
        layers.append({"name": nm, "type": s["type"], "channels": int(s["C"]),
                       "dead": dead, "weak": weak,
                       "dead_pct": 100.0 * dead / s["C"],
                       "mean_act": float(mean.mean()), "max_act": float(mx.max()),
                       "active_frac": float(afrac.mean())})
        tot_c += s["C"]; tot_dead += dead; tot_weak += weak
    totals = {"channels": tot_c, "dead": tot_dead, "weak": tot_weak,
              "dead_pct": 100.0 * tot_dead / max(tot_c, 1),
              "weak_pct": 100.0 * tot_weak / max(tot_c, 1), "n_images": n_imgs}
    return layers, totals


def grad_importance(model, files, input_size, device, bs=8, max_images=512):
    leaves = A.weighted_leaves(model)
    accum = {nm: torch.zeros_like(w) for nm, m, t, w in leaves}
    mods = {nm: m for nm, m, t, w in leaves}
    meta = {nm: {"type": t, "channels": int(w.shape[0])} for nm, m, t, w in leaves}
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    nb = n_imgs = 0
    capped = files[:max_images] if max_images else files
    with torch.enable_grad():
        for batch in iter_batches(capped, input_size, bs):
            for p in model.parameters():
                p.grad = None
            out = model(batch.to(device))
            ts = [t for t in A._tensors(out, []) if t.is_floating_point() and t.requires_grad]
            if not ts:
                for p in model.parameters():
                    p.grad = None
                return None, meta, 0
            loss = sum(t.float().pow(2).mean() for t in ts)
            loss.backward()
            for nm in accum:
                g = mods[nm].weight.grad
                if g is not None:
                    accum[nm] += g.detach().abs()
            nb += 1; n_imgs += batch.shape[0]
    for p in model.parameters():
        p.grad = None
    imp = {nm: (accum[nm].flatten(1).sum(1) / max(nb, 1)).cpu() for nm in accum}
    return imp, meta, n_imgs


def pruning_growing_potential(imp, meta, low_pct=0.20):
    ranked = []
    for nm, v in imp.items():
        v = v.float()
        nrm = v / (v.norm() + 1e-12)
        for ch in range(len(v)):
            ranked.append(float(nrm[ch]))
    ranked_sorted = sorted(ranked)
    n = len(ranked_sorted)
    cut = ranked_sorted[int(low_pct * n)] if n else 0.0

    desc = sorted(ranked, reverse=True)
    tot = sum(desc) or 1e-9
    def top_share(p):
        k = max(1, int(p * n))
        return 100.0 * sum(desc[:k]) / tot
    concentration = {"top10": top_share(0.10), "top25": top_share(0.25), "top50": top_share(0.50)}

    prune_layers, grow_layers = [], []
    for nm, v in imp.items():
        v = v.float()
        nrm = v / (v.norm() + 1e-12)
        low = int((nrm < cut).sum())
        prune_layers.append({"name": nm, "type": meta[nm]["type"], "channels": meta[nm]["channels"],
                             "low_imp": low, "low_pct": 100.0 * low / max(len(v), 1),
                             "mean_imp": float(v.mean())})
        grow_layers.append({"name": nm, "type": meta[nm]["type"], "channels": meta[nm]["channels"],
                            "benefit": float(v.mean())})
    bmax = max((g["benefit"] for g in grow_layers), default=1e-9) or 1e-9
    for g in grow_layers:
        g["benefit_norm"] = g["benefit"] / bmax
    prune_layers.sort(key=lambda r: (-r["low_pct"], r["mean_imp"]))
    grow_layers.sort(key=lambda r: -r["benefit"])

    total_low = sum(p["low_imp"] for p in prune_layers)
    total_ch = sum(p["channels"] for p in prune_layers)
    return {
        "low_pct_threshold": low_pct,
        "global_low_frac": 100.0 * total_low / max(total_ch, 1),
        "concentration": concentration,
        "prune_layers": prune_layers,
        "grow_layers": grow_layers,
    }


def run_deep(model_path, data_dir, input_size, code_dirs, device,
             grad_max_images=512, bs_act=16, bs_grad=8, eps=1e-6,
             weak_frac=0.01, low_pct=0.20):
    if cv2 is None:
        raise SystemExit("deep_analyze treba cv2 (opencv-python).")
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    model = A.load_model(model_path, dev, code_dirs)
    files = image_files(data_dir)
    if not files:
        raise SystemExit(f"Nema slika u {data_dir}")

    print(f"[deep] aktivacijske statistike preko {len(files)} slika...")
    act_layers, act_tot = activation_stats(model, files, input_size, dev, bs_act, eps, weak_frac)
    print(f"[deep] gradijentna vaznost (cap {grad_max_images} slika)...")
    imp, meta, n_grad = grad_importance(model, files, input_size, dev, bs_grad, grad_max_images)
    if imp is None:
        print("[deep] gradijentni signal nedostupan (izlaz odvojen od grafa) -> "
              "pruning/growing potencijal preskocen, mrtve jedinice ostaju.")
        pg = None
    else:
        pg = pruning_growing_potential(imp, meta, low_pct)

    return {
        "n_images_act": act_tot["n_images"],
        "n_images_grad": n_grad,
        "grad_available": imp is not None,
        "eps": eps, "weak_frac": weak_frac,
        "act_layers": act_layers, "act_totals": act_tot,
        "potential": pg,
    }


if __name__ == "__main__":
    if not A.MODEL_PATH:
        sys.exit("Postavi MODEL_PATH u analyze.py.")
    import json
    d = run_deep(A.MODEL_PATH, A.DATA_DIR, A.INPUT_SIZE, A.CODE_DIRS or None, A.DEVICE)
    print(json.dumps({k: v for k, v in d.items() if k not in ("act_layers", "potential")}, indent=2))
    print("dead total:", d["act_totals"]["dead"], "/", d["act_totals"]["channels"],
          f"({d['act_totals']['dead_pct']:.1f}%)")
    print("grow top-3:", [g["name"] for g in d["potential"]["grow_layers"][:3]])
    print("prune top-3:", [p["name"] for p in d["potential"]["prune_layers"][:3]])
