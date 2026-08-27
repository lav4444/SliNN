"""cost_profile_yolo26n.py — profil cijene reza po dubini mreze (slika za rad, 3.3.5).

Za svaki PRUNABLE sloj uzima SPREGNUTU cijenu uklanjanja JEDNOG izlaznog kanala
(morph.coupled_unit_cost) i crta tri krivulje po dubini:

    udio oslobodjenih GFLOPs   — visok na pocetku mreze, pada
    udio oslobodjenih params   — nizak na pocetku, raste prema kraju
    kombinirana cijena         — 0.60*GFLOPs_udio + 0.40*params_udio (PHASE2_COST_FLOPS_W)

Udjeli su normalizirani totalom preko prunable slojeva -> bezdimenzijski, usporedivi.

Bez argumenata: sve se podesava konstantama ispod, __main__ vrti cijeli posao.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =========================== KONFIGURACIJA =========================== #
SLINN_DIR = "/home/tomi/code/dipl/slinn"
MODEL_PATH = "/home/tomi/code/dipl/yolo26n.pt"
OUT_PNG = ("/home/tomi/latex_projects/DiplomskiTM/Quickstart Examples/"
           "Figures/cost_profile_yolo26n.png")

COST_FLOPS_W = 0.60          # isto kao settings.PHASE2_COST_FLOPS_W
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIOLET = "#7C5EB8"           # pviol iz rada
RED = "#C03E3E"              # pcut
LOGY = True                  # log os y (raspon je nekoliko redova velicine)
SMOOTH_W = 11                # prozor kliznog medijana za trend-krivulje (neparan)


sys.path.insert(0, SLINN_DIR)


def load_model(path):
    """Cijeli eager modul; hvata i ultralytics ckpt dict['model']. (kopija iz slinn/init.py)"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().float()
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().float()
    raise SystemExit(f"format nije podrzan: {type(obj).__name__}")


def collect():
    """Vrati listu redaka [{name, depth, flops, params}] za prunable slojeve, po dubini."""
    import classify as CL
    import position as P
    import morph as C
    import engine as E                    # VAZNO: instalira sizing-shimove za layer_table
    _ = E

    model = load_model(MODEL_PATH).to(DEVICE)
    adapter = CL.probe_adapter(model, DEVICE)
    print(f"[adapter] imgsz={adapter.imgsz} call={adapter._call} mode={adapter._mode}")

    pos, meta = P.positional(model, adapter, DEVICE)
    prunable = {n for n, v in pos.items() if v["morph"]}
    print(f"[klasifikacija] prunable={len(prunable)}  tapovi={len(meta['taps'])}  "
          f"terminalni={meta['terminal']}  kd_mode={meta['kd_mode']}")

    flops_per, params_per, units = C.coupled_unit_cost(model, adapter, DEVICE, prunable)

    order = {nm: i for i, (nm, _m) in enumerate(model.named_modules())}
    rows = [{"name": nm, "depth": order.get(nm, 0),
             "flops": float(flops_per[nm]), "params": float(params_per[nm])}
            for nm in flops_per if nm in prunable]
    rows.sort(key=lambda r: r["depth"])
    return rows


def _trend(y, w=SMOOTH_W):
    """Klizni medijan (neparni prozor), rubovi se skracuju simetricno."""
    n = len(y)
    out = np.empty(n)
    h = w // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = np.median(y[lo:hi])
    return out


def plot(rows):
    f = np.array([r["flops"] for r in rows], dtype=float)
    p = np.array([r["params"] for r in rows], dtype=float)
    sf = f / max(f.sum(), 1e-12)
    sp = p / max(p.sum(), 1e-12)
    cost = COST_FLOPS_W * sf + (1.0 - COST_FLOPS_W) * sp
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # sirovo mjerenje — tanko i blijedo, da se vidi da nije izmisljeno
    ax.plot(x, sf, "-", lw=0.7, color=RED, alpha=0.28)
    ax.plot(x, sp, "-", lw=0.7, color=VIOLET, alpha=0.28)
    # trend — klizni medijan
    ax.plot(x, _trend(sf), "-", lw=2.2, color=RED, label="udio oslobođenih GFLOPs")
    ax.plot(x, _trend(sp), "-", lw=2.2, color=VIOLET, label="udio oslobođenih parametara")
    ax.plot(x, _trend(cost), "--", lw=1.8, color="#333333",
            label=f"kombinirana cijena ({COST_FLOPS_W:.0%} / {1 - COST_FLOPS_W:.0%})")

    # granice okosnica / vrat / glava iz imena "model.N."
    blk = []
    for r in rows:
        parts = r["name"].split(".")
        blk.append(int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1)
    blk = np.array(blk)
    for lo, hi, txt in ((0, 10, "okosnica"), (11, 22, "vrat"), (23, 99, "glava")):
        idx = np.where((blk >= lo) & (blk <= hi))[0]
        if not len(idx):
            continue
        a, b = idx[0] - 0.5, idx[-1] + 0.5
        if lo:
            ax.axvline(a, color="black", lw=0.6, ls=":", alpha=0.5)
        ax.text((a + b) / 2, 0.965, txt, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color="black", alpha=0.65)

    if LOGY:
        ax.set_yscale("log")
    ax.set_xlabel("redni broj rezivog sloja (po dubini mreže)")
    ax.set_ylabel("udio cijene po uklonjenom kanalu")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"[slika] {OUT_PNG}")

    # kratka tablica u konzolu: prvih i zadnjih 5 slojeva
    print(f"\n{'#':>3}  {'sloj':<44}{'GFLOPs udio':>12}{'params udio':>12}{'cijena':>10}")
    for i in list(range(min(5, len(rows)))) + list(range(max(0, len(rows) - 5), len(rows))):
        r = rows[i]
        print(f"{i:>3}  {r['name'][:44]:<44}{sf[i]:>12.5f}{sp[i]:>12.5f}{cost[i]:>10.5f}")


if __name__ == "__main__":
    rows = collect()
    print(f"[podaci] {len(rows)} rezivih slojeva")
    plot(rows)
