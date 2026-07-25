"""
plot_pareto.py — čita pareto_run.csv (+ meta json) i crta krivulju sweepa:
  GLAVNE osi = apsolutne vrijednosti (x = GFLOPs, y = kvaliteta); SEKUNDARNE (suprotni rub) = postotci.

X-os je OBRNUTA (max/100% lijevo -> jaka kompresija desno). Dijagonala (gore-lijevo -> dolje-desno)
označava NEUTRALAN kompromis (kvaliteta pada razmjerno složenosti): iznad nje (zeleno) = bolje od
neutralnog (više kvalitete po jedinici složenosti), ispod (crveno) = gore. Sprema PNG. BEZ CLI argumenata.
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")              # headless (bez X servera)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV  = os.path.join(_HERE, "yolo26n_pareto_run.csv")
IN_META = os.path.join(_HERE, "yolo26n_pareto_run_meta.json")
OUT_PNG = os.path.join(_HERE, "yolo26n_pareto_front.png")

SWEEP_COLOR = "#15315e"            # tamno plava

# metrika kvalitete za Y-os: "val_map" | "val_acc" | "val_f1" (SchoolCNN ima sve tri u CSV-u;
# yolo ima samo val_map). Ako odabrani stupac ne postoji u CSV-u -> fallback na "quality".
QUALITY_COL = "val_map"


def _qlabel(col, meta):
    task = (meta.get("task") or "").lower()
    if col == "val_acc":
        return "val točnost (macro)"
    if col == "val_f1":
        return "val F1 (macro)"
    return "macro mAP" if "classif" in task else "val mAP@50:95"   # val_map / quality


def _load():
    with open(IN_CSV) as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        fields = set(rd.fieldnames or [])
    col = QUALITY_COL if QUALITY_COL in fields else "quality"
    pts = [{"step": int(r["step"]), "gflops": float(r["gflops"]),
            "pct": float(r["gflops_pct"]), "q": float(r[col])} for r in rows]
    meta = json.load(open(IN_META)) if os.path.exists(IN_META) else {}
    return pts, meta, col


def _pareto_front(pts):
    """Nedominirani skup za (min GFLOPs, max kvaliteta): točka nije dominirana ako ne postoji
    druga s <= GFLOPs I >= kvaliteta (uz barem jedno strogo). Vrati podskup, sortiran po GFLOPs."""
    nd = [p for p in pts if not any(
        (q["gflops"] <= p["gflops"] and q["q"] >= p["q"]) and (q["gflops"] < p["gflops"] or q["q"] > p["q"])
        for q in pts)]
    return sorted(nd, key=lambda d: d["gflops"])


def main():
    pts, meta, col = _load()
    if not pts:
        print(f"Nema podataka u {IN_CSV}"); return
    front = _pareto_front(pts)                               # crtamo SAMO nedominiranu frontu

    # baseline = vrijednost ODABRANE metrike u koraku 0 (original) -> točna normalizacija po stupcu
    z = [p for p in pts if p["step"] == 0]
    base_q = (z[0]["q"] if z else max((p["q"] for p in pts), default=1.0)) or 1.0
    base_g = meta.get("baseline_gflops") or max((p["gflops"] for p in pts), default=1.0)
    qlabel = _qlabel(col, meta)

    plt.rcParams.update({                      # veći naslov/osi/brojevi; legenda ostaje ista
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 10,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # --- overlay trokuti razdvojeni dijagonalom neutralnog kompromisa (0,0)->(base_g,base_q) ---
    ax.fill([0, base_g, 0], [0, base_q, base_q], color="green", alpha=0.2, lw=0, zorder=0)   # gornji: bolje od neutralnog
    ax.fill([0, base_g, base_g], [0, base_q, 0], color="red", alpha=0.2, lw=0, zorder=0)     # donji: gore od neutralnog
    ax.plot([0, base_g], [0, base_q], ls="--", color="gray", lw=1.5,
            label="neutralan kompromis", zorder=1)

    # --- Pareto fronta (jedna serija, tamno plava) ---
    ax.plot([p["gflops"] for p in front], [p["q"] for p in front],
            "-o", color=SWEEP_COLOR, ms=5, lw=1.8, label="Pareto fronta", zorder=3)
    ax.scatter([base_g], [base_q], marker="*", s=260, color="#2c7d3f",
               edgecolors="black", linewidths=0.6, label="original (baseline)", zorder=5)

    # GLAVNE osi: apsolutne; raspon 0..baseline; x obrnuta (max lijevo)
    ax.set_xlabel("Složenost — GFLOPs")
    ax.set_ylabel(f"Kvaliteta — {qlabel}")
    ax.set_xlim(base_g, 0)
    ax.set_ylim(0, base_q)
    title = "Kompromis kvalitete i složenosti"
    if meta.get("model"):
        title += f"  ({meta['model']})"
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.9)

    # SEKUNDARNE osi (suprotni rub): postotci 0–100%
    secx = ax.secondary_xaxis("top", functions=(lambda g: 100.0 * g / base_g,
                                                lambda pc: pc * base_g / 100.0))
    secx.set_xlabel("% originalnih GFLOPs")
    secx.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    secy = ax.secondary_yaxis("right", functions=(lambda q: 100.0 * q / base_q,
                                                  lambda pc: pc * base_q / 100.0))
    secy.set_ylabel("% baseline kvalitete")
    secy.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[save] {OUT_PNG}  ({len(front)} točaka na fronti / {len(pts)} ukupno)")


if __name__ == "__main__":
    main()
