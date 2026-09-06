
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("yolo26n",   "yolo26n — detekcija (slab model, težak zadatak)", "#d6452c"),
    ("schoolcnn", "SchoolCNN — klasifikacija (velik model, lagan zadatak)", "#2c6fd6"),
]
OUT_PNG = os.path.join(_HERE, "pareto_compare.png")


def _load(prefix):
    csv_path = os.path.join(_HERE, f"{prefix}_pareto_run.csv")
    meta_path = os.path.join(_HERE, f"{prefix}_pareto_run_meta.json")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pts = sorted(({"gflops": float(r["gflops"]), "pct": float(r["gflops_pct"]),
                   "q": float(r["quality"]), "step": int(r["step"])} for r in rows),
                 key=lambda d: d["step"])
    base_q = meta.get("baseline_quality") or (pts[0]["q"] if pts else 1.0)
    return {"pts": pts, "meta": meta, "base_q": base_q or 1.0}


def _pareto_front(pts):
    nd = [p for p in pts if not any(
        (q["pct"] <= p["pct"] and q["q"] >= p["q"]) and (q["pct"] < p["pct"] or q["q"] > p["q"])
        for q in pts)]
    return sorted(nd, key=lambda d: d["pct"])


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    for prefix, label, color in RUNS:
        d = _load(prefix)
        if d is None:
            print(f"  preskačem '{prefix}' (nema {prefix}_pareto_run.csv)")
            continue
        pts, base_q = d["pts"], d["base_q"]
        xs = [p["pct"] for p in pts]
        ys = [p["q"] / base_q for p in pts]
        ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.4, alpha=0.85, label=label)
        front = _pareto_front(pts)
        ax.plot([p["pct"] for p in front], [p["q"] / base_q for p in front],
                "-", color=color, lw=2.6, alpha=1.0)
        plotted += 1

    if not plotted:
        print("Nema niti jednog runa za crtanje."); return

    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Složenost — % originalnih GFLOPs")
    ax.set_ylabel("Kvaliteta relativna na originalni model (baseline = 1.0)")
    ax.set_title("Oblik Pareto krivulje: utjecaj prezatrpanosti modela i težine zadatka")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[save] {OUT_PNG}  ({plotted} krivulje)")


if __name__ == "__main__":
    main()
