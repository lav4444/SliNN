import math
import random

BETA = 0.2
M = 32
N_LAYERS = 5
SEED = 0
OUT_PNG = "align_score_demo_2.png"


def align_scores(width, m=M):
    r = width % m
    g = (-width) % m
    prune = (1 - r / m) if r != 0 else 0.0
    grow = (1 - g / m) if g != 0 else 0.0
    return prune, grow


def main():
    if SEED is not None:
        random.seed(SEED)
    widths = [random.randint(1, 100) for _ in range(N_LAYERS)]

    rows = []
    for i, w in enumerate(widths):
        ps, gs = align_scores(w)
        rows.append({"layer": i, "width": w, "tiles": math.ceil(w / M),
                     "ps": ps, "gs": gs, "pf": 1 - BETA * ps, "gf": 1 + BETA * gs})

    print(f"\n  M={M}  BETA={BETA}   (split: prune_score visok = tik IZNAD xM; grow_score visok = tik ISPOD xM)\n")
    print(f"  {'sloj':>4} {'width':>6} {'tiles':>6} {'prune_score':>12} {'grow_score':>11} {'align_PRUNE':>12} {'align_GROW':>11}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['layer']:>4} {r['width']:>6} {r['tiles']:>6} "
              f"{r['ps']:>12.3f} {r['gs']:>11.3f} {r['pf']:>12.3f} {r['gf']:>11.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = [r["layer"] for r in rows]
    labels = [f"L{r['layer']}\n({r['width']})" for r in rows]

    ax = axes[0, 0]
    ax.bar(x, [r["width"] for r in rows], color="tab:blue")
    for k in range(M, 101, M):
        ax.axhline(k, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_title(f"Broj kanala po sloju (sive crte = visekratnici {M})")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("width")

    ax = axes[0, 1]
    bw = 0.38
    ax.bar([xx - bw / 2 for xx in x], [r["ps"] for r in rows], width=bw, label="prune_score", color="tab:red")
    ax.bar([xx + bw / 2 for xx in x], [r["gs"] for r in rows], width=bw, label="grow_score", color="tab:green")
    ax.set_title("Direkcioni align scoreovi (0 = poravnat)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar([xx - bw / 2 for xx in x], [r["pf"] for r in rows], width=bw, label="align_PRUNE (1-B*ps)", color="tab:red")
    ax.bar([xx + bw / 2 for xx in x], [r["gf"] for r in rows], width=bw, label="align_GROW (1+B*gs)", color="tab:green")
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title("Alignment faktori po sloju (crta = 1.0 neutralno)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ws = list(range(1, 101))
    ps_curve = [align_scores(w)[0] for w in ws]
    gs_curve = [align_scores(w)[1] for w in ws]
    ax.plot(ws, ps_curve, color="tab:red", lw=1.3, label="prune_score")
    ax.plot(ws, gs_curve, color="tab:green", lw=1.3, label="grow_score")
    ax.scatter([r["width"] for r in rows], [r["ps"] for r in rows], color="tab:red", zorder=5, s=35)
    ax.scatter([r["width"] for r in rows], [r["gs"] for r in rows], color="tab:green", zorder=5, s=35)
    for r in rows:
        ax.annotate(f"L{r['layer']}", (r["width"], max(r["ps"], r["gs"])),
                    textcoords="offset points", xytext=(0, 6), fontsize=8)
    for k in range(M, 101, M):
        ax.axvline(k, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_title(f"Zrcalni sawtooth: prune vrh TIK NAKON xM, grow vrh TIK PRIJE xM (krizaju na sredini)")
    ax.set_xlabel("width"); ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)

    fig.suptitle(f"Alignment demo 2 — SPLIT direkcioni (M={M}, beta={BETA})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n  [plot] spremljeno: {OUT_PNG}\n")


if __name__ == "__main__":
    main()
