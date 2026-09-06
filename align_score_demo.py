import math
import random

BETA = 0.2
M = 32
N_LAYERS = 5
SEED = None
OUT_PNG = "align_score_demo.png"


def align_score(width, m=M):
    gap = (-width) % m
    return gap / (m - 1)


def main():
    if SEED is not None:
        random.seed(SEED)
    widths = [random.randint(1, 100) for _ in range(N_LAYERS)]

    rows = []
    for i, w in enumerate(widths):
        a = align_score(w)
        rows.append({"layer": i, "width": w, "tiles": math.ceil(w / M), "align": a,
                     "prune": 1 - BETA * a, "grow": 1 + BETA * a})

    print(f"\n  M={M}  BETA={BETA}   (align: 0 = poravnat na x{M}, vece = dalje od sljedeceg x{M})\n")
    print(f"  {'sloj':>4} {'width':>6} {'tiles':>6} {'align_score':>12} {'align_PRUNE':>12} {'align_GROW':>11}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['layer']:>4} {r['width']:>6} {r['tiles']:>6} "
              f"{r['align']:>12.3f} {r['prune']:>12.3f} {r['grow']:>11.3f}")

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
    ax.bar(x, [r["align"] for r in rows], color="tab:orange")
    ax.set_title("align_score po sloju (0 = poravnat, 1 = najgore)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)

    ax = axes[1, 0]
    bw = 0.38
    ax.bar([xx - bw / 2 for xx in x], [r["prune"] for r in rows], width=bw, label="align_PRUNE (1-B*a)", color="tab:red")
    ax.bar([xx + bw / 2 for xx in x], [r["grow"] for r in rows], width=bw, label="align_GROW (1+B*a)", color="tab:green")
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title("Alignment faktori po sloju (crta = 1.0 neutralno)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ws = list(range(1, 101))
    ax.plot(ws, [align_score(w) for w in ws], color="tab:purple", lw=1.2)
    ax.scatter([r["width"] for r in rows], [r["align"] for r in rows], color="tab:orange", zorder=5, s=45)
    for r in rows:
        ax.annotate(f"L{r['layer']}", (r["width"], r["align"]), textcoords="offset points", xytext=(0, 6), fontsize=8)
    for k in range(M, 101, M):
        ax.axvline(k, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_title(f"align_score(width) — pada na 0 na svakom x{M}")
    ax.set_xlabel("width"); ax.set_ylabel("align_score"); ax.set_ylim(0, 1.05)

    fig.suptitle(f"Alignment demo (M={M}, beta={BETA})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n  [plot] spremljeno: {OUT_PNG}\n")


if __name__ == "__main__":
    main()
