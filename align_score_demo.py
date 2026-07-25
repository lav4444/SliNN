"""
align_score_demo.py — IZOLIRAN demo (bez torcha, sve dummy): alignment score za poravnanje broja kanala
na visekratnik M (int8/CHW32 = 32). 5 dummy slojeva s random brojem kanala (1-100).
Ispis tablice u terminal + vizualizacija (PNG s par grafova).

Logika:
  align_score(w)      = udaljenost do SLJEDECEG visekratnika M, normirano 0..1 (0 = poravnat, 1 = najgore w%M==1)
  align_PRUNE_score   = 1 - BETA*align_score   (<1 za neporavnate -> nizi rang -> reze se ranije)
  align_GROW_score    = 1 + BETA*align_score   (>1 za neporavnate -> visi rang -> raste ranije)
"""
import math
import random

# ===================== POSTAVKE (mijenjaj ovdje) ===================== #
BETA = 0.2          # jacina alignment nudge-a (0 = iskljuceno)
M = 32              # ciljani visekratnik (int8/CHW32 -> 32; fp16 -> 8; arm -> 8)
N_LAYERS = 5        # broj dummy slojeva
SEED = None            # ponovljiv demo; stavi None za pravi random svaki put
OUT_PNG = "align_score_demo.png"


def align_score(width, m=M):
    """0 za visekratnike m; raste prema sljedecem visekratniku (prazni slotovi pocice). Max 1 (width % m == 1)."""
    gap = (-width) % m                      # koliko do sljedeceg xM (0 ako je vec xM)
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

    # ----------------------------- terminal ----------------------------- #
    print(f"\n  M={M}  BETA={BETA}   (align: 0 = poravnat na x{M}, vece = dalje od sljedeceg x{M})\n")
    print(f"  {'sloj':>4} {'width':>6} {'tiles':>6} {'align_score':>12} {'align_PRUNE':>12} {'align_GROW':>11}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['layer']:>4} {r['width']:>6} {r['tiles']:>6} "
              f"{r['align']:>12.3f} {r['prune']:>12.3f} {r['grow']:>11.3f}")

    # --------------------------- vizualizacija --------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = [r["layer"] for r in rows]
    labels = [f"L{r['layer']}\n({r['width']})" for r in rows]

    # 1) broj kanala po sloju
    ax = axes[0, 0]
    ax.bar(x, [r["width"] for r in rows], color="tab:blue")
    for k in range(M, 101, M):
        ax.axhline(k, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_title(f"Broj kanala po sloju (sive crte = visekratnici {M})")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("width")

    # 2) align_score po sloju
    ax = axes[0, 1]
    ax.bar(x, [r["align"] for r in rows], color="tab:orange")
    ax.set_title("align_score po sloju (0 = poravnat, 1 = najgore)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)

    # 3) prune vs grow faktor (grupirano)
    ax = axes[1, 0]
    bw = 0.38
    ax.bar([xx - bw / 2 for xx in x], [r["prune"] for r in rows], width=bw, label="align_PRUNE (1-B*a)", color="tab:red")
    ax.bar([xx + bw / 2 for xx in x], [r["grow"] for r in rows], width=bw, label="align_GROW (1+B*a)", color="tab:green")
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.7)
    ax.set_title("Alignment faktori po sloju (crta = 1.0 neutralno)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)

    # 4) align_score preko cijelog raspona (zubasti uzorak) + nasi slojevi
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
