"""
align_rank_smoke.py — IZOLIRAN smoke test koji koristi PRAVU logiku iz pipelinea (analysis + compress), bez dummyja:
  structural_flags -> prune_costs (coupled cost + width) -> grad_pass (importance + GradMax sigma)
  -> prune_candidates / grow_candidates  (oba s HW-align faktorom iz config.ALIGN_*)
Ispisuje TOP 5 za REZ (po filteru/neuronu) i TOP 5 za RAST (po sloju), s brojem kanala u sloju,
align faktorom (prune/grow) i UKUPNIM scoreom kojim se zapravo rangira.

Napomena: importance je GT-grad (analysis.grad_pass — isti put kao Overview). Kompresor u Fazi 2 koristi KD-grad,
ali RANKING kod (prune_candidates/grow_candidates/align_factors) je IDENTICAN.
"""
import torch
import analysis as A
import compress as C
import config

SPEC = config.MODEL_SPEC      # "fasterrcnn" ili YOLO_PATH iz config.py
IMAGES = 64                   # koliko slika za grad-pass (smoke -> malo, brzo)


def _short(nm, n=38):
    return nm if len(nm) <= n else "…" + nm[-(n - 1):]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A.load_any(SPEC, dev); adapter = A.pick_adapter(model)
    print(f"\n  model={A.model_name(SPEC)}  kind={adapter.kind}  |  ALIGN_M={config.ALIGN_M}  "
          f"BETA={config.ALIGN_BETA}  POW={config.ALIGN_POW}  |  grad-slika={IMAGES}\n")

    struct, note = A.structural_flags(model, adapter, dev)
    if note:
        print(f"  [flags] {note}")
    prunable = [n for n, _, _, _ in A.weighted_leaves(model) if struct.get(n)]
    cost, flops_per, units = C.prune_costs(model, adapter, dev, prunable, config.PHASE2_COST_FLOPS_W)
    info = {n: (m, w.dim()) for n, m, _, w in A.weighted_leaves(model)}

    loader = A.make_gt_loader("train")
    imp, gavg, nimg, dt = A.grad_pass(model, adapter, dev, loader, IMAGES)

    # ---------- PRUNE ranking (prava funkcija, s align faktorom) ----------
    pc = C.prune_candidates(imp, cost, info, struct)            # [(ukupni_score, layer, unit)] uzlazno
    print(f"\n  === TOP 5 PRUNE (filter/neuron; najmanji ukupni score = prvi za rez) ===")
    print(f"  {'sloj':<40}{'unit#':>6}{'#kanala':>8}{'align_PRUNE':>12}{'importance':>13}{'prune_score':>14}")
    for sc, nm, i in pc[:5]:
        width = info[nm][0].weight.shape[0]
        pf, _ = C.align_factors(width)
        print(f"  {_short(nm):<40}{i:>6}{width:>8}{pf:>12.3f}{float(imp[nm][i]):>13.3e}{sc:>14.3e}")

    # ---------- GROW ranking (prava funkcija, s align faktorom) ----------
    sigma = C.grow_potential({n: gavg[n] for n in prunable})
    gc = C.grow_candidates(sigma, flops_per, struct, units)     # [(ukupni_score, layer, sigma_max)] silazno
    print(f"\n  === TOP 5 GROW (po SLOJU; najveci ukupni score = prvi za rast) ===")
    print(f"  {'sloj':<40}{'#kanala':>8}{'align_GROW':>12}{'sigma_max':>13}{'grow_score':>14}")
    for sc, nm, smax in gc[:5]:
        width = units.get(nm, info[nm][0].weight.shape[0])
        _, gf = C.align_factors(width)
        print(f"  {_short(nm):<40}{width:>8}{gf:>12.3f}{smax:>13.3e}{sc:>14.3e}")

    # ---------- align faktori po NEPORAVNATIM slojevima (tu nudge zapravo varira) ----------
    mis = sorted((units.get(n, info[n][0].weight.shape[0]), n) for n in prunable)
    mis = [(w, n) for w, n in mis if w % config.ALIGN_M != 0]
    print(f"\n  === NEPORAVNATI prunabilni slojevi (width %% {config.ALIGN_M} != 0) — align faktori VARIRAJU ===")
    if not mis:
        print(f"  (nema — sve sirine su vec visekratnik {config.ALIGN_M}; nudge je no-op dok prune ne napravi neparne)")
    else:
        print(f"  {'sloj':<40}{'#kanala':>8}{'r=w%M':>7}{'align_PRUNE':>12}{'align_GROW':>12}")
        for w, n in mis[:12]:
            pf, gf = C.align_factors(w)
            print(f"  {_short(n):<40}{w:>8}{w % config.ALIGN_M:>7}{pf:>12.3f}{gf:>12.3f}")
    print()


if __name__ == "__main__":
    main()
