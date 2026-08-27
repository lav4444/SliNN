"""_prune1d55.py — smoke za Fazu 5.5 (1D-engine generalizacija) na speechcommands_m5 (Conv1d).
  A) morph_loop (prune+grow) na 1D: dokaz da Conv1d prune reže kanale (prije: tihi n_rem=0), forward OK, KD-FT.
  B) izravni grow-1D test (_try_grow_layer adapter-verzija): function-preserving +1 rast na Conv1d.
  C) GFLOPs sanity: A.weighted_leaves patch cini Conv1d vidljivim -> gflops > 0 (prije ~0, Conv1d nevidljiv).
-> REPORTS/prune1d55.txt
"""
import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "prune1d55.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import compress as C                                          # noqa: E402
import loss as L                                              # noqa: E402
import engine as E                                            # noqa: E402  (instalira shim: weighted_leaves 2/3/4 + try_grow)
from classify import probe_adapter, weighted_leaves as WL     # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


spec, cd, data = f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"
for d in cd:
    sys.path.insert(0, d)
m = A.load_any(spec, dev, code_dirs=cd)
ad = probe_adapter(m, dev, verbose=False)
ctx = prepare(m, ad, dev, data)
emit("===== FAZA 5.5 — 1D-engine generalizacija (speechcommands_m5, Conv1d) =====")
emit(f"mode={ad._mode}  taps={len(ctx['taps'])}  kd_mode={ctx['kd_mode']}  out_kind={ctx['out_kind']}  prunable={sorted(ctx['prunable'])}")
emit(f"shim: A.weighted_leaves={A.weighted_leaves.__name__} (leaves={len(A.weighted_leaves(m))}, uklj. Conv1d)  _try_grow_layer={C._try_grow_layer.__name__}")

# ---- C) GFLOPs sanity (Conv1d sad vidljiv) ----
g_full = E.gflops(m, ad, dev)
emit(f"\n[C] GFLOPs (Conv1d vidljiv preko WL-patcha) = {g_full:.5f}  ({'>0 OK' if g_full > 0 else 'NULA — Conv1d nevidljiv!'})")

# ---- A) morph_loop na 1D ----
teacher = copy.deepcopy(m).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(m).to(dev)
res = E.morph_loop(student, teacher, ad, dev, ctx, data, "speechcommands_m5",
                   target_frac=0.15, max_steps=12, ft_steps=5, batch_size=8, n_batches=6, imp_batches=3)
g0 = res["g0"]
emit(f"\n[A] morph_loop na Conv1d — baseline GFLOPs={g0:.5f}")
emit(f"{'korak':<7}{'GFLOPs':<12}{'ušteda%':<10}{'params':<11}{'rez kan':<9}{'grow':<7}{'KD':<9}")
emit("-" * 62)
tot_pruned = 0
for r in res["trajectory"]:
    saved = 100 * (g0 - r["gflops"]) / g0 if g0 > 0 else 0
    kd = f"{r['kd']:.4f}" if r.get("kd") is not None else "-"
    tot_pruned += r["removed_ch"]
    emit(f"{r['step']:<7}{r['gflops']:<12.5f}{saved:<10.2f}{r['params']:<11,}{r['removed_ch']:<9}{len(r.get('grown',[])):<7}{kd:<9}")
fwd = E._ag_forward_ok(res["student"], ad, dev)
emit(f"  ukupno rezano kanala (Conv1d): {tot_pruned}  |  forward-ok={'DA' if fwd else 'NE'}")

# ---- B) izravni grow-1D (function-preserving) ----
emit("\n[B] izravni +1 rast po Conv1d sloju (_try_grow_layer adapter-verzija):")
teacher2 = copy.deepcopy(m).to(dev).eval()
ref_in = ad.forward_example(dev)
with torch.no_grad():
    ref_out = ad.teacher_outputs(teacher2, ref_in)
ok_g, worst = 0, 0.0
for nm in sorted(ctx["prunable"]):
    grown = C._try_grow_layer(teacher2, ad, dev, nm, 1)
    if grown is None:
        emit(f"    {nm}: rollback (nije function-preserving)")
        continue
    with torch.no_grad():
        after = ad.teacher_outputs(grown, ref_in)
    diff = C._max_abs_diff(ref_out, after)
    okf = diff < 1e-3 and E._ag_forward_ok(grown, ad, dev)
    if okf:
        ok_g += 1; worst = max(worst, diff)
    emit(f"    {nm}: +1 -> {'OK' if okf else 'PAD'} (|Δ|={diff:.2e})")

# ---- verdikt ----
prune_ok = tot_pruned > 0 and fwd
grow_ok = ok_g > 0 and worst < 1e-3
emit("\n" + "-" * 62)
emit(f"VERDIKT: gflops>0={'DA' if g_full>0 else 'NE'}  1D-PRUNE(rezano {tot_pruned} kan, fwd)={'DA' if prune_ok else 'NE'}  "
     f"1D-GROW({ok_g} slojeva function-preserving)={'DA' if grow_ok else 'NE'}  ->  {'PROLAZI' if (g_full>0 and prune_ok and grow_ok) else 'PROVJERI'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
