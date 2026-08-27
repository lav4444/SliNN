"""_metric56.py — smoke za Fazu 5.6a (GENERALNOST s PRAVOM metrikom + real-metric quality-gate), voc segmentacija.
full_cycle (dead=OFF; KD-grad prune primaran) s metric_fn=mIoU gateom: best = najmanji GFLOPs čiji mIoU >=
tol×baseline. GT ULAZI SAMO u gate/izvještaj, nikad u loss. -> REPORTS/metric56.txt
(housing/tabular čeka odluku o preprocessing-ravnini — model traži standardizirani ulaz; v. izvještaj.)
"""
import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "metric56.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import engine as E                                            # noqa: E402
import metric as M                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit("===== FAZA 5.6a — GENERALNOST s pravom metrikom + real-metric gate (voc segmentacija) =====")
spec, cd, data = f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"
for d in cd:
    sys.path.insert(0, d)
m = A.load_any(spec, dev, code_dirs=cd)
ad = probe_adapter(m, dev, verbose=False)
ctx = prepare(m, ad, dev, data)
gate_pairs = M.pairs_segmentation(data, split="val", n=40, size=256)     # in-loop gate (brže)
rep_pairs = M.pairs_segmentation(data, split="val", n=100, size=256)     # završni izvještaj (veći uzorak)
miou = lambda mdl, pr: M.evaluate(mdl, ad, "segmentation", pr, dev)["mIoU"]  # noqa: E731

g0 = E.gflops(m, ad, dev)
m0 = miou(m, rep_pairs)
emit(f"original: mIoU={m0:.4f} (n=100)  GFLOPs={g0:.4f}  out_kind={ctx['out_kind']} (per-piksel KL)")

teacher = copy.deepcopy(m).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(m).to(dev)
res = E.full_cycle(student, teacher, ad, dev, ctx, data, "voc_deeplabv3",
                   target_frac=0.20, ft_steps=10, dead_ft_steps=0, batch_size=6, n_batches=12,
                   imp_batches=4, max_steps=16, dead=False,
                   metric_fn=lambda mdl: miou(mdl, gate_pairs), metric_tol=0.90)

emit(f"\nquality-gate: prava mIoU >= 0.90×baseline (dead-removal OFF; KD-grad prune primaran)")
emit(f"{'korak':<7}{'GFLOPs':<11}{'ušteda%':<10}{'rez':<6}{'grow':<7}{'mIoU(gate,n=40)':<17}{'KD':<8}")
emit("-" * 66)
for r in res["trajectory"]:
    saved = 100 * (g0 - r["gflops"]) / g0
    mt = f"{r['metric']:.4f}" if r.get("metric") is not None else "-"
    kd = f"{r['kd']:.4f}" if r.get("kd") is not None else "-"
    emit(f"{r['step']:<7}{r['gflops']:<11.4f}{saved:<10.2f}{r['removed_ch']:<6}{len(r.get('grown',[])):<7}{mt:<17}{kd:<8}")

best = res["best_model"]
g1 = res["best_gflops"]
m1 = miou(best, rep_pairs)                                    # provjeri best na VEĆEM uzorku
fwd = E._ag_forward_ok(best, ad, dev)
emit("-" * 66)
emit(f"BEST: korak {res['best_step']}  GFLOPs={g1:.4f} (ušteda {100*(g0-g1)/g0:.1f}%)  forward-ok={'DA' if fwd else 'NE'}")
emit(f"mIoU (n=100):  original={m0:.4f}  ->  best={m1:.4f}  (zadržano {100*m1/m0:.1f}%)")
ok = fwd and g1 < g0 and m1 >= 0.90 * m0
emit(f"\nVERDIKT 5.6a (voc generalnost, prava metrika): {'PROLAZI' if ok else 'PROVJERI'} "
     f"(mIoU zadržan ≥90% uz {100*(g0-g1)/g0:.0f}% ušteda GFLOPs)")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
