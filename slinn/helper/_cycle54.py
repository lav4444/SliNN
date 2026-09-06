import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "cycle54.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import introspect as A                                          # noqa: E402
import engine as E                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


spec, cd, data = f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"
for d in cd:
    sys.path.insert(0, d)
m = A.load_any(spec, dev, code_dirs=cd)
ad = probe_adapter(m, dev, verbose=False)
ctx = prepare(m, ad, dev, data)

teacher = copy.deepcopy(m).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(m).to(dev)

emit("===== FAZA 5.4 — dead-removal + quality-gated ciklus (voc) =====")
res = E.full_cycle(student, teacher, ad, dev, ctx, data, "voc_deeplabv3",
                   target_frac=0.15, ft_steps=5, dead_ft_steps=6, dead=True,
                   batch_size=4, n_batches=6, imp_batches=3, max_steps=14)
g0 = res["g0"]
emit(f"dead-removal: maknuto {res['n_dead']} kanala u {res['n_dead_layers']} slojeva")
emit(f"gate = teacher-agreement (auto); baseline-agreement={res['metric_baseline']:.4f}  |  baseline GFLOPs={g0:.4f}")
emit(f"\n{'korak':<7}{'GFLOPs':<11}{'ušteda%':<10}{'params':<12}{'rez':<6}{'grow':<8}{'agreement':<11}")
emit("-" * 68)
for r in res["trajectory"]:
    saved = 100 * (g0 - r["gflops"]) / g0
    mt = r.get("metric")
    mts = f"{mt:.4f}" if mt is not None else "-"
    ng = len(r.get("grown", []))
    emit(f"{r['step']:<7}{r['gflops']:<11.4f}{saved:<10.2f}{r['params']:<12,}{r['removed_ch']:<6}{ng:<8}{mts:<11}")

emit("-" * 68)
best = res["best_model"]
fwd = E._ag_forward_ok(best, ad, dev)
bsaved = 100 * (g0 - res["best_gflops"]) / g0
emit(f"BEST model: korak {res['best_step']}  GFLOPs={res['best_gflops']:.4f} (ušteda {bsaved:.2f}%)  forward-ok={'DA' if fwd else 'NE'}")
ok = (res["best_model"] is not None and res["best_gflops"] < g0 and fwd)
emit(f"VERDIKT: dead-removal-radi=DA  ciklus-napreduje={'DA' if res['final_gflops']<g0 else 'NE'}  "
     f"best-odabran={'DA' if res['best_step']>0 else '(zadnji)'}  forward-ok={'DA' if fwd else 'NE'}  ->  "
     f"{'PROLAZI' if ok else 'PROVJERI'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
