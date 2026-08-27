"""_sweep.py — stres-prolaz agnostic pipelinea preko svih zoo modela; matrica + pokrivenost registra.

Pokretanje:  conda activate dipl && python _sweep.py
Rezultat se ZAPISUJE u:  arch_agnostic/coverage_report.txt  (i ispisuje na kraju).
"""
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "coverage_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]

MODELS = [
    ("yolo26n", f"{BM}/yolo26n/yolo26n.pt", []),
    ("fasterrcnn", "fasterrcnn", []),
    ("schoolcnn", "/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", ["/home/tomi/code/dipl/pruning/critereum_experiment2"]),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"]),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"]),
    ("sst2_distilbert", f"{BM}/sst2_distilbert/model.pt", [f"{BM}/sst2_distilbert"]),
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"]),
    ("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS),
]

import introspect as A                                      # noqa: E402
import position as P                                      # noqa: E402
from classify import classify, fqn, load_register, probe_adapter   # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGT = set(load_register().get("types", {}).keys())
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


def reg_cov(model):
    seen = {}
    for _, m in model.named_modules():
        if not list(m.children()):
            seen[fqn(m)] = type(m).__name__
    present = [f for f in seen if f in REGT or seen[f] in REGT]
    missing = sorted({seen[f] for f in seen if f not in REGT and seen[f] not in REGT})
    return len(present), len(seen), missing


emit(f"{'model':<20}{'probe':<6}{'unk':>4}{'tap':>4}{'reg':>8}  nedostaju tipovi u registru")
emit("-" * 96)
for name, spec, cd in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        rp, rt, miss = reg_cov(m)
        pr = probe_adapter(m, dev, verbose=False)
        if pr is None:
            emit(f"{name:<20}{'NO':<6}{'-':>4}{'-':>4}{f'{rp}/{rt}':>8}  {', '.join(miss)}")
            del m
            continue
        cls = classify(m, pr, dev)
        pos, meta = P.positional(m, pr, dev, cls=cls)
        unk = sum(v["is_unknown"] for v in cls.values())
        emit(f"{name:<20}{'YES':<6}{unk:>4}{len(meta['taps']):>4}{f'{rp}/{rt}':>8}  {', '.join(miss)}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        emit(f"{name:<20}ERR: {type(e).__name__}: {str(e)[:55]}")

emit("")
emit("reg = koliko UNIKATNIH tipova leaf-slojeva je vec u LAYER_REGISTER.json / ukupno tipova u modelu.")
emit("'nedostaju' = tipovi koje registar jos ne pokriva (rules_decide) -> kandidati za Fazu 3 (populacija registra).")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
