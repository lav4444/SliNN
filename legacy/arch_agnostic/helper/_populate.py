import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
_M = "/home/tomi/code/dipl/morphology"
sys.path.insert(0, _M)
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "register_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]

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

import analysis as A                                                                # noqa: E402
import engine  # noqa: E402,F401  (import instalira 1D + adapter-size shim: A.weighted_leaves 2/3/4 + _try_grow_layer -> Conv1d capability trials rade)
from classify import capabilities_by_type, classify, load_register, merge_register, probe_adapter  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


before = set(load_register().get("types", {}).keys())
emit(f"baseline: {len(before)} tipova u registru")

for name, spec, cd in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        if pr is None:
            emit(f"\n### {name}: probe None — preskacem")
            del m
            continue
        cls = classify(m, pr, dev)
        caps = capabilities_by_type(m, pr, dev, cls)
        a, u, s = merge_register(caps)
        emit(f"\n### {name}: dodano={len(a)} azurirano={len(u)} hazard-skip={len(s)}")
        for ft, c in sorted(caps.items()):
            short = ft.split(".")[-1]
            emit("   {:24s} prune={!s:5s} grow={!s:5s} train={!s:5s}".format(
                short, c["prunable"], c["growable"], c["trainable"]))
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"\n### {name}: ERR {type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc()

after = load_register().get("types", {})
new = sorted(set(after.keys()) - before)
emit(f"\n{'=' * 70}\nNOVI TIPOVI ({len(new)}):")
for ft in new:
    c = after[ft]
    if c.get("status") == "hazard":
        emit(f"   {ft:52s} HAZARD")
    else:
        emit("   {:52s} prune={!s:5s} grow={!s:5s} train={!s:5s}".format(
            ft, c.get("prunable"), c.get("growable"), c.get("trainable")))
emit(f"\nukupno tipova sada: {len(after)}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
