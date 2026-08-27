"""_pipesweep.py — validacija Faze 4.5 (orkestracija). Za svaki (model, dataset_path) vrti CIJELI lanac u jednom
toku: prepare (probe->task->struktura->KD config) -> input_batch (pravi ulazi) -> kd_loss + kd_importance.
Student=kopija+sum (simulira razidjenost) da loss>0 i importance smislen. -> REPORTS/pipe_report.txt
"""
import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "pipe_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]
SUB10K = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"

import introspect as A                                          # noqa: E402
import dataset as DS                                          # noqa: E402
import loss as L                                              # noqa: E402
import pipeline as PP                                         # noqa: E402
from classify import probe_adapter                            # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("schoolcnn", "/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", ["/home/tomi/code/dipl/pruning/critereum_experiment2"], SUB10K),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], f"{BM}/housing_mlp/data"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"),
    ("sst2_distilbert", f"{BM}/sst2_distilbert/model.pt", [f"{BM}/sst2_distilbert"], f"{BM}/sst2_distilbert/data"),
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"),
    ("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS, f"{BM}/midas_depth/data"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'model':<18}{'task':<14}{'mode':<13}{'ulaz':<16}{'kd':<15}{'tap':<4}{'prun':<5}{'loss':<10}{'imp':<5}")
emit("-" * 110)
for name, spec, cd, dpath in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        ctx = PP.prepare(m, pr, dev, dpath)                  # JEDAN tok: probe->task->struktura->KD config
        if ctx["mode"] == "stop":
            emit(f"{name:<18}{ctx['task']:<14}{'STOP':<13} (nema ulaza)")
            continue
        split = "train" if ctx["splits"] and "train" in ctx["splits"] else None
        batch, source = DS.input_batch(dpath, pr, dev, split=split, n=4)

        teacher = copy.deepcopy(m).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = copy.deepcopy(m).to(dev)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(0.01 * torch.randn_like(p))

        total, _ = L.kd_loss(student, teacher, pr, batch, ctx["taps"], ctx["kd_mode"], ctx["out_kind"])
        imp, gavg = L.kd_importance(student, teacher, pr, [batch], ctx["taps"], ctx["kd_mode"],
                                    ctx["out_kind"], prunable=ctx["prunable"])
        lf = float(total)
        imp_ok = bool(imp) and all(torch.isfinite(v).all() for v in imp.values())
        emit(f"{name:<18}{ctx['task']:<14}{ctx['mode']:<13}{source:<16}{ctx['kd_mode']:<15}"
             f"{len(ctx['taps']):<4}{len(ctx['prunable']):<5}{lf:<10.4f}{'DA' if imp_ok else 'NE':<5}")
        emit(f"      metrike={ctx['metrics']}  enhaneri={ctx['enhancers']}  (task-izvor {ctx['task_source']})")
        del m, teacher, student
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"{name:<18}ERR {type(e).__name__}: {str(e)[:55]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
