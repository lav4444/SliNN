"""_tasksweep.py — validacija task-detektora preko zoo-a, SPOJENO s dataset_probe.

Bez per-model glue-a: `dataset.probe_dataset(dataset_path)` je JEDINI izvor oznaka (+ task_hint), a
`task.detect_task(model, adapter, probe=...)` ih konzumira i krsta s arhitekturom (B ogranicava A).
Po modelu se navodi samo (model_spec, code_dirs, DATASET_PATH) — pravi korisnicki ulaz, ne label-ekstrakcija.
Rezultat -> REPORTS/task_report.txt (+ ispis).
"""
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "task_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]
SUB10K = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"

import introspect as A                                          # noqa: E402
import dataset as DS                                          # noqa: E402
import task as TK                                             # noqa: E402
from classify import probe_adapter                            # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (naziv, model_spec, code_dirs, DATASET_PATH) — jedini per-model podatak je putanja (korisnicki ulaz)
MODELS = [
    ("yolo26n", f"{BM}/yolo26n/yolo26n.pt", [], SUB10K),
    ("fasterrcnn", "fasterrcnn", [], SUB10K),
    ("schoolcnn", "/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", ["/home/tomi/code/dipl/pruning/critereum_experiment2"], SUB10K),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], f"{BM}/housing_mlp/data"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"),
    ("sst2_distilbert", f"{BM}/sst2_distilbert/model.pt", [f"{BM}/sst2_distilbert"], f"{BM}/sst2_distilbert/data"),
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"),
    ("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS, f"{BM}/midas_depth/data"),
]

EXPECT = {"yolo26n": "detection", "fasterrcnn": "detection", "schoolcnn": "multilabel",
          "housing_mlp": "regression", "speechcommands_m5": "classification",
          "sst2_distilbert": "classification", "voc_deeplabv3": "segmentation", "midas_depth": "regression"}

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'model':<20}{'detektirano':<15}{'izvor':<11}{'ocek.':<14}{'OK':<4} metrike")
emit("-" * 104)
for name, spec, cd, dpath in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        probe = DS.probe_dataset(dpath)                       # JEDINI izvor oznaka + data-hint
        r = TK.detect_task(m, pr, dev, probe=probe)
        ok = "YES" if r["task"] == EXPECT.get(name) else "NE"
        emit(f"{name:<20}{r['task']:<15}{r['source']:<11}{EXPECT.get(name, '?'):<14}{ok:<4} {','.join(r['metrics'])}")
        emit(f"      probe: fmt={probe['format']} mode={probe['mode']} smpl={probe['samples_found']} lbl={probe['labels_found']}")
        emit(f"      why: {r['why']}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"{name:<20}ERR {type(e).__name__}: {str(e)[:60]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
