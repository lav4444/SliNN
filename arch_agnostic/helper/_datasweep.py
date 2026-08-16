"""_datasweep.py — validacija DatasetProbe-a nad postojecim datasetima na disku.
Za svaki path: detect_format -> label_sample -> _label_signature. Rezultat -> dataset_report.txt (+ ispis)."""
import os
import sys

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, _AA)
import dataset as DS                                          # noqa: E402
import task as TK                                             # noqa: E402

BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "dataset_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PATHS = [
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/data", "voc", "segmentation"),
    ("speechcommands", f"{BM}/speechcommands_m5/data", "folder_per_class", "classification"),
    ("sub10k (yolo/school)", "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7", "yolo", "detection"),
    ("housing", f"{BM}/housing_mlp/data", "tabular", "regression"),
    ("sst2 (hf cache)", f"{BM}/sst2_distilbert/data", "?", "?"),
    ("nyu (parquet cache)", f"{BM}/midas_depth/data", "?", "?"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'dataset':<22}{'format':<16}{'smpl':<6}{'lbl':<6}{'mode':<14}{'n':<8}splits")
emit("-" * 108)
for name, path, exp_fmt, exp_task in PATHS:
    r = DS.probe_dataset(path)
    emit(f"{name:<22}{r['format']:<16}{str(r['samples_found']):<6}{str(r['labels_found']):<6}"
         f"{r['mode']:<14}{r['n_samples']:<8}{r['splits']}")
    emit(f"      task_hint={r['task_hint']}  why: {r['why']}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
