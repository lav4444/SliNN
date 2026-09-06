import os
import sys

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, _AA)
import dataset as DS                                          # noqa: E402

BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "sniff_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PATHS = [
    ("sub10k (yolo/frcnn/school)", "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"),
    ("housing", f"{BM}/housing_mlp/data"),
    ("speechcommands", f"{BM}/speechcommands_m5/data"),
    ("sst2", f"{BM}/sst2_distilbert/data"),
    ("voc", f"{BM}/voc_deeplabv3/data"),
    ("nyu", f"{BM}/midas_depth/data"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'dataset':<28}{'modality':<10}{'n_smpl':<9}{'label_method':<16}{'task_hint':<14}")
emit("-" * 110)
for name, path in PATHS:
    r = DS.agnostic_sniffer(path, cap=500000)
    if r is None:
        emit(f"{name:<28}None — sniffer ne nalazi nista citljivo")
        continue
    emit(f"{name:<28}{str(r['modality']):<10}{r['n_samples']:<9}{r['label_method']:<16}{str(r['task_hint']):<14}")
    emit(f"      splits={r['splits']}  why: {r['why']}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
