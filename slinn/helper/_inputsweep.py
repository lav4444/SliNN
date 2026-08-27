"""_inputsweep.py — smoke za Fazu 4.0 (genericki input-reader). Za svaki (model, dataset_path): probe adapter,
input_batch dekodira PRAVE ulaze; provjeri oblik uskladjen s adapterom + da adapter.forward(model, batch) daje
konacan izlaz. -> REPORTS/input_report.txt
"""
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "input_report.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]
SUB10K = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"

import introspect as A                                          # noqa: E402
import dataset as DS                                          # noqa: E402
from classify import _finite, probe_adapter                   # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("schoolcnn", "/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", ["/home/tomi/code/dipl/pruning/critereum_experiment2"], SUB10K),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], f"{BM}/housing_mlp/data"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"),
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"),
    ("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS, f"{BM}/midas_depth/data"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'model':<20}{'mode':<9}{'source':<17}{'uzorak oblik':<20}{'forward OK':<11}")
emit("-" * 92)
for name, spec, cd, dpath in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        batch, source = DS.input_batch(dpath, pr, dev, n=4)
        shp = tuple(batch[0].shape)
        m.eval()
        with torch.no_grad():
            out = pr.forward(m, batch)
        fwd = _finite(out)
        emit(f"{name:<20}{getattr(pr, '_mode', '?'):<9}{source:<17}{str(shp):<20}{'DA' if fwd else 'NE':<11}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"{name:<20}ERR {type(e).__name__}: {str(e)[:55]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
