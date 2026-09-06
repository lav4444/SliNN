import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "kd_imp.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]

import analysis as A                                          # noqa: E402
import position as P                                          # noqa: E402
import loss as L                                              # noqa: E402
from classify import classify, probe_adapter                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("schoolcnn", "/home/tomi/code/dipl/pareto_sweep/schoolcnn_pareto_final.pt", ["/home/tomi/code/dipl/pruning/critereum_experiment2"], "mse"),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], "mse"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], "kl"),
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], "mse"),
    ("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS, "mse"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"{'model':<20}{'prunable':<10}{'imp_konacna':<13}{'diskrim slojeva':<17}{'nul-grad':<10}{'gavg OK':<8}")
emit("-" * 92)
for name, spec, cd, out_kind in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        cls = classify(m, pr, dev)
        pos, meta = P.positional(m, pr, dev, cls=cls)
        taps, kd_mode = meta["taps"], meta["kd_mode"]
        prunable = {n for n, v in pos.items() if v["morph"]}

        teacher = copy.deepcopy(m).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = copy.deepcopy(m).to(dev)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(0.01 * torch.randn_like(p))

        batches = [[pr._one(dev) for _ in range(4)] for _ in range(3)]
        imp, gavg = L.kd_importance(student, teacher, pr, batches, taps, kd_mode, out_kind, prunable=prunable)

        allv = torch.cat([v for v in imp.values()]) if imp else torch.zeros(1)
        finite = bool(torch.isfinite(allv).all()) and float(allv.abs().sum()) > 0
        ndisc = sum(1 for v in imp.values() if v.numel() > 1 and float(v.var()) > 0)
        nzero = sum(1 for v in imp.values() if float(v.abs().sum()) == 0)
        gavg_ok = all(gavg[n].dim() == 2 and gavg[n].shape[0] == imp[n].shape[0] for n in imp)
        emit(f"{name:<20}{len(imp):<10}{'DA' if finite else 'NE':<13}{f'{ndisc}/{len(imp)}':<17}{nzero:<10}{'DA' if gavg_ok else 'NE':<8}")
        ex = next(iter(imp))
        v = imp[ex]
        emit(f"      npr '{ex}': {v.numel()} kanala, imp min/mean/max = {float(v.min()):.2e}/{float(v.mean()):.2e}/{float(v.max()):.2e}")
        del m, teacher, student
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"{name:<20}ERR {type(e).__name__}: {str(e)[:60]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
