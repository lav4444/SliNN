"""_kdsmoke.py — smoke za Fazu 4.1 (genericki hook-KD). Za svaki model: teacher = smrznuta kopija,
student = kopija + sum na tezinama; par Adam koraka na KD-gubitku -> mora PASTI (student oponasa teachera).
Dokaz da genericki hook-KD racuna i da je valjan trening-signal na raznim taskovima. -> REPORTS/kd_smoke.txt
"""
import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "kd_smoke.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]

import introspect as A                                          # noqa: E402
import position as P                                          # noqa: E402
import loss as L                                              # noqa: E402
from classify import classify, probe_adapter                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (naziv, spec, code_dirs, out_kind) — out_kind hardkodiran za smoke (u 4.5 dolazi iz task_detector)
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


def _bn_eval(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()


emit(f"{'model':<20}{'kd_mode':<16}{'taps':<5}{'out':<5}{'loss0':<12}{'lossN':<12}{'pao?':<6}")
emit("-" * 90)
for name, spec, cd, out_kind in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        pr = probe_adapter(m, dev, verbose=False)
        cls = classify(m, pr, dev)
        _, meta = P.positional(m, pr, dev, cls=cls)
        taps, kd_mode = meta["taps"], meta["kd_mode"]

        teacher = copy.deepcopy(m).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = copy.deepcopy(m).to(dev)
        _bn_eval(teacher)
        _bn_eval(student)
        with torch.no_grad():                                # perturbiraj studenta da KD-loss bude > 0
            for p in student.parameters():
                p.add_(0.01 * torch.randn_like(p))

        imgs = [pr._one(dev) for _ in range(4)]              # fiksni random batch (4.0 pravi ulazi kasnije)
        import prodigyopt                                     # auto-LR (kao morphology KD-FT); clanovi su sad ~O(1)
        opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                                 lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
        losses = []
        for step in range(40):
            opt.zero_grad()
            total, info = L.kd_loss(student, teacher, pr, imgs, taps, kd_mode, out_kind)
            total.backward()
            opt.step()
            losses.append(float(total))
        l0, ln = losses[0], losses[-1]
        emit(f"{name:<20}{kd_mode:<16}{len(taps):<5}{out_kind:<5}{l0:<12.5f}{ln:<12.5f}{'DA' if ln < l0 else 'NE':<6}")
        emit(f"      grupe (zadnji korak): " + ", ".join(f"{k}={v:.4f}" for k, v in info.items()))
        del m, teacher, student
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"{name:<20}ERR {type(e).__name__}: {str(e)[:60]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
