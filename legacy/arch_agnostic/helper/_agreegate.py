import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "agree_gate.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
MIDAS = [os.path.join(torch.hub.get_dir(), d) for d in
         ("intel-isl_MiDaS_master", "rwightman_gen-efficientnet-pytorch_master")]

import analysis as A                                          # noqa: E402
import engine as E                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


def run(name, spec, cd, data):
    for d in cd:
        sys.path.insert(0, d)
    m = A.load_any(spec, dev, code_dirs=cd)
    ad = probe_adapter(m, dev, verbose=False)
    ctx = prepare(m, ad, dev, data)
    teacher = copy.deepcopy(m).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(m).to(dev)
    res = E.full_cycle(student, teacher, ad, dev, ctx, data, name,
                       target_frac=0.15, ft_steps=6, dead_ft_steps=0, batch_size=8, n_batches=8,
                       imp_batches=3, max_steps=12, dead=False)
    g0 = res["g0"]
    emit(f"\n[{name}]  out_kind={ctx['out_kind']}  gate=teacher-agreement (auto)  baseline-agreement={res['metric_baseline']:.4f}")
    emit(f"  {'korak':<6}{'GFLOPs':<11}{'ušteda%':<9}{'rez':<6}{'agreement':<11}")
    for r in res["trajectory"]:
        ag = f"{r['metric']:.4f}" if r.get("metric") is not None else "-"
        emit(f"  {r['step']:<6}{r['gflops']:<11.4f}{100*(g0-r['gflops'])/g0:<9.2f}{r['removed_ch']:<6}{ag:<11}")
    b = res["best_model"]; fwd = E._ag_forward_ok(b, ad, dev)
    emit(f"  BEST: korak {res['best_step']}  ušteda {100*(g0-res['best_gflops'])/g0:.1f}%  forward-ok={'DA' if fwd else 'NE'}  "
         f"(gate {'odabrao komprimiranu verziju' if res['best_step']>0 else 'čuva original — kompresija bi pala ispod praga'})")
    del m, teacher, student
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return fwd


emit("===== TEACHER-AGREEMENT GATE (unknown-task fallback) =====")
ok1 = run("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data")
ok2 = run("midas_depth", f"{BM}/midas_depth/model.pt", MIDAS, f"{BM}/midas_depth/data")
emit(f"\nVERDIKT: agreement-gate radi (kl+mse) -> {'PROLAZI' if (ok1 and ok2) else 'PROVJERI'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
