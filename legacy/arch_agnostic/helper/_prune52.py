import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "prune52.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import engine as E                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


PREP = [
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], f"{BM}/housing_mlp/data"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"),
]

emit("===== FAZA 5.2 — PREP (adapter-size shim + autobatch) =====")
emit(f"shim aktivan: layer_table={A.layer_table.__name__}  (adapter-size, ne hardkod 640)")
for name, spec, cd, data in PREP:
    for d in cd:
        sys.path.insert(0, d)
    m = A.load_any(spec, dev, code_dirs=cd)
    ad = probe_adapter(m, dev, verbose=False)
    ctx = prepare(m, ad, dev, data)
    g = E.gflops(m, ad, dev)
    bs = E.autobatch(m, ad, dev, ctx, data)
    emit(f"  {name:20s} mode={ad._mode:7s} GFLOPs={g:.4f}  autobatch={bs}")
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()

emit("\n===== FAZA 5.2 — KONTINUIRANI PRUNE (voc_deeplabv3) =====")
spec, cd, data = f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"
m = A.load_any(spec, dev, code_dirs=cd)
ad = probe_adapter(m, dev, verbose=False)
ctx = prepare(m, ad, dev, data)

teacher = copy.deepcopy(m).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(m).to(dev)

res = E.morph_loop(student, teacher, ad, dev, ctx, data, "voc_deeplabv3",
                   target_frac=0.12, reinvest_frac=0.0, max_steps=12, ft_steps=6, batch_size=4, n_batches=6, imp_batches=3)
g0 = res["g0"]
emit(f"baseline GFLOPs={g0:.4f}  target ušteda=12%  taps={len(ctx['taps'])}  prunable={len(ctx['prunable'])}")
emit(f"{'korak':<7}{'GFLOPs':<12}{'ušteda%':<10}{'params':<12}{'rez kan':<9}{'KD-loss':<10}{'banano':<7}")
emit("-" * 70)
for r in res["trajectory"]:
    saved = 100 * (g0 - r["gflops"]) / g0
    kd = f"{r['kd']:.4f}" if r.get("kd") is not None else "-"
    note = r.get("note", "")
    emit(f"{r['step']:<7}{r['gflops']:<12.4f}{saved:<10.2f}{r['params']:<12,}{r['removed_ch']:<9}{kd:<10}{r.get('banned',0):<7}{note}")

final = res["final_gflops"]
ok_fwd = E._ag_forward_ok(res["student"], ad, dev)
mono = all(res["trajectory"][i]["gflops"] <= res["trajectory"][i - 1]["gflops"] + 1e-9
           for i in range(1, len(res["trajectory"])))
emit("-" * 70)
emit(f"VERDIKT: ušteda={100*(g0-final)/g0:.2f}%  GFLOPs-monotono-pada={'DA' if mono else 'NE'}  "
     f"forward-ok={'DA' if ok_fwd else 'NE'}  banano={len(res['banned'])}  ->  "
     f"{'PROLAZI' if (final < g0 and mono and ok_fwd) else 'PADA'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
