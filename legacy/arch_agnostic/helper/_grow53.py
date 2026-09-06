import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "grow53.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import compress as C                                          # noqa: E402
import config as CFG                                          # noqa: E402
import loss as L                                              # noqa: E402
import engine as E                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


spec, cd, data = f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"
for d in cd:
    sys.path.insert(0, d)
m = A.load_any(spec, dev, code_dirs=cd)
ad = probe_adapter(m, dev, verbose=False)
ctx = prepare(m, ad, dev, data)


def fresh_pair():
    t = copy.deepcopy(m).to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    s = copy.deepcopy(m).to(dev)
    return t, s


emit("===== FAZA 5.3-A — morph_loop (prune+grow+cooldown+reinvest), voc =====")
teacher, student = fresh_pair()
res = E.morph_loop(student, teacher, ad, dev, ctx, data, "voc_deeplabv3",
                   target_frac=0.12, max_steps=14, ft_steps=5, batch_size=4, n_batches=6, imp_batches=3)
g0 = res["g0"]
emit(f"baseline GFLOPs={g0:.4f}  cooldown={CFG.PHASE2_CHURN_COOLDOWN}  reinvest={CFG.PHASE2_REINVEST_FRAC}  GROW_DOM={CFG.PHASE2_GROW_DOM}")
emit(f"{'korak':<7}{'GFLOPs':<11}{'ušteda%':<10}{'params':<12}{'rez':<6}{'grow(sloj,k)':<28}{'KD':<9}{'cd_ovr':<7}")
emit("-" * 92)
grow_events = []
for r in res["trajectory"]:
    saved = 100 * (g0 - r["gflops"]) / g0
    kd = f"{r['kd']:.4f}" if r.get("kd") is not None else "-"
    gstr = ",".join(f"{ln.split('.')[-1]}:{k}" for ln, k in r.get("grown", [])) or "-"
    for ln, k in r.get("grown", []):
        grow_events.append((r["step"], ln))
    emit(f"{r['step']:<7}{r['gflops']:<11.4f}{saved:<10.2f}{r['params']:<12,}{r['removed_ch']:<6}{gstr:<28}{kd:<9}{'DA' if r.get('cd_override') else '':<7}")

churn = []
for ln, gk in res["grown_at"].items():
    pk = res["pruned_at"].get(ln)
    if pk is not None and abs(pk - gk) <= CFG.PHASE2_CHURN_COOLDOWN and pk > gk:
        churn.append((ln, gk, pk))
mono_net = res["final_gflops"] <= g0
fwd_ok = E._ag_forward_ok(res["student"], ad, dev)
emit("-" * 92)
emit(f"grow-događaja: {len(grow_events)}  |  churn-prekršaja (narastao pa rezan u cooldownu): {len(churn)}")
emit(f"net GFLOPs pao: {'DA' if mono_net else 'NE'} ({100*(g0-res['final_gflops'])/g0:.2f}%)  forward-ok={'DA' if fwd_ok else 'NE'}")

emit("\n===== FAZA 5.3-B — function-preserving +1 rast po sloju (_try_grow_layer) =====")
teacher, student = fresh_pair()
prunable = sorted(ctx["prunable"])
ref_in = ad.forward_example(dev)
with torch.no_grad():
    ref_out = ad.teacher_outputs(student, ref_in)
ok_layers, worst = 0, 0.0
for nm in prunable:
    grown = C._try_grow_layer(student, ad, dev, nm, 1)
    if grown is None:
        continue
    with torch.no_grad():
        after = ad.teacher_outputs(grown, ref_in)
    diff = C._max_abs_diff(ref_out, after)
    if diff < 1e-3 and E._ag_forward_ok(grown, ad, dev):
        ok_layers += 1
        worst = max(worst, diff)
emit(f"  prunable slojeva: {len(prunable)}  |  function-preserving +1 rast USPIO na: {ok_layers}")
emit(f"  najveci |Δizlaz| medu uspjelima: {worst:.2e} (<1e-3 = function-preserving)")
grown_in_A = sorted({ln for _, ln in grow_events})
emit(f"  (u petlji A grow je materijalizirao na: {', '.join(g.split('.')[-1] for g in grown_in_A) or '-'})")

emit("\n" + "-" * 92)
b_ok = ok_layers > 0 and worst < 1e-3
emit(f"VERDIKT: A(cooldown-poštovan={'DA' if not churn else 'NE'}, net-pad={'DA' if mono_net else 'NE'}, fwd={'DA' if fwd_ok else 'NE'})  "
     f"B(grow-mehanika={'DA' if b_ok else 'NE'})  ->  {'PROLAZI' if (not churn and mono_net and fwd_ok and b_ok) else 'PROVJERI'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
