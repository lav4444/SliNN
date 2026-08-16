"""_spike50.py — FAZA 5.0 SPIKE: jedan prune-korak + KD-FT recovery na 2D modelu (schoolcnn) KROZ arch_agnostic ctx.

Cilj: DOKAZ da se morphology mehanike (kraljeznica) integriraju s NASIM slojevima bez lijepljenja adaptera:
  ctx            <- pipeline.prepare  (task/taps/kd_mode/out_kind/prunable/split_plan)
  ulazi          <- dataset.input_batch (pravi train uzorci, ne probe-random)
  PRUNE rang     <- loss.kd_importance (KD-grad, vs smrznuti ORIGINAL)
  cost/plan/apply<- morphology.compress (prune_costs / _select_prune_plan / _apply_prune_plan)  [REUSE]
  KD-FT recovery <- loss.kd_loss + Prodigy  (loss pada -> student sustize teachera nakon reza)

Model: schoolcnn (2D, 320x320 -> pogadja _forward_ok hardcode; dokazani prune-target, §4e). -> REPORTS/spike50.txt
"""
import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
_MORPH = "/home/tomi/code/dipl/morphology"
SCHOOL_CD = "/home/tomi/code/dipl/pruning/critereum_experiment2"
for d in (_MORPH, _AA, SCHOOL_CD):
    if d not in sys.path:
        sys.path.insert(0, d)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "spike50.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import compress as C                                          # noqa: E402
import config as CFG                                          # noqa: E402
import loss as L                                              # noqa: E402
import dataset as DS                                          # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BM = "/home/tomi/code/dipl/baseline_models"
# voc_deeplabv3: 2D, prave sirine (MobileNetV3), prune ima prostora; seg (dict-izlaz, mse out_kind, feature tapovi).
# (schoolcnn_pareto_final je vec MAKS. prorezan: sve sirine 6-9 < floor 16 -> nista za rezati; los spike-target.)
SPEC = f"{BM}/voc_deeplabv3/model.pt"
DATA = f"{BM}/voc_deeplabv3/data"
CODE_DIRS = [f"{BM}/voc_deeplabv3"]


# --- SPIKE-time generalizacija: layer_table mjeri na ADAPTEROVOM (probom izmjerenom) ulazu,
#     ne na hardkodiranom 640x640 detekcijskom tenzoru. Morphology fajl NETAKNUT (runtime monkeypatch =
#     izolacija ostaje). Faza 5.2/6 ovo formalizira: jedini izvor velicine ulaza = adapter, nikad hardkod. ---
import math                                                   # noqa: E402


def _layer_table_ag(model, adapter, device):
    leaves = A.weighted_leaves(model)
    by_id = {id(m): (name, tn, w) for name, m, tn, w in leaves}
    rec, handles = [], []

    def mk(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and o:
                o = o[0]
            name, tn, w = by_id[id(mod)]
            ishape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            oshape = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() >= 3:
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:
                    out_f, in_f = w.shape
                    flops = 2 * out_f * in_f * (o.numel() // out_f if out_f else 0)
            rec.append({"name": name, "type": tn,
                        "role": "neuron" if w.dim() == 2 else "filter",
                        "units": int(w.shape[0]),
                        "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                        "gflops": flops / 1e9, "in": ishape, "out": oshape})
        return hook

    for name, m, tn, w in leaves:
        handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        adapter.forward(model, adapter.forward_example(device))   # ADAPTEROV ulaz (ispravna velicina), ne hardkod 640
    for h in handles:
        h.remove()
    return rec


A.layer_table = _layer_table_ag                               # interni pozivatelji (gflops_total/coupled_unit_cost) vide patch

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"===== FAZA 5.0 SPIKE — prune + KD-FT kroz arch_agnostic ctx ({os.path.basename(os.path.dirname(SPEC))}) =====")

# 1) load + adapter + ctx (nasa orkestracija)
for d in CODE_DIRS:
    if d not in sys.path:
        sys.path.insert(0, d)
model = A.load_any(SPEC, dev, code_dirs=CODE_DIRS)
adapter = probe_adapter(model, dev, verbose=False)
ctx = prepare(model, adapter, dev, DATA)
taps, kd_mode, out_kind, prunable = ctx["taps"], ctx["kd_mode"], ctx["out_kind"], ctx["prunable"]
split = ctx["split_plan"]["train"]
emit(f"ctx: task={ctx['task']}  mode={ctx['mode']}  out_kind={out_kind}  kd_mode={kd_mode}  "
     f"taps={len(taps)}  prunable={len(prunable)}  train-split={split}")

# 2) teacher = smrznuti ORIGINAL; student = kopija koja se kompresira
teacher = copy.deepcopy(model).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(model).to(dev)

# 3) baseline mjere
g0 = A.gflops_total(student, adapter, dev)
p0 = A.count_params(student)
emit(f"baseline: GFLOPs={g0:.4f}  params={p0:,}")

# 4) PRAVI ulazni batch-evi iz train splita (nase citanje diska)
batches, src = [], None
for _ in range(3):
    b, src = DS.input_batch(DATA, adapter, dev, split=split, n=4)
    batches.append(b)
emit(f"input_batch: {len(batches)} batcha x {len(batches[0])} uzoraka  source={src}  shape={tuple(batches[0][0].shape)}")

# 5) KD-grad vaznost (PRUNE rang) + spregnuti cost (REUSE morphology)
imp, gavg = L.kd_importance(student, teacher, adapter, batches, taps, kd_mode, out_kind, prunable=prunable)
cost, flops_per, units = C.prune_costs(student, adapter, dev, prunable)
info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in prunable}
struct = {nm: True for nm in prunable}
emit(f"kd_importance: {len(imp)} slojeva rangirano  |  prune_costs: {len(cost)} slojeva")

# 6) select plan — jedan mali korak (PHASE2_PRUNE_STEP_FRAC baseline GFLOPs)
target = CFG.PHASE2_PRUNE_STEP_FRAC * g0
plan, removed = C._select_prune_plan(struct, imp, cost, flops_per, units, info, target,
                                     CFG.PHASE2_PRUNE_LAYER_CAP, CFG.PHASE2_MIN_ALIVE_FRAC, CFG.PHASE2_MIN_ALIVE)
n_ch = sum(len(v) for v in plan.values())
emit(f"_select_prune_plan: target={target:.4f} GFLOPs  ->  {len(plan)} slojeva, {n_ch} kanala, predvidjeno_freed={removed:.4f}")

# 7) apply (forward-safe trial/rollback) — REUSE morphology
student, n_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, dev, plan)
g1 = A.gflops_total(student, adapter, dev)
p1 = A.count_params(student)
emit(f"_apply_prune_plan: n_rem={n_rem} kanala, n_lay={n_lay} slojeva, n_bad={n_bad}, banano={len(bad)}")
emit(f"post-prune: GFLOPs={g1:.4f} ({100*(g0-g1)/g0:+.2f}%)  params={p1:,} ({100*(p1-p0)/p0:+.2f}%)")

# 8) forward jos radi?
ok = C._forward_ok(student, adapter, dev)
emit(f"forward-ok nakon reza: {ok}")

# 9) KD-FT recovery — Prodigy, prati da KD-loss pada (student sustize teachera)
import prodigyopt                                             # noqa: E402
student.eval()                                              # BN zamrznut (running-stats) -> konzistentno s teacher.eval();
for p in student.parameters():                              # izbjegava train/eval BN-mismatch koji lazno napuse KD-loss
    p.requires_grad_(True)                                  # eval NE gasi grad -> tezine se i dalje uce ([[bn-eval-detection-trainmode]])
kd0, _ = L.kd_loss(student, teacher, adapter, batches[0], taps, kd_mode, out_kind)
opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                         lr=1.0, d_coef=0.9, safeguard_warmup=True)
traj = [float(kd0)]
for step in range(8):
    opt.zero_grad(set_to_none=True)
    loss, _ = L.kd_loss(student, teacher, adapter, batches[step % len(batches)], taps, kd_mode, out_kind)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 5.0)
    opt.step()
    traj.append(float(loss))
emit(f"KD-FT (8 koraka): loss {traj[0]:.4f} -> {traj[-1]:.4f}  recovery={'DA' if traj[-1] < traj[0] else 'NE'}")
emit("  trajektorija: " + " ".join(f"{x:.3f}" for x in traj))

# verdikt
pruned_ok = g1 < g0 and p1 < p0
emit("-" * 78)
emit(f"VERDIKT: prune-smanjenje={'DA' if pruned_ok else 'NE'}  forward-ok={'DA' if ok else 'NE'}  "
     f"kd-ft-recovery={'DA' if traj[-1] < traj[0] else 'NE'}  ->  "
     f"{'SPIKE PROLAZI' if (pruned_ok and ok and traj[-1] < traj[0]) else 'SPIKE PADA'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
