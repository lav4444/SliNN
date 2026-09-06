import copy
import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
SCHOOL_CD = "/home/tomi/code/dipl/pruning/critereum_experiment2"
for d in (_AA, SCHOOL_CD):
    if d not in sys.path:
        sys.path.insert(0, d)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "spike50.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import introspect as A                                          # noqa: E402
import morph as C                                               # noqa: E402
import settings as CFG                                       # 6.4: slinn config.py -> settings.py                                          # noqa: E402
import loss as L                                              # noqa: E402
import dataset as DS                                          # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BM = "/home/tomi/code/dipl/baseline_models"
SPEC = f"{BM}/voc_deeplabv3/model.pt"
DATA = f"{BM}/voc_deeplabv3/data"
CODE_DIRS = [f"{BM}/voc_deeplabv3"]


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
        adapter.forward(model, adapter.forward_example(device))
    for h in handles:
        h.remove()
    return rec


A.layer_table = _layer_table_ag

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit(f"===== FAZA 5.0 SPIKE — prune + KD-FT kroz arch_agnostic ctx ({os.path.basename(os.path.dirname(SPEC))}) =====")

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

teacher = copy.deepcopy(model).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
student = copy.deepcopy(model).to(dev)

g0 = A.gflops_total(student, adapter, dev)
p0 = A.count_params(student)
emit(f"baseline: GFLOPs={g0:.4f}  params={p0:,}")

batches, src = [], None
for _ in range(3):
    b, src = DS.input_batch(DATA, adapter, dev, split=split, n=4)
    batches.append(b)
emit(f"input_batch: {len(batches)} batcha x {len(batches[0])} uzoraka  source={src}  shape={tuple(batches[0][0].shape)}")

imp, gavg = L.kd_importance(student, teacher, adapter, batches, taps, kd_mode, out_kind, prunable=prunable)
cost, flops_per, units = C.prune_costs(student, adapter, dev, prunable)
info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in prunable}
struct = {nm: True for nm in prunable}
emit(f"kd_importance: {len(imp)} slojeva rangirano  |  prune_costs: {len(cost)} slojeva")

target = CFG.F1_PRUNE_STEP_FRAC * g0
plan, removed = C._select_prune_plan(struct, imp, cost, flops_per, units, info, target,
                                     CFG.PHASE2_PRUNE_LAYER_CAP, CFG.PHASE2_MIN_ALIVE_FRAC, CFG.PHASE2_MIN_ALIVE)
n_ch = sum(len(v) for v in plan.values())
emit(f"_select_prune_plan: target={target:.4f} GFLOPs  ->  {len(plan)} slojeva, {n_ch} kanala, predvidjeno_freed={removed:.4f}")

student, n_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, dev, plan)
g1 = A.gflops_total(student, adapter, dev)
p1 = A.count_params(student)
emit(f"_apply_prune_plan: n_rem={n_rem} kanala, n_lay={n_lay} slojeva, n_bad={n_bad}, banano={len(bad)}")
emit(f"post-prune: GFLOPs={g1:.4f} ({100*(g0-g1)/g0:+.2f}%)  params={p1:,} ({100*(p1-p0)/p0:+.2f}%)")

ok = C._forward_ok(student, adapter, dev)
emit(f"forward-ok nakon reza: {ok}")

import prodigyopt                                             # noqa: E402
student.eval()
for p in student.parameters():
    p.requires_grad_(True)
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

pruned_ok = g1 < g0 and p1 < p0
emit("-" * 78)
emit(f"VERDIKT: prune-smanjenje={'DA' if pruned_ok else 'NE'}  forward-ok={'DA' if ok else 'NE'}  "
     f"kd-ft-recovery={'DA' if traj[-1] < traj[0] else 'NE'}  ->  "
     f"{'SPIKE PROLAZI' if (pruned_ok and ok and traj[-1] < traj[0]) else 'SPIKE PADA'}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
