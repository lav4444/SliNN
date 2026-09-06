
import copy
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP2 = os.path.join(os.path.dirname(_HERE), "pruning", "critereum_experiment2")
sys.path.insert(0, _EXP2)

import common
import pruning_lib2 as P
from model_cnn import SchoolCNN, INPUT_SIZE

MODEL_TAG   = "schoolcnn"
CKPT_PATH   = os.path.join(_EXP2, "checkpoints", "best.pt")

PRUNE_STEP_FRAC   = 0.03
PRUNE_LAYER_CAP   = 0.15
QUALITY_METRIC    = "map"
MIN_GFLOPS_FRAC   = 0.005
MIN_PROGRESS_FRAC = 0.0015
FT_EPOCHS_PER_STEP = 2

CRITERION   = "taylor"
MIN_KEEP    = 2
CALIB_BATCHES = 4
MAX_STEPS   = 500

LR          = 1e-3 / 3.0
BATCH_SIZE  = 32
NUM_WORKERS = 4

DEV_SUBSET  = None

OUT_CSV   = os.path.join(_HERE, f"{MODEL_TAG}_pareto_run.csv")
OUT_META  = os.path.join(_HERE, f"{MODEL_TAG}_pareto_run_meta.json")
OUT_MODEL = os.path.join(_HERE, f"{MODEL_TAG}_pareto_final.pt")

CSV_COLS = ["step", "phase", "gflops", "gflops_pct", "params", "size_mb",
            "val_map", "val_f1", "val_acc", "quality", "gflops_freed_cum",
            "ft_epochs", "pruned_params", "kd_loss", "wall_s"]


def _conv_out_hw(model, device):
    hw, handles = {}, []

    def hook(m, i, o):
        hw[id(m)] = o.shape[2] * o.shape[3]

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    was = model.training; model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device))
    for h in handles:
        h.remove()
    if was:
        model.train()
    return hw


def select_prune_gflops(model, entries, by_name, importance, target_gflops,
                        layer_cap, min_keep, device):
    prunable = [e for e in entries if e["prune_out"]]
    out_hw = _conv_out_hw(model, device)
    cur_out = {e["name"]: e["orig_out"] for e in entries}
    consumers = {}
    for c in entries:
        for src, block in c["in_sources"]:
            if src != "INPUT":
                consumers.setdefault(src, []).append((c, block))

    def in_eff(e):
        return sum((3 if src == "INPUT" else cur_out[src]) * block for src, block in e["in_sources"])

    def per_unit_gflops(e):
        own = (2 * in_eff(e) * e["kh"] * e["kw"] * out_hw[id(e["module"])]) if e["kind"] == "conv" \
            else (2 * in_eff(e))
        down = 0
        for c, block in consumers.get(e["name"], []):
            cpu = (2 * cur_out[c["name"]] * c["kh"] * c["kw"] * out_hw[id(c["module"])]) if c["kind"] == "conv" \
                else (2 * cur_out[c["name"]])
            down += cpu * block
        return (own + down) / 1e9

    fpu = {e["name"]: per_unit_gflops(e) for e in prunable}
    ranked = []
    for e in prunable:
        imp = importance[e["name"]].float()
        ranked.extend((float(imp[ch]), e["name"], ch) for ch in range(e["orig_out"]))
    ranked.sort(key=lambda t: t[0])

    cap = {e["name"]: int(math.floor(layer_cap * e["orig_out"])) for e in prunable}
    removed = {e["name"]: set() for e in prunable}
    freed = 0.0
    for _v, name, ch in ranked:
        if freed >= target_gflops:
            break
        if len(removed[name]) >= cap[name]:
            continue
        if cur_out[name] - len(removed[name]) <= min_keep:
            continue
        removed[name].add(ch); freed += fpu[name]
    kept_idx = {e["name"]: torch.tensor([i for i in range(e["orig_out"]) if i not in removed[e["name"]]],
                                        dtype=torch.long) for e in prunable}
    return kept_idx, freed


def _load_teacher(device):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = SchoolCNN().to(device)
    model.load_state_dict(state)
    return model


def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    qm = QUALITY_METRIC

    teacher = _load_teacher(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)

    train_loader = common.make_loader("train", BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, max_images=DEV_SUBSET)
    val_loader   = common.make_loader("val", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, max_images=DEV_SUBSET)
    calib_loader = common.make_loader("train", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, max_images=DEV_SUBSET)
    calib_batches = []
    for b in calib_loader:
        calib_batches.append(b)
        if len(calib_batches) >= CALIB_BATCHES:
            break

    def kd_soft(student_logits, x):
        with torch.no_grad():
            t_prob = torch.sigmoid(teacher(x))
        return F.binary_cross_entropy_with_logits(student_logits, t_prob)

    def kd_loss_fn(model, batch):
        x = batch[0].to(device, non_blocking=True)
        return kd_soft(model(x), x), x.size(0)

    def ft_kd(epochs):
        student.train()
        opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=LR)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        last = 0.0
        for _ in range(epochs):
            tot = 0.0; n = 0
            for x, _y in train_loader:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = kd_soft(student(x), x)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                tot += float(loss) * x.size(0); n += x.size(0)
            last = tot / max(n, 1)
        return last

    def gflops():
        return P.count_flops(student, device, INPUT_SIZE) / 1e9

    def measure():
        ev = common.evaluate(student, val_loader, device)
        return {"map": ev["map"], "f1": ev["f1"], "acc": ev["acc"]}

    base_gflops = gflops()
    base = measure()
    base_q = base.get(qm, 0.0)
    floor_gflops = MIN_GFLOPS_FRAC * base_gflops
    min_progress = MIN_PROGRESS_FRAC * base_gflops
    target_gflops = PRUNE_STEP_FRAC * base_gflops

    print(f"\n########## PARETO-SWEEP ({MODEL_TAG}) — klasifikacija, cilj: cijela Pareto krivulja ##########")
    if DEV_SUBSET:
        print(f"⚠️ DEV_SUBSET={DEV_SUBSET}: samo prvih {DEV_SUBSET} slika po splitu (niska vjernost — makni za pravi run).")
    print(f"  baseline: GFLOPs={base_gflops:.3f}  val_mAP={base['map']:.4f}  F1={base['f1']:.4f}  acc={base['acc']:.4f}  | metrika='{qm}'")
    print(f"  rez/korak={target_gflops:.3f} GFLOPs ({PRUNE_STEP_FRAC:.1%} orig.) | kriterij={CRITERION} | cap={PRUNE_LAYER_CAP:.0%}/sloj | "
          f"POD={floor_gflops:.3f} GFLOPs ({MIN_GFLOPS_FRAC:.0%} orig.) | FT/korak={FT_EPOCHS_PER_STEP}" +
          (f" | early-stop ako rez < {min_progress:.4f} GFLOPs ({MIN_PROGRESS_FRAC:.2%} orig.)" if MIN_PROGRESS_FRAC > 0 else ""))

    csv_f = open(OUT_CSV, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_COLS); writer.writeheader()

    def record(step, phase, gf, params, m, pruned, kd_loss, ft_eps):
        row = {"step": step, "phase": phase, "gflops": round(gf, 5),
               "gflops_pct": round(100.0 * gf / base_gflops, 3), "params": params,
               "size_mb": round(params * 4 / (1024 ** 2), 3),
               "val_map": round(m.get("map", 0.0), 5), "val_f1": round(m.get("f1", 0.0), 5),
               "val_acc": round(m.get("acc", 0.0), 5), "quality": round(m.get(qm, 0.0), 5),
               "gflops_freed_cum": round(base_gflops - gf, 5), "ft_epochs": ft_eps,
               "pruned_params": pruned, "kd_loss": round(kd_loss, 5),
               "wall_s": round(time.time() - t_start, 1)}
        writer.writerow(row); csv_f.flush()

    p0 = P.count_params(student)
    print(f"[ep 0] [ORIGINAL] val_mAP={base['map']:.4f} F1={base['f1']:.4f} acc={base['acc']:.4f} | GFLOPs={base_gflops:.3f}")
    record(0, "original", base_gflops, p0, base, 0, 0.0, 0)

    end_reason = f"dosegnut MAX_STEPS={MAX_STEPS}"
    for step in range(1, MAX_STEPS + 1):
        gf_start = gflops()
        if gf_start <= floor_gflops:
            end_reason = f"GFLOPs {gf_start:.3f} <= POD {floor_gflops:.3f} ({MIN_GFLOPS_FRAC:.0%} orig.)"
            break

        params_before = P.count_params(student)
        entries, by_name = P.build_plan(student)
        importance, _ = P.compute_importance(student, entries, CRITERION, kd_loss_fn, calib_batches, device)
        kept_idx, est_freed = select_prune_gflops(student, entries, by_name, importance, target_gflops,
                                                  PRUNE_LAYER_CAP, MIN_KEEP, device)
        P.apply_pruning(student, entries, by_name, kept_idx)
        params_after = P.count_params(student)
        pruned = params_before - params_after
        if pruned <= 0:
            end_reason = "nema više rezivih jedinica (svi slojevi na floor-u MIN_KEEP)"
            break

        P.recalibrate_bn(student, calib_batches, lambda b: b[0], device, reset=True)

        kd_loss = ft_kd(FT_EPOCHS_PER_STEP)

        m = measure()
        gf = gflops(); params = P.count_params(student)
        record(step, "prune+ft", gf, params, m, pruned, kd_loss, FT_EPOCHS_PER_STEP)
        print(f"[ep{step:3d}] [PRUNE+FT] -{pruned} params -> {params} ({params/p0*100:.1f}% orig.) | kd={kd_loss:.4f} | "
              f"val_mAP={m['map']:.4f} F1={m['f1']:.4f} acc={m['acc']:.4f} | GFLOPs={gf:.3f} ({gf/base_gflops*100:.1f}% orig.)")

        freed_step = gf_start - gf
        if MIN_PROGRESS_FRAC > 0 and freed_step < min_progress:
            end_reason = (f"early-stop: korak oslobodio {freed_step:.4f} GFLOPs < prag "
                          f"{min_progress:.4f} ({MIN_PROGRESS_FRAC:.2%} orig.) — diminishing returns")
            print(f"  >>> KRAJ: {end_reason}")
            break
    else:
        end_reason = f"dosegnut MAX_STEPS={MAX_STEPS}"

    csv_f.close()
    torch.save(student, OUT_MODEL)
    total_time = time.time() - t_start
    final_gf = gflops()
    meta = {
        "model": MODEL_TAG, "arch": "SchoolCNN", "task": "multilabel-classification",
        "ckpt": CKPT_PATH, "device": str(device), "dataset_root": str(common.DATASET_ROOT),
        "dev_subset": DEV_SUBSET, "quality_metric": qm,
        "baseline_gflops": base_gflops, "baseline_quality": base_q,
        "baseline_map": base["map"], "baseline_f1": base["f1"], "baseline_acc": base["acc"],
        "baseline_params": p0,
        "params": {"prune_step_frac": PRUNE_STEP_FRAC, "min_gflops_frac": MIN_GFLOPS_FRAC,
                   "min_progress_frac": MIN_PROGRESS_FRAC, "ft_epochs_per_step": FT_EPOCHS_PER_STEP,
                   "criterion": CRITERION, "min_keep": MIN_KEEP, "lr": LR, "batch_size": BATCH_SIZE},
        "n_steps": step, "final_gflops": final_gf, "final_gflops_pct": 100.0 * final_gf / base_gflops,
        "final_params": P.count_params(student), "end_reason": end_reason,
        "total_time_s": round(total_time, 1), "csv": OUT_CSV, "final_model": OUT_MODEL,
    }
    json.dump(meta, open(OUT_META, "w"), indent=2)

    print(f"\n########## PARETO-SWEEP GOTOV ({MODEL_TAG}) ##########")
    print(f"  razlog kraja: {end_reason}")
    print(f"  koraka: {step} | GFLOPs {base_gflops:.3f} -> {final_gf:.3f} ({final_gf/base_gflops*100:.1f}% orig.) | "
          f"params {p0} -> {P.count_params(student)}")
    print(f"  CSV:   {OUT_CSV}")
    print(f"  meta:  {OUT_META}")
    print(f"  model: {OUT_MODEL}")
    print(f"  UKUPNO VRIJEME: {total_time:.1f} s  ({total_time/60:.1f} min)")
    return meta


if __name__ == "__main__":
    run_sweep()
