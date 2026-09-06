
import sys
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch_pruning as tp

sys.path.insert(0, str(Path(__file__).parent.parent))
import common3 as C
import train_baseline3 as TB
import run_experiment3 as R
import inloss_lib as IL


HERE = Path(__file__).parent
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"
POSTHOC_SUMMARY = HERE.parent / "summary.json"

MODES = ["l1", "l0"]
KEEP_PARAM_FRAC = R.KEEP_PARAM_FRAC
DEVICE = R.DEVICE

PRUNE_STEPS = 6
PATIENCE_FT = 2
MAX_EP_PER_STEP = 5
PRE_FT_PATIENCE = 2
MAX_EP_PRE = 3
PATIENCE_FINAL = 3
MAX_EP_FINAL = 10
BATCH_SIZE = 16
NUM_WORKERS = 2
LR_MODEL = 0.005
LR_GATE = 0.2
KD_WEIGHT = 1.0
LAMBDA = 0.1
GRAD_CLIP = 5.0
WARMUP_ITERS = 500
POST_FT_EPOCHS = 8
EVAL_RECALIB_BATCHES = max(4, 128 // BATCH_SIZE)
MON_VAL_IMAGES = 400
CALIB_BATCHES = max(8, 256 // BATCH_SIZE)
SAVE_MODELS = True


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def plot_results(summary, posthoc):
    base_M = summary["baseline"]["params"] / 1e6
    base_map = summary["baseline"]["val_map"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, mode in zip(axes, MODES):
        s = summary.get(mode, {})
        hist = s.get("history", [])
        if hist:
            xs = [h["eff_frac"] * base_M for h in hist]
            ys = [h["val_map"] for h in hist]
            ax.plot(xs, ys, "-o", color="C0", ms=4, lw=1.5,
                    label="training (soft-prune trajectory)")
            ax.annotate(f"ep{hist[0]['epoch']}", (xs[0], ys[0]), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
            ax.annotate(f"ep{hist[-1]['epoch']}", (xs[-1], ys[-1]), fontsize=8,
                        textcoords="offset points", xytext=(4, -10))
        if "params" in s:
            ax.scatter([s["params"] / 1e6], [s.get("ft_val_map", float("nan"))],
                       marker="*", s=280, color="C3", zorder=5, edgecolor="k",
                       label="final (materialized + post-FT)")
        ax.scatter([base_M], [base_map], marker="D", s=70, color="gray",
                   zorder=4, label="baseline (unpruned)")
        if posthoc:
            ax.scatter([posthoc["params"] / 1e6], [posthoc.get("ft_val_map", float("nan"))],
                       marker="s", s=70, color="green", zorder=4, label="post-hoc Taylor (exp3)")
        ax.axvline(KEEP_PARAM_FRAC * base_M, ls="--", color="k", alpha=0.35)
        ax.text(KEEP_PARAM_FRAC * base_M, ax.get_ylim()[0], f" target {KEEP_PARAM_FRAC*100:.0f}%",
                fontsize=8, va="bottom", alpha=0.6)
        ax.set_title(f"{mode.upper()}  (pruning embedded in loss)")
        ax.set_xlabel("# parameters (M)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("val mAP@[.5:.95]")
    fig.suptitle("In-loss pruning: # parameters vs performance (L1 vs L0)", fontsize=13)
    fig.tight_layout()
    out = HERE / "params_vs_map.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def move_batch(imgs, targets):
    imgs = [im.to(DEVICE, non_blocking=True) for im in imgs]
    targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
    return imgs, targets


def build_fixed_train_loader():
    import random as _random
    from torch.utils.data import DataLoader
    ds = C.DetDataset("train", drop_empty=True)
    _random.Random(42).shuffle(ds.items)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=True, collate_fn=C.det_collate,
                      persistent_workers=NUM_WORKERS > 0)


@torch.no_grad()
def eval_with_bn_recalib(model, gc, calib_small, val_loader):
    gc.eval()
    moms = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            moms[m] = m.momentum
            m.reset_running_stats()
            m.momentum = None
    model.train()
    for imgs, targets in calib_small:
        imgs, targets = move_batch(imgs, targets)
        model(imgs, targets)
    for m, mm in moms.items():
        m.momentum = mm
    model.eval()
    return C.evaluate(model, val_loader, DEVICE)


def train_variant(mode, teacher, student, gc, kd, train_loader, val_loader, total_params, calib_small):
    student.to(DEVICE); teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model_params = [p for p in student.parameters() if p.requires_grad]
    gate_params = [p for p in gc.parameters() if p.requires_grad]
    optim = torch.optim.SGD([
        {"params": model_params, "lr": LR_MODEL},
        {"params": gate_params, "lr": LR_GATE},
    ], momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    warmup_iters = min(WARMUP_ITERS, max(1, steps_per_epoch - 1))
    final_removed = (1.0 - KEEP_PARAM_FRAC) * total_params
    st = {"it": 0, "epoch": 0}
    history = []
    teacher_cache = []

    def lr_scale(i):
        return (i + 1) / warmup_iters if i < warmup_iters else 1.0

    def run_epoch():
        student.train(); gc.train()
        t_ep = time.time(); rd = rk = rp = 0.0; nb = 0
        for bi, (imgs, targets) in enumerate(train_loader):
            imgs, targets = move_batch(imgs, targets)
            if bi < len(teacher_cache):
                t_feat = {k: v.to(DEVICE, non_blocking=True) for k, v in teacher_cache[bi].items()}
            else:
                with torch.no_grad():
                    t_imgs, _ = teacher.transform(imgs)
                    teacher.backbone(t_imgs.tensors)
                t_feat = kd.t_feat
                teacher_cache.append({k: v.detach().half().cpu() for k, v in t_feat.items()})
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss_dict = student(imgs, targets)
                det = sum(loss_dict.values())
                kd_l = kd.loss_with(t_feat)
                pen = gc.penalty()
                loss = det + KD_WEIGHT * kd_l + LAMBDA * pen
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model_params + gate_params, GRAD_CLIP)
            sc = lr_scale(st["it"])
            optim.param_groups[0]["lr"] = LR_MODEL * sc
            optim.param_groups[1]["lr"] = LR_GATE * sc
            scaler.step(optim); scaler.update()
            st["it"] += 1
            rd += float(det); rk += float(kd_l); rp += float(pen); nb += 1
        return rd / nb, rk / nb, rp / nb, time.time() - t_ep

    def ft_until_plateau(phase, patience, max_ep, track_best=False):
        best_map = -1.0; no_improve = 0; best_state = None; ep = 0
        while ep < max_ep and no_improve < patience:
            ep += 1; st["epoch"] += 1
            det, kdl, pen, dt = run_epoch()
            val = eval_with_bn_recalib(student, gc, calib_small, val_loader)
            eff = gc.effective_params(total_params); alive, tot = gc.alive_fraction()
            history.append({"epoch": st["epoch"], "val_map": val["map"], "eff_frac": eff / total_params})
            append_block(f"    [{mode} {phase} ep{st['epoch']:2d}] det={det:.3f} kd={kdl:.3f} pen={pen:.3f} "
                         f"| eff={eff/total_params*100:.1f}% alive={alive}/{tot} | val_mAP={val['map']:.4f} ({dt:.0f}s)")
            if val["map"] > best_map + 1e-4:
                best_map = val["map"]; no_improve = 0
                if track_best:
                    best_state = copy.deepcopy(student.state_dict())
            else:
                no_improve += 1
        return best_state, best_map

    ft_until_plateau("pre", PRE_FT_PATIENCE, MAX_EP_PRE)
    for k in range(1, PRUNE_STEPS + 1):
        gc.prune_to_removed_params(k / PRUNE_STEPS * final_removed)
        eff = gc.effective_params(total_params)
        append_block(f"    [{mode}] >>> REZ {k}/{PRUNE_STEPS} -> eff={eff/total_params*100:.1f}% params; FT do platoa...")
        ft_until_plateau(f"ft{k}", PATIENCE_FT, MAX_EP_PER_STEP)
    best_state, best = ft_until_plateau("final", PATIENCE_FINAL, MAX_EP_FINAL, track_best=True)
    if best_state is not None:
        student.load_state_dict(best_state)
        append_block(f"    [{mode}] finalni best val mAP={best:.4f} (ucitano za materijalizaciju)")
    return history


def materialize(student, gc, target_frac):
    gc.fold_into_weights()
    gc.remove()
    student.eval()
    base = C.count_params(student)
    target = target_frac * base

    def trial_ratio(ratio):
        m = copy.deepcopy(student)
        pr = tp.pruner.MetaPruner(m, example_inputs=R.EXAMPLE(), importance=tp.importance.MagnitudeImportance(p=1, normalizer=None, group_reduction="mean",
                                                                target_types=[nn.BatchNorm2d, nn.Linear]),
                                  pruning_ratio=ratio, global_pruning=R.GLOBAL_PRUNING,
                                  round_to=R.ROUND_TO, ignored_layers=C.prunable_ignored_layers(m))
        pr.step()
        p = C.count_params(m)
        del m, pr
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return p

    lo, hi, best_r, best_d = 0.05, 0.95, 0.5, float("inf")
    for _ in range(R.RATIO_SEARCH_ITERS):
        mid = (lo + hi) / 2
        p = trial_ratio(mid)
        if abs(p - target) < best_d:
            best_d = abs(p - target); best_r = mid
        if p > target:
            lo = mid
        else:
            hi = mid
    pruner = tp.pruner.MetaPruner(student, example_inputs=R.EXAMPLE(), importance=tp.importance.MagnitudeImportance(p=1, normalizer=None, group_reduction="mean",
                                                                target_types=[nn.BatchNorm2d, nn.Linear]),
                                  pruning_ratio=best_r, global_pruning=R.GLOBAL_PRUNING,
                                  round_to=R.ROUND_TO, ignored_layers=C.prunable_ignored_layers(student))
    pruner.step()
    student.eval()
    return best_r


def run_variant(mode, loaders, calib, base_params, base_gmacs):
    append_block("=" * 80)
    append_block(f"VARIJANTA: {mode.upper()}  (in-loss {'L1 |gate|' if mode == 'l1' else 'L0 hard-concrete'} "
                 f"+ cost-weighted, KD=FPN-MSE)")

    teacher = C.build_model(pretrained=True, coco_map=True).to(DEVICE).eval()
    student = C.build_model(pretrained=True, coco_map=True).to(DEVICE)
    gc = IL.GateController(student, mode=mode).to(DEVICE)
    IL.compute_costs(gc, student, DEVICE)
    gc.attach()
    kd = IL.FPNFeatureKD(teacher, student)

    t0 = time.perf_counter()
    history = train_variant(mode, teacher, student, gc, kd,
                            loaders["ft_train"], loaders["mon_val"], base_params,
                            calib_small=calib[:EVAL_RECALIB_BATCHES])
    kd.remove()
    best_r = materialize(student, gc, KEEP_PARAM_FRAC)
    train_min = (time.perf_counter() - t0) / 60.0

    pruned_params = C.count_params(student)
    pruned_gmacs = C.backbone_gmacs(student, DEVICE)
    append_block(f"  [materijalizacija] tp ratio={best_r:.3f} | params {base_params:,} -> {pruned_params:,} "
                 f"({pruned_params/base_params*100:.1f}%) | bbGMACs {base_gmacs:.3f} -> {pruned_gmacs:.3f}")
    append_block(f"  fc6: {student.roi_heads.box_head.fc6.in_features}->{student.roi_heads.box_head.fc6.out_features} "
                 f"| fc7: {student.roi_heads.box_head.fc7.in_features}->{student.roi_heads.box_head.fc7.out_features}")

    R.recalibrate_bn(student, calib)
    noft = R.eval_all(student, {s: loaders[s] for s in R.EVAL_SPLITS})
    append_block("  --- prije recovery FT (BN rekalibriran) ---")
    append_block(R.fmt_eval(noft))

    best_state, ft = TB.train_loop(student, DEVICE, loaders["ft_train"], loaders["mon_val"],
                                   lr=R.FT_LR, max_epochs=POST_FT_EPOCHS, patience=TB.PATIENCE,
                                   label=f"postFT-{mode}")
    student.load_state_dict(best_state)
    ft_eval = R.eval_all(student, {s: loaders[s] for s in R.EVAL_SPLITS})
    append_block(f"  --- nakon recovery FT ({ft['epochs_run']} ep, early-stop) | best val mAP={ft['best_val_map']:.4f} ---")
    append_block(R.fmt_eval(ft_eval))

    bench = C.benchmark_latency(student)
    dead = C.count_dead(student, calib, DEVICE)
    append_block(f"  [brzina] GPU={bench['cuda']:.2f} ms/img  CPU={bench['cpu']:.2f} ms/img")
    append_block(f"  [mrtve jedinice] filteri={dead['dead_filters']}/{dead['total_filters']}  "
                 f"neuroni={dead['dead_neurons']}/{dead['total_neurons']}")
    append_block(f"  [vrijeme treninga varijante] {train_min:.1f} min")
    append_block("")

    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"mode": mode, "model": best_state, "pruned_params": pruned_params},
                   MODELS_DIR / f"{mode}.pt")

    out = {"params": pruned_params, "gmacs": pruned_gmacs,
           "noft_val_map": noft["val"]["map"], "ft_val_map": ft_eval["val"]["map"],
           "ft_test_map": ft_eval["test"]["map"], "gpu_ms": bench["cuda"], "cpu_ms": bench["cpu"],
           "dead_filters": dead["dead_filters"], "total_filters": dead["total_filters"],
           "dead_neurons": dead["dead_neurons"], "total_neurons": dead["total_neurons"],
           "train_min": train_min, "history": history}
    del teacher, student, gc, kd
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    t_script = time.perf_counter()
    loaders = {
        "ft_train": build_fixed_train_loader(),
        "ft_val": C.make_loader("val", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        "mon_val": C.make_loader("val", BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, max_images=MON_VAL_IMAGES),
    }
    for s in R.EVAL_SPLITS:
        loaders[s] = C.make_loader(s, BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, max_images=R.MAX_EVAL_IMAGES)
    from torch.utils.data import DataLoader as _DL
    calib_ds = C.DetDataset("train", max_images=CALIB_BATCHES * BATCH_SIZE, drop_empty=True)
    calib = list(_DL(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                     pin_memory=False, collate_fn=C.det_collate))

    RESULTS_FILE.write_text("")
    append_block("=" * 80)
    append_block("PRUNING UGRADEN U LOSS (train-time, risk-reward) — L1 vs L0 + KD(FPN-MSE)")
    append_block("Model: fasterrcnn_mobilenet_v3_large_320_fpn | dataset: sub10k_open_images_v7 (6 kl)")
    append_block("=" * 80)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Cilj: zadrzi {KEEP_PARAM_FRAC*100:.0f}% total params | iterativno: "
                 f"(pre-FT) -> [REZ -> FT do platoa (patience {PATIENCE_FT})] x {PRUNE_STEPS} -> finalni FT")
    append_block(f"reward=cost (0.5*FLOPs+0.5*params), risk=gate grad | LAMBDA={LAMBDA}, KD_W={KD_WEIGHT}")
    append_block(f"LR_model={LR_MODEL}, LR_gate={LR_GATE} (warmup->const) | postFT max {POST_FT_EPOCHS} ep | "
                 f"eval cap {R.MAX_EVAL_IMAGES}/split")
    append_block("")

    base = C.build_model(pretrained=True, coco_map=True).to(DEVICE).eval()
    base_params = C.count_params(base)
    base_gmacs = C.backbone_gmacs(base, DEVICE)
    base_eval = R.eval_all(base, {s: loaders[s] for s in R.EVAL_SPLITS})
    base_bench = C.benchmark_latency(base)
    append_block("-" * 80)
    append_block("BASELINE (COCO-mapped, nepruned, BEZ treninga)")
    append_block(f"  params={base_params:,}  bbGMACs={base_gmacs:.3f}")
    append_block(f"  GPU={base_bench['cuda']:.2f} ms/img  CPU={base_bench['cpu']:.2f} ms/img")
    append_block(R.fmt_eval(base_eval))
    append_block("")
    summary = {"baseline": {"params": base_params, "gmacs": base_gmacs,
                            "val_map": base_eval["val"]["map"], "test_map": base_eval["test"]["map"],
                            "gpu_ms": base_bench["cuda"], "cpu_ms": base_bench["cpu"]}}
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for mode in MODES:
        print(f"\n>>> VARIJANTA: {mode}")
        summary[mode] = run_variant(mode, loaders, calib, base_params, base_gmacs)

    posthoc = {}
    if POSTHOC_SUMMARY.exists():
        try:
            ph = json.loads(POSTHOC_SUMMARY.read_text())
            if "taylor" in ph:
                posthoc = ph["taylor"]
        except Exception:
            pass

    append_block("=" * 80)
    append_block("USPOREDNA TABLICA — in-loss (L1/L0) vs post-hoc Taylor (exp3)")
    append_block("=" * 80)
    append_block(f"{'pristup':<18}{'params':>9}{'bbGMACs':>9}{'noFTval':>8}{'FTval':>7}"
                 f"{'FTtest':>8}{'GPUms':>7}{'CPUms':>7}{'deadF':>10}")
    append_block("-" * 86)
    b = summary["baseline"]
    append_block(f"{'baseline':<18}{b['params']/1e6:>8.2f}M{b['gmacs']:>9.3f}{'-':>8}"
                 f"{b['val_map']:>7.3f}{b['test_map']:>8.3f}{b['gpu_ms']:>7.1f}{b['cpu_ms']:>7.1f}{'-':>10}")
    for mode in MODES:
        s = summary[mode]
        append_block(f"{'in-loss ' + mode.upper():<18}{s['params']/1e6:>8.2f}M{s['gmacs']:>9.3f}"
                     f"{s['noft_val_map']:>8.3f}{s['ft_val_map']:>7.3f}{s['ft_test_map']:>8.3f}"
                     f"{s['gpu_ms']:>7.1f}{s['cpu_ms']:>7.1f}{s['dead_filters']:>4}/{s['total_filters']:<5}")
    if posthoc:
        append_block(f"{'post-hoc Taylor':<18}{posthoc['params']/1e6:>8.2f}M{posthoc['gmacs']:>9.3f}"
                     f"{posthoc['noft_val_map']:>8.3f}{posthoc['ft_val_map']:>7.3f}{posthoc['ft_test_map']:>8.3f}"
                     f"{posthoc['gpu_ms']:>7.1f}{posthoc['cpu_ms']:>7.1f}"
                     f"{posthoc['dead_filters']:>4}/{posthoc['total_filters']:<5}")
    append_block("-" * 86)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        graf = plot_results(summary, posthoc)
        append_block(f"Graf spremljen: {graf}")
    except Exception as e:
        append_block(f"(graf preskocen: {type(e).__name__}: {e})")
    append_block(f"\nUKUPNO VRIJEME SKRIPTE: {(time.perf_counter()-t_script)/60:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
