
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp

sys.path.insert(0, str(Path(__file__).parent.parent))
import common3 as C
import train_baseline3 as TB
import run_experiment3 as R


HERE = Path(__file__).parent
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"

NORMALIZERS = ["none", "mean", "max", "standarization", "gaussian", "sentinel_0.5", "lamp"]

NORM_NOTE = {
    "none":           "bez normalizacije (sirova vaznost; kontrola).",
    "mean":           "x / mean(x) (default iz exp3).",
    "max":            "x / max(x).",
    "standarization": "min-max u [0,1] (izjednaci raspon).",
    "gaussian":       "z-score (x-mu)/sigma (izjednaci srednju I varijancu).",
    "sentinel_0.5":   "x / medijan (robusno na outliere).",
    "lamp":           "LAMP layer-adaptive (ICLR'21).",
}

DEVICE = R.DEVICE
SAVE_MODELS = True


def make_taylor(norm):
    n = None if norm == "none" else norm
    return tp.importance.TaylorImportance(normalizer=n)


def _prune_trial(norm, ratio, calib):
    m = R.fresh_baseline_model()
    imp = make_taylor(norm)
    pruner = tp.pruner.MetaPruner(
        m, example_inputs=R.EXAMPLE(), importance=imp, pruning_ratio=ratio,
        global_pruning=R.GLOBAL_PRUNING, round_to=R.ROUND_TO,
        ignored_layers=C.prunable_ignored_layers(m))
    R.prepare_grads(m, imp, "taylor", calib)
    pruner.step()
    p = C.count_params(m)
    del m, imp, pruner
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return p


def find_param_ratio(norm, search_calib, target_params):
    lo, hi = 0.05, 0.95
    best_ratio, best_diff = 0.5, float("inf")
    for _ in range(R.RATIO_SEARCH_ITERS):
        mid = (lo + hi) / 2
        p = _prune_trial(norm, mid, search_calib)
        if abs(p - target_params) < best_diff:
            best_diff = abs(p - target_params); best_ratio = mid
        if p > target_params:
            lo = mid
        else:
            hi = mid
    return best_ratio


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def main():
    t_script = time.perf_counter()
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    eval_loaders = {s: C.make_loader(s, TB.BATCH_SIZE, shuffle=False,
                                     num_workers=TB.NUM_WORKERS, max_images=R.MAX_EVAL_IMAGES)
                    for s in R.EVAL_SPLITS}
    ft_train_loader = C.make_loader("train", TB.BATCH_SIZE, shuffle=True,
                                    num_workers=TB.NUM_WORKERS, drop_empty=True)
    ft_val_loader = C.make_loader("val", TB.BATCH_SIZE, shuffle=False, num_workers=TB.NUM_WORKERS)
    calib = []
    for i, b in enumerate(C.make_loader("train", TB.BATCH_SIZE, shuffle=True,
                                        num_workers=TB.NUM_WORKERS, drop_empty=True)):
        if i >= R.CALIB_BATCHES:
            break
        calib.append(b)

    RESULTS_FILE.write_text("")
    append_block("=" * 80)
    append_block("ABLACIJA NORMALIZACIJE VAZNOSTI — kriterij FIKSAN = Taylor")
    append_block("Model: fasterrcnn_mobilenet_v3_large_320_fpn | dataset: sub10k_open_images_v7 (6 kl)")
    append_block("=" * 80)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Normalizeri: {', '.join(NORMALIZERS)}")
    append_block(f"Param-target: zadrzi {R.KEEP_PARAM_FRAC*100:.0f}% total params (binarna pretraga ratija) | "
                 f"global={R.GLOBAL_PRUNING} | calib batcheva: {R.CALIB_BATCHES} (batch {TB.BATCH_SIZE})")
    append_block(f"FT: max {R.FT_MAX_EPOCHS} ep, LR={R.FT_LR}, early-stop patience {TB.PATIENCE} (val mAP) | "
                 f"eval cap {R.MAX_EVAL_IMAGES}/split | group_reduction=mean (tp default)")
    append_block("Metrika: mAP@50:95 / mAP@50 / mAP@75 / mAR@100 (torchmetrics)")
    append_block("")

    base = R.fresh_baseline_model()
    base_params = C.count_params(base)
    base_gmacs = C.backbone_gmacs(base, DEVICE)
    base_eval = R.eval_all(base, eval_loaders)
    base_bench = C.benchmark_latency(base)
    base_dead = C.count_dead(base, calib, DEVICE)
    append_block("-" * 80)
    append_block("BASELINE (COCO-mapped na 6 klasa, zero-shot, BEZ treninga)")
    append_block(f"  params={base_params:,}  backbone GMACs={base_gmacs:.3f}  size={base_params*4/1e6:.2f} MB")
    append_block(f"  GPU={base_bench['cuda']:.2f} ms/img  CPU={base_bench['cpu']:.2f} ms/img")
    append_block(f"  mrtvi filteri={base_dead['dead_filters']}/{base_dead['total_filters']}  "
                 f"mrtvi neuroni={base_dead['dead_neurons']}/{base_dead['total_neurons']}")
    append_block(R.fmt_eval(base_eval))
    append_block("")
    summary = {"baseline": {"params": base_params, "gmacs": base_gmacs,
                            "val_map": base_eval["val"]["map"], "test_map": base_eval["test"]["map"],
                            "gpu_ms": base_bench["cuda"], "cpu_ms": base_bench["cpu"],
                            "dead_filters": base_dead["dead_filters"], "total_filters": base_dead["total_filters"],
                            "dead_neurons": base_dead["dead_neurons"], "total_neurons": base_dead["total_neurons"]}}
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    target_params = R.KEEP_PARAM_FRAC * base_params
    search_calib = calib[:R.SEARCH_CALIB_BATCHES]
    append_block(f"PARAM-TARGET: zadrzi {R.KEEP_PARAM_FRAC*100:.0f}% total params "
                 f"(~{int(target_params):,}) | ratio se trazi binarno po normalizeru")
    append_block("")

    for norm in NORMALIZERS:
        print(f"\n>>> NORMALIZER: {norm}")
        append_block("=" * 80)
        append_block(f"NORMALIZER: {norm.upper()}  ({NORM_NOTE.get(norm, '')})")
        try:
            found_ratio = find_param_ratio(norm, search_calib, target_params)
            append_block(f"  [param-target] nadeni channel-ratio={found_ratio:.3f} za ~{R.KEEP_PARAM_FRAC*100:.0f}% params")

            model = R.fresh_baseline_model()
            imp = make_taylor(norm)
            ignored = C.prunable_ignored_layers(model)
            pruner = tp.pruner.MetaPruner(
                model, example_inputs=R.EXAMPLE(), importance=imp, pruning_ratio=found_ratio,
                global_pruning=R.GLOBAL_PRUNING, round_to=R.ROUND_TO, ignored_layers=ignored)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)
            t0 = time.perf_counter()
            n_imgs = R.prepare_grads(model, imp, "taylor", calib)
            pruner.step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            dt = time.perf_counter() - t0
            peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / 1e6) if DEVICE.type == "cuda" else float("nan")
            model.zero_grad(set_to_none=True)
            model.eval()

            pruned_params = C.count_params(model)
            pruned_gmacs = C.backbone_gmacs(model, DEVICE)
            ms_per_img = (dt * 1e3 / n_imgs) if n_imgs else 0.0
            append_block(f"  [vaznost+rez] vrijeme={dt:.2f}s ({ms_per_img:.2f} ms/slika, n={n_imgs}) | "
                         f"peak GPU={peak_mb:.0f} MB")
            append_block(f"  [rezanje] params {base_params:,} -> {pruned_params:,} "
                         f"({pruned_params/base_params*100:.1f}%) | backbone GMACs {base_gmacs:.3f} -> {pruned_gmacs:.3f}")
            append_block(f"  fc6: {model.roi_heads.box_head.fc6.in_features}->{model.roi_heads.box_head.fc6.out_features} "
                         f"| fc7: {model.roi_heads.box_head.fc7.in_features}->{model.roi_heads.box_head.fc7.out_features}")

            R.recalibrate_bn(model, calib)
            noft_eval = R.eval_all(model, eval_loaders)
            append_block("  --- BEZ FT (BN rekalibriran) ---")
            append_block(R.fmt_eval(noft_eval))

            print("  recovery fine-tuning...")
            best_state, ft = TB.train_loop(model, DEVICE, ft_train_loader, ft_val_loader,
                                           lr=R.FT_LR, max_epochs=R.FT_MAX_EPOCHS, patience=TB.PATIENCE,
                                           label=f"FT-{norm}")
            model.load_state_dict(best_state)
            append_block(f"  --- FINE-TUNE: epoha={ft['epochs_run']} ({ft['sec_per_epoch']:.1f} s/ep, "
                         f"ukupno {ft['total_time_s']/60:.1f} min) | best val mAP={ft['best_val_map']:.4f} @ {ft['best_epoch']} ---")

            ft_eval = R.eval_all(model, eval_loaders)
            append_block("  --- NAKON FINE-TUNEA ---")
            append_block(R.fmt_eval(ft_eval))
            bench = C.benchmark_latency(model)
            append_block(f"  [brzina nakon FT] GPU={bench['cuda']:.2f} ms/img  CPU={bench['cpu']:.2f} ms/img")
            dead = C.count_dead(model, calib, DEVICE)
            append_block(f"  [mrtve jedinice] filteri={dead['dead_filters']}/{dead['total_filters']}  "
                         f"neuroni={dead['dead_neurons']}/{dead['total_neurons']}")
            append_block("")

            if SAVE_MODELS:
                torch.save({"normalizer": norm, "model": best_state, "pruned_params": pruned_params},
                           MODELS_DIR / f"{norm}.pt")

            summary[norm] = {
                "ratio": found_ratio, "params": pruned_params, "gmacs": pruned_gmacs,
                "imp_ms_per_img": ms_per_img, "imp_peak_mb": peak_mb,
                "noft_val_map": noft_eval["val"]["map"],
                "ft_val_map": ft_eval["val"]["map"], "ft_test_map": ft_eval["test"]["map"],
                "ft_test_map50": ft_eval["test"]["map_50"],
                "ft_sec_per_epoch": ft["sec_per_epoch"], "ft_epochs": ft["epochs_run"],
                "gpu_ms": bench["cuda"], "cpu_ms": bench["cpu"],
                "dead_filters": dead["dead_filters"], "total_filters": dead["total_filters"],
                "dead_neurons": dead["dead_neurons"], "total_neurons": dead["total_neurons"]}
            del model, pruner, imp
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            append_block(f"  !!! GRESKA za normalizer '{norm}': {type(e).__name__}: {e}")
            append_block("")
            summary[norm] = {"error": f"{type(e).__name__}: {e}"}
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            continue

    append_block("=" * 80)
    append_block("USPOREDNA TABLICA (Taylor + razliciti normalizeri)")
    append_block("=" * 80)
    append_block(f"{'normalizer':<15}{'ratio':>7}{'params':>9}{'bbGMACs':>9}{'noFTval':>8}"
                 f"{'FTval':>7}{'FTtest':>8}{'GPUms':>7}{'CPUms':>7}{'deadF':>9}")
    append_block("-" * 86)
    b = summary["baseline"]
    append_block(f"{'baseline':<15}{'-':>7}{b['params']/1e6:>8.2f}M{b['gmacs']:>9.3f}{'-':>8}"
                 f"{b['val_map']:>7.3f}{b['test_map']:>8.3f}{b['gpu_ms']:>7.1f}{b['cpu_ms']:>7.1f}"
                 f"{b['dead_filters']:>4}/{b['total_filters']:<4}")
    for norm in NORMALIZERS:
        s = summary.get(norm, {})
        if "error" in s:
            append_block(f"{norm:<15}{'GRESKA: ' + s['error']:>0}")
            continue
        if not s:
            continue
        append_block(f"{norm:<15}{s['ratio']:>7.3f}{s['params']/1e6:>8.2f}M{s['gmacs']:>9.3f}"
                     f"{s['noft_val_map']:>8.3f}{s['ft_val_map']:>7.3f}{s['ft_test_map']:>8.3f}"
                     f"{s['gpu_ms']:>7.1f}{s['cpu_ms']:>7.1f}"
                     f"{s['dead_filters']:>4}/{s['total_filters']:<4}")
    append_block("-" * 86)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    total_min = (time.perf_counter() - t_script) / 60.0
    append_block(f"\nUKUPNO VRIJEME SKRIPTE: {total_min:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
