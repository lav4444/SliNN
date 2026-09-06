
import sys
import time
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

STUDENT_DIR = Path("/home/tomi/code/dipl/custom_models/student_2_m")
PURE_KD_DIR = STUDENT_DIR / "pure_KD"
sys.path.insert(0, str(PURE_KD_DIR))
sys.path.insert(0, str(STUDENT_DIR))

from model_arch import StudentYOLO                                  # noqa: E402
import train_kd as TK                                              # noqa: E402
import evaluate_student as EV                                      # noqa: E402

import pruning_lib as PL                                           # noqa: E402


HERE = Path(__file__).parent
BASE_CKPT = PURE_KD_DIR / "checkpoints" / "best.pt"
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"

METHODS = ["magnitude", "gradient", "taylor", "hessian"]

KEEP_PARAM_FRAC = 0.50
MIN_KEEP = 2
ALLOC = "global"

CALIB_BATCHES = 32
DO_RAW_EVAL = False
DO_RECALIB_EVAL = True

EVAL_SPLITS = ("val", "test")
MAX_EVAL_IMAGES = 600
DO_BASELINE_EVAL = False

FT_MAX_EPOCHS = 5
FT_LR = 1e-4

SAVE_MODELS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_orig_list_images = EV.list_images
def _capped_list_images(img_dir):
    imgs = _orig_list_images(img_dir)
    return imgs[:MAX_EVAL_IMAGES] if MAX_EVAL_IMAGES else imgs
EV.list_images = _capped_list_images

COMPLEXITY_NOTE = {
    "magnitude": "O(#params); data-free; bez backward-a (najjeftiniji).",
    "gradient":  "1 backward prolaz po batchu; O(#params) memorije za grad.",
    "taylor":    "1 backward (g*W); slican trosak kao gradient.",
    "hessian":   "1 backward + g^2 (empirijski Fisher ~ dijag. Hessiana); ~ kao gradient.",
}


def fresh_baseline_model():
    ckpt = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    model = StudentYOLO(num_classes=TK.NUM_CLASSES, input_size=TK.IMG_SIZE).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def make_calib_batches(n_batches):
    ds = TK.KDTrainDataset(TK.TRAIN_IMG_DIR, TK.TEACHER_SOFT_DIR)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=TK.BATCH_SIZE, shuffle=True, num_workers=TK.NUM_WORKERS,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(TK.SEED),
    )
    batches = []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(b)
    return batches


def kd_loss_fn(model, batch):
    imgs = batch["image"].to(DEVICE, non_blocking=True)
    tb = batch["teacher_boxes"].to(DEVICE, non_blocking=True)
    tp = batch["teacher_probs"].to(DEVICE, non_blocking=True)
    raw = model(imgs)
    loss, _, _ = TK.kd_loss(raw, tb, tp, model.anchor_xy, model.anchor_stride)
    return loss, imgs.size(0)


def eval_all_splits(model):
    model.eval()
    out = {}
    for split in EVAL_SPLITS:
        out[split] = EV.eval_split(model, split, DEVICE)
    return out


def fine_tune(model):
    train_ds = TK.KDTrainDataset(TK.TRAIN_IMG_DIR, TK.TEACHER_SOFT_DIR)
    val_ds = TK.ValDataset(TK.VAL_IMG_DIR, TK.VAL_LBL_DIR, TK.VAL_TEACHER_SOFT_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=TK.BATCH_SIZE, shuffle=True, num_workers=TK.NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=TK.NUM_WORKERS > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=TK.BATCH_SIZE, shuffle=False, num_workers=TK.NUM_WORKERS,
        pin_memory=True, collate_fn=TK.val_collate, persistent_workers=TK.NUM_WORKERS > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=TK.WEIGHT_DECAY)
    iters_per_epoch = len(train_loader)
    total_iters = FT_MAX_EPOCHS * iters_per_epoch
    warmup_iters = TK.WARMUP_EPOCHS * iters_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, TK.make_lr_lambda(total_iters, warmup_iters))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    anchor_xy, anchor_stride = model.anchor_xy, model.anchor_stride

    best_val_map = -1.0
    best_epoch = None
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    lr_reduced_in_streak = False

    epoch_times = []
    t_start = time.time()
    epochs_run = 0

    for epoch in range(1, FT_MAX_EPOCHS + 1):
        model.train()
        t_ep = time.time()
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            tb = batch["teacher_boxes"].to(DEVICE, non_blocking=True)
            tp = batch["teacher_probs"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                raw = model(imgs)
                loss, _, _ = TK.kd_loss(raw, tb, tp, anchor_xy, anchor_stride)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), TK.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        epoch_times.append(time.time() - t_ep)
        epochs_run = epoch

        is_val_epoch = ((epoch - 1) % TK.VAL_EVERY_N_EPOCHS) == 0
        if not is_val_epoch:
            continue

        val_metrics = TK.validate(model, val_loader, DEVICE, anchor_xy, anchor_stride)
        vm = val_metrics["map"]
        print(f"    [FT epoch {epoch:3d}] val_mAP@50:95={vm:.4f} "
              f"val_kd_loss={val_metrics['kd_loss']:.4f} ({epoch_times[-1]:.1f}s)")

        if vm > best_val_map:
            best_val_map = vm
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            lr_reduced_in_streak = False
        else:
            epochs_without_improvement += TK.VAL_EVERY_N_EPOCHS
            if (epochs_without_improvement >= TK.LR_REDUCE_PATIENCE
                    and not lr_reduced_in_streak):
                scheduler.base_lrs = [lr * TK.LR_REDUCE_FACTOR for lr in scheduler.base_lrs]
                for pg in optimizer.param_groups:
                    pg["lr"] *= TK.LR_REDUCE_FACTOR
                lr_reduced_in_streak = True
            if epochs_without_improvement >= TK.EARLY_STOP_PATIENCE:
                print(f"    [FT] early stop @ epoch {epoch} (best {best_val_map:.4f} @ {best_epoch})")
                break

    total_time = time.time() - t_start
    ft_stats = {
        "epochs_run": epochs_run,
        "total_time_s": total_time,
        "sec_per_epoch": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
        "best_val_map": best_val_map,
        "best_epoch": best_epoch,
    }
    return best_state, ft_stats


def fmt_split_table(per_split):
    lines = []
    hdr = f"  {'split':<7}{'mAP50:95':>10}{'mAP50':>9}{'mAP75':>9}{'mAR100':>9}{'inf ms':>9}"
    lines.append(hdr)
    for split in EVAL_SPLITS:
        if split not in per_split:
            continue
        s = per_split[split]
        lines.append(f"  {split:<7}{s['map']:>10.4f}{s['map_50']:>9.4f}"
                     f"{s['map_75']:>9.4f}{s['mar_100']:>9.4f}{s['avg_inference_ms']:>9.2f}")
    return "\n".join(lines)


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def main():
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_FILE.write_text("")
    append_block("=" * 78)
    append_block("STRUCTURED PRUNING — usporedba kriterija vaznosti kanala/filtera")
    append_block("Model: StudentYOLO (pure_KD)  |  dataset: sub10k_open_images_v7")
    append_block("=" * 78)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Kriteriji: {', '.join(METHODS)}")
    append_block(f"Ciljani budzet: zadrzi {KEEP_PARAM_FRAC*100:.0f}% params "
                 f"(izbaci {100-KEEP_PARAM_FRAC*100:.0f}%) | alokacija: {ALLOC} | min_keep={MIN_KEEP}")
    append_block(f"Vaznost na {CALIB_BATCHES} batcheva (batch={TK.BATCH_SIZE}) | "
                 f"FT (oporavak, NE od nule): max {FT_MAX_EPOCHS} epoha, LR={FT_LR}, "
                 f"early-stop kao u KD treningu")
    append_block(f"No-FT eval: raw={DO_RAW_EVAL}, BN-recalib={DO_RECALIB_EVAL} | "
                 f"splitovi: {', '.join(EVAL_SPLITS)}")
    append_block("Napomena: StudentYOLO je cisto konvolucijski -> rezu se conv filteri (kanali).")
    append_block("Napomena: 'hessian' = dijagonala aproksimirana empirijskim Fisherom (OBD).")
    append_block("")

    print(">>> BASELINE (neprunani model)")
    base_model, base_ckpt = fresh_baseline_model()
    base_params = PL.count_params(base_model)
    base_flops = PL.count_flops(base_model, DEVICE)

    append_block("-" * 78)
    append_block("BASELINE (bez rezanja)")
    append_block(f"  params={base_params:,}  GFLOPs={base_flops/1e9:.3f}  "
                 f"size={base_params*4/1e6:.2f} MB")

    base_eval = None
    base_gpu = base_cpu = None
    if DO_BASELINE_EVAL:
        base_eval = eval_all_splits(base_model)
        base_bench = EV.benchmark_cpu_vs_gpu_latency(base_model, EV.DATASET_ROOT / "images" / "train")
        base_gpu = base_bench.get("cuda")
        base_cpu = base_bench.get("cpu")
        append_block(f"  GPU={base_gpu['fast_mean_ms']:.2f} ms/img  "
                     f"CPU={base_cpu['fast_mean_ms']:.2f} ms/img")
        append_block(fmt_split_table(base_eval))
    else:
        append_block("  (baseline eval preskocen -> referenca: pure_KD/eval_result.txt)")
    append_block("")

    calib_batches = make_calib_batches(CALIB_BATCHES)
    get_imgs = lambda b: b["image"]

    summary = {"baseline": {
        "params": base_params, "gflops": base_flops / 1e9,
        "val_map": base_eval["val"]["map"] if base_eval else float("nan"),
        "test_map": base_eval["test"]["map"] if base_eval else float("nan"),
        "gpu_ms": base_gpu["fast_mean_ms"] if base_gpu else float("nan"),
        "cpu_ms": base_cpu["fast_mean_ms"] if base_cpu else float("nan"),
    }}

    del base_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for method in METHODS:
        print(f"\n>>> METODA: {method}")
        append_block("=" * 78)
        append_block(f"METODA: {method.upper()}")
        append_block(f"  Slozenost: {COMPLEXITY_NOTE[method]}")

        model, _ = fresh_baseline_model()
        entries, by_name = PL.build_plan(model)

        importance, overhead = PL.compute_importance(
            model, entries, method, kd_loss_fn, calib_batches, DEVICE)
        append_block(f"  [vaznost] vrijeme={overhead['time_s']:.3f}s "
                     f"({overhead['time_ms_per_image']:.3f} ms/slika, n={overhead['n_images']}) "
                     f"| peak GPU mem={overhead['peak_mem_mb']:.1f} MB "
                     f"| backward={overhead['needs_backward']}")

        t_sel = time.perf_counter()
        kept, sel_info = PL.select_kept_indices(
            entries, by_name, importance, KEEP_PARAM_FRAC, MIN_KEEP, ALLOC)
        sel_time = time.perf_counter() - t_sel
        PL.apply_pruning(model, entries, by_name, kept)
        pruned_params = PL.count_params(model)
        pruned_flops = PL.count_flops(model, DEVICE)
        append_block(f"  [rezanje] params {sel_info['baseline_params']:,} -> {pruned_params:,} "
                     f"({pruned_params/sel_info['baseline_params']*100:.1f}% zadrzano) "
                     f"| GFLOPs {base_flops/1e9:.3f} -> {pruned_flops/1e9:.3f} "
                     f"| izbor={sel_time*1e3:.1f} ms")
        layer_str = "  ".join(f"{n}:{sel_info['per_layer_kept'][n]}/{sel_info['per_layer_orig'][n]}"
                              for n in sel_info["per_layer_kept"])
        append_block(f"  [po sloju zadrzano] {layer_str}")

        if DO_RAW_EVAL:
            raw_eval = eval_all_splits(model)
            append_block("  --- NAKON REZANJA, BEZ FT (sirovi BN) ---")
            append_block(fmt_split_table(raw_eval))

        recalib_eval = None
        if DO_RECALIB_EVAL:
            recal_model = copy.deepcopy(model)
            PL.recalibrate_bn(recal_model, calib_batches, get_imgs, DEVICE, reset=True)
            recalib_eval = eval_all_splits(recal_model)
            append_block("  --- NAKON REZANJA, BEZ FT (BN rekalibriran) ---")
            append_block(fmt_split_table(recalib_eval))
            del recal_model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        print("  fine-tuning...")
        best_state, ft_stats = fine_tune(model)
        model.load_state_dict(best_state)
        append_block(f"  --- FINE-TUNE: epoha={ft_stats['epochs_run']} "
                     f"({ft_stats['sec_per_epoch']:.1f} s/epoha, ukupno {ft_stats['total_time_s']/60:.1f} min) "
                     f"| best val mAP={ft_stats['best_val_map']:.4f} @ {ft_stats['best_epoch']} ---")

        ft_eval = eval_all_splits(model)
        append_block("  --- NAKON FINE-TUNEA ---")
        append_block(fmt_split_table(ft_eval))

        bench = EV.benchmark_cpu_vs_gpu_latency(model, EV.DATASET_ROOT / "images" / "train")
        gpu_ms = bench["cuda"]["fast_mean_ms"] if bench.get("cuda") else float("nan")
        cpu_ms = bench["cpu"]["fast_mean_ms"] if bench.get("cpu") else float("nan")
        append_block(f"  [brzina nakon FT] GPU={gpu_ms:.2f} ms/img  CPU={cpu_ms:.2f} ms/img")

        pc = ft_eval["test"]["per_class_map"]
        pc_str = "  ".join(f"{TK.CLASS_NAMES[i]}:{pc.get(i, float('nan')):.3f}"
                           for i in range(TK.NUM_CLASSES))
        append_block(f"  [per-class mAP test, FT] {pc_str}")
        append_block("")

        if SAVE_MODELS:
            torch.save({"method": method, "model": best_state,
                        "kept": {k: v.tolist() for k, v in kept.items()},
                        "pruned_params": pruned_params}, MODELS_DIR / f"{method}.pt")

        summary[method] = {
            "params": pruned_params, "gflops": pruned_flops / 1e9,
            "imp_ms_per_img": overhead["time_ms_per_image"],
            "imp_peak_mb": overhead["peak_mem_mb"],
            "noft_recalib_val_map": (recalib_eval["val"]["map"] if recalib_eval else float("nan")),
            "ft_val_map": ft_eval["val"]["map"], "ft_test_map": ft_eval["test"]["map"],
            "ft_sec_per_epoch": ft_stats["sec_per_epoch"], "ft_epochs": ft_stats["epochs_run"],
            "gpu_ms": gpu_ms, "cpu_ms": cpu_ms,
        }

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    append_block("=" * 78)
    append_block("USPOREDNA TABLICA (head-to-head)")
    append_block("=" * 78)
    cols = (f"{'metoda':<11}{'params':>9}{'GFLOPs':>8}{'impMS/img':>10}"
            f"{'noFT val':>9}{'FT val':>8}{'FT test':>8}{'s/epoha':>8}{'GPUms':>7}{'CPUms':>7}")
    append_block(cols)
    append_block("-" * 78)
    b = summary["baseline"]
    append_block(f"{'baseline':<11}{b['params']/1e6:>8.3f}M{b['gflops']:>8.2f}{'-':>10}"
                 f"{'-':>9}{b['val_map']:>8.4f}{b['test_map']:>8.4f}{'-':>8}"
                 f"{b['gpu_ms']:>7.2f}{b['cpu_ms']:>7.2f}")
    for m in METHODS:
        s = summary[m]
        append_block(f"{m:<11}{s['params']/1e6:>8.3f}M{s['gflops']:>8.2f}"
                     f"{s['imp_ms_per_img']:>10.3f}{s['noft_recalib_val_map']:>9.4f}"
                     f"{s['ft_val_map']:>8.4f}{s['ft_test_map']:>8.4f}"
                     f"{s['ft_sec_per_epoch']:>8.1f}{s['gpu_ms']:>7.2f}{s['cpu_ms']:>7.2f}")
    append_block("-" * 78)
    append_block("Legenda: noFT val = val mAP@50:95 nakon rezanja (BN rekalib, bez FT); "
                 "FT val/test = nakon fine-tunea; impMS/img = overhead izracuna vaznosti.")

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    append_block(f"\nSpremljeno: {RESULTS_FILE}")
    append_block(f"Spremljeno: {HERE / 'summary.json'}")
    if SAVE_MODELS:
        append_block(f"Spremljeni modeli: {MODELS_DIR}/<metoda>.pt")


if __name__ == "__main__":
    main()
