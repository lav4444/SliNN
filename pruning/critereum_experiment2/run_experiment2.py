
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from model_cnn import SchoolCNN, INPUT_SIZE, NUM_CLASSES, CLASS_NAMES
import common
import pruning_lib2 as PL
import train_baseline as TB


HERE = Path(__file__).parent
BASE_CKPT = HERE / "checkpoints" / "best.pt"
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"

METHODS = ["magnitude", "gradient", "taylor", "hessian"]

KEEP_PARAM_FRAC = 0.05
MIN_KEEP = 2
ALLOC = "global"

CALIB_BATCHES = 32
DO_RECALIB_EVAL = True

EVAL_SPLITS = ("val", "test")
MAX_EVAL_IMAGES = 2000

FT_MAX_EPOCHS = 10
FT_LR = TB.LR
SAVE_MODELS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPLEXITY_NOTE = {
    "magnitude": "O(#params); data-free; bez backward-a (najjeftiniji).",
    "gradient":  "1 backward prolaz po batchu; O(#params) memorije za grad.",
    "taylor":    "1 backward (g*W); slican trosak kao gradient.",
    "hessian":   "1 backward + g^2 (empirijski Fisher ~ dijag. Hessiana); ~ kao gradient.",
}


def fresh_baseline_model():
    ckpt = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
    model = SchoolCNN().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def bce_loss_fn(model, batch):
    x, y = batch
    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
    loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
    return loss, x.size(0)


def fmt_eval_table(per_split):
    lines = [f"  {'split':<7}{'mAP':>9}{'F1':>9}{'acc':>9}{'loss':>9}"]
    for split in EVAL_SPLITS:
        s = per_split[split]
        lines.append(f"  {split:<7}{s['map']:>9.4f}{s['f1']:>9.4f}{s['acc']:>9.4f}{s['loss']:>9.4f}")
    return "\n".join(lines)


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def main():
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_CKPT.exists():
        TB.train_baseline_and_save(DEVICE, BASE_CKPT)

    eval_loaders = {s: common.make_loader(s, TB.BATCH_SIZE, shuffle=False,
                                           num_workers=TB.NUM_WORKERS, max_images=MAX_EVAL_IMAGES)
                    for s in EVAL_SPLITS}
    ft_train_loader = common.make_loader("train", TB.BATCH_SIZE, shuffle=True, num_workers=TB.NUM_WORKERS)
    ft_val_loader = common.make_loader("val", TB.BATCH_SIZE, shuffle=False, num_workers=TB.NUM_WORKERS)
    calib_batches = []
    for i, b in enumerate(ft_train_loader):
        if i >= CALIB_BATCHES:
            break
        calib_batches.append(b)
    get_imgs = lambda b: b[0]

    def eval_all(model):
        return {s: common.evaluate(model, eval_loaders[s], DEVICE) for s in EVAL_SPLITS}

    RESULTS_FILE.write_text("")
    append_block("=" * 78)
    append_block("STRUCTURED PRUNING exp2 — kriteriji vaznosti (CONV FILTERI + DENSE NEURONI)")
    append_block("Model: SchoolCNN (multi-label, 6 klasa) | dataset: sub10k_open_images_v7")
    append_block("=" * 78)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Kriteriji: {', '.join(METHODS)}")
    append_block(f"Ciljani budzet: zadrzi {KEEP_PARAM_FRAC*100:.0f}% params "
                 f"(izbaci {100-KEEP_PARAM_FRAC*100:.0f}%) | alokacija: {ALLOC} | min_keep={MIN_KEEP}")
    append_block(f"Vaznost na {CALIB_BATCHES} batcheva (batch={TB.BATCH_SIZE}) | "
                 f"FT: max {FT_MAX_EPOCHS} epoha, LR={FT_LR:.2e} (baseline i FT, original/3), early-stop patience {TB.PATIENCE} (val_loss)")
    append_block(f"No-FT eval: BN-recalib={DO_RECALIB_EVAL} | splitovi: {', '.join(EVAL_SPLITS)} "
                 f"(cap {MAX_EVAL_IMAGES}/split)")
    append_block("Metrike: macro mAP (AP po klasi) + macro F1@0.5 + macro acc@0.5")
    append_block("Napomena: 'hessian' = dijagonala aproksimirana empirijskim Fisherom (OBD).")
    append_block("")

    base_model, base_ckpt = fresh_baseline_model()
    base_params = PL.count_params(base_model)
    base_flops = PL.count_flops(base_model, DEVICE, INPUT_SIZE)
    base_eval = eval_all(base_model)
    base_bench = common.benchmark_latency(base_model)
    append_block("-" * 78)
    append_block("BASELINE (bez rezanja)")
    append_block(f"  params={base_params:,}  GFLOPs={base_flops/1e9:.3f}  size={base_params*4/1e6:.2f} MB")
    append_block(f"  GPU={base_bench['cuda']:.2f} ms/img  CPU={base_bench['cpu']:.2f} ms/img")
    append_block(fmt_eval_table(base_eval))
    append_block("")

    summary = {"baseline": {
        "params": base_params, "gflops": base_flops / 1e9,
        "val_map": base_eval["val"]["map"], "test_map": base_eval["test"]["map"],
        "gpu_ms": base_bench["cuda"], "cpu_ms": base_bench["cpu"],
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

        importance, ov = PL.compute_importance(model, entries, method, bce_loss_fn, calib_batches, DEVICE)
        append_block(f"  [vaznost] vrijeme={ov['time_s']:.3f}s ({ov['time_ms_per_image']:.3f} ms/slika, "
                     f"n={ov['n_images']}) | peak GPU mem={ov['peak_mem_mb']:.1f} MB | backward={ov['needs_backward']}")

        t_sel = time.perf_counter()
        kept, info = PL.select_kept_indices(entries, by_name, importance, KEEP_PARAM_FRAC, MIN_KEEP, ALLOC)
        sel_ms = (time.perf_counter() - t_sel) * 1e3
        PL.apply_pruning(model, entries, by_name, kept)
        pruned_params = PL.count_params(model)
        pruned_flops = PL.count_flops(model, DEVICE, INPUT_SIZE)
        append_block(f"  [rezanje] params {info['baseline_params']:,} -> {pruned_params:,} "
                     f"({pruned_params/info['baseline_params']*100:.1f}% zadrzano) | "
                     f"GFLOPs {base_flops/1e9:.3f} -> {pruned_flops/1e9:.3f} | izbor={sel_ms:.1f} ms")
        append_block("  [po jedinici zadrzano] " +
                     "  ".join(f"{n}:{info['per_layer_kept'][n]}/{info['per_layer_orig'][n]}"
                              for n in info["per_layer_kept"]))

        recalib_eval = None
        if DO_RECALIB_EVAL:
            rec = copy.deepcopy(model)
            PL.recalibrate_bn(rec, calib_batches, get_imgs, DEVICE, reset=True)
            recalib_eval = eval_all(rec)
            append_block("  --- NAKON REZANJA, BEZ FT (BN rekalibriran) ---")
            append_block(fmt_eval_table(recalib_eval))
            del rec
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        print("  fine-tuning...")
        best_state, ft = TB.train_loop(model, DEVICE, ft_train_loader, ft_val_loader,
                                       lr=FT_LR, max_epochs=FT_MAX_EPOCHS, patience=TB.PATIENCE,
                                       label=f"FT-{method}")
        model.load_state_dict(best_state)
        append_block(f"  --- FINE-TUNE: epoha={ft['epochs_run']} ({ft['sec_per_epoch']:.1f} s/epoha, "
                     f"ukupno {ft['total_time_s']/60:.1f} min) | best val_loss={ft['best_val_loss']:.4f} "
                     f"@ {ft['best_epoch']} (val mAP={ft['best_val_map']:.4f}) ---")

        ft_eval = eval_all(model)
        append_block("  --- NAKON FINE-TUNEA ---")
        append_block(fmt_eval_table(ft_eval))
        bench = common.benchmark_latency(model)
        append_block(f"  [brzina nakon FT] GPU={bench['cuda']:.2f} ms/img  CPU={bench['cpu']:.2f} ms/img")
        pc = ft_eval["test"]["per_class_ap"]
        append_block("  [per-class AP test, FT] " + "  ".join(f"{k}:{v:.3f}" for k, v in pc.items()))
        append_block("")

        if SAVE_MODELS:
            torch.save({"method": method, "model": best_state,
                        "kept": {k: v.tolist() for k, v in kept.items()},
                        "pruned_params": pruned_params}, MODELS_DIR / f"{method}.pt")

        summary[method] = {
            "params": pruned_params, "gflops": pruned_flops / 1e9,
            "imp_ms_per_img": ov["time_ms_per_image"], "imp_peak_mb": ov["peak_mem_mb"],
            "noft_recalib_val_map": (recalib_eval["val"]["map"] if recalib_eval else float("nan")),
            "ft_val_map": ft_eval["val"]["map"], "ft_test_map": ft_eval["test"]["map"],
            "ft_test_f1": ft_eval["test"]["f1"], "ft_test_acc": ft_eval["test"]["acc"],
            "ft_sec_per_epoch": ft["sec_per_epoch"], "ft_epochs": ft["epochs_run"],
            "gpu_ms": bench["cuda"], "cpu_ms": bench["cpu"],
        }
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    append_block("=" * 78)
    append_block("USPOREDNA TABLICA (head-to-head)")
    append_block("=" * 78)
    append_block(f"{'metoda':<11}{'params':>9}{'GFLOPs':>8}{'impMS/img':>10}"
                 f"{'noFT val':>9}{'FT val':>8}{'FT test':>8}{'testF1':>8}{'s/epoha':>8}{'GPUms':>7}{'CPUms':>7}")
    append_block("-" * 78)
    b = summary["baseline"]
    append_block(f"{'baseline':<11}{b['params']/1e6:>8.3f}M{b['gflops']:>8.2f}{'-':>10}"
                 f"{'-':>9}{b['val_map']:>8.4f}{b['test_map']:>8.4f}{'-':>8}{'-':>8}"
                 f"{b['gpu_ms']:>7.2f}{b['cpu_ms']:>7.2f}")
    for m in METHODS:
        s = summary[m]
        append_block(f"{m:<11}{s['params']/1e6:>8.3f}M{s['gflops']:>8.2f}"
                     f"{s['imp_ms_per_img']:>10.3f}{s['noft_recalib_val_map']:>9.4f}"
                     f"{s['ft_val_map']:>8.4f}{s['ft_test_map']:>8.4f}{s['ft_test_f1']:>8.4f}"
                     f"{s['ft_sec_per_epoch']:>8.1f}{s['gpu_ms']:>7.2f}{s['cpu_ms']:>7.2f}")
    append_block("-" * 78)
    append_block("Legenda: noFT val = val mAP nakon rezanja (BN rekalib, bez FT); "
                 "FT val/test = macro mAP nakon fine-tunea; impMS/img = overhead vaznosti.")

    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    append_block(f"\nSpremljeno: {RESULTS_FILE}")
    append_block(f"Spremljeno: {HERE / 'summary.json'}")
    if SAVE_MODELS:
        append_block(f"Spremljeni modeli: {MODELS_DIR}/<metoda>.pt")


if __name__ == "__main__":
    main()
