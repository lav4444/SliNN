"""
run_experiment3.py — usporedba 4 kriterija vaznosti za STRUCTURED pruning na
PRODUCTION detektoru (fasterrcnn_mobilenet_v3_large_320_fpn), lokalizacija+klasifikacija.

Dependency + surgery: Torch-Pruning (tp) — automatski rjesava depthwise/SE/residual/FPN.
Kriteriji (NASI, preko tp group-aware importancea):
    magnitude -> tp MagnitudeImportance(p=1)
    gradient  -> custom GradientImportance (|grad|, swap weight<->grad)
    taylor    -> tp TaylorImportance (g*w)
    hessian   -> tp HessianImportance (Fisher ~ dijag. Hessiana)

Scope: prune-amo backbone.body (MobileNetV3) + roi_heads.box_head (fc6/fc7).
Zamrznuto (ignored): FPN izlaz (256-interfejs), RPN glava, box predictor.

Protokol po metodi: vaznost(+overhead) -> tp prune (global) -> BN rekalib ->
eval bez FT -> recovery fine-tune (max 10 ep) -> eval -> latencija (CPU/GPU).

Pokretanje:  python run_experiment3.py   (baseline se trenira ako fali)
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp

import common3 as C
import train_baseline3 as TB


# =========================== KONFIGURACIJA =========================== #
HERE = Path(__file__).parent
BASE_CKPT = HERE / "checkpoints" / "best.pt"
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"

METHODS = ["magnitude", "gradient", "taylor", "hessian"]
KEEP_PARAM_FRAC = 0.70      # CILJ: zadrzi X% TOTAL params (izjednaceno za sve 4 metode)
GLOBAL_PRUNING = True
ROUND_TO = 8                # zaokruzi kanale na visekratnik 8 (uskladi depthwise/grupe -> bez conv/BN mismatcha)
SEARCH_CALIB_BATCHES = 16   # manji calib za binarnu pretragu ratija (brze)
RATIO_SEARCH_ITERS = 7      # koraka binarne pretrage
CALIB_BATCHES = 128         # ~500 slika (125 x batch 4) za vaznost, BN rekalib i dead-count
MAX_EVAL_IMAGES = 1000
EVAL_SPLITS = ("val", "test")
FT_MAX_EPOCHS = 2           # recovery FT (max 2 epohe; best se prati po epohi)
FT_LR = 0.0005              # recovery peak LR (blag fine-tune; + warmup->cosine + grad clip)
SAVE_MODELS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLE = lambda: torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device=DEVICE)

COMPLEXITY_NOTE = {
    "magnitude": "data-free, bez backward-a (najjeftiniji).",
    "gradient":  "akumulirani backward (|grad|).",
    "taylor":    "akumulirani backward (g*w).",
    "hessian":   "per-batch Fisher (g^2) akumulacija.",
}


# ---- custom gradient importance: |grad| preko tp group logike ----
class GradientImportance(tp.importance.GroupMagnitudeImportance):
    @torch.no_grad()
    def __call__(self, group):
        saved = []
        for dep, idxs in group:
            layer = dep.layer
            for t in (getattr(layer, "weight", None), getattr(layer, "bias", None)):
                if t is not None:
                    saved.append((t, t.data))
                    t.data = t.grad.detach().abs() if t.grad is not None else torch.zeros_like(t.data)
        try:
            return super().__call__(group)
        finally:
            for t, d in saved:
                t.data = d


def make_importance(method):
    if method == "magnitude":
        return tp.importance.MagnitudeImportance(p=1)
    if method == "gradient":
        return GradientImportance(p=1)
    if method == "taylor":
        return tp.importance.TaylorImportance()
    if method == "hessian":
        return tp.importance.HessianImportance()
    raise ValueError(method)


def _prune_trial(method, ratio, calib, ignored_fn):
    """Svjezi model -> prune na zadani channel-ratio -> vrati broj params (za pretragu)."""
    m = fresh_baseline_model()
    imp = make_importance(method)
    pruner = tp.pruner.MetaPruner(m, example_inputs=EXAMPLE(), importance=imp,
                                  pruning_ratio=ratio, global_pruning=GLOBAL_PRUNING,
                                  round_to=ROUND_TO, ignored_layers=ignored_fn(m))
    prepare_grads(m, imp, method, calib)
    pruner.step()
    p = C.count_params(m)
    del m, imp, pruner
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return p


def find_param_ratio(method, search_calib, target_params):
    """Binarna pretraga channel-ratija koji daje ~target_params (TOTAL). Vraca (ratio)."""
    lo, hi = 0.05, 0.95
    best_ratio, best_diff = 0.5, float("inf")
    for _ in range(RATIO_SEARCH_ITERS):
        mid = (lo + hi) / 2
        p = _prune_trial(method, mid, search_calib, C.prunable_ignored_layers)
        if abs(p - target_params) < best_diff:
            best_diff = abs(p - target_params); best_ratio = mid
        if p > target_params:      # zadrzano previse -> rezi vise (veci ratio)
            lo = mid
        else:                      # zadrzano premalo -> rezi manje (manji ratio)
            hi = mid
    return best_ratio


# =========================== POMOCNE =========================== #
def fresh_baseline_model():
    # baseline BEZ treninga: COCO-pretrained + mapiranje na nasih 6 klasa (zero-shot)
    model = C.build_model(pretrained=True, coco_map=True).to(DEVICE)
    model.eval()
    return model


def move_batch(imgs, targets):
    imgs = [im.to(DEVICE, non_blocking=True) for im in imgs]
    targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
    return imgs, targets


def prepare_grads(model, imp, method, calib):
    """Pripremi gradijente za grad-bazirane kriterije. Vrati broj slika.
    Svi grad tenzori se inic. na 0 (ne None) -> neaktivni parametri dobiju
    vaznost 0 umjesto pada (TaylorImportance bi pao na None.grad)."""
    if method == "magnitude":
        return 0
    model.train()
    n = 0
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    if method == "hessian":
        imp.zero_grad()
        for imgs, targets in calib:
            imgs, targets = move_batch(imgs, targets)
            for p in model.parameters():        # reset na 0 (per-batch Fisher = g^2)
                p.grad.zero_()
            loss = sum(model(imgs, targets).values())
            loss.backward()
            imp.accumulate_grad(model)
            n += len(imgs)
    else:  # gradient, taylor -> akumuliraj .grad (bez zeroiranja izmedu)
        for imgs, targets in calib:
            imgs, targets = move_batch(imgs, targets)
            loss = sum(model(imgs, targets).values())
            loss.backward()
            n += len(imgs)
    return n


@torch.no_grad()
def recalibrate_bn(model, calib):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    model.train()
    for imgs, targets in calib:
        imgs, targets = move_batch(imgs, targets)
        model(imgs, targets)        # train mode -> osvjezi BN statistike
    model.eval()


def eval_all(model, loaders):
    return {s: C.evaluate(model, loaders[s], DEVICE) for s in EVAL_SPLITS}


def fmt_eval(per_split):
    lines = [f"  {'split':<7}{'mAP':>9}{'mAP50':>9}{'mAP75':>9}{'mAR100':>9}"]
    for s in EVAL_SPLITS:
        e = per_split[s]
        lines.append(f"  {s:<7}{e['map']:>9.4f}{e['map_50']:>9.4f}{e['map_75']:>9.4f}{e['mar_100']:>9.4f}")
    return "\n".join(lines)


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


# =========================== MAIN =========================== #
def main():
    t_script = time.perf_counter()
    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Bez treninga baseline-a: koristimo COCO-mapped model (fresh_baseline_model).

    eval_loaders = {s: C.make_loader(s, TB.BATCH_SIZE, shuffle=False,
                                     num_workers=TB.NUM_WORKERS, max_images=MAX_EVAL_IMAGES)
                    for s in EVAL_SPLITS}
    ft_train_loader = C.make_loader("train", TB.BATCH_SIZE, shuffle=True,
                                    num_workers=TB.NUM_WORKERS, drop_empty=True)
    ft_val_loader = C.make_loader("val", TB.BATCH_SIZE, shuffle=False, num_workers=TB.NUM_WORKERS)
    calib = []
    for i, b in enumerate(C.make_loader("train", TB.BATCH_SIZE, shuffle=True,
                                        num_workers=TB.NUM_WORKERS, drop_empty=True)):
        if i >= CALIB_BATCHES:
            break
        calib.append(b)

    # ---- header ----
    RESULTS_FILE.write_text("")
    append_block("=" * 80)
    append_block("STRUCTURED PRUNING exp3 — production detektor (Torch-Pruning + nasi kriteriji)")
    append_block("Model: fasterrcnn_mobilenet_v3_large_320_fpn | dataset: sub10k_open_images_v7 (6 kl)")
    append_block("=" * 80)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Kriteriji: {', '.join(METHODS)} (preko tp group-aware importancea)")
    append_block(f"Scope: backbone.body + roi_heads.box_head | ignored: FPN izlaz, RPN, box_predictor")
    append_block(f"Param-target: zadrzi {KEEP_PARAM_FRAC*100:.0f}% total params (binarna pretraga ratija) | "
                 f"global={GLOBAL_PRUNING} | calib batcheva: {CALIB_BATCHES} (batch {TB.BATCH_SIZE})")
    append_block(f"FT: max {FT_MAX_EPOCHS} ep, LR={FT_LR}, early-stop patience {TB.PATIENCE} (val mAP) | "
                 f"eval cap {MAX_EVAL_IMAGES}/split")
    append_block("Metrika: mAP@50:95 / mAP@50 / mAP@75 / mAR@100 (torchmetrics)")
    append_block("")

    # ---- baseline ----
    base = fresh_baseline_model()
    base_params = C.count_params(base)
    base_gmacs = C.backbone_gmacs(base, DEVICE)
    base_eval = eval_all(base, eval_loaders)
    base_bench = C.benchmark_latency(base)
    base_dead = C.count_dead(base, calib, DEVICE)
    append_block("-" * 80)
    append_block("BASELINE (COCO-mapped na 6 klasa, zero-shot, BEZ treninga)")
    append_block(f"  params={base_params:,}  backbone GMACs={base_gmacs:.3f}  size={base_params*4/1e6:.2f} MB")
    append_block(f"  GPU={base_bench['cuda']:.2f} ms/img  CPU={base_bench['cpu']:.2f} ms/img")
    append_block(f"  mrtvi filteri={base_dead['dead_filters']}/{base_dead['total_filters']}  "
                 f"mrtvi neuroni={base_dead['dead_neurons']}/{base_dead['total_neurons']}")
    append_block(fmt_eval(base_eval))
    append_block("")
    summary = {"baseline": {"params": base_params, "gmacs": base_gmacs,
                            "val_map": base_eval["val"]["map"], "test_map": base_eval["test"]["map"],
                            "gpu_ms": base_bench["cuda"], "cpu_ms": base_bench["cpu"],
                            "dead_filters": base_dead["dead_filters"], "total_filters": base_dead["total_filters"],
                            "dead_neurons": base_dead["dead_neurons"], "total_neurons": base_dead["total_neurons"]}}
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    target_params = KEEP_PARAM_FRAC * base_params
    search_calib = calib[:SEARCH_CALIB_BATCHES]
    append_block(f"PARAM-TARGET: zadrzi {KEEP_PARAM_FRAC*100:.0f}% total params "
                 f"(~{int(target_params):,}) | ratio se trazi binarno po metodi")
    append_block("")

    # ---- po metodi ----
    for method in METHODS:
        print(f"\n>>> METODA: {method}")
        append_block("=" * 80)
        append_block(f"METODA: {method.upper()}  ({COMPLEXITY_NOTE[method]})")

        # nadji channel-ratio koji daje ciljani % TOTAL params (izjednaceno za sve metode)
        found_ratio = find_param_ratio(method, search_calib, target_params)
        append_block(f"  [param-target] nadeni channel-ratio={found_ratio:.3f} za ~{KEEP_PARAM_FRAC*100:.0f}% params")

        model = fresh_baseline_model()
        imp = make_importance(method)
        ignored = C.prunable_ignored_layers(model)
        pruner = tp.pruner.MetaPruner(
            model, example_inputs=EXAMPLE(), importance=imp, pruning_ratio=found_ratio,
            global_pruning=GLOBAL_PRUNING, round_to=ROUND_TO, ignored_layers=ignored)

        # (1) vaznost (grad-prep) + (2) prune  -- mjeri overhead zajedno
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        n_imgs = prepare_grads(model, imp, method, calib)
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

        # (3) BN rekalib + eval bez FT
        recalibrate_bn(model, calib)
        noft_eval = eval_all(model, eval_loaders)
        append_block("  --- BEZ FT (BN rekalibriran) ---")
        append_block(fmt_eval(noft_eval))

        # (4) recovery fine-tune
        print("  recovery fine-tuning...")
        best_state, ft = TB.train_loop(model, DEVICE, ft_train_loader, ft_val_loader,
                                       lr=FT_LR, max_epochs=FT_MAX_EPOCHS, patience=TB.PATIENCE,
                                       label=f"FT-{method}")
        model.load_state_dict(best_state)
        append_block(f"  --- FINE-TUNE: epoha={ft['epochs_run']} ({ft['sec_per_epoch']:.1f} s/ep, "
                     f"ukupno {ft['total_time_s']/60:.1f} min) | best val mAP={ft['best_val_map']:.4f} @ {ft['best_epoch']} ---")

        ft_eval = eval_all(model, eval_loaders)
        append_block("  --- NAKON FINE-TUNEA ---")
        append_block(fmt_eval(ft_eval))
        bench = C.benchmark_latency(model)
        append_block(f"  [brzina nakon FT] GPU={bench['cuda']:.2f} ms/img  CPU={bench['cpu']:.2f} ms/img")
        dead = C.count_dead(model, calib, DEVICE)
        append_block(f"  [mrtve jedinice] filteri={dead['dead_filters']}/{dead['total_filters']}  "
                     f"neuroni={dead['dead_neurons']}/{dead['total_neurons']}")
        append_block("")

        if SAVE_MODELS:
            torch.save({"method": method, "model": best_state, "pruned_params": pruned_params},
                       MODELS_DIR / f"{method}.pt")

        summary[method] = {
            "params": pruned_params, "gmacs": pruned_gmacs,
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

    # ---- usporedna tablica ----
    append_block("=" * 80)
    append_block("USPOREDNA TABLICA (head-to-head)")
    append_block("=" * 80)
    append_block(f"{'metoda':<11}{'params':>9}{'bbGMACs':>9}{'impMS':>7}{'noFTval':>8}"
                 f"{'FTval':>7}{'FTtest':>8}{'GPUms':>7}{'CPUms':>7}{'deadF':>9}{'deadN':>9}")
    append_block("-" * 88)
    b = summary["baseline"]
    append_block(f"{'baseline':<11}{b['params']/1e6:>8.2f}M{b['gmacs']:>9.3f}{'-':>7}{'-':>8}"
                 f"{b['val_map']:>7.3f}{b['test_map']:>8.3f}{b['gpu_ms']:>7.1f}{b['cpu_ms']:>7.1f}"
                 f"{b['dead_filters']:>4}/{b['total_filters']:<4}{b['dead_neurons']:>4}/{b['total_neurons']:<4}")
    for m in METHODS:
        s = summary[m]
        append_block(f"{m:<11}{s['params']/1e6:>8.2f}M{s['gmacs']:>9.3f}{s['imp_ms_per_img']:>7.1f}"
                     f"{s['noft_val_map']:>8.3f}{s['ft_val_map']:>7.3f}{s['ft_test_map']:>8.3f}"
                     f"{s['gpu_ms']:>7.1f}{s['cpu_ms']:>7.1f}"
                     f"{s['dead_filters']:>4}/{s['total_filters']:<4}{s['dead_neurons']:>4}/{s['total_neurons']:<4}")
    append_block("-" * 88)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    total_min = (time.perf_counter() - t_script) / 60.0
    append_block(f"\nUKUPNO VRIJEME SKRIPTE: {total_min:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
