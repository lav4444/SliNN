
import random
import time
from pathlib import Path

import cv2
import torch
from torchmetrics.detection import MeanAveragePrecision

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from KD_first import StudentYOLO
from train_kd import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    VAL_CONF_THRESH,
    VAL_IOU_THRESH,
    VAL_MAX_DET,
    boxes_letter_to_orig,
    postprocess_for_eval,
    preprocess_image,
)


SCRIPT_DIR = Path(__file__).parent
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
CKPT_PATH = SCRIPT_DIR / "checkpoints" / "best.pt"
RESULT_FILE = SCRIPT_DIR / "eval_result.txt"

SPLITS = ("train", "val", "test")

BENCHMARK_N_IMAGES = 10
BENCHMARK_N_DISCARD = 2
BENCHMARK_SEED = 42


def list_images(img_dir: Path):
    return sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_gt_yolo(label_file: Path, img_w: int, img_h: int):
    boxes, labels = [], []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes), torch.tensor(labels, dtype=torch.int64)


@torch.no_grad()
def eval_split(model, split: str, device: torch.device) -> dict:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    img_paths = list_images(img_dir)
    n_total = len(img_paths)
    print(f"[{split}] running inference on {n_total} images")

    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False

    sum_inf_ms = 0.0
    sum_total_ms = 0.0
    n = 0
    is_cuda = device.type == "cuda"

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]

        if is_cuda:
            torch.cuda.synchronize()
        t_pre = time.perf_counter()

        img_t, ratio, (dw, dh) = preprocess_image(img_bgr)
        img_t = img_t.unsqueeze(0).to(device, non_blocking=True)

        if is_cuda:
            torch.cuda.synchronize()
        t_inf_start = time.perf_counter()

        raw = model(img_t)
        boxes_dec, probs_dec = model.decode(raw)

        if is_cuda:
            torch.cuda.synchronize()
        t_inf_end = time.perf_counter()

        b_xyxy, sc, cl = postprocess_for_eval(
            boxes_dec[0].cpu(), probs_dec[0].cpu(),
            conf_thresh=VAL_CONF_THRESH, iou_thresh=VAL_IOU_THRESH, max_det=VAL_MAX_DET,
        )
        b_orig = boxes_letter_to_orig(b_xyxy, ratio, (dw, dh))

        if is_cuda:
            torch.cuda.synchronize()
        t_post = time.perf_counter()

        gt_boxes, gt_labels = load_gt_yolo(lbl_dir / f"{img_path.stem}.txt", orig_w, orig_h)
        metric.update(
            preds=[{"boxes": b_orig, "scores": sc, "labels": cl}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}],
        )

        sum_inf_ms += (t_inf_end - t_inf_start) * 1000.0
        sum_total_ms += (t_post - t_pre) * 1000.0
        n += 1

        if n % 200 == 0:
            print(f"  [{split}] {n}/{n_total}")

    print(f"[{split}] computing mAP...")
    m = metric.compute()

    per_cls = m.get("map_per_class")
    classes_present = m.get("classes")
    per_class_map = {}
    if per_cls is not None and classes_present is not None and per_cls.numel() > 0:
        per_class_map = {int(c): float(p) for c, p in zip(classes_present.tolist(), per_cls.tolist())}

    return {
        "n_images": n,
        "avg_inference_ms": sum_inf_ms / max(n, 1),
        "avg_total_ms": sum_total_ms / max(n, 1),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
        "mar_100": float(m["mar_100"]),
        "per_class_map": per_class_map,
    }


def write_eval_result(per_split: dict, ckpt_info: dict, ckpt_path: Path,
                      device: torch.device, n_params: int):
    lines = [
        f"Model: StudentYOLO ({n_params:,} params)",
        f"Checkpoint: {ckpt_path}",
        f"Trained epoch: {ckpt_info.get('epoch', '?')} "
        f"(val_kd_loss={ckpt_info.get('val_kd_loss', float('nan')):.4f}, "
        f"val_mAP@50:95={ckpt_info.get('val_map', float('nan')):.4f})",
        f"Image size: {IMG_SIZE}",
        f"Device: {device}",
        f"Eval conf threshold: {VAL_CONF_THRESH}",
        f"NMS IoU threshold: {VAL_IOU_THRESH}",
        "",
    ]
    for split in SPLITS:
        if split not in per_split:
            continue
        s = per_split[split]
        lines.append(f"=========== {split.upper()} ({s['n_images']} images) ===========")
        lines.append("")
        lines.append("Speed:")
        avg_inf = s["avg_inference_ms"]
        avg_total = s["avg_total_ms"]
        lines.append(f"  Inference (model only):       {avg_inf:.2f} ms/image  "
                     f"({1000.0 / max(avg_inf, 1e-9):.1f} FPS)")
        lines.append(f"  Total (preproc+inf+postproc): {avg_total:.2f} ms/image  "
                     f"({1000.0 / max(avg_total, 1e-9):.1f} FPS)")
        lines.append("")
        lines.append("Combined mAP:")
        lines.append(f"  mAP@50:95 = {s['map']:.4f}")
        lines.append(f"  mAP@50    = {s['map_50']:.4f}")
        lines.append(f"  mAP@75    = {s['map_75']:.4f}")
        lines.append(f"  mAR@100   = {s['mar_100']:.4f}")
        lines.append("")
        lines.append("Per-class mAP@50:95:")
        for i, name in enumerate(CLASS_NAMES):
            v = s["per_class_map"].get(i)
            if v is None:
                lines.append(f"  {name:<12} (no predictions or GT)")
            else:
                lines.append(f"  {name:<12} {v:.4f}")
        lines.append("")

    out = "\n".join(lines)
    RESULT_FILE.write_text(out)
    print("\n" + out)
    print(f"Saved: {RESULT_FILE}")


def benchmark_cpu_vs_gpu_latency(model: StudentYOLO, train_img_dir: Path) -> dict:
    rng = random.Random(BENCHMARK_SEED)
    all_imgs = list_images(train_img_dir)
    if len(all_imgs) < BENCHMARK_N_IMAGES:
        raise RuntimeError(
            f"Need {BENCHMARK_N_IMAGES} images, only {len(all_imgs)} found in {train_img_dir}"
        )
    sampled = rng.sample(all_imgs, BENCHMARK_N_IMAGES)

    print(f"\nBenchmark: preprocessing {BENCHMARK_N_IMAGES} random train images...")
    inputs = []
    for img_path in sampled:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img_t, _, _ = preprocess_image(img_bgr)
        inputs.append(img_t.unsqueeze(0))

    results: dict = {}

    for device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            results[device_name] = None
            continue

        device = torch.device(device_name)
        print(f"  benchmark on {device_name.upper()}...")
        model.to(device).eval()

        times_ms = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        sorted_times = sorted(times_ms)
        fast_times = sorted_times[:-BENCHMARK_N_DISCARD] if BENCHMARK_N_DISCARD > 0 else sorted_times
        results[device_name] = {
            "all_ms": times_ms,
            "fast_mean_ms": sum(fast_times) / len(fast_times),
            "n_fast": len(fast_times),
        }

    return results


def format_benchmark_results(results: dict) -> str:
    lines = [
        "",
        "=" * 64,
        f"CPU vs GPU latency benchmark "
        f"({BENCHMARK_N_IMAGES} random train images, seed={BENCHMARK_SEED})",
        "=" * 64,
        "",
    ]
    for device_name in ("cpu", "cuda"):
        r = results.get(device_name)
        label = "CPU" if device_name == "cpu" else "GPU"
        if r is None:
            lines.append(f"{label}: not available")
            continue
        times_str = ", ".join(f"{t:7.2f}" for t in r["all_ms"])
        lines.append(f"{label} per-image times (ms): [{times_str}]")
    lines.append("")
    lines.append(f"Mean of {BENCHMARK_N_IMAGES - BENCHMARK_N_DISCARD} fastest "
                 f"({BENCHMARK_N_DISCARD} slowest discarded as warmup):")

    cpu_r = results.get("cpu")
    gpu_r = results.get("cuda")
    if cpu_r is not None:
        cpu_ms = cpu_r["fast_mean_ms"]
        lines.append(f"  CPU:  {cpu_ms:>8.2f} ms/image  ({1000.0 / cpu_ms:.1f} FPS)")
    if gpu_r is not None:
        gpu_ms = gpu_r["fast_mean_ms"]
        lines.append(f"  GPU:  {gpu_ms:>8.2f} ms/image  ({1000.0 / gpu_ms:.1f} FPS)")
        if cpu_r is not None:
            lines.append(f"  Speedup (GPU vs CPU):  {cpu_ms / gpu_ms:.1f}x")
    return "\n".join(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_kd_loss={ckpt.get('val_kd_loss', float('nan')):.4f}, "
          f"val_mAP@50:95={ckpt.get('val_map', float('nan')):.4f})")

    model = StudentYOLO(num_classes=NUM_CLASSES, input_size=IMG_SIZE).to(device).eval()
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    per_split = {}
    for split in SPLITS:
        t0 = time.time()
        per_split[split] = eval_split(model, split, device)
        print(f"[{split}] done in {time.time() - t0:.1f}s "
              f"(mAP@50:95 = {per_split[split]['map']:.4f})\n")

    write_eval_result(per_split, ckpt, CKPT_PATH, device, n_params)

    train_img_dir = DATASET_ROOT / "images" / "train"
    bench_results = benchmark_cpu_vs_gpu_latency(model, train_img_dir)
    bench_text = format_benchmark_results(bench_results)
    print(bench_text)
    with RESULT_FILE.open("a") as f:
        f.write("\n" + bench_text + "\n")


if __name__ == "__main__":
    main()
