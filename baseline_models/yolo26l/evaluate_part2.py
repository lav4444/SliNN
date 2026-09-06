
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).parent
MODEL_NAME = "yolo26l.pt"
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7_part2")
PRED_ROOT = DATASET_ROOT / "yolo26l"

IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUR_CLASSES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
OURS_TO_COCO = [0, 2, 7, 5, 3, 1]
COCO_TO_OURS = {coco: ours for ours, coco in enumerate(OURS_TO_COCO)}
KEEP_COCO_IDS = list(COCO_TO_OURS.keys())

EVAL_CONF = 0.001
HARD_LABEL_CONF = 0.25
NMS_IOU = 0.7
MAX_DET = 300

BENCHMARK_N_IMAGES = 10
BENCHMARK_N_DISCARD = 2
BENCHMARK_SEED = 42

EXTRACT_BACKBONE_FEATURES = True
BACKBONE_P3_LAYER_IDX = 4

SPLITS = ("train", "val", "test")
META_FILE = PRED_ROOT / "meta.json"
EVAL_RESULT_FILE = SCRIPT_DIR / "eval_result_part2.txt"


def split_paths(split: str) -> dict:
    return {
        "img_dir": DATASET_ROOT / "images" / split,
        "lbl_dir": DATASET_ROOT / "labels" / split,
        "soft_dir": PRED_ROOT / split / "soft",
        "hard_dir": PRED_ROOT / split / "labels",
    }


def list_images(img_dir: Path):
    return sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def find_detect_head(model: YOLO):
    candidates = [m for m in model.model.modules()
                  if m.__class__.__name__ in ("Detect", "v10Detect", "DetectV2")]
    if not candidates:
        raise RuntimeError("Could not locate a Detect head on the model.")
    return candidates[-1]


def letterbox_meta(orig_w: int, orig_h: int) -> dict:
    ratio = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))
    pad_x = (IMG_SIZE - new_w) / 2.0
    pad_y = (IMG_SIZE - new_h) / 2.0
    return {
        "ratio": (ratio, ratio),
        "pad": (pad_x, pad_y),
        "orig_size": (orig_w, orig_h),
        "input_size": (IMG_SIZE, IMG_SIZE),
    }


def letterbox_image(img_bgr, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (IMG_SIZE - new_unpad[0]) / 2.0
    dh = (IMG_SIZE - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_bgr, r, (dw, dh)


def preprocess_image(img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {img_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    img_lb, r, (dw, dh) = letterbox_image(img_bgr)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).float() / 255.0
    return tensor, r, (dw, dh), orig_w, orig_h


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


def write_hard_yolo(label_file: Path, boxes_xyxy, scores, labels, img_w: int, img_h: int):
    lines = []
    for (x1, y1, x2, y2), conf, cls in zip(boxes_xyxy, scores, labels):
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(conf):.6f}")
    label_file.write_text("\n".join(lines) + ("\n" if lines else ""))


def nms_post_process(boxes_xywh: torch.Tensor, class_probs: torch.Tensor,
                     conf_thresh: float, iou_thresh: float, max_det: int):
    scores, classes = class_probs.max(dim=-1)
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return (torch.zeros((0, 4)), torch.zeros((0,)),
                torch.zeros((0,), dtype=torch.int64))
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    classes = classes[keep]
    cx, cy, w, h = boxes_xywh.unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    max_coord = boxes_xyxy.max() if boxes_xyxy.numel() else 0.0
    offsets = classes.float() * (max_coord + 1)
    boxes_for_nms = boxes_xyxy + offsets.unsqueeze(-1)
    keep_idx = tvops.nms(boxes_for_nms, scores, iou_thresh)[:max_det]
    return boxes_xyxy[keep_idx], scores[keep_idx], classes[keep_idx]


def boxes_letter_to_orig(boxes_xyxy: torch.Tensor, lb: dict) -> torch.Tensor:
    rx = lb["ratio"][0]
    pad_x, pad_y = lb["pad"]
    out = boxes_xyxy.clone()
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= rx
    return out


def is_split_complete(paths: dict) -> bool:
    soft_dir = paths["soft_dir"]
    if not soft_dir.exists():
        return False
    image_stems = {p.stem for p in list_images(paths["img_dir"])}
    soft_stems = {p.stem for p in soft_dir.glob("*.pt")}
    if not (image_stems.issubset(soft_stems) and len(image_stems) > 0):
        return False
    if EXTRACT_BACKBONE_FEATURES:
        sample_pt = next(iter(sorted(soft_dir.glob("*.pt"))))
        sample_data = torch.load(sample_pt, weights_only=False)
        if "p3_features" not in sample_data:
            print(f"  [{soft_dir.name}] cache present but missing p3_features → will re-infer")
            return False
    return True


@torch.no_grad()
def run_inference_for_split(model: YOLO, split: str, paths: dict) -> dict:
    paths["soft_dir"].mkdir(parents=True, exist_ok=True)
    paths["hard_dir"].mkdir(parents=True, exist_ok=True)

    img_paths = list_images(paths["img_dir"])
    n_total = len(img_paths)
    print(f"[{split}] running inference on {n_total} images")

    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False
    sum_inf_ms = 0.0
    sum_total_ms = 0.0
    n = 0
    anchor_count = None
    is_cuda = DEVICE == "cuda"

    model.model.eval()

    features_cache: dict = {}
    hook_handle = None
    if EXTRACT_BACKBONE_FEATURES:
        def _p3_hook(module, inp, output):
            features_cache["p3"] = output.detach()
        backbone_layers = model.model.model
        hook_handle = backbone_layers[BACKBONE_P3_LAYER_IDX].register_forward_hook(_p3_hook)
        print(f"[setup] registered P3 hook on backbone layer {BACKBONE_P3_LAYER_IDX} "
              f"(feature shape will be reported on first sample)")

    for img_path in img_paths:
        n += 1
        stem = img_path.stem

        t_pre = time.perf_counter()
        tensor, ratio, (pad_x, pad_y), orig_w, orig_h = preprocess_image(img_path)
        tensor = tensor.unsqueeze(0).to(DEVICE, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize()
        t_inf_start = time.perf_counter()

        out = model.model(tensor)
        if is_cuda:
            torch.cuda.synchronize()
        t_inf_end = time.perf_counter()

        if isinstance(out, (tuple, list)):
            out = out[0]
        dense = out[0]
        n_classes = dense.shape[0] - 4
        if anchor_count is None:
            anchor_count = int(dense.shape[1])

        boxes_xywh = dense[:4].T.contiguous().cpu()
        class_probs_full = dense[4:4 + n_classes].T
        class_probs_ours = class_probs_full[:, OURS_TO_COCO].contiguous().cpu()
        t_post = time.perf_counter()

        inf_ms = (t_inf_end - t_inf_start) * 1000.0
        total_ms = (t_post - t_pre) * 1000.0
        sum_inf_ms += inf_ms
        sum_total_ms += total_ms

        lb = {
            "ratio": (ratio, ratio),
            "pad": (pad_x, pad_y),
            "orig_size": (orig_w, orig_h),
            "input_size": (IMG_SIZE, IMG_SIZE),
        }
        save_dict = {
            "boxes_xywh": boxes_xywh,
            "class_probs": class_probs_ours,
            "letterbox": lb,
            "inference_ms": inf_ms,
        }
        if EXTRACT_BACKBONE_FEATURES and "p3" in features_cache:
            p3_feat = features_cache["p3"][0].cpu().half()
            save_dict["p3_features"] = p3_feat
            if n == 1:
                print(f"  [{split}] P3 feature shape: {tuple(p3_feat.shape)}, dtype: {p3_feat.dtype}, "
                      f"size: {p3_feat.numel() * 2 / 1024 / 1024:.2f} MB")
            features_cache.clear()
        torch.save(save_dict, paths["soft_dir"] / f"{stem}.pt")

        boxes_xyxy_letter, scores, our_cls = nms_post_process(
            boxes_xywh, class_probs_ours,
            conf_thresh=EVAL_CONF, iou_thresh=NMS_IOU, max_det=MAX_DET,
        )
        boxes_orig = boxes_letter_to_orig(boxes_xyxy_letter, lb)

        gt_boxes, gt_labels = load_gt_yolo(paths["lbl_dir"] / f"{stem}.txt", orig_w, orig_h)
        metric.update(
            preds=[{"boxes": boxes_orig, "scores": scores, "labels": our_cls}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}],
        )

        keep_hard = scores >= HARD_LABEL_CONF
        write_hard_yolo(
            paths["hard_dir"] / f"{stem}.txt",
            boxes_orig[keep_hard].tolist(),
            scores[keep_hard].tolist(),
            our_cls[keep_hard].tolist(),
            orig_w, orig_h,
        )

        if n % 200 == 0:
            print(f"  [{split}] {n}/{n_total}")

    print(f"[{split}] computing mAP...")
    m = metric.compute()
    if hook_handle is not None:
        hook_handle.remove()
    return {
        "n_images": n,
        "anchor_count": anchor_count,
        "avg_inference_ms": sum_inf_ms / max(n, 1),
        "avg_total_ms": sum_total_ms / max(n, 1),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
        "mar_100": float(m["mar_100"]),
        "per_class_map": _per_class_dict(m),
    }


def load_and_eval_split(split: str, paths: dict) -> dict:
    img_paths = list_images(paths["img_dir"])
    n_total = len(img_paths)
    print(f"[{split}] outputs already complete -> recomputing metrics from {n_total} saved .pt files")

    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False
    sum_inf_ms = 0.0
    n = 0
    anchor_count = None

    for img_path in img_paths:
        stem = img_path.stem
        data = torch.load(paths["soft_dir"] / f"{stem}.pt", weights_only=False)
        boxes_xywh = data["boxes_xywh"]
        class_probs = data["class_probs"]
        lb = data["letterbox"]
        sum_inf_ms += float(data.get("inference_ms", 0.0))
        n += 1
        if anchor_count is None:
            anchor_count = int(boxes_xywh.shape[0])

        boxes_xyxy_letter, scores, our_cls = nms_post_process(
            boxes_xywh, class_probs,
            conf_thresh=EVAL_CONF, iou_thresh=NMS_IOU, max_det=MAX_DET,
        )
        boxes_orig = boxes_letter_to_orig(boxes_xyxy_letter, lb)

        orig_w, orig_h = lb["orig_size"]
        gt_boxes, gt_labels = load_gt_yolo(paths["lbl_dir"] / f"{stem}.txt", orig_w, orig_h)
        metric.update(
            preds=[{"boxes": boxes_orig, "scores": scores, "labels": our_cls}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}],
        )

        if n % 1000 == 0:
            print(f"  [{split}] {n}/{n_total}")

    print(f"[{split}] computing mAP...")
    m = metric.compute()
    return {
        "n_images": n,
        "anchor_count": anchor_count,
        "avg_inference_ms": sum_inf_ms / max(n, 1),
        "avg_total_ms": None,
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
        "mar_100": float(m["mar_100"]),
        "per_class_map": _per_class_dict(m),
    }


def _per_class_dict(m) -> dict:
    per_cls = m.get("map_per_class")
    classes_present = m.get("classes")
    if per_cls is None or classes_present is None or per_cls.numel() == 0:
        return {}
    return {int(c): float(p) for c, p in zip(classes_present.tolist(), per_cls.tolist())}


def write_meta(per_split: dict, anchor_count_global: int | None):
    meta = {
        "model": MODEL_NAME,
        "imgsz": IMG_SIZE,
        "device": DEVICE,
        "our_classes": OUR_CLASSES,
        "ours_to_coco_index": OURS_TO_COCO,
        "anchor_count": anchor_count_global,
        "eval_conf_thresh": EVAL_CONF,
        "hard_label_conf_thresh": HARD_LABEL_CONF,
        "nms_iou": NMS_IOU,
        "max_det": MAX_DET,
        "format_version": 2,
        "splits": {
            split: {
                "n_images": s["n_images"],
                "images_dir": str(split_paths(split)["img_dir"]),
                "labels_dir": str(split_paths(split)["lbl_dir"]),
                "soft_dir": str(split_paths(split)["soft_dir"]),
                "hard_dir": str(split_paths(split)["hard_dir"]),
                "metrics": {
                    "map_50_95": s["map"],
                    "map_50": s["map_50"],
                    "map_75": s["map_75"],
                    "mar_100": s["mar_100"],
                    "per_class_map_50_95": {
                        OUR_CLASSES[i]: s["per_class_map"].get(i)
                        for i in range(len(OUR_CLASSES))
                    },
                    "avg_inference_ms": s["avg_inference_ms"],
                    "avg_total_ms": s["avg_total_ms"],
                },
            }
            for split, s in per_split.items()
        },
    }
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2))


def write_eval_result(per_split: dict):
    lines = [
        f"Model: {MODEL_NAME}",
        f"Image size: {IMG_SIZE}",
        f"Device: {DEVICE}",
        f"Eval conf threshold: {EVAL_CONF}",
        f"Hard pseudo-label threshold: {HARD_LABEL_CONF}",
        "",
    ]
    for split in SPLITS:
        if split not in per_split:
            continue
        s = per_split[split]
        header = f"=========== {split.upper()} ({s['n_images']} images) ==========="
        lines.append(header)
        lines.append("")
        lines.append("Speed:")
        avg_inf = s["avg_inference_ms"]
        lines.append(f"  Inference (model only):       {avg_inf:.2f} ms/image  "
                     f"({1000.0 / max(avg_inf, 1e-9):.1f} FPS)")
        if s["avg_total_ms"] is not None:
            avg_total = s["avg_total_ms"]
            lines.append(f"  Total (preproc+inf+postproc): {avg_total:.2f} ms/image  "
                         f"({1000.0 / max(avg_total, 1e-9):.1f} FPS)")
        else:
            lines.append("  Total (preproc+inf+postproc): n/a (loaded from disk)")
        lines.append("")
        lines.append("Combined mAP:")
        lines.append(f"  mAP@50:95 = {s['map']:.4f}")
        lines.append(f"  mAP@50    = {s['map_50']:.4f}")
        lines.append(f"  mAP@75    = {s['map_75']:.4f}")
        lines.append(f"  mAR@100   = {s['mar_100']:.4f}")
        lines.append("")
        lines.append("Per-class mAP@50:95:")
        for i, name in enumerate(OUR_CLASSES):
            v = s["per_class_map"].get(i)
            if v is None:
                lines.append(f"  {name:<12} (no predictions or GT)")
            else:
                lines.append(f"  {name:<12} {v:.4f}")
        lines.append("")
    EVAL_RESULT_FILE.write_text("\n".join(lines))


def benchmark_cpu_vs_gpu_latency(model: YOLO, train_img_dir: Path) -> dict:
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
        tensor, _, _, _, _ = preprocess_image(img_path)
        inputs.append(tensor.unsqueeze(0))

    torch_model = model.model
    results: dict = {}

    for device_name in ("cpu", "cuda"):
        if device_name == "cuda" and not torch.cuda.is_available():
            results[device_name] = None
            continue

        device = torch.device(device_name)
        print(f"  benchmark on {device_name.upper()}...")
        torch_model.to(device).eval()

        times_ms = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = torch_model(x)
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
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_NAME}")

    plan = []
    for split in SPLITS:
        paths = split_paths(split)
        if is_split_complete(paths):
            plan.append((split, paths, "load"))
        else:
            plan.append((split, paths, "infer"))
    needs_model = any(action == "infer" for _, _, action in plan)

    model = None
    if needs_model:
        model = YOLO(MODEL_NAME)
        model.to(DEVICE)
        detect_head = find_detect_head(model)
        if getattr(detect_head, "end2end", False):
            detect_head.end2end = False
            print("[setup] disabled end2end mode on Detect head (need pre-NMS dense)")
    else:
        print("All splits already have outputs -- skipping model load.")

    per_split: dict = {}
    anchor_count_global: int | None = None
    for split, paths, action in plan:
        t0 = time.time()
        if action == "infer":
            per_split[split] = run_inference_for_split(model, split, paths)
        else:
            per_split[split] = load_and_eval_split(split, paths)
        print(f"[{split}] done in {time.time() - t0:.1f}s "
              f"(mAP@50:95 = {per_split[split]['map']:.4f})")
        if anchor_count_global is None:
            anchor_count_global = per_split[split]["anchor_count"]

    write_meta(per_split, anchor_count_global)
    write_eval_result(per_split)

    if model is None:
        print("\n[benchmark] loading model (was skipped because all splits loaded from disk)...")
        model = YOLO(MODEL_NAME)
        detect_head = find_detect_head(model)
        if getattr(detect_head, "end2end", False):
            detect_head.end2end = False

    bench_results = benchmark_cpu_vs_gpu_latency(model, split_paths("train")["img_dir"])
    bench_text = format_benchmark_results(bench_results)
    print(bench_text)
    with EVAL_RESULT_FILE.open("a") as f:
        f.write("\n" + bench_text + "\n")

    print(f"\nSaved: {META_FILE}")
    print(f"Saved: {EVAL_RESULT_FILE}")
    print("\n" + EVAL_RESULT_FILE.read_text())


if __name__ == "__main__":
    main()
