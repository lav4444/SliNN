"""
tmp_viz_teacher_vs_gt.py  --  TEMP debug vizualizacija (obrisati kad ne treba)

Za jednu sliku iz mini_seta prikazuje jedno do drugog:
    lijevo  = ground truth  (labels/<split>/<stem>.txt)
    desno   = teacher (yolo26l) precomputed soft predictions
              (yolo26l/<split>/soft/<stem>.pt  ->  pre-NMS dense, pa NMS ovdje)

Prikazuje samo bbox + class + conf.
Bez argumenata -- sve se namjesta konstantama ispod.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===== Config =====
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
TEACHER      = "yolo26l"
SPLIT        = "train"          # train / val / test
IMAGE_STEM   = "5ecaadb09f80b007"   # None = slucajna slika koja ima i GT i soft cache
SEED         = None              # None = svaki put druga slika
CONF_THRESH  = 0.25             # = HARD_LABEL_CONF iz evaluate.py
NMS_IOU      = 0.7
MAX_DET      = 300
MIN_GT_BOXES = 1                # preskoci slike s manje GT kutija od ovoga
OUT_PNG      = Path("/home/tomi/code/dipl/help/teacher_vs_gt.png")

CLASSES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
COLORS  = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00b8d4"]


# ===== Ucitavanje =====
def pick_stem() -> str:
    if IMAGE_STEM:
        return IMAGE_STEM
    img_dir  = DATASET_ROOT / "images" / SPLIT
    gt_dir   = DATASET_ROOT / "labels" / SPLIT
    soft_dir = DATASET_ROOT / TEACHER / SPLIT / "soft"
    stems = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        gt = gt_dir / f"{p.stem}.txt"
        if not gt.exists() or not (soft_dir / f"{p.stem}.pt").exists():
            continue
        if len(gt.read_text().split("\n")) - 1 < MIN_GT_BOXES:
            continue
        stems.append(p.stem)
    if not stems:
        raise RuntimeError(f"Nijedna slika u splitu '{SPLIT}' nema i GT i soft cache.")
    rng = random.Random(SEED)
    return rng.choice(stems)


def find_image(stem: str) -> Path:
    img_dir = DATASET_ROOT / "images" / SPLIT
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Slika za stem '{stem}' nije nadena u {img_dir}")


def load_gt(stem: str, img_w: int, img_h: int):
    """-> lista (x1, y1, x2, y2, cls_id, None)"""
    f = DATASET_ROOT / "labels" / SPLIT / f"{stem}.txt"
    out = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        out.append(((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h, cls, None))
    return out


def nms_post_process(boxes_xywh, class_probs, conf_thresh, iou_thresh, max_det):
    """Class-aware NMS na letterbox-space dense izlazima (isto kao evaluate.py)."""
    scores, classes = class_probs.max(dim=-1)
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return (torch.zeros((0, 4)), torch.zeros((0,)),
                torch.zeros((0,), dtype=torch.int64))
    boxes_xywh, scores, classes = boxes_xywh[keep], scores[keep], classes[keep]
    cx, cy, w, h = boxes_xywh.unbind(-1)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    max_coord = boxes_xyxy.max() if boxes_xyxy.numel() else 0.0
    offsets = classes.float() * (max_coord + 1)
    keep_idx = tvops.nms(boxes_xyxy + offsets.unsqueeze(-1), scores, iou_thresh)[:max_det]
    return boxes_xyxy[keep_idx], scores[keep_idx], classes[keep_idx]


def load_teacher(stem: str, img_w: int, img_h: int):
    """-> (lista (x1,y1,x2,y2,cls,conf), broj sidara, broj razreda)"""
    f = DATASET_ROOT / TEACHER / SPLIT / "soft" / f"{stem}.pt"
    d = torch.load(f, map_location="cpu", weights_only=False)
    boxes_xywh  = d["boxes_xywh"].float()
    class_probs = d["class_probs"].float()
    lb = d["letterbox"]

    boxes_xyxy, scores, classes = nms_post_process(
        boxes_xywh, class_probs, CONF_THRESH, NMS_IOU, MAX_DET
    )
    # letterbox -> original
    rx = lb["ratio"][0]
    pad_x, pad_y = lb["pad"]
    b = boxes_xyxy.clone()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b /= rx
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, img_w)      # rezanje na granice slike
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, img_h)

    out = [(float(x1), float(y1), float(x2), float(y2), int(c), float(s))
           for (x1, y1, x2, y2), c, s in zip(b.tolist(), classes.tolist(), scores.tolist())]
    return out, tuple(class_probs.shape), lb


# ===== Crtanje =====
def draw(ax, img_rgb, boxes, title):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")
    for x1, y1, x2, y2, cls, conf in boxes:
        color = COLORS[cls % len(COLORS)]
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, edgecolor=color, linewidth=2))
        name = CLASSES[cls] if cls < len(CLASSES) else f"cls{cls}"
        label = name if conf is None else f"{name} {conf:.2f}"
        ax.text(x1, max(y1 - 4, 8), label, fontsize=8, color="white",
                bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.9))


def main():
    stem = pick_stem()
    img_path = find_image(stem)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"cv2 ne moze procitati {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    gt = load_gt(stem, w, h)
    teacher, probs_shape, lb = load_teacher(stem, w, h)

    print(f"slika        : {img_path}")
    print(f"originalno   : {w}x{h}   letterbox ratio={lb['ratio'][0]:.4f} pad={lb['pad']}")
    print(f"soft cache   : class_probs {probs_shape}  (sidra x razredi)")
    print(f"GT kutija    : {len(gt)}")
    print(f"teacher      : {len(teacher)} kutija @ conf>{CONF_THRESH}")
    print()
    print(f"{'':4} {'GT':<28} | teacher")
    for i in range(max(len(gt), len(teacher))):
        g = f"{CLASSES[gt[i][4]]}" if i < len(gt) else ""
        if i < len(teacher):
            t = f"{CLASSES[teacher[i][4]]} {teacher[i][5]:.3f}"
        else:
            t = ""
        print(f"{i:4} {g:<28} | {t}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    draw(axes[0], img_rgb, gt, "GROUND TRUTH")
    draw(axes[1], img_rgb, teacher, f"TEACHER ({TEACHER})")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight")
    print(f"\nspremljeno -> {OUT_PNG}")


if __name__ == "__main__":
    main()
