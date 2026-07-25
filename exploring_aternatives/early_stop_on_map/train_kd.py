"""
Train StudentYOLO via knowledge distillation from YOLO26l teacher.

Loss:
  L = w_cls * focal_loss(student_logits, teacher_probs) / num_pos
    + w_box * weighted_smooth_l1(student_raw, encoded_teacher_raw)
  where num_pos = anchors with max teacher prob > 0.5 (RetinaNet-style normalization).
  Focal loss params: alpha=0.25, gamma=2.0 — down-weights easy background examples,
  focuses learning on hard/foreground anchors that determine mAP.

Validation each epoch: val KD loss (vs teacher) + val mAP (vs hard GT).
Early stops on val mAP@50:95 with patience (in contrast to the original `pure_KD`
training script, which stops on val KD loss). val_kd_loss is still logged each
epoch for diagnostic purposes but no longer drives selection or stopping.
Saves best checkpoint by val mAP@50:95.

Run:
    conda activate dipl
    pip install opencv-python torchmetrics pycocotools
    python train_kd.py
"""

import json
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision

from KD_first import StudentYOLO


# ===== Config =====
SCRIPT_DIR = Path(__file__).parent
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
TEACHER_SOFT_DIR = DATASET_ROOT / "yolo26l" / "train" / "soft"
VAL_TEACHER_SOFT_DIR = DATASET_ROOT / "yolo26l" / "val" / "soft"

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"
VAL_IMG_DIR = DATASET_ROOT / "images" / "val"
VAL_LBL_DIR = DATASET_ROOT / "labels" / "val"

CKPT_DIR = SCRIPT_DIR / "checkpoints"
LOG_FILE = SCRIPT_DIR / "training_log.txt"

IMG_SIZE = 640
NUM_CLASSES = 6
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]

EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 2
GRAD_CLIP = 10.0

LOSS_W_CLS = 2.0  # 1.0
LOSS_W_BOX = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

VAL_CONF_THRESH = 0.001
VAL_IOU_THRESH = 0.6
VAL_MAX_DET = 300

EARLY_STOP_PATIENCE = 10  # stop if val mAP@50:95 doesn't improve for this many epochs (mAP is noisier than loss, so patience is higher than in pure_KD)

SEED = 42


# ===== Letterbox (matches Ultralytics) =====
def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess_image(img_bgr):
    img_lb, r, pad = letterbox(img_bgr)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    img_t = torch.from_numpy(img_chw).float() / 255.0
    return img_t, r, pad


# ===== Datasets =====
class KDTrainDataset(Dataset):
    def __init__(self, images_dir: Path, soft_dir: Path):
        all_imgs = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.items = [p for p in all_imgs if (soft_dir / f"{p.stem}.pt").exists()]
        self.soft_dir = soft_dir
        if not self.items:
            raise RuntimeError(f"No (image, soft_label) pairs found in {images_dir} / {soft_dir}")
        print(f"[KDTrainDataset] {len(self.items)}/{len(all_imgs)} images have teacher soft labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        img_t, _, _ = preprocess_image(img_bgr)
        soft = torch.load(self.soft_dir / f"{p.stem}.pt", weights_only=True)
        return {
            "image": img_t,
            "teacher_boxes": soft["boxes_xywh"].float(),     # [8400, 4] in 640x640 letterbox px
            "teacher_probs": soft["class_probs"].float(),    # [8400, 6]
        }


class ValDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, soft_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.soft_dir = soft_dir
        all_imgs = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.items = [p for p in all_imgs if (soft_dir / f"{p.stem}.pt").exists()]
        if len(self.items) != len(all_imgs):
            print(f"[ValDataset] {len(self.items)}/{len(all_imgs)} val images have soft labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img_bgr = cv2.imread(str(p))
        orig_h, orig_w = img_bgr.shape[:2]
        img_t, r, (dw, dh) = preprocess_image(img_bgr)
        gt_boxes, gt_labels = self._load_gt(self.labels_dir / f"{p.stem}.txt", orig_w, orig_h)
        soft = torch.load(self.soft_dir / f"{p.stem}.pt", weights_only=False)
        return {
            "image": img_t,
            "ratio": r,
            "pad": (dw, dh),
            "orig_size": (orig_w, orig_h),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "teacher_boxes": soft["boxes_xywh"].float(),
            "teacher_probs": soft["class_probs"].float(),
        }

    @staticmethod
    def _load_gt(label_file: Path, img_w: int, img_h: int):
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


def val_collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "ratios": [b["ratio"] for b in batch],
        "pads": [b["pad"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "teacher_boxes": torch.stack([b["teacher_boxes"] for b in batch], dim=0),
        "teacher_probs": torch.stack([b["teacher_probs"] for b in batch], dim=0),
    }


# ===== KD loss =====
def encode_box_to_raw(teacher_boxes, anchor_xy, anchor_stride):
    """Invert StudentYOLO.decode for boxes so we can SmoothL1 in raw space."""
    cx = teacher_boxes[..., 0]
    cy = teacher_boxes[..., 1]
    w = teacher_boxes[..., 2]
    h = teacher_boxes[..., 3]
    raw_x = (cx - anchor_xy[:, 0]) / anchor_stride
    raw_y = (cy - anchor_xy[:, 1]) / anchor_stride
    raw_w = torch.log((w / anchor_stride).clamp(min=1e-6))
    raw_h = torch.log((h / anchor_stride).clamp(min=1e-6))
    return torch.stack([raw_x, raw_y, raw_w, raw_h], dim=-1)


def sigmoid_focal_loss_sum(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Sigmoid focal loss with soft targets (Lin et al., RetinaNet).

    L_i = alpha_t * (1 - p_t)^gamma * BCE(p_i, t_i)
        where p_t = p_i if t_i==1 else 1-p_i, generalized for soft t_i in [0,1].

    Returns the SUM over all elements; caller normalizes by number of positives.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def kd_loss(raw_student, teacher_boxes, teacher_probs, anchor_xy, anchor_stride):
    """
    raw_student:    [B, 4+nc, 8400]
    teacher_boxes:  [B, 8400, 4]   (cx,cy,w,h in 640x640 letterbox px)
    teacher_probs:  [B, 8400, nc]  (sigmoid'd, our 6 classes)
    anchor_xy:      [8400, 2]
    anchor_stride:  [8400]
    """
    student_box_raw = raw_student[:, :4, :].permute(0, 2, 1)      # [B, 8400, 4]
    student_cls_logits = raw_student[:, 4:, :].permute(0, 2, 1)   # [B, 8400, nc]

    # Class loss: focal loss, normalized by number of effective positive anchors
    # (any anchor where teacher is confident about some class).
    pos_mask = teacher_probs.amax(dim=-1) > 0.5      # [B, 8400]
    num_pos = pos_mask.float().sum().clamp(min=1.0)
    loss_cls = sigmoid_focal_loss_sum(student_cls_logits, teacher_probs) / num_pos

    # Box loss: SmoothL1 in raw-target space, weighted by max teacher prob.
    target_raw = encode_box_to_raw(teacher_boxes, anchor_xy, anchor_stride)
    box_diff = F.smooth_l1_loss(student_box_raw, target_raw, reduction="none").sum(-1)
    weights = teacher_probs.amax(dim=-1)
    loss_box = (box_diff * weights).sum() / weights.sum().clamp(min=1.0)

    loss = LOSS_W_CLS * loss_cls + LOSS_W_BOX * loss_box
    return loss, loss_cls.detach(), loss_box.detach()


# ===== Validation =====
def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def boxes_letter_to_orig(boxes_xyxy, ratio, pad):
    dw, dh = pad
    out = boxes_xyxy.clone()
    out[:, [0, 2]] -= dw
    out[:, [1, 3]] -= dh
    out /= ratio
    return out


def postprocess_for_eval(decoded_boxes, decoded_probs, conf_thresh, iou_thresh, max_det):
    """boxes [8400,4] xywh in letterbox space; probs [8400, nc]. Returns NMSed (xyxy, scores, labels) in letterbox space."""
    scores, classes = decoded_probs.max(dim=-1)
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return (torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.int64))
    boxes_xywh = decoded_boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    max_coord = boxes_xyxy.max() if boxes_xyxy.numel() > 0 else 0.0
    offsets = classes.float() * (max_coord + 1)
    boxes_for_nms = boxes_xyxy + offsets.unsqueeze(-1)
    keep_idx = tvops.nms(boxes_for_nms, scores, iou_thresh)[:max_det]
    return boxes_xyxy[keep_idx], scores[keep_idx], classes[keep_idx]


@torch.no_grad()
def validate(model, val_loader, device, anchor_xy, anchor_stride):
    """Single pass over val: computes (a) val KD loss vs teacher, (b) val mAP vs GT."""
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False

    sum_loss = 0.0
    sum_cls = 0.0
    sum_box = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["images"].to(device, non_blocking=True)
        tboxes = batch["teacher_boxes"].to(device, non_blocking=True)
        tprobs = batch["teacher_probs"].to(device, non_blocking=True)

        raw = model(imgs)

        loss, lcls, lbox = kd_loss(raw, tboxes, tprobs, anchor_xy, anchor_stride)
        sum_loss += float(loss)
        sum_cls += float(lcls)
        sum_box += float(lbox)
        n_batches += 1

        boxes, probs = model.decode(raw)
        boxes = boxes.cpu()
        probs = probs.cpu()
        for i in range(imgs.shape[0]):
            b_xyxy, sc, cl = postprocess_for_eval(
                boxes[i], probs[i],
                conf_thresh=VAL_CONF_THRESH,
                iou_thresh=VAL_IOU_THRESH,
                max_det=VAL_MAX_DET,
            )
            b_orig = boxes_letter_to_orig(b_xyxy, batch["ratios"][i], batch["pads"][i])
            metric.update(
                preds=[{"boxes": b_orig, "scores": sc, "labels": cl}],
                target=[{"boxes": batch["gt_boxes"][i], "labels": batch["gt_labels"][i]}],
            )

    m = metric.compute()
    return {
        "kd_loss": sum_loss / max(n_batches, 1),
        "kd_cls": sum_cls / max(n_batches, 1),
        "kd_box": sum_box / max(n_batches, 1),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
        "map_per_class": m.get("map_per_class"),
        "classes": m.get("classes"),
    }


# ===== LR schedule =====
def make_lr_lambda(total_iters, warmup_iters):
    def fn(it):
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return fn


# ===== Logging =====
def log(line: str):
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


# ===== Plotting =====
def make_training_plots(history: dict, out_path: Path, best_epoch: int | None):
    """Generate train-vs-val curves at end of training."""
    epochs = history["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Student KD training", fontsize=13)

    # (a) Total loss: train vs val KD
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue", linewidth=1.6)
    ax.plot(epochs, history["val_kd_loss"], label="val (KD)", color="tab:orange", linewidth=1.6)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KD loss")
    ax.set_title("Total KD loss (train vs val)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Loss components
    ax = axes[0, 1]
    ax.plot(epochs, history["train_cls"], label="train cls", color="tab:blue", linewidth=1.4)
    ax.plot(epochs, history["val_cls"], label="val cls", color="tab:blue", linewidth=1.4, linestyle="--")
    ax.plot(epochs, history["train_box"], label="train box", color="tab:red", linewidth=1.4)
    ax.plot(epochs, history["val_box"], label="val box", color="tab:red", linewidth=1.4, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss component")
    ax.set_title("Loss components (cls = BCE, box = SmoothL1)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Val mAP curves
    ax = axes[1, 0]
    ax.plot(epochs, history["val_map"], label="mAP@50:95", color="tab:green", linewidth=1.6)
    ax.plot(epochs, history["val_map_50"], label="mAP@50", color="tab:olive", linewidth=1.4)
    ax.plot(epochs, history["val_map_75"], label="mAP@75", color="tab:cyan", linewidth=1.4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP vs ground truth")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # (d) Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="tab:purple", linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule (warmup + cosine)")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ===== Main =====
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")  # truncate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    train_ds = KDTrainDataset(TRAIN_IMG_DIR, TEACHER_SOFT_DIR)
    val_ds = ValDataset(VAL_IMG_DIR, VAL_LBL_DIR, VAL_TEACHER_SOFT_DIR)
    log(f"Train: {len(train_ds)}    Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=val_collate, persistent_workers=NUM_WORKERS > 0,
    )

    model = StudentYOLO(num_classes=NUM_CLASSES, input_size=IMG_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch
    warmup_iters = WARMUP_EPOCHS * iters_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_iters, warmup_iters))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    anchor_xy = model.anchor_xy
    anchor_stride = model.anchor_stride

    best_val_kd = float("inf")
    best_val_map = -1.0
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False
    global_iter = 0

    history = {
        "epoch": [], "train_loss": [], "train_cls": [], "train_box": [],
        "val_kd_loss": [], "val_cls": [], "val_box": [],
        "val_map": [], "val_map_50": [], "val_map_75": [],
        "lr": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "cls": 0.0, "box": 0.0}
        n_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            tboxes = batch["teacher_boxes"].to(device, non_blocking=True)
            tprobs = batch["teacher_probs"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw = model(imgs)
                loss, loss_cls, loss_box = kd_loss(raw, tboxes, tprobs, anchor_xy, anchor_stride)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += float(loss.detach())
            running["cls"] += float(loss_cls)
            running["box"] += float(loss_box)
            n_batches += 1
            global_iter += 1

            if global_iter % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log(
                    f"  iter {global_iter:>6} | epoch {epoch:>3}/{EPOCHS} "
                    f"| loss {running['loss']/n_batches:.4f} "
                    f"| cls {running['cls']/n_batches:.4f} "
                    f"| box {running['box']/n_batches:.4f} "
                    f"| lr {lr_now:.2e}"
                )

        epoch_dur = time.time() - t_epoch
        log(
            f"[epoch {epoch:>3}] train_loss={running['loss']/n_batches:.4f} "
            f"cls={running['cls']/n_batches:.4f} box={running['box']/n_batches:.4f} "
            f"({epoch_dur:.1f}s)"
        )

        log(f"[epoch {epoch:>3}] validating...")
        t_val = time.time()
        val_metrics = validate(model, val_loader, device, anchor_xy, anchor_stride)
        log(
            f"[epoch {epoch:>3}] val_kd_loss={val_metrics['kd_loss']:.4f} "
            f"(cls={val_metrics['kd_cls']:.4f} box={val_metrics['kd_box']:.4f}) | "
            f"val_mAP@50:95={val_metrics['map']:.4f} "
            f"mAP@50={val_metrics['map_50']:.4f} ({time.time() - t_val:.1f}s)"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_kd": best_val_kd,
            "best_val_map": best_val_map,
            "val_kd_loss": val_metrics["kd_loss"],
            "val_map": val_metrics["map"],
        }
        torch.save(ckpt, CKPT_DIR / "last.pt")

        history["epoch"].append(epoch)
        history["train_loss"].append(running["loss"] / max(n_batches, 1))
        history["train_cls"].append(running["cls"] / max(n_batches, 1))
        history["train_box"].append(running["box"] / max(n_batches, 1))
        history["val_kd_loss"].append(val_metrics["kd_loss"])
        history["val_cls"].append(val_metrics["kd_cls"])
        history["val_box"].append(val_metrics["kd_box"])
        history["val_map"].append(val_metrics["map"])
        history["val_map_50"].append(val_metrics["map_50"])
        history["val_map_75"].append(val_metrics["map_75"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_metrics["map"] > best_val_map
        if improved:
            best_val_map = val_metrics["map"]
            best_val_kd = val_metrics["kd_loss"]
            best_epoch = epoch
            ckpt["best_val_kd"] = best_val_kd
            ckpt["best_val_map"] = best_val_map
            torch.save(ckpt, CKPT_DIR / "best.pt")
            epochs_without_improvement = 0
            log(f"[epoch {epoch:>3}] new best (val_mAP@50:95 = {best_val_map:.4f}, "
                f"val_kd_loss = {best_val_kd:.4f})")
        else:
            epochs_without_improvement += 1
            log(f"[epoch {epoch:>3}] no improvement "
                f"({epochs_without_improvement}/{EARLY_STOP_PATIENCE} patience)")
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                log(f"\nEarly stopping at epoch {epoch}: val_mAP@50:95 has not improved "
                    f"for {EARLY_STOP_PATIENCE} consecutive epochs.")
                stopped_early = True
                break

    if stopped_early:
        log(f"\nStopped early.")
    else:
        log(f"\nReached max epochs ({EPOCHS}).")
    log(f"Best val_mAP@50:95 = {best_val_map:.4f}  (epoch {best_epoch})")
    log(f"Best val_kd_loss   = {best_val_kd:.4f}  (same epoch)")
    log(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    # Save history + plots
    history_path = SCRIPT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    plot_path = SCRIPT_DIR / "training_plots.png"
    make_training_plots(history, plot_path, best_epoch)
    log(f"Saved: {history_path}")
    log(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
