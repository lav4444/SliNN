
import copy
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_arch import StudentYOLO


SCRIPT_DIR = Path(__file__).parent
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")

TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"
VAL_IMG_DIR = DATASET_ROOT / "images" / "val"
TRAIN_LBL_DIR = DATASET_ROOT / "labels" / "train"
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

LOSS_W_CLS = 1.5
LOSS_W_BOX = 2.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

EMA_DECAY = 0.9999
EMA_TAU = 2000

VAL_CONF_THRESH = 0.001
VAL_IOU_THRESH = 0.6
VAL_MAX_DET = 300

VAL_EVERY_N_EPOCHS = 3
EARLY_STOP_PATIENCE = 9
LR_REDUCE_PATIENCE = 6
LR_REDUCE_FACTOR = 0.6

SEED = 42


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


def load_gt_yolo_letterboxed(label_file: Path, orig_w: int, orig_h: int,
                             ratio: float, pad: tuple) -> tuple:
    boxes, labels = [], []
    dw, dh = pad
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx_n, cy_n, w_n, h_n = map(float, parts[1:])
            cx = cx_n * orig_w
            cy = cy_n * orig_h
            w = w_n * orig_w
            h = h_n * orig_h
            x1 = (cx - w / 2) * ratio + dw
            y1 = (cy - h / 2) * ratio + dh
            x2 = (cx + w / 2) * ratio + dw
            y2 = (cy + h / 2) * ratio + dh
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_id)
    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


def assign_gt_to_anchors(gt_boxes_xyxy: torch.Tensor, gt_labels: torch.Tensor,
                         anchor_xy: torch.Tensor, anchor_stride: torch.Tensor,
                         n_classes: int) -> tuple:
    n_anchors = anchor_xy.shape[0]
    cls_targets = torch.zeros(n_anchors, n_classes, dtype=torch.float32)
    box_targets = torch.zeros(n_anchors, 4, dtype=torch.float32)
    pos_mask = torch.zeros(n_anchors, dtype=torch.bool)

    if len(gt_boxes_xyxy) == 0:
        return cls_targets, box_targets, pos_mask

    half = (anchor_stride / 2).unsqueeze(-1)
    anchor_xyxy = torch.cat([anchor_xy - half, anchor_xy + half], dim=-1)

    ious = tvops.box_iou(gt_boxes_xyxy, anchor_xyxy)
    best_iou, best_anchor = ious.max(dim=1)

    order = torch.argsort(best_iou, descending=True)
    for i in order.tolist():
        anchor_idx = best_anchor[i].item()
        if pos_mask[anchor_idx]:
            continue
        cls_id = int(gt_labels[i].item())
        cls_targets[anchor_idx, cls_id] = 1.0
        x1, y1, x2, y2 = gt_boxes_xyxy[i].tolist()
        box_targets[anchor_idx, 0] = (x1 + x2) * 0.5
        box_targets[anchor_idx, 1] = (y1 + y2) * 0.5
        box_targets[anchor_idx, 2] = x2 - x1
        box_targets[anchor_idx, 3] = y2 - y1
        pos_mask[anchor_idx] = True

    return cls_targets, box_targets, pos_mask


def encode_box_to_raw(boxes_xywh, anchor_xy, anchor_stride):
    cx = boxes_xywh[..., 0]
    cy = boxes_xywh[..., 1]
    w = boxes_xywh[..., 2]
    h = boxes_xywh[..., 3]
    raw_x = (cx - anchor_xy[:, 0]) / anchor_stride
    raw_y = (cy - anchor_xy[:, 1]) / anchor_stride
    raw_w = torch.log((w / anchor_stride).clamp(min=1e-6))
    raw_h = torch.log((h / anchor_stride).clamp(min=1e-6))
    return torch.stack([raw_x, raw_y, raw_w, raw_h], dim=-1)


def sigmoid_focal_loss_sum(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()


def _decode_box_xywh(raw_box, anchor_xy, anchor_stride):
    cx = anchor_xy[:, 0] + raw_box[..., 0] * anchor_stride
    cy = anchor_xy[:, 1] + raw_box[..., 1] * anchor_stride
    w = torch.exp(raw_box[..., 2].clamp(min=-8.0, max=8.0)) * anchor_stride
    h = torch.exp(raw_box[..., 3].clamp(min=-8.0, max=8.0)) * anchor_stride
    return torch.stack([cx, cy, w, h], dim=-1)


def _giou(a, b):
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)
    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_b = (bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0)
    ix1 = torch.max(ax1, bx1); iy1 = torch.max(ay1, by1)
    ix2 = torch.min(ax2, bx2); iy2 = torch.min(ay2, by2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    union = area_a + area_b - inter + 1e-7
    iou = inter / union
    cx1 = torch.min(ax1, bx1); cy1 = torch.min(ay1, by1)
    cx2 = torch.max(ax2, bx2); cy2 = torch.max(ay2, by2)
    carea = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
    return iou - (carea - union) / carea


def gt_supervision_loss(raw_student, cls_targets, box_targets, pos_mask, anchor_xy, anchor_stride,
                        box_format="raw"):
    student_box_raw = raw_student[:, :4, :].permute(0, 2, 1)
    student_cls_logits = raw_student[:, 4:, :].permute(0, 2, 1)

    num_pos = pos_mask.float().sum().clamp(min=1.0)
    loss_cls = sigmoid_focal_loss_sum(student_cls_logits, cls_targets) / num_pos

    if pos_mask.any():
        if box_format == "decoded":
            student_xywh = student_box_raw.float()
        elif box_format == "raw":
            student_xywh = _decode_box_xywh(student_box_raw.float(), anchor_xy, anchor_stride)
        else:
            raise ValueError(f"Unknown box_format: {box_format!r} (use 'raw' or 'decoded')")
        student_xyxy = xywh_to_xyxy(student_xywh)[pos_mask]
        target_xyxy = xywh_to_xyxy(box_targets.float())[pos_mask]
        loss_box = (1.0 - _giou(student_xyxy, target_xyxy)).mean()
    else:
        loss_box = torch.tensor(0.0, device=raw_student.device)

    loss = LOSS_W_CLS * loss_cls + LOSS_W_BOX * loss_box
    return loss, loss_cls.detach(), loss_box.detach()


class GTTrainDataset(Dataset):

    def __init__(self, images_dir: Path, gt_dir: Path,
                 anchor_xy: torch.Tensor, anchor_stride: torch.Tensor, n_classes: int):
        all_imgs = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.items = [p for p in all_imgs if (gt_dir / f"{p.stem}.txt").exists()]
        self.gt_dir = gt_dir
        self.anchor_xy = anchor_xy.cpu()
        self.anchor_stride = anchor_stride.cpu()
        self.n_classes = n_classes
        if not self.items:
            raise RuntimeError(f"No (image, GT label) pairs found in {images_dir} / {gt_dir}")
        print(f"[GTTrainDataset] {len(self.items)}/{len(all_imgs)} images have GT labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        orig_h, orig_w = img_bgr.shape[:2]
        img_t, ratio, pad = preprocess_image(img_bgr)

        gt_boxes_xyxy, gt_labels = load_gt_yolo_letterboxed(
            self.gt_dir / f"{p.stem}.txt", orig_w, orig_h, ratio, pad
        )
        cls_targets, box_targets, pos_mask = assign_gt_to_anchors(
            gt_boxes_xyxy, gt_labels, self.anchor_xy, self.anchor_stride, self.n_classes
        )
        return {
            "image": img_t,
            "gt_cls_targets": cls_targets,
            "gt_box_targets": box_targets,
            "gt_pos_mask": pos_mask,
        }


class ValDataset(Dataset):

    def __init__(self, images_dir: Path, labels_dir: Path,
                 anchor_xy: torch.Tensor, anchor_stride: torch.Tensor, n_classes: int):
        self.labels_dir = labels_dir
        self.anchor_xy = anchor_xy.cpu()
        self.anchor_stride = anchor_stride.cpu()
        self.n_classes = n_classes
        all_imgs = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        self.items = [p for p in all_imgs if (labels_dir / f"{p.stem}.txt").exists()]
        if len(self.items) != len(all_imgs):
            print(f"[ValDataset] {len(self.items)}/{len(all_imgs)} val images have GT labels")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img_bgr = cv2.imread(str(p))
        orig_h, orig_w = img_bgr.shape[:2]
        img_t, r, (dw, dh) = preprocess_image(img_bgr)

        gt_boxes, gt_labels = self._load_gt(self.labels_dir / f"{p.stem}.txt", orig_w, orig_h)
        gt_xyxy_lb, gt_lab_lb = load_gt_yolo_letterboxed(
            self.labels_dir / f"{p.stem}.txt", orig_w, orig_h, r, (dw, dh)
        )
        cls_targets, box_targets, pos_mask = assign_gt_to_anchors(
            gt_xyxy_lb, gt_lab_lb, self.anchor_xy, self.anchor_stride, self.n_classes
        )
        return {
            "image": img_t,
            "ratio": r,
            "pad": (dw, dh),
            "orig_size": (orig_w, orig_h),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_cls_targets": cls_targets,
            "gt_box_targets": box_targets,
            "gt_pos_mask": pos_mask,
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
        "gt_cls_targets": torch.stack([b["gt_cls_targets"] for b in batch], dim=0),
        "gt_box_targets": torch.stack([b["gt_box_targets"] for b in batch], dim=0),
        "gt_pos_mask": torch.stack([b["gt_pos_mask"] for b in batch], dim=0),
    }


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
    model.eval()
    box_format = getattr(model, "BOX_OUTPUT_FORMAT", "raw")
    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False

    sum_loss = 0.0
    sum_cls = 0.0
    sum_box = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["images"].to(device, non_blocking=True)
        cls_t = batch["gt_cls_targets"].to(device, non_blocking=True)
        box_t = batch["gt_box_targets"].to(device, non_blocking=True)
        pos_m = batch["gt_pos_mask"].to(device, non_blocking=True)

        raw = model(imgs)
        loss, lcls, lbox = gt_supervision_loss(raw, cls_t, box_t, pos_m, anchor_xy, anchor_stride,
                                              box_format=box_format)
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
        "gt_loss": sum_loss / max(n_batches, 1),
        "gt_cls": sum_cls / max(n_batches, 1),
        "gt_box": sum_box / max(n_batches, 1),
        "map": float(m["map"]),
        "map_50": float(m["map_50"]),
        "map_75": float(m["map_75"]),
        "map_per_class": m.get("map_per_class"),
        "classes": m.get("classes"),
    }


def make_lr_lambda(total_iters, warmup_iters):
    def fn(it):
        if it < warmup_iters:
            return (it + 1) / max(1, warmup_iters)
        progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return fn


def log(line: str):
    print(line)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def make_training_plots(history: dict, out_path: Path, best_epoch):
    epochs = history["epoch"]
    val_epochs = history["val_epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Student GT-only training (0.5M)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue", linewidth=1.6)
    ax.plot(val_epochs, history["val_gt_loss"], label="val (GT)", color="tab:orange", linewidth=1.6, marker="o", markersize=4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("GT loss")
    ax.set_title("Total GT loss (train vs val)"); ax.set_ylim(0, 2); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_cls"], label="train cls", color="tab:blue", linewidth=1.4)
    ax.plot(val_epochs, history["val_cls"], label="val cls", color="tab:blue", linewidth=1.4, linestyle="--", marker="o", markersize=4)
    ax.plot(epochs, history["train_box"], label="train box", color="tab:red", linewidth=1.4)
    ax.plot(val_epochs, history["val_box"], label="val box", color="tab:red", linewidth=1.4, linestyle="--", marker="o", markersize=4)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss component")
    ax.set_title("Loss components (cls = focal, box = 1-GIoU)"); ax.set_ylim(0, 2); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(val_epochs, history["val_map"], label="mAP@50:95", color="tab:green", linewidth=1.6, marker="o", markersize=4)
    ax.plot(val_epochs, history["val_map_50"], label="mAP@50", color="tab:olive", linewidth=1.4, marker="o", markersize=4)
    ax.plot(val_epochs, history["val_map_75"], label="mAP@75", color="tab:cyan", linewidth=1.4, marker="o", markersize=4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP")
    ax.set_title("Validation mAP vs ground truth"); ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="tab:purple", linewidth=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule (warmup + cosine)"); ax.grid(alpha=0.3); ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


class ModelEMA:
    def __init__(self, model, decay=EMA_DECAY, tau=EMA_TAU):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    model = StudentYOLO(num_classes=NUM_CLASSES, input_size=IMG_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    anchor_xy = model.anchor_xy
    anchor_stride = model.anchor_stride

    box_format = getattr(model, "BOX_OUTPUT_FORMAT", "raw")
    log(f"Box output format: {box_format}")

    train_ds = GTTrainDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, anchor_xy, anchor_stride, NUM_CLASSES)
    val_ds = ValDataset(VAL_IMG_DIR, VAL_LBL_DIR, anchor_xy, anchor_stride, NUM_CLASSES)
    log(f"Train: {len(train_ds)}    Val: {len(val_ds)}")

    n_probe = min(200, len(train_ds))
    pos_counts = [int(train_ds[i]["gt_pos_mask"].sum()) for i in range(n_probe)]
    log(f"[assign probe] avg positives/img over {n_probe}: "
        f"{np.mean(pos_counts):.2f} (min {min(pos_counts)}, max {max(pos_counts)}, "
        f"zero-GT imgs {sum(c == 0 for c in pos_counts)})")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=val_collate, persistent_workers=NUM_WORKERS > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch
    warmup_iters = WARMUP_EPOCHS * iters_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_iters, warmup_iters))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model)
    log(f"EMA enabled (decay={EMA_DECAY}, tau={EMA_TAU}) — validating & saving EMA weights")

    best_val_gt = float("inf")
    best_val_map = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    lr_reduced_in_current_streak = False
    stopped_early = False
    global_iter = 0

    history = {
        "epoch": [], "train_loss": [], "train_cls": [], "train_box": [], "lr": [],
        "val_epoch": [],
        "val_gt_loss": [], "val_cls": [], "val_box": [],
        "val_map": [], "val_map_50": [], "val_map_75": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "cls": 0.0, "box": 0.0}
        n_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            cls_t = batch["gt_cls_targets"].to(device, non_blocking=True)
            box_t = batch["gt_box_targets"].to(device, non_blocking=True)
            pos_m = batch["gt_pos_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw = model(imgs)
                loss, loss_cls, loss_box = gt_supervision_loss(raw, cls_t, box_t, pos_m, anchor_xy, anchor_stride,
                                                              box_format=box_format)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            running["loss"] += float(loss.detach())
            running["cls"] += float(loss_cls)
            running["box"] += float(loss_box)
            n_batches += 1
            global_iter += 1

            if global_iter % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log(f"  iter {global_iter:>6} | epoch {epoch:>3}/{EPOCHS} "
                    f"| loss {running['loss']/n_batches:.4f} "
                    f"| cls {running['cls']/n_batches:.4f} "
                    f"| box {running['box']/n_batches:.4f} "
                    f"| lr {lr_now:.2e}")

        epoch_dur = time.time() - t_epoch
        log(f"[epoch {epoch:>3}] train_loss={running['loss']/n_batches:.4f} "
            f"cls={running['cls']/n_batches:.4f} box={running['box']/n_batches:.4f} "
            f"({epoch_dur:.1f}s)")

        history["epoch"].append(epoch)
        history["train_loss"].append(running["loss"] / max(n_batches, 1))
        history["train_cls"].append(running["cls"] / max(n_batches, 1))
        history["train_box"].append(running["box"] / max(n_batches, 1))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_gt": best_val_gt,
            "best_val_map": best_val_map,
        }

        is_val_epoch = ((epoch - 1) % VAL_EVERY_N_EPOCHS) == 0
        if is_val_epoch:
            log(f"[epoch {epoch:>3}] validating (EMA weights)...")
            t_val = time.time()
            val_metrics = validate(ema.ema, val_loader, device, anchor_xy, anchor_stride)
            log(f"[epoch {epoch:>3}] val_gt_loss={val_metrics['gt_loss']:.4f} "
                f"(cls={val_metrics['gt_cls']:.4f} box={val_metrics['gt_box']:.4f}) | "
                f"val_mAP@50:95={val_metrics['map']:.4f} "
                f"mAP@50={val_metrics['map_50']:.4f} ({time.time() - t_val:.1f}s)")

            history["val_epoch"].append(epoch)
            history["val_gt_loss"].append(val_metrics["gt_loss"])
            history["val_cls"].append(val_metrics["gt_cls"])
            history["val_box"].append(val_metrics["gt_box"])
            history["val_map"].append(val_metrics["map"])
            history["val_map_50"].append(val_metrics["map_50"])
            history["val_map_75"].append(val_metrics["map_75"])

            ckpt["val_gt_loss"] = val_metrics["gt_loss"]
            ckpt["val_map"] = val_metrics["map"]

        torch.save(ckpt, CKPT_DIR / "last.pt")

        if is_val_epoch:
            improved = val_metrics["map"] > best_val_map
            if improved:
                best_val_map = val_metrics["map"]
                best_val_gt = val_metrics["gt_loss"]
                best_epoch = epoch
                ckpt["best_val_gt"] = best_val_gt
                ckpt["best_val_map"] = best_val_map
                best_ckpt = dict(ckpt)
                best_ckpt["model"] = ema.ema.state_dict()
                torch.save(best_ckpt, CKPT_DIR / "best.pt")
                epochs_without_improvement = 0
                lr_reduced_in_current_streak = False
                log(f"[epoch {epoch:>3}] new best (val_mAP@50:95 = {best_val_map:.4f}, "
                    f"val_gt_loss = {best_val_gt:.4f})")
            else:
                epochs_without_improvement += VAL_EVERY_N_EPOCHS
                log(f"[epoch {epoch:>3}] no improvement "
                    f"({epochs_without_improvement}/{EARLY_STOP_PATIENCE} epochs patience)")
                if epochs_without_improvement >= LR_REDUCE_PATIENCE and not lr_reduced_in_current_streak:
                    old_lr = optimizer.param_groups[0]["lr"]
                    scheduler.base_lrs = [lr * LR_REDUCE_FACTOR for lr in scheduler.base_lrs]
                    for pg in optimizer.param_groups:
                        pg["lr"] *= LR_REDUCE_FACTOR
                    lr_reduced_in_current_streak = True
                    new_lr = optimizer.param_groups[0]["lr"]
                    log(f"[epoch {epoch:>3}] LR reduced (x{LR_REDUCE_FACTOR}): "
                        f"{old_lr:.2e} -> {new_lr:.2e}")
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
    log(f"Best val_gt_loss   = {best_val_gt:.4f}  (same epoch)")
    log(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    history_path = SCRIPT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    plot_path = SCRIPT_DIR / "training_plots.png"
    make_training_plots(history, plot_path, best_epoch)
    log(f"Saved: {history_path}")
    log(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
