"""
Train StudentYOLOFeat via PURE KD (no GT) from YOLO26l teacher, using the MORPHOLOGY-pipeline
YOLO recipe: feature(neck MSE) + dense_cls(sigmoid focal) + box(GIoU, conf-weighted).

  L = w_feat·MSE(feat) + w_cls·dense_cls(probs) + w_box·box_giou(xyxy, conf)

Loss funkcije su portane DOSLOVNO iz morphology/kd.py (_feature_loss, _dense_cls_loss,
_giou/_box_giou_loss) pa je gubitak identičan pipelineu. Težine = morphology default (sve 1.0).

Razlike vs pure_KD baseline:
  * feature-tap (egzaktni MSE na neck izlazima 256/512/512 — student je tap-matched, vidi model_arch_feat.py)
  * response = dense_cls + box_giou (morphology), umjesto focal + SmoothL1 (pure_KD)
  * BEZ ground-trutha (kao pure_KD); GT samo kao val-metrika (mAP) / early-stop.

Targeti:
  * response (boxes_xywh, class_probs) -> postojeći .../yolo26l/<split>/soft/<stem>.pt (NE recomputiramo)
  * feature  (3 neck mape fp16)        -> .../yolo26l/train/feat/<stem>.pt  (precompute_feats.py)

Trening recept (AdamW + warmup-cosine + AMP + early-stop) IDENTIČAN pure_KD-u radi fer usporedbe.

VARIJANTA 20k (ovaj folder)
--------------------------
Isti recept i ista arhitektura kao KD_featlogit/, ali trening ide nad UNIJOM
part1 + part2 train splitova (5860 + 8625 = 14485 slika). Datasetovi ostaju
fizicki odvojeni na disku -- unija se radi samo u KDTrainDataset.

Validacija ostaje na part1 val (837 slika), NEPROMIJENJENA, da mAP bude direktno
usporediv s postojecim KD_featlogit rezultatima. Mijenja se samo kolicina
trening podataka -- to je cijela poanta eksperimenta.

Topli start: RESUME_FROM ucitava tezine iz part1 KD_featlogit/checkpoints/best.pt.
Optimizer i scheduler se NE nastavljaju (broj iteracija po epohi je drugaciji),
nego se krecu iznova s RESUME_LR. Postavi RESUME_FROM = None za trening od nule.

Svi izlazi (checkpoints/, training_log.txt, training_history.json,
training_plots.png) su SCRIPT_DIR-relativni pa nista iz KD_featlogit/ ne dira.

Run:
    conda activate dipl
    # prije ovoga: baseline_models/yolo26l/precompute_feats_part2.py
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from KD_first import StudentYOLO  # noqa: F401  (referenca; 0.5M arh je u KD_first.py)
from model_arch_feat import StudentYOLOFeat


# ===== Config =====
SCRIPT_DIR = Path(__file__).parent
MINI_SET = Path("/home/tomi/code/dipl/datasets/mini_set")

PART1 = MINI_SET / "sub10k_open_images_v7"
PART2 = MINI_SET / "sub10k_open_images_v7_part2"

# Trening = UNIJA train splitova. Datasetovi ostaju odvojeni na disku.
TRAIN_ROOTS = [PART1, PART2]

# Validacija OSTAJE samo na part1 -> mAP direktno usporediv s KD_featlogit runom.
VAL_ROOT = PART1
DATASET_ROOT = PART1                      # zadrzano zbog evaluate_student.py importa
VAL_TEACHER_SOFT_DIR = VAL_ROOT / "yolo26l" / "val" / "soft"
VAL_IMG_DIR = VAL_ROOT / "images" / "val"
VAL_LBL_DIR = VAL_ROOT / "labels" / "val"

# Topli start iz part1 runa. None = trening od nule.
RESUME_FROM = Path("/home/tomi/code/dipl/custom_models/student_0_5_m/KD_featlogit/checkpoints/best.pt")
RESUME_LR = 5e-4                          # topli start ne treba puni LR (LR nize je 1e-3)

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

# Težine — morphology default (sve 1.0). _feature_loss je MEAN-MSE pa se ne skalira s br. kanala.
LOSS_W_FEAT = 1.0
LOSS_W_CLS = 1.0
LOSS_W_BOX = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

VAL_CONF_THRESH = 0.001
VAL_IOU_THRESH = 0.6
VAL_MAX_DET = 300

VAL_EVERY_N_EPOCHS = 3
EARLY_STOP_PATIENCE = 9
LR_REDUCE_PATIENCE = 6
LR_REDUCE_FACTOR = 0.6

SEED = 42


# ===== Letterbox (matches Ultralytics / evaluate.py) =====
def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
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
    """Image + teacher response (boxes_xywh, class_probs) + teacher neck feat (3 maps)."""

    def __init__(self, roots):
        """roots = lista DATASET_ROOT-ova; svaki nosi svoj images/train + yolo26l/train/{soft,feat}."""
        self.items = []                      # [(img_path, soft_dir, feat_dir)]
        seen_stems = set()
        for root in roots:
            images_dir = root / "images" / "train"
            soft_dir = root / "yolo26l" / "train" / "soft"
            feat_dir = root / "yolo26l" / "train" / "feat"
            all_imgs = sorted(p for p in images_dir.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            have = [p for p in all_imgs
                    if (soft_dir / f"{p.stem}.pt").exists() and (feat_dir / f"{p.stem}.pt").exists()]
            print(f"[KDTrainDataset] {root.name}: {len(have)}/{len(all_imgs)} slika ima soft+feat")
            if not have:
                raise RuntimeError(
                    f"Nula (image, soft, feat) trojki u {root.name}. "
                    f"Jesi li pokrenuo precompute_feats za taj dataset? ({feat_dir})")
            dup = {p.stem for p in have} & seen_stems
            if dup:
                raise RuntimeError(f"Isti stemovi u dva roota ({len(dup)}), npr. {sorted(dup)[:3]}")
            seen_stems |= {p.stem for p in have}
            self.items += [(p, soft_dir, feat_dir) for p in have]
        print(f"[KDTrainDataset] UKUPNO {len(self.items)} trening slika iz {len(roots)} dataseta")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, soft_dir, feat_dir = self.items[idx]
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        img_t, _, _ = preprocess_image(img_bgr)
        soft = torch.load(soft_dir / f"{p.stem}.pt", weights_only=True)
        feat = torch.load(feat_dir / f"{p.stem}.pt", weights_only=True)["feat"]
        return {
            "image": img_t,
            "teacher_boxes": soft["boxes_xywh"].float(),     # [8400, 4] cx,cy,w,h (640 letterbox px)
            "teacher_probs": soft["class_probs"].float(),    # [8400, 6] sigmoid
            "teacher_feat": [f.float() for f in feat],       # [256x80x80, 512x40x40, 512x20x20]
        }


def train_collate(batch):
    n_lvl = len(batch[0]["teacher_feat"])
    return {
        "images": torch.stack([b["image"] for b in batch], dim=0),
        "teacher_boxes": torch.stack([b["teacher_boxes"] for b in batch], dim=0),
        "teacher_probs": torch.stack([b["teacher_probs"] for b in batch], dim=0),
        "teacher_feat": [torch.stack([b["teacher_feat"][l] for b in batch], dim=0) for l in range(n_lvl)],
    }


class ValDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, soft_dir: Path):
        self.labels_dir = labels_dir
        self.soft_dir = soft_dir
        all_imgs = sorted(p for p in images_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
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
        soft = torch.load(self.soft_dir / f"{p.stem}.pt", weights_only=True)
        return {
            "image": img_t, "ratio": r, "pad": (dw, dh), "orig_size": (orig_w, orig_h),
            "gt_boxes": gt_boxes, "gt_labels": gt_labels,
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
                boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                              (cx + w / 2) * img_w, (cy + h / 2) * img_h])
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


# ===== KD loss — PORTANO IZ morphology/kd.py (identično) =====
def _feature_loss(s, t):
    """MEAN-MSE preko feature mapa (lista), prosjek po razinama. Kanali se egzaktno poklapaju."""
    tot, n = None, 0
    for sk, tk in zip(s, t):
        l = F.mse_loss(sk, tk.to(sk.dtype).detach())
        tot = l if tot is None else tot + l
        n += 1
    return tot / max(n, 1)


def _dense_cls_loss(s, t, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Sigmoid focal sa SOFT metama; s,t su VJEROJATNOSTI [0,1]. Norm. po efektivnim pozitivima (teacher>0.5).
    fp32 redukcija: .sum() preko ~800K elem. bi u fp16 (AMP) preljevao u inf."""
    s = s.float().clamp(1e-6, 1 - 1e-6)
    t = t.float()
    ce = -(t * torch.log(s) + (1 - t) * torch.log(1 - s))
    p_t = s * t + (1 - s) * (1 - t)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * t + (1 - alpha) * (1 - t)) * loss
    num_pos = (t.amax(dim=-1) > 0.5).float().sum().clamp(min=1.0)
    return loss.sum() / num_pos


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


def _box_giou_loss(s, t, w):
    """1 - GIoU na DEKODIRANIM xyxy okvirima, ponderiran per-box tezinom w (teacher conf)."""
    g = _giou(s, t.to(s.dtype))
    w = w.to(s.dtype)
    return ((1 - g) * w).sum() / w.sum().clamp(min=1e-6)


def xywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def kd_loss(student_feat, raw_student, decode_fn, teacher_feat, teacher_boxes, teacher_probs):
    """Morphology yolo recept: feature + dense_cls + box_giou (bez GT)."""
    boxes_xywh, class_probs = decode_fn(raw_student)        # [B,8400,4], [B,8400,6] (sigmoid)
    s_box = xywh_to_xyxy(boxes_xywh)
    t_box = xywh_to_xyxy(teacher_boxes)
    t_conf = teacher_probs.amax(dim=-1)                     # per-box težina

    l_feat = _feature_loss(student_feat, teacher_feat)
    l_cls = _dense_cls_loss(class_probs, teacher_probs)
    l_box = _box_giou_loss(s_box, t_box, t_conf)
    total = LOSS_W_FEAT * l_feat + LOSS_W_CLS * l_cls + LOSS_W_BOX * l_box
    return total, l_feat.detach(), l_cls.detach(), l_box.detach()


# ===== Validation (response KD diag + mAP vs GT; feature se NE računa u val) =====
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
    boxes_xywh = decoded_boxes[keep]; scores = scores[keep]; classes = classes[keep]
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    max_coord = boxes_xyxy.max() if boxes_xyxy.numel() > 0 else 0.0
    offsets = classes.float() * (max_coord + 1)
    keep_idx = tvops.nms(boxes_xyxy + offsets.unsqueeze(-1), scores, iou_thresh)[:max_det]
    return boxes_xyxy[keep_idx], scores[keep_idx], classes[keep_idx]


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    metric = MeanAveragePrecision(class_metrics=True, box_format="xyxy")
    metric.warn_on_many_detections = False
    sum_loss = sum_cls = sum_box = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["images"].to(device, non_blocking=True)
        tboxes = batch["teacher_boxes"].to(device, non_blocking=True)
        tprobs = batch["teacher_probs"].to(device, non_blocking=True)

        raw = model(imgs)
        boxes_xywh, class_probs = model.decode(raw)
        # response-KD diag (dense_cls + box_giou), bez feature
        s_box = xywh_to_xyxy(boxes_xywh); t_box = xywh_to_xyxy(tboxes)
        l_cls = _dense_cls_loss(class_probs, tprobs)
        l_box = _box_giou_loss(s_box, t_box, tprobs.amax(dim=-1))
        sum_loss += float(LOSS_W_CLS * l_cls + LOSS_W_BOX * l_box)
        sum_cls += float(l_cls); sum_box += float(l_box)
        n_batches += 1

        boxes = boxes_xywh.cpu(); probs = class_probs.cpu()
        for i in range(imgs.shape[0]):
            b_xyxy, sc, cl = postprocess_for_eval(boxes[i], probs[i], VAL_CONF_THRESH, VAL_IOU_THRESH, VAL_MAX_DET)
            b_orig = boxes_letter_to_orig(b_xyxy, batch["ratios"][i], batch["pads"][i])
            metric.update(preds=[{"boxes": b_orig, "scores": sc, "labels": cl}],
                          target=[{"boxes": batch["gt_boxes"][i], "labels": batch["gt_labels"][i]}])

    m = metric.compute()
    return {
        "kd_loss": sum_loss / max(n_batches, 1),
        "kd_cls": sum_cls / max(n_batches, 1),
        "kd_box": sum_box / max(n_batches, 1),
        "map": float(m["map"]), "map_50": float(m["map_50"]), "map_75": float(m["map_75"]),
    }


# ===== LR schedule / logging =====
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


# ===== Plotting =====
def make_training_plots(history: dict, out_path: Path, best_epoch):
    epochs = history["epoch"]; val_epochs = history["val_epoch"]
    if not epochs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("StudentYOLOFeat — KD featlogit (feature + dense_cls + box_giou, no GT)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train (total)", color="tab:blue", linewidth=1.6)
    ax.plot(val_epochs, history["val_kd_loss"], label="val (cls+box)", color="tab:orange", linewidth=1.6, marker="o", markersize=4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label=f"best ({best_epoch})")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Total train vs val response loss")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_feat"], label="feat (MSE)", color="tab:green", linewidth=1.4)
    ax.plot(epochs, history["train_cls"], label="cls (focal)", color="tab:blue", linewidth=1.4)
    ax.plot(epochs, history["train_box"], label="box (1-GIoU)", color="tab:red", linewidth=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("component"); ax.set_title("Train loss components (raw)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    ax.plot(val_epochs, history["val_map"], label="mAP@50:95", color="tab:green", linewidth=1.6, marker="o", markersize=4)
    ax.plot(val_epochs, history["val_map_50"], label="mAP@50", color="tab:olive", linewidth=1.4, marker="o", markersize=4)
    ax.plot(val_epochs, history["val_map_75"], label="mAP@75", color="tab:cyan", linewidth=1.4, marker="o", markersize=4)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.set_title("Validation mAP vs GT")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="tab:purple", linewidth=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("lr"); ax.set_title("LR (warmup + cosine + reduce-on-plateau)")
    ax.grid(alpha=0.3); ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ===== Main =====
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    log("Train rootovi: " + ", ".join(r.name for r in TRAIN_ROOTS))
    log(f"Val root:      {VAL_ROOT.name}  (nepromijenjen -> mAP usporediv s KD_featlogit)")
    train_ds = KDTrainDataset(TRAIN_ROOTS)
    val_ds = ValDataset(VAL_IMG_DIR, VAL_LBL_DIR, VAL_TEACHER_SOFT_DIR)
    log(f"Train: {len(train_ds)}    Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True, collate_fn=train_collate,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, collate_fn=val_collate, persistent_workers=NUM_WORKERS > 0)

    model = StudentYOLOFeat(num_classes=NUM_CLASSES, input_size=IMG_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # ---- topli start: samo tezine, optimizer/scheduler krecu iznova ----
    lr_start = LR
    if RESUME_FROM is not None:
        if not RESUME_FROM.exists():
            raise FileNotFoundError(f"RESUME_FROM ne postoji: {RESUME_FROM}")
        prev = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        model.load_state_dict(prev["model"], strict=True)
        lr_start = RESUME_LR
        log(f"Topli start iz: {RESUME_FROM}")
        log(f"  prethodna epoha={prev.get('epoch', '?')}  "
            f"val_mAP={prev.get('val_map', float('nan')):.4f}")
        log(f"  optimizer/scheduler NE nastavljaju se; LR krece od {lr_start:.2e}")
    else:
        log(f"Trening od nule; LR krece od {lr_start:.2e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=WEIGHT_DECAY)
    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch
    warmup_iters = WARMUP_EPOCHS * iters_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_iters, warmup_iters))
    log(f"Iteracija/epoha: {iters_per_epoch}  (part1-only run je imao ~366)")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_map = -1.0
    best_val_kd = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    lr_reduced_in_current_streak = False
    stopped_early = False
    global_iter = 0

    history = {
        "epoch": [], "train_loss": [], "train_feat": [], "train_cls": [], "train_box": [], "lr": [],
        "val_epoch": [], "val_kd_loss": [], "val_cls": [], "val_box": [],
        "val_map": [], "val_map_50": [], "val_map_75": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_epoch = time.time()
        running = {"loss": 0.0, "feat": 0.0, "cls": 0.0, "box": 0.0}
        n_batches = 0

        for batch in train_loader:
            imgs = batch["images"].to(device, non_blocking=True)
            tboxes = batch["teacher_boxes"].to(device, non_blocking=True)
            tprobs = batch["teacher_probs"].to(device, non_blocking=True)
            tfeat = [f.to(device, non_blocking=True) for f in batch["teacher_feat"]]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                raw, sfeat = model(imgs, return_feats=True)
                loss, l_feat, l_cls, l_box = kd_loss(sfeat, raw, model.decode, tfeat, tboxes, tprobs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += float(loss.detach())
            running["feat"] += float(l_feat); running["cls"] += float(l_cls); running["box"] += float(l_box)
            n_batches += 1
            global_iter += 1
            if global_iter % 50 == 0:
                log(f"  iter {global_iter:>6} | epoch {epoch:>3}/{EPOCHS} "
                    f"| loss {running['loss']/n_batches:.4f} | feat {running['feat']/n_batches:.4f} "
                    f"| cls {running['cls']/n_batches:.4f} | box {running['box']/n_batches:.4f} "
                    f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        log(f"[epoch {epoch:>3}] train_loss={running['loss']/n_batches:.4f} "
            f"feat={running['feat']/n_batches:.4f} cls={running['cls']/n_batches:.4f} "
            f"box={running['box']/n_batches:.4f} ({time.time()-t_epoch:.1f}s)")

        history["epoch"].append(epoch)
        history["train_loss"].append(running["loss"] / max(n_batches, 1))
        history["train_feat"].append(running["feat"] / max(n_batches, 1))
        history["train_cls"].append(running["cls"] / max(n_batches, 1))
        history["train_box"].append(running["box"] / max(n_batches, 1))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "best_val_kd": best_val_kd, "best_val_map": best_val_map}

        is_val_epoch = ((epoch - 1) % VAL_EVERY_N_EPOCHS) == 0
        if is_val_epoch:
            log(f"[epoch {epoch:>3}] validating...")
            t_val = time.time()
            vm = validate(model, val_loader, device)
            log(f"[epoch {epoch:>3}] val_resp={vm['kd_loss']:.4f} (cls={vm['kd_cls']:.4f} box={vm['kd_box']:.4f}) "
                f"| val_mAP@50:95={vm['map']:.4f} mAP@50={vm['map_50']:.4f} ({time.time()-t_val:.1f}s)")
            history["val_epoch"].append(epoch)
            history["val_kd_loss"].append(vm["kd_loss"]); history["val_cls"].append(vm["kd_cls"])
            history["val_box"].append(vm["kd_box"]); history["val_map"].append(vm["map"])
            history["val_map_50"].append(vm["map_50"]); history["val_map_75"].append(vm["map_75"])
            ckpt["val_kd_loss"] = vm["kd_loss"]; ckpt["val_map"] = vm["map"]

        torch.save(ckpt, CKPT_DIR / "last.pt")

        if is_val_epoch:
            if vm["map"] > best_val_map:
                best_val_map = vm["map"]; best_val_kd = vm["kd_loss"]; best_epoch = epoch
                ckpt["best_val_map"] = best_val_map; ckpt["best_val_kd"] = best_val_kd
                torch.save(ckpt, CKPT_DIR / "best.pt")
                epochs_without_improvement = 0; lr_reduced_in_current_streak = False
                log(f"[epoch {epoch:>3}] new best (val_mAP@50:95 = {best_val_map:.4f})")
            else:
                epochs_without_improvement += VAL_EVERY_N_EPOCHS
                log(f"[epoch {epoch:>3}] no improvement "
                    f"({epochs_without_improvement}/{EARLY_STOP_PATIENCE} patience)")
                if epochs_without_improvement >= LR_REDUCE_PATIENCE and not lr_reduced_in_current_streak:
                    old_lr = optimizer.param_groups[0]["lr"]
                    scheduler.base_lrs = [lr * LR_REDUCE_FACTOR for lr in scheduler.base_lrs]
                    for pg in optimizer.param_groups:
                        pg["lr"] *= LR_REDUCE_FACTOR
                    lr_reduced_in_current_streak = True
                    log(f"[epoch {epoch:>3}] LR reduced (x{LR_REDUCE_FACTOR}): "
                        f"{old_lr:.2e} -> {optimizer.param_groups[0]['lr']:.2e}")
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    log(f"\nEarly stopping at epoch {epoch}.")
                    stopped_early = True
                    break

    log("\nStopped early." if stopped_early else f"\nReached max epochs ({EPOCHS}).")
    log(f"Best val_mAP@50:95 = {best_val_map:.4f}  (epoch {best_epoch})")
    log(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    history_path = SCRIPT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    plot_path = SCRIPT_DIR / "training_plots.png"
    make_training_plots(history, plot_path, best_epoch)
    log(f"Saved: {history_path}")
    log(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
