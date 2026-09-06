
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import (
    MultilabelAveragePrecision, MultilabelF1Score, MultilabelAccuracy,
)

from model_cnn import NUM_CLASSES, INPUT_SIZE, CLASS_NAMES

DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def img_dir(split):
    return DATASET_ROOT / "images" / split


def lbl_dir(split):
    return DATASET_ROOT / "labels" / split


def list_images(d):
    return sorted(p for p in Path(d).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_multilabel_target(label_file: Path):
    t = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                c = int(parts[0])
                if 0 <= c < NUM_CLASSES:
                    t[c] = 1.0
    return t


def preprocess(img_bgr, img_size=INPUT_SIZE):
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class MultiLabelDataset(Dataset):
    def __init__(self, split, img_size=INPUT_SIZE, max_images=None, seed=42):
        self.split = split
        self.img_size = img_size
        imgs = list_images(img_dir(split))
        if max_images is not None and len(imgs) > max_images:
            rng = random.Random(seed)
            imgs = sorted(rng.sample(imgs, max_images))
        self.items = imgs
        self.lbl_dir = lbl_dir(split)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        x = preprocess(img_bgr, self.img_size)
        y = load_multilabel_target(self.lbl_dir / f"{p.stem}.txt")
        return x, y


def make_loader(split, batch_size, shuffle, num_workers=4, max_images=None):
    ds = MultiLabelDataset(split, max_images=max_images)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    ap = MultilabelAveragePrecision(num_labels=NUM_CLASSES, average=None).to(device)
    f1 = MultilabelF1Score(num_labels=NUM_CLASSES, average="macro", threshold=0.5).to(device)
    acc = MultilabelAccuracy(num_labels=NUM_CLASSES, average="macro", threshold=0.5).to(device)

    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += float(criterion(logits, y)) * x.size(0)
        n += x.size(0)
        probs = torch.sigmoid(logits)
        ap.update(probs, y.int())
        f1.update(probs, y.int())
        acc.update(probs, y.int())

    per_class_ap = ap.compute().cpu()
    return {
        "loss": loss_sum / max(n, 1),
        "map": float(per_class_ap.mean()),
        "f1": float(f1.compute()),
        "acc": float(acc.compute()),
        "per_class_ap": {CLASS_NAMES[i]: float(per_class_ap[i]) for i in range(NUM_CLASSES)},
        "n_images": n,
    }


def benchmark_latency(model, n_images=10, n_discard=2, seed=42, img_size=INPUT_SIZE):
    rng = random.Random(seed)
    imgs = list_images(img_dir("train"))
    sampled = rng.sample(imgs, min(n_images, len(imgs)))
    inputs = []
    for p in sampled:
        b = cv2.imread(str(p))
        if b is None:
            continue
        inputs.append(preprocess(b, img_size).unsqueeze(0))

    results = {}
    for dev_name in ("cpu", "cuda"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            results[dev_name] = None
            continue
        device = torch.device(dev_name)
        model.to(device).eval()
        times = []
        with torch.no_grad():
            for inp in inputs:
                x = inp.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
        fast = sorted(times)[:-n_discard] if n_discard > 0 else sorted(times)
        results[dev_name] = sum(fast) / len(fast)
    return results
