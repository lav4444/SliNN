"""
common3.py — dataset / model / eval / benchmark za exp3
(production detektor: fasterrcnn_mobilenet_v3_large_320_fpn, lokalizacija+klasifikacija).

Dataset: isti Open Images subset (6 klasa), YOLO labeli -> torchvision detekcijski
format ({boxes xyxy u pikselima, labels 1..6}; 0 = background).
"""

from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchmetrics.detection import MeanAveragePrecision

DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
NUM_CLASSES = len(CLASS_NAMES) + 1          # +1 za background (torchvision konvencija)
IMG_SIZE = 320


def img_dir(split):
    return DATASET_ROOT / "images" / split


def lbl_dir(split):
    return DATASET_ROOT / "labels" / split


def list_images(d):
    return sorted(p for p in Path(d).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_yolo_boxes(label_file: Path, w: int, h: int):
    """Vrati (boxes xyxy u pikselima, labels 1..6). Background = bez okvira."""
    boxes, labels = [], []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0]); cx, cy, bw, bh = map(float, p[1:])
            x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2]); labels.append(c + 1)   # +1: 0=background
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class DetDataset(Dataset):
    """Vraca (image[0,1] CHW float, target{boxes,labels,image_id}).
    Torchvision detektor sam radi resize/normalizaciju (GeneralizedRCNNTransform)."""

    def __init__(self, split, max_images=None, drop_empty=False, seed=42):
        imgs = list_images(img_dir(split))
        if drop_empty:
            imgs = [p for p in imgs if (lbl_dir(split) / f"{p.stem}.txt").exists()
                    and (lbl_dir(split) / f"{p.stem}.txt").read_text().strip()]
        if max_images is not None and len(imgs) > max_images:
            imgs = sorted(random.Random(seed).sample(imgs, max_images))
        self.items = imgs
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        boxes, labels = load_yolo_boxes(lbl_dir(self.split) / f"{p.stem}.txt", w, h)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img, target


def det_collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return imgs, targets


def make_loader(split, batch_size, shuffle, num_workers=4, max_images=None, drop_empty=False):
    ds = DetDataset(split, max_images=max_images, drop_empty=drop_empty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, collate_fn=det_collate, persistent_workers=num_workers > 0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def unfreeze_bn(module):
    """Zamijeni sve FrozenBatchNorm2d (torchvision detekcijski backbone) ekvivalentnim
    nn.BatchNorm2d (kopiraj statistike). Nuzno da ih Torch-Pruning prepozna i reze
    zajedno s pripadnim conv-om (inace: mismatch kanala nakon rezanja)."""
    for name, child in module.named_children():
        if isinstance(child, FrozenBatchNorm2d):
            C = child.weight.shape[0]
            eps = float(getattr(child, "eps", 1e-5))
            bn = nn.BatchNorm2d(C, eps=eps)
            with torch.no_grad():
                bn.weight.copy_(child.weight); bn.bias.copy_(child.bias)
                bn.running_mean.copy_(child.running_mean); bn.running_var.copy_(child.running_var)
            setattr(module, name, bn)
        else:
            unfreeze_bn(child)
    return module


# COCO label indeksi (torchvision) za nasih 6 klasa -> mapiranje umjesto treninga
COCO_IDS = {"Person": 1, "Car": 3, "Truck": 8, "Bus": 6, "Motorcycle": 4, "Bicycle": 2}


def build_model(num_classes=NUM_CLASSES, pretrained=True, coco_map=True):
    """Vraca detektor s nasih 6 klasa.
    coco_map=True (uz pretrained): KOPIRA COCO tezine predictora za nasih 6 klasa
    (+background) -> near-zero-shot baseline BEZ treninga. Inace random glava."""
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=("DEFAULT" if pretrained else None))
    in_feat = model.roi_heads.box_predictor.cls_score.in_features

    if pretrained and coco_map:
        old = model.roi_heads.box_predictor                    # 91-klasni (COCO)
        new = FastRCNNPredictor(in_feat, num_classes)
        coco_idx = [0] + [COCO_IDS[n] for n in CLASS_NAMES]    # nas label 0..6 -> COCO idx
        with torch.no_grad():
            new.cls_score.weight.copy_(old.cls_score.weight[coco_idx])
            new.cls_score.bias.copy_(old.cls_score.bias[coco_idx])
            rows = [k * 4 + j for k in coco_idx for j in range(4)]   # bbox: 4 retka/klasi
            new.bbox_pred.weight.copy_(old.bbox_pred.weight[rows])
            new.bbox_pred.bias.copy_(old.bbox_pred.bias[rows])
        model.roi_heads.box_predictor = new
    else:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    unfreeze_bn(model)          # FrozenBatchNorm2d -> nn.BatchNorm2d (da tp moze rezati)
    return model


def prunable_ignored_layers(model):
    """Slojevi koje NE diramo: FPN izlaz (256-interfejs), RPN glava, box predictor.
    Time prune-amo samo backbone.body (MobileNetV3) + roi_heads.box_head (fc6/fc7).

    Dodatno ignoriramo STEM blok (backbone.body['0'] = conv 3->16 + BN): on je na
    samom ulazu (3-kanalni) pa ga tp ne sprega s njegovim BN-om u istu grupu ->
    BN bi se rezao neovisno o convu i davao conv(16)/BN(8) mismatch (crash u
    recalibrate_bn). Stem je sitan (zanemarivo za params), pa ga pinnamo na 16
    eksplicitnim ignoriranjem conv-a I BN-a (jednako za sve 4 metode)."""
    ignored = []
    for sub in [model.backbone.fpn, model.rpn, model.roi_heads.box_predictor]:
        for leaf in sub.modules():
            if isinstance(leaf, (nn.Conv2d, nn.Linear)):
                ignored.append(leaf)
    body = model.backbone.body
    stem = body["0"] if "0" in dict(body.named_children()) else getattr(body, "0")
    for leaf in stem.modules():
        if isinstance(leaf, (nn.Conv2d, nn.BatchNorm2d)):
            ignored.append(leaf)
    return ignored


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def backbone_gmacs(model, device, imgsz=IMG_SIZE):
    """GMACs backbone-a (staticni dio; proxy za conv racunanje). Detektorski
    puni MACs su loše definirani (dinamicni RoI), pa mjerimo backbone + latenciju."""
    try:
        import torch_pruning as tp
        macs, _ = tp.utils.count_ops_and_params(
            model.backbone, torch.randn(1, 3, imgsz, imgsz, device=device))
        return macs / 1e9
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Evaluacija (torchmetrics mAP)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.05):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.warn_on_many_detections = False
    for imgs, targets in loader:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        outputs = model(imgs)
        preds, tgts = [], []
        for out, tgt in zip(outputs, targets):
            keep = out["scores"] >= score_thresh
            preds.append({"boxes": out["boxes"][keep].cpu(),
                          "scores": out["scores"][keep].cpu(),
                          "labels": out["labels"][keep].cpu()})
            tgts.append({"boxes": tgt["boxes"], "labels": tgt["labels"]})
        metric.update(preds, tgts)
    m = metric.compute()
    return {"map": float(m["map"]), "map_50": float(m["map_50"]),
            "map_75": float(m["map_75"]), "mar_100": float(m["mar_100"])}


# --------------------------------------------------------------------------- #
# Benchmark latencije (CPU/GPU): 2 warmup odbaceni
# --------------------------------------------------------------------------- #
def benchmark_latency(model, n_images=10, n_discard=2, seed=42, imgsz=IMG_SIZE):
    rng = random.Random(seed)
    imgs = list_images(img_dir("train"))
    sampled = rng.sample(imgs, min(n_images, len(imgs)))
    inputs = []
    for p in sampled:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (imgsz, imgsz))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inputs.append(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))))

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
                x = [inp.to(device)]
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


# --------------------------------------------------------------------------- #
# Mrtve jedinice: filteri (backbone aktivacije nikad >0) i neuroni (fc6/fc7 ReLU nikad >0)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def count_dead(model, batches, device, eps=1e-6):
    """Vrati broj mrtvih FILTERA (backbone.body aktivacije koje nikad ne predu 0)
    i mrtvih NEURONA (fc6/fc7 nakon ReLU nikad >0), preko `batches` (eval forward).
    'Mrtvo' = per-kanal/per-neuron MAKSIMUM aktivacije preko svih podataka <= eps."""
    model.eval().to(device)
    ACT = (nn.ReLU, nn.ReLU6, nn.Hardswish)         # glavne feature aktivacije (ne Hardsigmoid/SE gate)
    acts = {}
    handles = []

    def make_hook(key, relu=False):
        def hook(m, inp, out):
            o = torch.relu(out) if relu else out
            if o.dim() == 4:
                ch_max = o.amax(dim=(0, 2, 3))
            elif o.dim() == 2:
                ch_max = o.amax(dim=0)
            else:
                ch_max = o.transpose(0, 1).flatten(1).amax(1)
            prev = acts.get(key)
            acts[key] = ch_max.detach() if prev is None else torch.maximum(prev, ch_max.detach())
        return hook

    for name, mod in model.backbone.body.named_modules():
        if isinstance(mod, ACT):
            handles.append(mod.register_forward_hook(make_hook(("F", name))))
    handles.append(model.roi_heads.box_head.fc6.register_forward_hook(make_hook(("N", "fc6"), relu=True)))
    handles.append(model.roi_heads.box_head.fc7.register_forward_hook(make_hook(("N", "fc7"), relu=True)))

    for imgs, _ in batches:
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        model(imgs)
    for h in handles:
        h.remove()

    df = tf = dn = tn = 0
    for key, mx in acts.items():
        dead = int((mx <= eps).sum()); tot = mx.numel()
        if key[0] == "F":
            df += dead; tf += tot
        else:
            dn += dead; tn += tot
    return {"dead_filters": df, "total_filters": tf, "dead_neurons": dn, "total_neurons": tn}
