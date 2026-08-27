"""
analysis.py — morphology dijagnostika (SAMOSTALNO: bez importa iz drugih foldera;
sve potrebno se kopira ovamo, ne importa).

Cilj: na proizvoljnom eager modelu (fasterrcnn baseline ILI yolo26n) izmjeriti, na
CIJELOM train skupu, per-layer statiku, aktivnost (mrtve/near-dead jedinice),
pruning-potencijal (gradient) i growing-potencijal (pravi GradMax). Modul je
opcionalno importabilan prije morph-treninga ako nas zanima analiza.

Blok 1a: statistika dataseta (slike po splitu, broj klasa, objekti po klasi).
"""

from pathlib import Path
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

# Sve postavke dolaze iz config.py (isti folder = "samostalno"; jedinstveni izvor istine).
from config import (DATASET_ROOT, CLASS_NAMES, IMG_EXTS, NUM_CLASSES, COCO_IDS, COCO_YOLO_IDS,
                    YOLO_PATH, MODEL_SPEC, GRAD_MAX_IMAGES, GRAD_BATCH, SHOW_LAYER_TABLE,
                    TMP_ROOT, EVAL_MAX, EVAL_BATCH, DEV_DATA_SUBSET, USE_PROFILE_ADAPTERS)


def dev_subset_note():
    """Tekst upozorenja ako je DEV_DATA_SUBSET aktivan (inace None). Koristi terminal i GUI."""
    if DEV_DATA_SUBSET:
        return f"⚠️ DEV_DATA_SUBSET={DEV_DATA_SUBSET}: pipeline koristi samo prvih {DEV_DATA_SUBSET} slika PO SPLITU (niska vjernost — makni prije pravih runova)."
    return None


def _img_dir(split):
    return DATASET_ROOT / "images" / split


def _lbl_dir(split):
    return DATASET_ROOT / "labels" / split


def list_splits():
    """Auto-detektiraj splitove iz images/<split>/ (ne hardkodiraj train/val/test)."""
    base = DATASET_ROOT / "images"
    return sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []


def scan_split(split):
    """Prebroji slike, prazne (bez labela), objekte ukupno i po klasi (YOLO .txt)."""
    d = _img_dir(split)
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []
    if DEV_DATA_SUBSET:
        imgs = imgs[:DEV_DATA_SUBSET]                    # dev: stats odrazavaju isti podskup kao loaderi
    per_class = [0] * len(CLASS_NAMES)
    n_obj = n_empty = 0
    for p in imgs:
        lf = _lbl_dir(split) / f"{p.stem}.txt"
        txt = lf.read_text().strip() if lf.exists() else ""
        if not txt:
            n_empty += 1
            continue
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            if 0 <= c < len(CLASS_NAMES):
                per_class[c] += 1
                n_obj += 1
    return {"split": split, "n_images": len(imgs), "n_empty": n_empty,
            "n_objects": n_obj, "per_class": per_class}


def dataset_stats():
    return [scan_split(s) for s in list_splits()]


def print_dataset_stats(stats):
    print(f"\n=== DATASET: {DATASET_ROOT.name} | klasa: {len(CLASS_NAMES)} ===")
    hdr = f"{'split':<8}{'slike':>8}{'prazne':>8}{'objekti':>9}  " + "".join(f"{c[:7]:>9}" for c in CLASS_NAMES)
    print(hdr)
    print("-" * len(hdr))
    tot = [0] * len(CLASS_NAMES)
    ti = te = to = 0
    for s in stats:
        print(f"{s['split']:<8}{s['n_images']:>8}{s['n_empty']:>8}{s['n_objects']:>9}  "
              + "".join(f"{v:>9}" for v in s["per_class"]))
        tot = [a + b for a, b in zip(tot, s["per_class"])]
        ti += s["n_images"]; te += s["n_empty"]; to += s["n_objects"]
    print("-" * len(hdr))
    print(f"{'UKUPNO':<8}{ti:>8}{te:>8}{to:>9}  " + "".join(f"{v:>9}" for v in tot))


# --------------------------------------------------------------------------- #
# Blok 1b: ucitavanje modela (oba) + introspekcija (broj klasa, conv-slojevi)
# --------------------------------------------------------------------------- #
def set_bn_eval(model):
    """Drzi SVE BatchNorm slojeve na running-statistici (eval), cak i kad je model u train() modu.
    KRITICNO za detekciju: torchvision RCNN loss radi samo u train modu, a tada bi BN AZURIRAO
    running_mean/var batch-statistikom i POKVARIO pretrenirane COCO stat. -> rusi mAP (lazno nizak baseline).
    Zovi odmah nakon model.train() u grad-prolazu i FT-u."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    return model


def unfreeze_bn(module):
    """FrozenBatchNorm2d -> nn.BatchNorm2d (kopiraj statistike). Kopirano iz common3."""
    from torchvision.ops.misc import FrozenBatchNorm2d
    for name, child in module.named_children():
        if isinstance(child, FrozenBatchNorm2d):
            bn = nn.BatchNorm2d(child.weight.shape[0], eps=float(getattr(child, "eps", 1e-5)))
            with torch.no_grad():
                bn.weight.copy_(child.weight); bn.bias.copy_(child.bias)
                bn.running_mean.copy_(child.running_mean); bn.running_var.copy_(child.running_var)
            setattr(module, name, bn)
        else:
            unfreeze_bn(child)
    return module


def build_fasterrcnn(num_classes=NUM_CLASSES):
    """Baseline fasterrcnn_mobilenet_v3_large_320_fpn, COCO-mapped na 6 klasa. Kopirano iz common3."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT", weights_backbone="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    old, new = model.roi_heads.box_predictor, FastRCNNPredictor(in_feat, num_classes)
    coco_idx = [0] + [COCO_IDS[n] for n in CLASS_NAMES]
    with torch.no_grad():
        new.cls_score.weight.copy_(old.cls_score.weight[coco_idx])
        new.cls_score.bias.copy_(old.cls_score.bias[coco_idx])
        rows = [k * 4 + j for k in coco_idx for j in range(4)]
        new.bbox_pred.weight.copy_(old.bbox_pred.weight[rows])
        new.bbox_pred.bias.copy_(old.bbox_pred.bias[rows])
    model.roi_heads.box_predictor = new
    return unfreeze_bn(model)


def load_eager(path, device, code_dirs=None):
    """Ucitaj CIJELI eager modul (torch.save(model)); hvata i ultralytics dict['model']. Iz analyze.py."""
    for d in (code_dirs or []):
        if d not in sys.path:
            sys.path.insert(0, d)
    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().to(device).float()
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().to(device).float()
    raise SystemExit(f"FORMAT NIJE PODRZAN: {type(obj).__name__} (treba cijeli eager modul).")


def load_any(spec, device, code_dirs=None):
    """spec='fasterrcnn' -> build; inace putanja .pt -> load_eager. Vrati model (adapter se bira posebno)."""
    return build_fasterrcnn().to(device).eval() if spec == "fasterrcnn" else load_eager(spec, device, code_dirs)


# === MODEL ADAPTERI ========================================================== #
# JEDINO mjesto s model-specificnom logikom. Jezgra (layer-tablica, aktivnost,
# prune, grow) je UNIVERZALNA i radi na bilo kojem nn.Module. Univerzalni ulaz
# svuda = lista [0,1] CHW tenzora. Novi model = novi ModelAdapter + registracija.
class ModelAdapter:
    kind = "base"; imgsz = 640
    SUPPORTS = []                                        # ljudski-citljiv popis sto adapter podrzava (za About)

    @staticmethod
    def matches(model):
        return False

    @staticmethod
    def forward(model, imgs):
        """Census/FLOPs forward (eval). imgs: lista [0,1] CHW. Vrati izlaz modela."""
        raise NotImplementedError

    @staticmethod
    def gt_loss(model, imgs, targets):
        """Skalarni GT-loss za gradijente (SAMO analiza). targets: lista {boxes xyxy px, labels 1..K}."""
        raise NotImplementedError

    @staticmethod
    def tp_example(device):
        """Primjer ulaza za torch-pruning DependencyGraph (RAW arg za model(...))."""
        raise NotImplementedError

    @staticmethod
    def predict(model, imgs):
        """Detekcije za mAP: lista {boxes xyxy px (orig. koord.), labels, scores} po slici (eval)."""
        raise NotImplementedError

    @staticmethod
    def teacher_outputs(teacher, imgs):
        """KD-meta teachera za ovaj batch (cpu, detached): sve sto kd_loss treba. Za precompute cache."""
        raise NotImplementedError

    @staticmethod
    def kd_loss(student, td, imgs, w_feat=1.0, w_rpn=1.0, T=1.0, box_w=1.0):
        """Pure-KD loss student vs (cached) teacher td. Vrati (total, info). td vec na uredaju+float."""
        raise NotImplementedError


class FrcnnAdapter(ModelAdapter):
    kind = "frcnn"; imgsz = 320; task = "detection"     # task -> bira evaluator/metriku (evaluate())
    SUPPORTS = ["analiza (dead/prune/grow)", "dead+near-dead rez", "KD fine-tune (feature+RPN+logit)", "mAP eval"]

    @staticmethod
    def matches(model):
        return hasattr(model, "roi_heads")

    @staticmethod
    def forward(model, imgs):
        return model(imgs)                               # interni GeneralizedRCNNTransform resize-a

    @staticmethod
    def gt_loss(model, imgs, targets):
        model.train(); set_bn_eval(model)                # train mod za loss, ALI BN ostaje na running-stats (ne kvari ih)
        return sum(model(imgs, targets).values())        # torchvision detekcijska loss

    @staticmethod
    def tp_example(device):
        return [torch.rand(3, 320, 320, device=device)]   # frcnn forward prima listu

    @staticmethod
    def predict(model, imgs):
        model.eval()
        return model(imgs)                               # torchvision: lista {boxes,labels,scores} u orig. koord.

    @staticmethod
    def roi_fg_boxes(roi_heads, bbox, props, cls):
        """Dekodiraj ROI box-delte -> xyxy okviri za FOREGROUND argmax klasu (bez bg=0). Za univerzalni box_giou.
        Vrati (box [K,4] xyxy, conf [K] = fg max prob, cidx [K] = odabrana klasa 1..)."""
        import torch.nn.functional as F
        dec = roi_heads.box_coder.decode(bbox, props)    # [K, nc, 4] (okviri po klasi)
        conf, fgi = F.softmax(cls, -1)[:, 1:].max(dim=-1)  # max preko foreground klasa
        cidx = fgi + 1                                   # stvarni indeks klase (preskoci bg=0)
        k = torch.arange(dec.shape[0], device=dec.device)
        return dec[k, cidx], conf, cidx

    @staticmethod
    @torch.no_grad()
    def teacher_outputs(teacher, imgs):
        """frcnn KD-meta: FPN feat, RPN izlazi, ROI cls + DEKODIRAN ROI box (xyxy, fg klasa) + conf — za box_giou."""
        teacher.eval()
        ti, _ = teacher.transform(imgs)
        feat = teacher.backbone(ti.tensors)
        obj, box = teacher.rpn.head(list(feat.values()))
        props, _ = teacher.rpn(ti, feat)
        bf = teacher.roi_heads.box_roi_pool(feat, props, ti.image_sizes)
        cls, bbox = teacher.roi_heads.box_predictor(teacher.roi_heads.box_head(bf))
        rbox, rconf, rcidx = FrcnnAdapter.roi_fg_boxes(teacher.roi_heads, bbox, props, cls)
        cpu = lambda t: t.detach().cpu()
        return {"feat": {k: cpu(v) for k, v in feat.items()},
                "rpn_obj": [cpu(o) for o in obj], "rpn_box": [cpu(b) for b in box],
                "props": [cpu(p) for p in props], "cls": cpu(cls),
                "roi_box": cpu(rbox), "roi_conf": cpu(rconf), "roi_cidx": cpu(rcidx),
                "sizes": [tuple(s) for s in ti.image_sizes]}

    @staticmethod
    def kd_loss(student, td, imgs, w_feat=1.0, w_rpn=1.0, T=1.0, box_w=1.0):
        """Pure-KD (exp2): feature(FPN-MSE) + RPN-KD(obj BCE + box L1) + logit(ROI cls-KL + box L1).
        EKSTRAKCIJA signala je model-specificna (ostaje ovdje); matematika ide kroz genericki kd.kd_total."""
        import kd
        s_imgs, _ = student.transform(imgs)
        s_feat = student.backbone(s_imgs.tensors)
        s_obj, s_box = student.rpn.head(list(s_feat.values()))
        s_bf = student.roi_heads.box_roi_pool(s_feat, td["props"], s_imgs.image_sizes)
        s_cls, s_box2 = student.roi_heads.box_predictor(student.roi_heads.box_head(s_bf))
        terms = [
            {"group": "feat", "type": "feature", "student": s_feat, "teacher": td["feat"], "w": w_feat},
            {"group": "rpn", "type": "objectness", "student": s_obj, "teacher": td["rpn_obj"], "w": w_rpn},
            {"group": "rpn", "type": "box", "student": s_box, "teacher": td["rpn_box"], "w": w_rpn},
            {"group": "logit", "type": "logit", "student": s_cls, "teacher": td["cls"], "w": 1.0, "kw": {"T": T}},
            {"group": "logit", "type": "box", "student": s_box2, "teacher": td["box"], "w": box_w},
        ]
        return kd.kd_total(terms)


class YoloAdapter(ModelAdapter):
    kind = "yolo"; imgsz = 640; task = "detection"
    SUPPORTS = ["analiza (dead/prune/grow)"]             # KD fine-tune / mAP eval: TBD

    @staticmethod
    def matches(model):
        return type(model).__module__.startswith("ultralytics")

    @staticmethod
    def _batch(imgs, sz=640):
        import torch.nn.functional as F
        return torch.stack([F.interpolate(im.unsqueeze(0), (sz, sz), mode="bilinear", align_corners=False)[0]
                            for im in imgs])

    @staticmethod
    def forward(model, imgs):
        return model(YoloAdapter._batch(imgs))

    @staticmethod
    def gt_loss(model, imgs, targets):
        from types import SimpleNamespace
        a = getattr(model, "args", None)                 # v8 loss cita hyp.box/cls/dfl (atribut); dict -> namespace
        a = dict(a) if isinstance(a, dict) else (vars(a) if a is not None else {})
        model.args = SimpleNamespace(**a); model.criterion = None   # BEZ defaulta: ako gain fali, neka pukne
        o2c = [COCO_YOLO_IDS[n] for n in CLASS_NAMES]     # nas 0..5 -> COCO-80 idx
        dev = imgs[0].device
        cls_l, box_l, bidx_l = [], [], []
        for bi, t in enumerate(targets):
            _, h0, w0 = imgs[bi].shape
            for (x1, y1, x2, y2), lab in zip(t["boxes"].tolist(), t["labels"].tolist()):
                cls_l.append([o2c[int(lab) - 1]])         # lab 1..K (0=bg) -> our 0.. -> COCO
                box_l.append([(x1 + x2) / 2 / w0, (y1 + y2) / 2 / h0, (x2 - x1) / w0, (y2 - y1) / h0])
                bidx_l.append(bi)
        batch = {"img": YoloAdapter._batch(imgs),
                 "cls": torch.tensor(cls_l, dtype=torch.float32, device=dev).reshape(-1, 1),
                 "bboxes": torch.tensor(box_l, dtype=torch.float32, device=dev).reshape(-1, 4),
                 "batch_idx": torch.tensor(bidx_l, dtype=torch.float32, device=dev).reshape(-1)}
        model.train(); set_bn_eval(model)                # BN na running-stats i za yolo (ne kvari ga grad-prolaz)
        out = model.loss(batch)
        loss = out[0] if isinstance(out, (tuple, list)) else out
        return loss.sum()                                # [box,cls,dfl] (ili e2e komponente) -> skalar za backward

    @staticmethod
    def tp_example(device):
        return torch.rand(1, 3, 640, 640, device=device)

    @staticmethod
    def _letterbox(im, sz=640):
        """Aspect-preserving resize + centrirani pad (114/255). Vrati (chw sz×sz, ratio, pad_x, pad_y)."""
        import torch.nn.functional as F
        _, h, w = im.shape
        r = min(sz / h, sz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img = F.interpolate(im.unsqueeze(0), (nh, nw), mode="bilinear", align_corners=False)[0]
        pw, ph = sz - nw, sz - nh
        left, top = pw // 2, ph // 2
        img = F.pad(img, (left, pw - left, top, ph - top), value=114.0 / 255.0)
        return img, r, left, top

    @staticmethod
    @torch.no_grad()
    def predict(model, imgs, conf=0.001, iou=0.7, max_det=300):
        """Gusta glava (end2end OFF) -> dekodirani xywh(640px) + sigmoid-probs; class-aware NMS;
        COCO->nasih 6 (label = idx+1, kao GT); rescale letterbox->original. Vrati [{boxes,scores,labels}]."""
        import torchvision.ops as tvops
        o2c = [COCO_YOLO_IDS[n] for n in CLASS_NAMES]    # nas idx 0..5 -> COCO stupac
        lbs = [YoloAdapter._letterbox(im) for im in imgs]
        batch = torch.stack([l[0] for l in lbs])
        head = list(model.model)[-1]; saved = getattr(head, "end2end", None)
        model.eval()
        try:
            head.end2end = False                         # gusta glava [B, 4+nc, 8400]
            out = model(batch)
        finally:
            if saved is not None:
                head.end2end = saved
        dense = out[0] if isinstance(out, (tuple, list)) else out
        preds = []
        for bi in range(dense.shape[0]):
            _, r, px, py = lbs[bi]
            d = dense[bi]
            box = d[:4].T                                # [N,4] xywh (640 px)
            probs = d[4:4 + head.nc].T[:, o2c]           # [N,6] nase klase (vec sigmoid)
            scores, cls = probs.max(dim=1)
            keep = scores > conf
            box, scores, cls = box[keep], scores[keep], cls[keep]
            if box.numel() == 0:
                preds.append({"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                              "labels": torch.zeros((0,), dtype=torch.int64)})
                continue
            cx, cy, bw, bh = box.unbind(1)
            xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            off = cls.float() * (xyxy.max() + 1)         # class-aware NMS (offset trik)
            k = tvops.nms(xyxy + off[:, None], scores, iou)[:max_det]
            xyxy, scores, cls = xyxy[k], scores[k], cls[k]
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r  # letterbox -> original px
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
            preds.append({"boxes": xyxy.cpu(), "scores": scores.cpu(),
                          "labels": (cls + 1).cpu().to(torch.int64)})  # +1 -> baza kao GT (1..6)
        return preds

    @staticmethod
    def _dense_decode(model, imgs, train_bn=False):
        """Gusta glava (end2end OFF -> DEKODIRAN izlaz): xywh(norm [0,1]) + sigmoid-probs naših 6 + NECK feature mape.
        Vrati (box [B,A,4], cls [B,A,6], feats list[[B,C,h,w]]). `train_bn`=True: glava eval (dekodira), BN trenira."""
        o2c = [COCO_YOLO_IDS[n] for n in CLASS_NAMES]
        batch = torch.stack([YoloAdapter._letterbox(im)[0] for im in imgs])
        head = list(model.model)[-1]; saved = getattr(head, "end2end", None)
        model.eval()                                     # Detect eval -> dekodiran izlaz
        if train_bn:                                     # ...ali BN se adaptira (ne zamrznut) tijekom FT-a
            for mm in model.modules():
                if isinstance(mm, nn.modules.batchnorm._BatchNorm):
                    mm.train()
        grab = {}
        h = head.register_forward_pre_hook(lambda m, inp: grab.__setitem__("f", list(inp[0])))  # neck izlazi = ulaz glave
        try:
            head.end2end = False
            out = model(batch)
        finally:
            h.remove()
            if saved is not None:
                head.end2end = saved
        dense = out[0] if isinstance(out, (tuple, list)) else out   # [B, 4+nc, A]
        xywh = dense[:, :4, :].permute(0, 2, 1)                       # [B, A, 4] dekodiran xywh (px 640)
        cx, cy, w, h = xywh.unbind(-1)                               # -> xyxy (univerzalni box_giou ocekuje xyxy)
        box = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        cls = dense[:, 4:4 + head.nc, :].permute(0, 2, 1)[:, :, o2c]  # [B, A, 6] (vec sigmoid)
        return box, cls, grab.get("f", [])

    @staticmethod
    @torch.no_grad()
    def teacher_outputs(teacher, imgs):
        """KD-meta teachera: box(xyxy px) + cls(probs naših 6) + conf(per-anchor max prob) + NECK feature mape."""
        box, cls, feats = YoloAdapter._dense_decode(teacher, imgs)
        return {"box": box.detach().cpu(), "cls": cls.detach().cpu(),
                "conf": cls.amax(dim=-1).detach().cpu(),             # per-box tezina za box_giou
                "feat": [f.detach().cpu() for f in feats]}

    @staticmethod
    def kd_loss(student, td, imgs, w_feat=1.0, w_rpn=1.0, T=1.0, box_w=1.0):
        raise NotImplementedError("yolo kd_loss jos nije — krecemo s frcnn")


ADAPTERS = [FrcnnAdapter, YoloAdapter]


def pick_adapter(model):
    if USE_PROFILE_ADAPTERS:                              # novi put: tanki profili nad dijeljenim handlerima
        import profiles
        pa = profiles.pick_profile(model)
        if pa is not None:
            return pa
    for A in ADAPTERS:                                    # sigurnosni fallback: stari monolitni adapteri
        if A.matches(model):
            return A
    raise SystemExit(f"Nema ModelAdaptera za {type(model).__name__} — dodaj adapter (matches/forward/gt_loss).")


# =========================== REGISTRI ZA About (izvori istine — model-agnosticno) =========================== #
# Metrike PO TASKU (dijeli ih i FT/Overview). headline = glavne koje se uvijek prikazuju; ostatak iza expandera.
# Task je "podrzan" tek kad ima I evaluator (_EVALUATORS, mjerljivo) I profil (profiles.PROFILES, kompresibilno).
TASK_METRICS = {
    "detection": {
        "headline": ["map", "mar_100"],
        "all": {"map": "mAP@[.50:.95]", "mar_100": "mAR@100", "map_50": "mAP@.50", "map_75": "mAP@.75",
                "map_small": "mAP (mali)", "map_medium": "mAP (srednji)", "map_large": "mAP (veliki)"},
    },
    "classification": {
        "headline": ["acc"],
        "all": {"acc": "Top-1 točnost", "top5": "Top-5 točnost"},
    },
}

# LAYER-ROLE POLICY (dvije nezavisne osi: STRUKTURNO prune/grow↔off-limits, TRENING trainable↔frozen).
# Opceniti popis TIPOVA (ne brojanje po modelu). Iste oznake/razlozi koje koristi i structural_flags.
LAYER_POLICY = [
    {"tip": "Conv1d/2d/3d, Linear (s downstream potrošačem)", "strukturno": "prunable/growable",
     "trening": "trainable", "razlog": "jezgra rezanja (torch-pruning DependencyGraph)"},
    {"tip": "feature-tap (FPN / neck-izlaz)", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "egzaktni feature-KD — kanali student↔teacher moraju ostati poravnati"},
    {"tip": "RPN / anchor sučelje", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "anchor geometrija (2-stage)"},
    {"tip": "izlazna glava (klasifikator / detektor)", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "fiksan broj klasa"},
    {"tip": "depthwise konvolucija", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "reže se kroz producera (grupe se ne diraju izravno)"},
    {"tip": "attention", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "torch-pruning nesiguran (qkv/head spregnutost, reshape-i)"},
    {"tip": "nepoznat / nepodržan tip", "strukturno": "off-limits", "trening": "trainable",
     "razlog": "siguran default — ne prepoznajemo ga pa ne diramo strukturu, ali se fine-tuna"},
]

# FAZE pipelinea (implementirano vs planirano).
PHASES = {
    "done": ["analiza (census aktivnosti + grad-importance)",
             "mjerenje performansi (mAP / params / GFLOPs / CPU+GPU brzina)",
             "dead/near-dead removal + KD FT recovery"],
    "planned": ["kontinuirani prune + uvjetni grow", "Pareto trajektorija + selektor verzija",
                "kvantizacija (placeholder)"],
}


def capabilities():
    """Dinamicki popis PODRZANOG (za About) — model-agnosticno, NE ucitava nikakav model.
    CITA iz registara: kd._LOSS(+_DOC), _EVALUATORS, profiles.PROFILES, TASK_METRICS, LAYER_POLICY, PHASES.
    Dodaj task/handler/profil/fazu -> stranica se sama azurira."""
    import kd
    import profiles as P

    profile_tasks = {p["task"] for p in P.PROFILES}
    tasks = []
    for t in sorted(set(_EVALUATORS) | profile_tasks):
        has_metric, has_profile = t in _EVALUATORS, t in profile_tasks
        tm = TASK_METRICS.get(t, {})
        tasks.append({"task": t, "metric": has_metric, "profile": has_profile,
                      "status": "full" if (has_metric and has_profile) else "partial",
                      "headline": tm.get("headline", []), "metrics": tm.get("all", {})})

    profiles_out = []
    for p in P.PROFILES:
        prot = p.get("protect", [])
        prot_desc = ("dinamički (sučelja koja hrane glavu + glava)" if callable(prot)
                     else (", ".join(prot) if prot else "—"))
        profiles_out.append({"kind": p["kind"], "task": p["task"], "imgsz": p["imgsz"],
                             "kd_types": sorted({tap["type"] for tap in p.get("kd_taps", [])}),
                             "protect": prot_desc})

    return {
        "what": "Kompresija mreza (structured prune + opcionalni grow) uz ocuvanje kvalitete kroz "
                "KD imitaciju zamrznutog originala. Budzet u GFLOPs, mali inkrementalni koraci.",
        "training_loss": "pure-KD (feature + klase [+ box/RPN]) vs zamrznuti original — BEZ GT u treningu",
        "optimizer": "Prodigy (auto-LR) + warmup",
        "metric_note": "metrika se bira PO TASKU; GT se koristi SAMO za metriku, ne u treningu",
        "budget": "GFLOPs (total)",
        "tasks": tasks,
        "kd_types": [{"type": k, "doc": kd._LOSS_DOC.get(k, "")} for k in sorted(kd._LOSS)],
        "profiles": profiles_out,
        "layer_policy": LAYER_POLICY,
        "phases": PHASES,
    }


def weighted_leaves(model):
    """[(name, module, type, weight)] za leaf module s conv/linear weightom. Iz analyze.py."""
    out = []
    for name, m in model.named_modules():
        if list(m.children()):
            continue
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() in (2, 4):
            out.append((name, m, type(m).__name__, w))
    return out


def layer_table(model, adapter, device):
    """Forward s hookovima -> per-layer zapis (role/units/params/GFLOPs/in-out shape),
    u FORWARD redoslijedu. Generalizira: conv (dim>=3) = filteri, linear (dim==2) = neuroni."""
    leaves = weighted_leaves(model)
    by_id = {id(m): (name, tn, w) for name, m, tn, w in leaves}
    rec, handles = [], []

    def mk(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and o:
                o = o[0]
            name, tn, w = by_id[id(mod)]
            ishape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            oshape = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() >= 3:                                  # conv (1d/2d/3d)
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:                                # linear
                    out_f, in_f = w.shape
                    flops = 2 * out_f * in_f * (o.numel() // out_f if out_f else 0)
            rec.append({"name": name, "type": tn,
                        "role": "neuron" if w.dim() == 2 else "filter",
                        "units": int(w.shape[0]),
                        "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                        "gflops": flops / 1e9, "in": ishape, "out": oshape})
        return hook

    for name, m, tn, w in leaves:
        handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        adapter.forward(model, [torch.rand(3, 640, 640, device=device)])   # adapter rjesava resize/batch
    for h in handles:
        h.remove()
    return rec


def model_num_classes(model):
    """Best-effort broj klasa koje model podrzava (+ izvor). None ako nepoznato — radi na bilo cemu."""
    if isinstance(getattr(model, "nc", None), int):
        return int(model.nc), "model.nc"
    cs = getattr(getattr(getattr(model, "roi_heads", None), "box_predictor", None), "cls_score", None)
    if cs is not None and hasattr(cs, "out_features"):
        return cs.out_features - 1, "roi cls_score.out_features-1 (bez background)"
    for m in model.modules():                          # ultralytics Detect glava ima .nc
        v = getattr(m, "nc", None)
        if isinstance(v, int) and v > 0:
            return v, f"{type(m).__name__}.nc"
    names = getattr(model, "names", None)
    if names is not None:
        try:
            return len(names), "len(model.names)"
        except Exception:
            pass
    return None, "nepoznato"


def trainable_leaves(model):
    """name -> bool: MOZE li se sloj kasnije fine-tunati (ima ucljive parametre).
    To NIJE 'requires_grad pri loadu' — frozen-at-load (inference ckpt / torchvision default) se SVEJEDNO
    moze odmrznuti i fine-tunati (RPN: niti se pruna niti grow-a, ali se fine-tuna da prati izmjene arhitekture).
    Zamrzavanje je morph-ODLUKA (politika), ne svojstvo modela -> 'frozen' je ~0 osim ako zadamo freeze-listu."""
    return {name: any(p.numel() > 0 for p in m.parameters(recurse=False))
            for name, m, _, _ in weighted_leaves(model)}


def structural_flags(model, adapter, device):
    """prunable/growable po sloju preko tp.DependencyGraph (graf povezanosti).
    prunable = validna pruning-grupa KOJA dira >=1 drugi weighted (Conv/Linear) sloj (= ima potrosaca;
    terminalne izlazne glave diraju samo sebe -> N). growable = isti skup (ista potreba za rewiringom).
    Vrati (name->bool, note_or_None). struct=None -> tp graf nije izgradiv (degradiraj na '?');
    note != None uz validan dict -> info (npr. attention slojevi preskoceni)."""
    for p in model.parameters():
        p.requires_grad_(True)        # tp tracer treba grad-graf; inference ckpt je frozen -> 0 ovisnosti.
                                      # (trainable-status je vec uhvacen PRIJE ovog poziva)
    # ATTENTION slojevi: tp.get_pruning_group na multi-head QKV-u TVRDO obori proces (nije Python iznimka,
    # pa nehvatljivo). Detektiraj attention module i preskoci njihove leafove (prunable=N).
    att_pref = [nm + "." for nm, mod in model.named_modules()
                if hasattr(mod, "num_heads") or "attention" in type(mod).__name__.lower()
                or "attn" in type(mod).__name__.lower()]
    under_att = lambda name: any(name.startswith(p) for p in att_pref)
    try:
        import torch_pruning as tp
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    prot = []                                             # OFF-LIMITS sucelja (feature-tap/RPN): ne reze se (B pristup)
    gp = getattr(adapter, "protect_prefixes", None)
    if gp:
        try:
            prot = gp(model)
        except Exception:
            prot = []
    under_prot = lambda name: any(name.startswith(p) for p in prot)
    # SE / channel-attention: conv/linear koji radi nad GLOBALNO-POOLANIM deskriptorom (ULAZNI prostorni oblik 1×1)
    # -> nije prostorni feature-sloj nego "paznja po kanalima". Rast mu ne dodaje kapacitet (FLOPs~0) i vara score;
    # rez ga ionako ne bira (cost~0 -> rangiran zadnji). Generalno (SE/CBAM, neovisno o imenima); off-limits za oboje.
    # (Pointwise 1×1 conv u glavnom putu ima ulaz H×W>1 -> NIJE pogodjen; razlika je ulazni prostorni oblik, ne kernel.)
    se_names = set()
    try:
        for r in layer_table(model, adapter, device):
            ish = r.get("in")
            if ish is not None and len(ish) >= 3 and all(int(d) == 1 for d in ish[2:]):
                se_names.add(r["name"])
    except Exception:
        pass
    out, n_skip, n_prot, n_se = {}, 0, 0, 0
    for name, m, _, w in weighted_leaves(model):
        if under_att(name):                               # attention -> tp-unsafe, preskoci
            out[name] = False; n_skip += 1; continue
        if name in se_names:                              # SE/channel-attention (ulaz 1×1) -> off-limits (prune i grow)
            out[name] = False; n_se += 1; continue
        if under_prot(name):                              # zasticeno sucelje -> off-limits (ali trainable)
            out[name] = False; n_prot += 1; continue
        fn = pconv if w.dim() >= 3 else plin
        ok = False
        try:
            g = DG.get_pruning_group(m, fn, idxs=[0])
            if DG.check_pruning_group(g):
                wmods = set()
                for item in g:
                    dep = getattr(item, "dep", None) or (item[0] if isinstance(item, (tuple, list)) else None)
                    mm = getattr(getattr(dep, "target", None), "module", None)
                    if isinstance(mm, (nn.Conv2d, nn.Linear)):
                        wmods.add(id(mm))
                ok = len(wmods) > 1                       # root + >=1 weighted potrosac
        except BaseException:
            ok = False
        out[name] = ok
    bits = []
    if n_skip:
        bits.append(f"{n_skip} attention (tp-unsafe)")
    if n_se:
        bits.append(f"{n_se} SE/channel-attention (ulaz 1×1, off-limits)")
    if n_prot:
        bits.append(f"{n_prot} zasticeno sucelje (feature-tap/RPN/glava, off-limits)")
    note = (" · ".join(bits) + " -> prunable=N") if bits else None
    return out, note


# --------------------------------------------------------------------------- #
# Blok 4: PRUNING potencijal (GT-grad, mean-norm) — GT SAMO za analizu!
# --------------------------------------------------------------------------- #
class _DetDataset(torch.utils.data.Dataset):
    """(image[0,1] CHW, target{boxes xyxy px, labels 1..6}). Kompaktno kopirano iz common3."""
    def __init__(self, split):
        self.items = sorted(p for p in _img_dir(split).iterdir() if p.suffix.lower() in IMG_EXTS)
        if DEV_DATA_SUBSET:
            self.items = self.items[:DEV_DATA_SUBSET]    # ⚠️ dev: samo prvih N po splitu (sve nizvodno se kapira ovdje)
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import cv2
        import numpy as np
        p = self.items[idx]
        bgr = cv2.imread(str(p)); h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        boxes, labels = [], []
        lf = _lbl_dir(self.split) / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text().splitlines():
                q = line.split()
                if len(q) != 5:
                    continue
                c = int(q[0]); cx, cy, bw, bh = map(float, q[1:])
                x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2]); labels.append(c + 1)
        return img, {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                     "labels": torch.tensor(labels, dtype=torch.int64)}


def make_gt_loader(split="train", bs=GRAD_BATCH, workers=2):
    return torch.utils.data.DataLoader(_DetDataset(split), batch_size=bs, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]))


# =========================== PERFORMANSE (mjerenje + persist) =========================== #
# Mjerenje zivi OVDJE (analyze/Overview ga racuna i sprema na disk), a compress ga samo CITA
# za usporedbu s originalom -> baseline se NE racuna dvaput.
def model_name(spec):
    """Kratko ime modela (bez ekstenzije) = kljuc za tmp/<name>/ (teacher cache + baseline_perf.json)."""
    return spec if spec == "fasterrcnn" else os.path.basename(str(spec)).split(".")[0]


def perf_path(name):
    suff = f"_dev{DEV_DATA_SUBSET}" if DEV_DATA_SUBSET else ""   # dev baseline odvojen od pravog (razlicit broj slika)
    return os.path.join(TMP_ROOT, name, f"baseline_perf{suff}.json")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def gflops_total(model, adapter, device):
    """Suma GFLOPs po svim weighted slojevima (per-layer iz layer_table)."""
    return sum(r["gflops"] for r in layer_table(model, adapter, device))


@torch.no_grad()
def eval_map(model, adapter, loader, device, max_images=None):
    """mAP (sve torchmetrics metrike) na splitu. preds preko adapter.predict; GT iz loadera (xyxy px)."""
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.warn_on_many_detections = False
    model.eval()
    n = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        preds = adapter.predict(model, imgs)
        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
        n += len(imgs)
        if max_images and n >= max_images:
            break
    out = metric.compute()
    return {k: float(v) for k, v in out.items()
            if hasattr(v, "numel") and v.numel() == 1 and "per_class" not in k}, n   # bez per-class (ionako -1)


@torch.no_grad()
def eval_classification(model, adapter, loader, device, max_images=None):
    """Top-1/Top-5 tocnost (klasifikacija). adapter.predict -> logiti [B,nc]; loader vraca (inputs, labels)."""
    model.eval()
    c1 = c5 = n = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device).view(-1)
        logits = adapter.predict(model, x)
        k = min(5, logits.shape[1])
        top = logits.topk(k, dim=1).indices
        c1 += (top[:, 0] == y).sum().item()
        c5 += (top == y.unsqueeze(1)).any(1).sum().item()
        n += y.numel()
        if max_images and n >= max_images:
            break
    return {"acc": c1 / max(n, 1), "top5": c5 / max(n, 1)}, n


# Metrika se bira PO TASKU (detection->mAP, classification->acc, segmentation->mIoU TBD) — model-agnosticno.
_EVALUATORS = {"detection": eval_map, "classification": eval_classification}


def evaluate(model, adapter, loader, device, max_images=None):
    """Dispatch evaluatora po adapter.task. Vrati (metric_dict, n) — kljucevi ovise o tasku."""
    task = getattr(adapter, "task", "detection")
    fn = _EVALUATORS.get(task)
    if fn is None:
        raise NotImplementedError(f"Nema evaluatora za task '{task}' (dodaj u _EVALUATORS).")
    return fn(model, adapter, loader, device, max_images)


@torch.no_grad()
def bench_speed(model, adapter, dev_str, reps=10, warmup=3, drop_worst=2, imgsz=640):
    """ms/sliku: warmup pa `reps` mjerenja; izbaci `drop_worst` najsporijih, vrati median ostalih."""
    dev = torch.device(dev_str)
    m = model.to(dev).eval()
    img = torch.rand(3, imgsz, imgsz, device=dev)
    for _ in range(warmup):
        adapter.forward(m, [img])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        adapter.forward(m, [img])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = sorted(ts)[:max(1, reps - drop_worst)]          # izbaci najsporije
    return statistics.median(ts)


def perf_report(model, adapter, loaders, device, eval_max=None, step=None):
    """Cjeloviti izvjestaj: {params, gflops, maps:{split:{...}}, n_eval:{split:n}, gpu_ms, cpu_ms}.
    step(frac,msg) opcionalni progress callback (za Overview % loading)."""
    def _s(f, m):
        if step:
            step(f, m)
    rep = {"params": count_params(model), "gflops": gflops_total(model, adapter, device),
           "maps": {}, "n_eval": {}}
    splits = list(loaders.items())
    for i, (split, loader) in enumerate(splits):
        _s(0.1 + 0.7 * i / max(len(splits), 1), f"mAP {split}")
        m, n = evaluate(model, adapter, loader, device, eval_max)   # metrika po tasku (detection->mAP)
        rep["maps"][split] = m; rep["n_eval"][split] = n
    _s(0.85, "brzina GPU/CPU")
    rep["gpu_ms"] = bench_speed(model, adapter, "cuda") if torch.cuda.is_available() else None
    rep["cpu_ms"] = bench_speed(model, adapter, "cpu")
    model.to(device)                                     # vrati na glavni uredaj nakon CPU bencha
    return rep


def print_perf(rep, tag=""):
    print(f"\n  === PERF{(' ' + tag) if tag else ''} ===")
    print(f"  params={rep['params']:,}  GFLOPs={rep['gflops']:.3f}  "
          f"GPU={rep['gpu_ms']:.2f} ms/img  CPU={rep['cpu_ms']:.2f} ms/img")
    keys = []                                            # metrike: unija kljuceva svih splitova
    for m in rep["maps"].values():
        for k in m:
            if k not in keys:
                keys.append(k)
    print(f"  {'split':<8}{'n':>7}  " + "".join(f"{k:>10}" for k in keys))
    for split, m in rep["maps"].items():
        print(f"  {split:<8}{rep['n_eval'][split]:>7}  " + "".join(f"{m.get(k, float('nan')):>10.4f}" for k in keys))


def baseline_perf(spec, device="cuda", recompute=False, eval_max=EVAL_MAX, progress=None, model=None, adapter=None):
    """Baseline perf ORIGINALNOG modela: iz cachea (tmp/<name>/baseline_perf.json) ako postoji, inace izracunaj+spremi.
    Ako su model/adapter vec ucitani (npr. u analyze_report) -> koristi njih (bez ponovnog load-a)."""
    name = model_name(spec)
    p = perf_path(name)
    if not recompute and os.path.exists(p):
        return json.load(open(p))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_any(spec, dev); adapter = pick_adapter(model)
    loaders = {s: make_gt_loader(s, bs=EVAL_BATCH) for s in list_splits()}
    rep = perf_report(model, adapter, loaders, dev, eval_max, step=progress)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rep, open(p, "w"))
    return rep


def grad_pass(model, adapter, device, loader, max_images=None):
    """Jedan GT-backward prolaz preko skupa. Vrati:
       imp[name]  = mean|grad| po jedinici (PRUNE; mean-norm, usporedivo medu slojevima),
       gavg[name] = prosjecni (signed) grad matrice [O, ...] (GROW: GradMax-style SVD),
       n_imgs, dt. Generalizira na conv (dim>=3) i linear (dim==2)."""
    leaves = weighted_leaves(model)
    abs_acc = {nm: torch.zeros(w.shape[0]) for nm, _, _, w in leaves}
    sgn_acc = {nm: torch.zeros(w.shape, device="cpu") for nm, _, _, w in leaves}
    for p in model.parameters():
        p.requires_grad_(True)
    # UNIVERZALNA zastita: snapshot BN bufera (running_mean/var/num_batches) -> vrati ih netaknute nakon prolaza.
    # gt_loss radi u train modu (RCNN loss); cak i ako neki (bud$ci) adapter zaboravi set_bn_eval, baseline ostaje cist.
    bn_snap = [(m, m.running_mean.clone(), m.running_var.clone(),
                None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
               for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None]
    nb = nimg = 0; t0 = time.time()
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for p in model.parameters():
            p.grad = None
        adapter.gt_loss(model, imgs, targets).backward()
        for nm, m, _, w in leaves:
            g = m.weight.grad
            if g is not None:
                gd = g.detach().cpu()
                abs_acc[nm] += gd.abs().flatten(1).mean(1)         # prune: mean|grad| po jedinici
                sgn_acc[nm] += gd                                  # grow: signed grad (za SVD)
        nb += 1; nimg += len(imgs)
        if max_images and nimg >= max_images:
            break
    for p in model.parameters():
        p.grad = None
    for m, rm, rv, nbt in bn_snap:                        # vrati BN bufere (neovisno o adapteru) -> nema korupcije baseline-a
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        if nbt is not None:
            m.num_batches_tracked.copy_(nbt)
    nb = max(nb, 1)
    imp = {nm: abs_acc[nm] / nb for nm in abs_acc}
    gavg = {nm: sgn_acc[nm] / nb for nm in sgn_acc}
    return imp, gavg, nimg, time.time() - t0


# --------------------------------------------------------------------------- #
# Blok 3: aktivnost (dead / near-dead PO LOKACIJI) — post-aktivacijski, cijeli skup
# --------------------------------------------------------------------------- #
ACT_TYPES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.GELU, nn.Hardswish,
             nn.Hardsigmoid, nn.ELU, nn.Mish)


@torch.no_grad()
def activation_stats(model, adapter, device, loader, max_images=None, eps=1e-6, weak=0.01):
    """Post-aktivacijska aktivnost PO LOKACIJI preko skupa. dead = max<=eps (nikad ne opali);
    near-dead = opali na <1% lokacija. Statistike se kljuce po PRETHODNOM conv/linear (forward redoslijed,
    st['last']); time radi i kad je aktivacija DIJELJEN modul (ultralytics: jedna SiLU za sve Conv-ove)."""
    leaf_id = {id(m): nm for nm, m, _, _ in weighted_leaves(model)}
    stats, handles = {}, []
    st = {"last": None}

    def leaf_hook(m, i, o):
        st["last"] = leaf_id[id(m)]                          # prati zadnji conv/linear (svaki forward)

    def act_hook(m, i, o):
        leaf = st["last"]
        if leaf is None or not isinstance(o, torch.Tensor) or o.dim() not in (2, 4):
            return
        dims = (0, 2, 3) if o.dim() == 4 else (0,)
        nel = o.shape[0] * (o.shape[2] * o.shape[3] if o.dim() == 4 else 1)
        pos = (o > eps).sum(dim=dims).float(); mx = o.amax(dim=dims)
        s = stats.get(leaf)
        if s is None:
            stats[leaf] = {"pos": pos.clone(), "nel": nel, "max": mx.clone()}
        elif s["pos"].numel() == pos.numel():                # ista jedinica -> akumuliraj
            s["pos"] += pos; s["nel"] += nel; s["max"] = torch.maximum(s["max"], mx)

    for nm, m in model.named_modules():
        if id(m) in leaf_id:
            handles.append(m.register_forward_hook(leaf_hook))
        elif isinstance(m, ACT_TYPES):
            handles.append(m.register_forward_hook(act_hook))
    model.eval()
    nimg = 0
    for imgs, _ in loader:
        imgs = [im.to(device) for im in imgs]
        adapter.forward(model, imgs)
        nimg += len(imgs)
        if max_images and nimg >= max_images:
            break
    for h in handles:
        h.remove()

    out = {}
    for leaf, s in stats.items():
        afrac = s["pos"] / max(s["nel"], 1)
        dead_mask = s["max"] <= eps
        near_mask = (afrac < weak) & (s["max"] > eps)
        out[leaf] = {"dead": int(dead_mask.sum()), "near": int(near_mask.sum()), "C": s["pos"].numel(),
                     "dead_idx": dead_mask.nonzero(as_tuple=False).flatten().tolist(),
                     "near_idx": near_mask.nonzero(as_tuple=False).flatten().tolist()}
    return out, nimg


def print_combined(rec, imp, act, grow, train_set, struct, flops_per):
    """Po sloju (forward redoslijed) 5 redaka: [general]/[flags]/[dead]/[prune]/[grow]. Prazan red izmedu; TOTAL na kraju.
    flags: prunable/growable (tp graf), trainable (requires_grad). dead/near = post-aktivacijski po lokaciji;
    prune mean_imp = mean-norm; imp/GFLOP = vaznost po SPREGNUTOM FLOP-u (nizak = jeftin rez)."""
    print("\n  === PER-LAYER: general / flags / dead / prune / grow (redni broj = forward redoslijed) ===")
    tot_c = tot_dead = tot_near = 0
    for i, r in enumerate(rec):
        ish = "x".join(map(str, r["in"][1:])) if r["in"] else "?"
        osh = "x".join(map(str, r["out"][1:])) if r["out"] else "?"
        print(f"  [{i:>3}] {r['name']}")
        print(f"    [general] {r['type']:<10} {r['role']:<6} units={r['units']:<5} params={r['params']:<9} "
              f"GFLOPs={r['gflops']:.4f}  {ish} -> {osh}")
        pr = struct.get(r["name"]) if struct is not None else None
        pr_s = "Y" if pr else ("N" if pr is not None else "?")
        tr_s = "Y" if train_set.get(r["name"], False) else "N"
        print(f"    [flags]   prunable={pr_s} growable={pr_s} trainable={tr_s}")
        a = act.get(r["name"])
        if a is not None:
            tot_c += a["C"]; tot_dead += a["dead"]; tot_near += a["near"]
            print(f"    [dead]    dead={a['dead']}/{a['C']}  near-dead<1%={a['near']}/{a['C']}")
        else:
            print("    [dead]    - (nema aktivacije iza sloja)")
        fu = flops_per.get(r["name"], r["gflops"] / max(r["units"], 1))   # spregnuti per-kanal GFLOPs (fallback: vlastiti)
        v = imp.get(r["name"])
        if v is not None:
            mi = float(v.float().mean())
            ipf = mi / fu if fu > 0 else float("inf")
            print(f"    [prune]   mean_imp={mi:.3e}  GFLOP/unit={fu:.5f}  imp/GFLOP={ipf:.3e}")
        else:
            print("    [prune]   -")
        g = grow.get(r["name"])
        if g is not None and len(g):
            s1 = float(g[0])
            bpf = s1 / fu if fu > 0 else float("inf")
            nuse = int((g > 0.01 * g[0]).sum())          # ~koliko novih neurona bi smisleno pomoglo
            print(f"    [grow]    benefit(σ1)={s1:.3e}  benefit/GFLOP={bpf:.3e}  korisnih_σ={nuse}/{len(g)}")
        else:
            print("    [grow]    -")
        print()
    print("  === TOTAL (mjereni slojevi s aktivacijom) ===")
    print(f"  dead: {tot_dead}/{tot_c} ({100*tot_dead/max(tot_c,1):.1f}%)  |  "
          f"near-dead<1%: {tot_near}/{tot_c} ({100*tot_near/max(tot_c,1):.1f}%)")

    n_total = len(rec)
    n_train = sum(1 for r in rec if train_set.get(r["name"], False))
    prun_s = str(sum(1 for r in rec if struct.get(r["name"]))) if struct is not None else "n/a (tp graf nedostupan)"
    print("\n  === LAYER FLAGS TOTAL ===")
    print(f"  Total layers: {n_total}")
    print(f"  prunable:  {prun_s}")
    print(f"  growable:  {prun_s}")
    print(f"  trainable: {n_train}  (moze se fine-tunati; frozen-at-load se svejedno moze odmrznuti)")
    print(f"  frozen:    {n_total - n_train}  (nema ucljivih param; zamrzavanje je inace morph-odluka)")


def print_top_prune(cand, imp, flops_per, k=10):
    """Top-k filtera/neurona za rez iz ZAJEDNICKOG rankinga (prune_candidates — IDENTICNO kompresoru):
    najjeftiniji prvo (nizak score = importance/blended-cost). Prikaz: importance, SPREGNUTI GFLOPs, score."""
    rows = []
    print(f"\n  === TOP-{k} PRUNE (filteri/neuroni; ZAJEDNICKI ranking s kompresorom; nizak score = najjeftiniji rez) ===")
    print(f"  {'sloj':<42}{'unit#':>6}{'importance':>13}{'GFLOPs':>11}{'score':>12}")
    for score, nm, i in cand[:k]:
        ii = float(imp[nm].float()[i]); fu = float(flops_per.get(nm, 0.0))
        nms = nm if len(nm) <= 41 else "…" + nm[-40:]
        print(f"  {nms:<42}{i:>6}{ii:>13.3e}{fu:>11.5f}{score:>12.3e}")
        rows.append({"layer": nm, "unit": i, "importance": ii, "gflops": fu, "score": score})
    return rows


def print_top_grow(gcand, sigma, flops_per, k=10):
    """Top-k SLOJEVA za rast iz ZAJEDNICKOG rankinga (compress.grow_candidates — IDENTICNO kompresoru):
    najveci score = σ_max/coupled-flops prvo. σ_max = korist najboljeg novog neurona (GradMax); korisnih_σ = koliko
    novih neurona bi smisleno pomoglo (σ_i > 1% σ_max). Prikaz po sloju (kompresor i odlucuje po sloju)."""
    rows = []
    print(f"\n  === TOP-{k} GROW (slojevi; ZAJEDNICKI ranking s kompresorom; visok score = najbolji rast/FLOP) ===")
    print(f"  {'sloj':<42}{'σ_max':>13}{'GFLOPs':>11}{'score':>12}{'korisnih_σ':>11}")
    for score, nm, smax in gcand[:k]:
        s = sigma.get(nm)
        nuse = int((s > 0.01 * s[0]).sum()) if (s is not None and len(s)) else 0
        fu = float(flops_per.get(nm, 0.0))
        nms = nm if len(nm) <= 41 else "…" + nm[-40:]
        print(f"  {nms:<42}{smax:>13.3e}{fu:>11.5f}{score:>12.3e}{nuse:>11}")
        rows.append({"layer": nm, "benefit_sigma_max": smax, "gflops": fu, "score": score, "useful_sigma": nuse})
    return rows


def plot_layers(rec, imp, grow, struct, out_path, title, flops_per, cap_mean=False):
    """3 subplota (zajednicki x = layer index, FORWARD order): GFLOPs / prune-imp-per-GFLOP /
    grow-benefit-per-GFLOP. imp/benefit-per-GFLOP koriste SPREGNUTI per-kanal trosak (root+potrosaci).
    CRVENI stupci = sloj NIJE prunable/growable (off-limits).
    cap_mean=True -> y-os svakog subplota cappana na mean te metrike (vidi se struktura ispod spikeova)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    n = len(rec); x = list(range(n))
    gf, ipf, gbf = [], [], []
    for r in rec:
        v = imp.get(r["name"]); m = float(v.float().mean()) if v is not None else 0.0
        s = grow.get(r["name"]); s1 = float(s[0]) if (s is not None and len(s)) else 0.0
        fu = flops_per.get(r["name"], r["gflops"] / max(r["units"], 1))   # spregnuti per-kanal GFLOPs (fallback: vlastiti)
        gf.append(r["gflops"]); ipf.append(m / fu if fu > 0 else 0.0); gbf.append(s1 / fu if fu > 0 else 0.0)
    locked = [bool(struct is not None and not struct.get(r["name"])) for r in rec]   # crveno = off-limits
    def cols(base):
        return ["tab:red" if lk else base for lk in locked]
    series = [                                            # zadnji flag = smije li se cappati (grow: NE)
        (gf, "GFLOPs", "tab:blue", True),
        (ipf, "prune imp / GFLOP  (low = cheap to prune)", "tab:green", True),
        (gbf, "grow benefit / GFLOP  (high = best gain)", "tab:purple", False)]   # grow: gledamo vrhove
    fig, axes = plt.subplots(len(series), 1, figsize=(max(12, n * 0.14), 9), sharex=True)
    for ax, (data, lab, base, cappable) in zip(axes, series):
        ax.bar(x, data, width=0.9, color=cols(base))
        ax.set_ylabel(lab, fontsize=8); ax.grid(axis="y", alpha=0.3)
        if cap_mean and cappable:
            mean = sum(data) / max(len(data), 1)
            ax.set_ylim(0, mean if mean > 0 else 1)
            ax.axhline(mean, color="black", lw=0.8, ls="--", alpha=0.6)
    axes[0].legend(handles=[mpatches.Patch(color="tab:red", label="NOT prunable/growable")],
                   fontsize=8, loc="upper right")
    axes[-1].set_xlabel("layer index (forward order: input -> output)")
    axes[-1].set_xticks(range(0, n, max(1, n // 25)))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def analyze_report(spec, device="cuda", progress=None):
    """Analiza JEDNOG modela. PRINTA u terminal (kao i dosad) I vraca strukturirani dict (za GUI).
    progress(frac, msg) callback (opcionalno) za % loading. Sav compute je ovdje; GUI samo renderira dict."""
    def step(f, msg):
        if progress:
            progress(min(f, 1.0), msg)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    tag = spec if spec == "fasterrcnn" else Path(str(spec)).name
    print(f"\n=== ANALIZA: {tag} (device={dev}) ===")
    if dev_subset_note():
        print("  " + dev_subset_note())
    step(0.03, "ucitavanje modela")
    model = load_any(spec, dev); adapter = pick_adapter(model)
    step(0.10, "per-layer tablica (forward)")
    rec = layer_table(model, adapter, dev)
    types = {}
    for r in rec:
        types[r["type"]] = types.get(r["type"], 0) + 1
    n_filt = sum(r["units"] for r in rec if r["role"] == "filter")
    n_neur = sum(r["units"] for r in rec if r["role"] == "neuron")
    gflops = sum(r["gflops"] for r in rec)
    n_par = sum(p.numel() for p in model.parameters())
    nc, src = model_num_classes(model)
    print(f"[{tag}] task={adapter.task} | kind={adapter.kind} | klase={nc} ({src})")
    print(f"  tipovi slojeva (weighted): " + ", ".join(f"{t}×{n}" for t, n in sorted(types.items())))
    print(f"  filteri={n_filt} | neuroni={n_neur} | params={n_par/1e6:.2f}M | GFLOPs@{adapter.imgsz}px={gflops:.3f}")

    step(0.20, "trainable + tp graf (prunable/growable)")
    train_set = trainable_leaves(model)
    struct, struct_note = structural_flags(model, adapter, dev)
    if struct is None:
        print(f"  [flags] tp graf nedostupan -> prunable/growable='?'  ({struct_note})")
    elif struct_note:
        print(f"  [flags] {struct_note}")
    # Overview je SAMO demo viz -> posuduje ISTI kod iz KOMPRESIJE (compress je autoritativan; lazy import izbjegava cikl).
    import compress as C
    prunable_nm = [r["name"] for r in rec if struct is not None and struct.get(r["name"])]
    cost, flops_per, units = C.prune_costs(model, adapter, dev, prunable_nm)   # units = sirine (za grow align faktor)
    info = {nm: (m, w.dim()) for nm, m, _, w in weighted_leaves(model)}

    step(0.25, "aktivnost (dead/near-dead)")
    loader = make_gt_loader()
    act, n_act = activation_stats(model, adapter, dev, loader, GRAD_MAX_IMAGES)
    print(f"\n  [aktivnost] {n_act} slika")
    step(0.40, "gradijenti (prune/grow)")
    try:
        imp, gavg, nimg, t_imp = grad_pass(model, adapter, dev, loader, GRAD_MAX_IMAGES)
        grow = C.grow_potential(gavg)                     # σ po sloju (kod u compress, Overview posuduje)
        print(f"  [prune+grow] GT-grad {nimg} slika u {t_imp:.0f}s")
    except NotImplementedError as e:
        imp, grow = {}, {}
        print(f"  [prune+grow] preskoceno ({e})")

    # PERFORMANSE: mjeri se OVDJE i sprema na disk; compress ovo cita kao baseline (bez ponovnog racunanja).
    step(0.55, "performanse (mAP train/val/test + brzina)")
    perf = baseline_perf(spec, dev, recompute=True, model=model, adapter=adapter,   # analiza = autoritativno mjerenje
                         progress=lambda f, m: step(0.55 + 0.33 * f, m))            # -> spremi; compress samo cita
    print_perf(perf, tag="original")

    step(0.90, "tablica + top-10 + plotovi")
    if SHOW_LAYER_TABLE:
        print_combined(rec, imp, act, grow, train_set, struct, flops_per)
    cand = C.prune_candidates(imp, cost, info, struct)        # ISTI prune ranking kao kompresor (_select_prune_plan ga zove)
    top_p = print_top_prune(cand, imp, flops_per)
    gcand = C.grow_candidates(grow, flops_per, struct, units)  # STATICNI grow ranking (viz); kompresor selektira per-kanal dinamicki (_select_grow_plan)
    top_g = print_top_grow(gcand, grow, flops_per)
    here = Path(__file__).parent
    ttl = f"{tag} - per-layer GFLOPs / prune-imp-per-GFLOP / grow-benefit-per-GFLOP (coupled cost; forward order; red = not prunable/growable)"
    out = here / f"layers_{adapter.kind}.png"
    plot_layers(rec, imp, grow, struct, out, ttl, flops_per)
    out_cap = here / f"layers_{adapter.kind}_mean_capped.png"
    plot_layers(rec, imp, grow, struct, out_cap, ttl + " [y capped at mean]", flops_per, cap_mean=True)
    print(f"\n  [plot] spremljeno: {out}  |  {out_cap}")

    # --- strukturirani report za GUI ---
    layers = []
    for i, r in enumerate(rec):
        v = imp.get(r["name"]); s = grow.get(r["name"]); a = act.get(r["name"])
        mi = float(v.float().mean()) if v is not None else None
        s1 = float(s[0]) if (s is not None and len(s)) else None
        fu = flops_per.get(r["name"], r["gflops"] / max(r["units"], 1))   # spregnuti per-kanal GFLOPs (kao kompresor)
        layers.append({"idx": i, "name": r["name"], "type": r["type"], "role": r["role"],
                       "units": r["units"], "params": r["params"], "gflops": r["gflops"],
                       "prunable": (struct.get(r["name"]) if struct is not None else None),
                       "trainable": train_set.get(r["name"], False),
                       "dead": (a["dead"] if a else None), "near": (a["near"] if a else None),
                       "prune_imp_per_gflop": (mi / fu if (mi is not None and fu > 0) else None),
                       "grow_benefit_per_gflop": (s1 / fu if (s1 is not None and fu > 0) else None)})
    report = {
        "name": tag, "kind": adapter.kind, "task": adapter.task, "classes": nc, "imgsz": adapter.imgsz,
        "params": n_par, "gflops": gflops, "n_filters": n_filt, "n_neurons": n_neur, "types": types,
        "total_layers": len(rec),
        "prunable": (sum(1 for r in rec if struct.get(r["name"])) if struct is not None else None),
        "trainable": sum(1 for r in rec if train_set.get(r["name"], False)),
        "dead_total": sum(a["dead"] for a in act.values()), "near_total": sum(a["near"] for a in act.values()),
        "act_channels": sum(a["C"] for a in act.values()),
        "layers": layers, "top_prune": top_p, "top_grow": top_g,
        "plots": [str(out), str(out_cap)], "struct_note": struct_note,
        "perf": perf,                                    # mAP (train/val/test) + brzina CPU/GPU (spremljeno na disk)
    }
    step(1.0, "gotovo")
    return report


def run_analysis(spec, device="cuda"):
    """Terminalni ulaz (kao dosad) — analyze_report printa sve; povratni dict se ignorira."""
    analyze_report(spec, device)


if __name__ == "__main__":
    t0 = time.time()
    print_dataset_stats(dataset_stats())               # Blok 1a
    print(f"\n[vrijeme] dataset scan: {time.time() - t0:.1f}s")

    run_analysis(MODEL_SPEC)                            # Blok 1b/2/4: introspekcija + per-layer + prune
