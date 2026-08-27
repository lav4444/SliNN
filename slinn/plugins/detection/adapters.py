"""slinn/plugins/detection/adapters.py — detekcijski decode plug (iz morphology/analysis.py).

JEDINI priznati per-obitelj dio: dekodiranje izlaza (yolo DFL/sidra, frcnn ROI), NMS i mAP.
To se NE moze izvesti mjerenjem, pa je ogradeno ovdje. Jezgra ovo doseze samo kroz
backend.build_metric_fn; bez ovog foldera jezgra pada na teacher-agreement gate i radi dalje.
"""

import math
import os
import sys

import torch
import torch.nn as nn

_SLINN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SLINN not in sys.path:
    sys.path.insert(0, _SLINN)

from introspect import unfreeze_bn                            # noqa: E402  (jezgra)
from .dsconfig import (DATASET_ROOT, CLASS_NAMES, IMG_EXTS, DEV_DATA_SUBSET, NUM_CLASSES,
                       COCO_IDS, COCO_YOLO_IDS, GRAD_BATCH, USE_PROFILE_ADAPTERS)


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
        import kdterms as kd
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


def pick_adapter(model, sample_input=None, strict=False):
    """Odaberi detekcijski decode. TRI RAZINE, tim redom:

      1. POZNATA OBITELJ — `matches()` (ultralytics paket / `roi_heads` atribut). Brzo i sigurno
         za modele koje vec podrzavamo.
      2. FALLBACK PO FORMATU IZLAZA — za model koji nikad nismo vidjeli: pokreni ga na `sample_input`,
         klasificiraj OBLIK izlaza (`outfmt.describe`) i posudi decode koji taj oblik zna citati.
         Time nova yolo-slicna mreza iz druge biblioteke prolazi bez ijedne nove linije koda.
      3. NEPREPOZNATO -> vrati None uz GLASNO upozorenje. Jezgra tada degradira na KD-only
         (gate = teacher-agreement) i KOMPRIMIRA DALJE — bez mAP-a, ali radi.

    `strict=True` pretvara 3. u tvrdi prekid (kad je korisnik izricito trazio mAP-gated kompresiju).
    `sample_input=None` preskace 2. razinu (nemamo cime pokrenuti model)."""
    if USE_PROFILE_ADAPTERS:                              # 1a: tanki profili nad dijeljenim handlerima
        from . import profiles
        pa = profiles.pick_profile(model)
        if pa is not None:
            return pa
    for A in ADAPTERS:                                    # 1b: monolitni adapteri (matches)
        if A.matches(model):
            return A

    if sample_input is not None:                          # 2: nepoznat model -> pitaj OBLIK izlaza
        from . import outfmt
        rep = outfmt.describe(model, sample_input)
        print(outfmt.explain(rep))
        name = rep.get("adapter")
        if name:
            cls = {"YoloAdapter": YoloAdapter, "FrcnnAdapter": FrcnnAdapter}[name]
            print("[outfmt] posudujem {} za nepoznat model {}".format(name, type(model).__name__))
            return cls
        why = rep["why"]
    else:
        why = "nema uzorka ulaza za probu izlaza"

    msg = ("Nepoznat detekcijski model {} — {}. Nijedan decode ga ne zna procitati."
           .format(type(model).__name__, why))
    if strict:
        raise SystemExit(msg + " (strict=True)")
    print("[outfmt] UPOZORENJE: " + msg)
    print("[outfmt] nastavljam BEZ detekcijske metrike: gate = teacher-agreement, KD = generic core.")
    return None


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


def make_gt_loader(split="train", bs=GRAD_BATCH, workers=2, frac=None, seed=0):
    """`frac` in (0,1) -> FIKSAN podskup splita, za brzi monitor metrike (PLAN_KOMPRESIJA 1.7).
    Seedani NASUMICNI izbor, ne svaki k-ti: redoslijed datoteka zna biti grupiran (po klasi/izvoru),
    pa bi korak sustavno izostavio dio distribucije. Seed cini izbor ponovljivim, a racuna se jednom
    pri gradnji loadera -> isti uzorci svaku epohu (patience se ne odlucuje na sumu uzorkovanja)."""
    ds = _DetDataset(split)
    if frac and 0.0 < frac < 1.0 and len(ds) > 1:
        import random
        k = max(1, int(round(frac * len(ds))))
        ds = torch.utils.data.Subset(ds, sorted(random.Random(seed).sample(range(len(ds)), k)))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]))


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
