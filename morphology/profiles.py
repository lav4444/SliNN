"""
profiles.py — PER-KOMPONENTA pristup: model se opisuje TANKIM deklarativnim profilom (gdje su tapovi,
kako se grade signali), a sva teska logika (KD math, metrika, prune, BN) je dijeljena. ProfileAdapter
izlaze isto sucelje kao stari adapteri, pa pipeline ne primjecuje razliku.

Stari FrcnnAdapter/YoloAdapter (analysis.py) OSTAJU netaknuti kao sigurnost/orakl (config.USE_PROFILE_ADAPTERS).
"""

import torch

import analysis as A
import kd


def _frcnn_student_signals(student, td, imgs):
    """Studentovi KD-signali za frcnn (ekstrakcija). ROI box se DEKODIRA u xyxy (za istu, teacherovu klasu) -> box_giou."""
    s_imgs, _ = student.transform(imgs)
    s_feat = student.backbone(s_imgs.tensors)
    s_obj, s_box = student.rpn.head(list(s_feat.values()))
    s_bf = student.roi_heads.box_roi_pool(s_feat, td["props"], s_imgs.image_sizes)
    s_cls, s_box2 = student.roi_heads.box_predictor(student.roi_heads.box_head(s_bf))
    dec = student.roi_heads.box_coder.decode(s_box2, td["props"])   # [K, nc, 4]
    cidx = td["roi_cidx"].to(dec.device).long()                    # ISTA klasa kao teacher
    k = torch.arange(dec.shape[0], device=dec.device)
    roi_box = dec[k, cidx]                                          # [K, 4] xyxy
    return {"feat": s_feat, "obj": s_obj, "box": s_box, "cls": s_cls, "roi_box": roi_box}


# DEKLARATIVNA tablica KD-tapova. box (ROI) sad UNIVERZALNI box_giou (kao yolo); RPN box ostaje SmoothL1 (2-stage extra).
FRCNN_KD_TAPS = [
    {"group": "feat",  "type": "feature",    "src": "feat",    "td": "feat",    "w": "w_feat"},
    {"group": "rpn",   "type": "objectness", "src": "obj",     "td": "rpn_obj", "w": "w_rpn"},
    {"group": "rpn",   "type": "box",        "src": "box",     "td": "rpn_box", "w": "w_rpn"},
    {"group": "logit", "type": "logit",      "src": "cls",     "td": "cls",     "w": "one", "kwk": {"T": "T"}},
    {"group": "logit", "type": "box_giou",   "src": "roi_box", "td": "roi_box", "w": "box_w", "kw_td": {"w": "roi_conf"}},
]

FRCNN_PROFILE = {
    "kind": "frcnn", "task": "detection", "imgsz": 320, "SUPPORTS": A.FrcnnAdapter.SUPPORTS,
    "matches": A.FrcnnAdapter.matches,                  # reuse provjerene procedure (identicni rezultati)
    "forward": A.FrcnnAdapter.forward, "gt_loss": A.FrcnnAdapter.gt_loss,
    "tp_example": A.FrcnnAdapter.tp_example, "predict": A.FrcnnAdapter.predict,
    "teacher_outputs": A.FrcnnAdapter.teacher_outputs,
    "student_signals": _frcnn_student_signals, "kd_taps": FRCNN_KD_TAPS,
    # off-limits strukturno (ne prune/grow): FPN = feature-tap (egzaktni feature-KD), RPN = anchor sucelje,
    # roi_heads = detekcijska glava (box_head fc6/fc7 + predictor; FLOPs su ROI-ovisni -> nemjerljivi dummy forwardom,
    # cost~0 bi varao score). Sve trainable. (Glavni-kanalni rez svejedno spregnuto smanji fc2/predictor preko tp.)
    "protect": ["backbone.fpn", "rpn", "roi_heads"],
}

def _yolo_student_signals(student, td, imgs):
    """Studentovi yolo KD-signali: gusti decode (xywh norm + probs nasih 6 + neck feat). BN se adaptira."""
    box, cls, feat = A.YoloAdapter._dense_decode(student, imgs, train_bn=True)
    return {"box": box, "cls": cls, "feat": feat}


def _yolo_protect(model):
    """Off-limits za yolo: SAMO izlazni channel-count blokova koji hrane Detect glavu (= feature-tap), da feature-KD
    egzaktno pase. U ultralytics bloku izlazna projekcija je `cv2` -> stitimo samo nju (unutrasnjost bloka + backbone
    se i dalje rezu). Fallback: cijeli blok ako nema cv2. + sama glava (terminalna). Izvori iz Detect.f."""
    import torch.nn as nn
    seq = list(model.model)
    f = getattr(seq[-1], "f", [])
    idxs = list(f) if isinstance(f, (list, tuple)) else [f]
    pref = []
    for i in idxs:
        if not (isinstance(i, int) and 0 <= i < len(seq)):
            continue
        blk = seq[i]
        if isinstance(getattr(blk, "cv2", None), nn.Module):
            pref.append(f"model.{i}.cv2.")           # izlazna projekcija bloka (fiksiran broj kanala)
        else:
            pref.append(f"model.{i}.")               # fallback: cijeli blok
    pref.append(f"model.{len(seq) - 1}.")            # Detect glava (terminalna)
    return pref


# Yolo KD = feature(neck MSE) + logit(head cls focal) + box(GIoU, UNIVERZALNI handler) — kao exp2 "featlogit", bez GT.
YOLO_KD_TAPS = [
    {"group": "feat", "type": "feature",  "src": "feat", "td": "feat", "w": "w_feat"},
    {"group": "cls",  "type": "dense_cls", "src": "cls", "td": "cls",  "w": "w_cls"},
    {"group": "box",  "type": "box_giou", "src": "box", "td": "box",  "w": "box_w", "kw_td": {"w": "conf"}},
]

YOLO_PROFILE = {
    "kind": "yolo", "task": "detection", "imgsz": 640, "SUPPORTS": A.YoloAdapter.SUPPORTS,
    "matches": A.YoloAdapter.matches, "forward": A.YoloAdapter.forward, "gt_loss": A.YoloAdapter.gt_loss,
    "tp_example": A.YoloAdapter.tp_example, "predict": A.YoloAdapter.predict,
    "teacher_outputs": A.YoloAdapter.teacher_outputs,
    "student_signals": _yolo_student_signals, "kd_taps": YOLO_KD_TAPS,
    "protect": _yolo_protect,                            # neck-izlaz (feature-tap) + glava = off-limits (B, ne FGD)
}

PROFILES = [FRCNN_PROFILE, YOLO_PROFILE]


class ProfileAdapter:
    """Tanki wrapper nad profilom -> isto sucelje kao stari adapteri."""
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec["kind"]; self.task = spec["task"]; self.imgsz = spec["imgsz"]
        self.SUPPORTS = spec.get("SUPPORTS", [])

    def matches(self, model):
        return self.spec["matches"](model)

    def protect_prefixes(self, model):
        """Imena-prefiksi slojeva koji su OFF-LIMITS strukturno (ne prune/grow): feature-tap/RPN/sucelja."""
        p = self.spec.get("protect", [])
        return list(p(model)) if callable(p) else list(p)

    def forward(self, model, imgs):
        return self.spec["forward"](model, imgs)

    def gt_loss(self, model, imgs, targets):
        return self.spec["gt_loss"](model, imgs, targets)

    def tp_example(self, device):
        return self.spec["tp_example"](device)

    def predict(self, model, imgs):
        return self.spec["predict"](model, imgs)

    def teacher_outputs(self, teacher, imgs):
        return self.spec["teacher_outputs"](teacher, imgs)

    def kd_loss(self, student, td, imgs, **weights):
        """Genericki: ekstrahiraj signale -> sastavi terme iz DEKLARATIVNE tablice -> kd.kd_total."""
        if "kd_taps" not in self.spec or "student_signals" not in self.spec:
            raise NotImplementedError(f"KD jos nije definiran za profil '{self.kind}'")
        W = {"w_feat": 1.0, "w_rpn": 1.0, "w_cls": 1.0, "T": 1.0, "box_w": 1.0, "one": 1.0}
        W.update(weights)
        S = self.spec["student_signals"](student, td, imgs)
        terms = []
        for t in self.spec["kd_taps"]:
            term = {"group": t["group"], "type": t["type"], "student": S[t["src"]],
                    "teacher": td[t["td"]], "w": W[t["w"]]}
            if "kwk" in t:                                   # kw iz tezina (npr. T)
                term["kw"] = {k: W[v] for k, v in t["kwk"].items()}
            if "kw_td" in t:                                 # kw iz teacher-tapova (npr. confidence za dense_box)
                term.setdefault("kw", {}).update({k: td[v] for k, v in t["kw_td"].items()})
            terms.append(term)
        return kd.kd_total(terms)


def pick_profile(model):
    for p in PROFILES:
        if p["matches"](model):
            return ProfileAdapter(p)
    return None
