
import torch

from . import adapters as A
import kdterms as kd


def _frcnn_student_signals(student, td, imgs):
    s_imgs, _ = student.transform(imgs)
    s_feat = student.backbone(s_imgs.tensors)
    s_obj, s_box = student.rpn.head(list(s_feat.values()))
    s_bf = student.roi_heads.box_roi_pool(s_feat, td["props"], s_imgs.image_sizes)
    s_cls, s_box2 = student.roi_heads.box_predictor(student.roi_heads.box_head(s_bf))
    dec = student.roi_heads.box_coder.decode(s_box2, td["props"])
    cidx = td["roi_cidx"].to(dec.device).long()
    k = torch.arange(dec.shape[0], device=dec.device)
    roi_box = dec[k, cidx]
    return {"feat": s_feat, "obj": s_obj, "box": s_box, "cls": s_cls, "roi_box": roi_box}


FRCNN_KD_TAPS = [
    {"group": "feat",  "type": "feature",    "src": "feat",    "td": "feat",    "w": "w_feat"},
    {"group": "rpn",   "type": "objectness", "src": "obj",     "td": "rpn_obj", "w": "w_rpn"},
    {"group": "rpn",   "type": "box",        "src": "box",     "td": "rpn_box", "w": "w_rpn"},
    {"group": "logit", "type": "logit",      "src": "cls",     "td": "cls",     "w": "one", "kwk": {"T": "T"}},
    {"group": "logit", "type": "box_giou",   "src": "roi_box", "td": "roi_box", "w": "box_w", "kw_td": {"w": "roi_conf"}},
]

FRCNN_PROFILE = {
    "kind": "frcnn", "task": "detection", "imgsz": 320, "SUPPORTS": A.FrcnnAdapter.SUPPORTS,
    "matches": A.FrcnnAdapter.matches,
    "forward": A.FrcnnAdapter.forward, "gt_loss": A.FrcnnAdapter.gt_loss,
    "tp_example": A.FrcnnAdapter.tp_example, "predict": A.FrcnnAdapter.predict,
    "teacher_outputs": A.FrcnnAdapter.teacher_outputs,
    "student_signals": _frcnn_student_signals, "kd_taps": FRCNN_KD_TAPS,
    "protect": ["backbone.fpn", "rpn", "roi_heads"],
}

def _yolo_student_signals(student, td, imgs):
    box, cls, feat = A.YoloAdapter._dense_decode(student, imgs, train_bn=False)
    return {"box": box, "cls": cls, "feat": feat}


def _yolo_protect(model):
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
            pref.append(f"model.{i}.cv2.")
        else:
            pref.append(f"model.{i}.")
    pref.append(f"model.{len(seq) - 1}.")
    return pref


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
    "protect": _yolo_protect,
}

PROFILES = [FRCNN_PROFILE, YOLO_PROFILE]


class ProfileAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec["kind"]; self.task = spec["task"]; self.imgsz = spec["imgsz"]
        self.SUPPORTS = spec.get("SUPPORTS", [])

    def matches(self, model):
        return self.spec["matches"](model)

    def protect_prefixes(self, model):
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
        if "kd_taps" not in self.spec or "student_signals" not in self.spec:
            raise NotImplementedError(f"KD jos nije definiran za profil '{self.kind}'")
        W = {"w_feat": 1.0, "w_rpn": 1.0, "w_cls": 1.0, "T": 1.0, "box_w": 1.0, "one": 1.0}
        W.update(weights)
        S = self.spec["student_signals"](student, td, imgs)
        terms = []
        for t in self.spec["kd_taps"]:
            term = {"group": t["group"], "type": t["type"], "student": S[t["src"]],
                    "teacher": td[t["td"]], "w": W[t["w"]]}
            if "kwk" in t:
                term["kw"] = {k: W[v] for k, v in t["kwk"].items()}
            if "kw_td" in t:
                term.setdefault("kw", {}).update({k: td[v] for k, v in t["kw_td"].items()})
            terms.append(term)
        return kd.kd_total(terms)


def pick_profile(model):
    for p in PROFILES:
        if p["matches"](model):
            return ProfileAdapter(p)
    return None
