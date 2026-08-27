"""
slinn/kdterms.py — GENERICKI KD engine po TIPU tapa (model-agnosticno). Preseljeno iz morphology/kd.py (6.4).
model-specificna je SAMO ekstrakcija signala (u profilu/adapteru), a matematika je ovdje.

Tipovi tapova:
  feature     — feature mape (dict ili lista [B,C,H,W]) -> MSE (po razini, prosjek). 1x1 proj ako kanali ne pasu.
  objectness  — per-level objectness logiti -> BCE(student, sigmoid(teacher))
  box         — box delte/koord (tensor ili lista) -> SmoothL1
  logit       — klase (softmax) -> KL@T * T^2
  dense_cls   — per-anchor klase (sigmoid, yolo) -> sigmoid focal sa soft metama

kd_total(terms): terms = [{group, type, student, teacher, w=1.0, kw={}}]. Vrati (total, info):
  total = Σ w * loss(type)(student, teacher, **kw);  info[group] = Σ raw loss (neponderirano).
"""

import torch
import torch.nn.functional as F


def _feature_loss(s, t):
    """MSE preko feature mapa (dict ili lista). Feature-tap je OFF-LIMITS (zasticen u profile['protect']) pa kanali
    UVIJEK pasu -> egzaktni MSE. Ako se ne slazu -> GLASNA greska (taj sloj nije zasticen; nema adaptera/FGD-a)."""
    keys = list(s.keys()) if hasattr(s, "keys") else range(len(s))
    tot = None; n = 0
    for k in keys:
        sk = s[k]; tk = t[k].to(sk.dtype)
        if sk.shape[1] != tk.shape[1]:
            raise RuntimeError(f"feature-KD: kanali se ne slazu na '{k}' (student {sk.shape[1]} vs teacher {tk.shape[1]}). "
                               f"Feature-tap mora ostati off-limits — dodaj taj sloj u profile['protect'].")
        if sk.shape[0] != tk.shape[0]:                     # batch mismatch -> NE broadcastaj tiho (trenirao bi na smecu)
            raise RuntimeError(f"feature-KD: batch se ne slaze na '{k}' (student {sk.shape[0]} vs teacher {tk.shape[0]}). "
                               f"Teacher cache je gradjen na DRUGOM batchu -> obrisi tmp/<model>/train* i rerun (meta sad ukljucuje batch_size).")
        l = F.mse_loss(sk, tk.detach())
        tot = l if tot is None else tot + l; n += 1
    return tot / max(n, 1) if tot is not None else torch.zeros((), device=s[keys[0]].device)


def _objectness_loss(s, t):
    return sum(F.binary_cross_entropy_with_logits(so, torch.sigmoid(to.to(so.dtype)))
               for so, to in zip(s, t)) / max(len(s), 1)


def _box_loss(s, t):
    if isinstance(s, (list, tuple)):
        return sum(F.smooth_l1_loss(sb, tb.to(sb.dtype)) for sb, tb in zip(s, t)) / max(len(s), 1)
    return F.smooth_l1_loss(s, t.to(s.dtype))


def _logit_loss(s, t, T=1.0):
    return F.kl_div(F.log_softmax(s / T, -1), F.softmax(t.to(s.dtype) / T, -1), reduction="batchmean") * (T * T)


def _dense_cls_loss(s, t, alpha=0.25, gamma=2.0):
    """Sigmoid focal sa SOFT metama; s,t su VJEROJATNOSTI [0,1] (yolo eval daje sigmoidirane klase).
    Normaliziran po efektivnim pozitivima (teacher prob > 0.5)."""
    s = s.clamp(1e-6, 1 - 1e-6); t = t.to(s.dtype)
    ce = -(t * torch.log(s) + (1 - t) * torch.log(1 - s))
    p_t = s * t + (1 - s) * (1 - t)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * t + (1 - alpha) * (1 - t)) * loss
    num_pos = (t.amax(dim=-1) > 0.5).float().sum().clamp(min=1.0) if t.dim() >= 2 else float(t.numel())
    return loss.sum() / num_pos


def _giou(a, b):
    """Pairwise GIoU (isti indeks) na xyxy okvirima [..,4]. Vrati [..] u [-1,1]. Scale-invariant."""
    ax1, ay1, ax2, ay2 = a.unbind(-1); bx1, by1, bx2, by2 = b.unbind(-1)
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
    """UNIVERZALNI box-KD: GIoU (1 - GIoU) na DEKODIRANIM xyxy okvirima, ponderiran per-box tezinom w (teacher conf).
    Isti handler za yolo (dense anchori) i frcnn (ROI) — svaki profil dekodira SVOJE okvire u xyxy + da w.
    s,t: [...,4] xyxy; w: [...]. Scale-invariant; pravilno reagira na greske lokalizacije."""
    g = _giou(s, t.to(s.dtype))                          # [...]
    w = w.to(s.dtype)
    return ((1 - g) * w).sum() / w.sum().clamp(min=1e-6)


_LOSS = {"feature": _feature_loss, "objectness": _objectness_loss, "box": _box_loss,
         "logit": _logit_loss, "dense_cls": _dense_cls_loss, "box_giou": _box_giou_loss}

# 1-line opis po tipu tapa (izvor istine za About — auto se azurira kad dodamo handler u _LOSS).
_LOSS_DOC = {
    "feature":    "feature mape → MSE (kanali poravnati jer je tap off-limits)",
    "objectness": "RPN objectness → BCE(student, sigmoid(teacher))",
    "box":        "box delte/koordinate → SmoothL1 (npr. RPN, 2-stage)",
    "logit":      "klase (softmax) → KL@T · T² (npr. frcnn ROI glava)",
    "dense_cls":  "per-anchor klase (sigmoid) → focal sa soft metama (npr. yolo gusta glava)",
    "box_giou":   "dekodirani xyxy okviri → 1−GIoU, conf-težinski (UNIVERZALNI box-KD)",
}


def kd_total(terms):
    """terms: [{group, type, student, teacher, w=1.0, kw={}}]. Vrati (total, info)."""
    total = None; info = {}
    for tm in terms:
        l = _LOSS[tm["type"]](tm["student"], tm["teacher"], **tm.get("kw", {}))
        w = tm.get("w", 1.0)
        total = l * w if total is None else total + l * w
        g = tm["group"]; info[g] = info.get(g, 0.0) + float(l)
    if total is None:
        total = torch.zeros(())
    return total, info
