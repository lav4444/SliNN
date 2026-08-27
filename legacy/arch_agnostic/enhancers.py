"""
enhancers.py — TASK-UVJETNI KD enhaneri (IZOLIRAN plug, biran AUTO-detektiranim taskom, NE per-model u jezgri).

Dogovor: loss = KD-core (generic feature+output, loss.py) + ENHANERI ovisni o tasku. Enhaneri su decode-svjesni
(traze poznavanje strukture glave) pa ne mogu u genericku jezgru — zato su ovdje, kao plug koji pipeline kaci
UVJETNO kad `task` ima enhanere (SUPPORTED_TASKS). Jezgra nikad ne zna "ovo je yolo".

Detekcija (dense_cls + box_giou + feature) = REUSE dokazanog morphology decode-a (profiles.ProfileAdapter ->
YoloAdapter/FrcnnAdapter._dense_decode + kd._LOSS). Decode se bira po OBITELJI (yolo grid vs frcnn ROI) unutar
morphology `pick_profile` (auto po arhitekturi), sto je jedini priznati per-obitelj plug (PLAN Korak 4).
"""
import sys

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)


def _to_dev(o, device):
    """Rekurzivno premjesti tenzore (dict/list/tuple) na uredjaj, float (teacher_outputs zna vratiti cpu/half)."""
    import torch
    if isinstance(o, torch.Tensor):
        return o.to(device).float() if o.is_floating_point() else o.to(device)
    if isinstance(o, dict):
        return {k: _to_dev(v, device) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_dev(v, device) for v in o)
    return o


def detection_kd(student, teacher, imgs):
    """Detekcijski enhancer-KD (feature + dense_cls + box_giou) preko morphology ProfileAdapter (reuse decode).
    teacher/student = yolo/frcnn eager moduli; imgs = lista [C,H,W]. Vrati (loss, info) ili None ako nema profila."""
    import profiles as PF
    ad = PF.pick_profile(teacher)                            # decode profil po OBITELJI (auto po arhitekturi)
    if ad is None:
        return None
    device = next(student.parameters()).device
    td = _to_dev(ad.teacher_outputs(teacher, imgs), device)  # teacher signali -> na uredjaj (bili cpu/half)
    return ad.kd_loss(student, td, imgs)                     # feature + dense_cls + box_giou (kd.kd_total)


def has_enhancers(ctx):
    """Ima li ovaj task decode-svjesne enhanere (iz SUPPORTED_TASKS preko pipeline ctx)?"""
    return bool(ctx.get("enhancers"))


# registar plugova po tasku (prosiruje se; segmentation/regression nemaju enhanere -> None = samo core)
ENHANCERS = {"detection": detection_kd}


def enhancer_loss_fn(ctx, teacher):
    """Vrati loss_fn(student, imgs)->(loss,info) za enhanere ovog taska, ili None (samo KD-core).
    Biran ISKLJUCIVO po ctx['task'] (auto-detektiran), nikad po imenu modela."""
    fn = ENHANCERS.get(ctx.get("task")) if has_enhancers(ctx) else None
    if fn is None:
        return None
    return lambda student, imgs: fn(student, teacher, imgs)
