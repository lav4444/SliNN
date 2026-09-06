import sys

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)


def _to_dev(o, device):
    import torch
    if isinstance(o, torch.Tensor):
        return o.to(device).float() if o.is_floating_point() else o.to(device)
    if isinstance(o, dict):
        return {k: _to_dev(v, device) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_dev(v, device) for v in o)
    return o


def detection_kd(student, teacher, imgs):
    import profiles as PF
    ad = PF.pick_profile(teacher)
    if ad is None:
        return None
    device = next(student.parameters()).device
    td = _to_dev(ad.teacher_outputs(teacher, imgs), device)
    return ad.kd_loss(student, td, imgs)


def has_enhancers(ctx):
    return bool(ctx.get("enhancers"))


ENHANCERS = {"detection": detection_kd}


def enhancer_loss_fn(ctx, teacher):
    fn = ENHANCERS.get(ctx.get("task")) if has_enhancers(ctx) else None
    if fn is None:
        return None
    return lambda student, imgs: fn(student, teacher, imgs)
