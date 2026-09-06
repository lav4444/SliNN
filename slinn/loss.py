import sys


import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import kdterms as kd                                                  # noqa: E402
from classify import _unwrap, weighted_leaves                # noqa: E402


def _main_out(o):
    o = _unwrap(o)
    if isinstance(o, dict):
        o = o.get("out", next(iter(o.values())))
    while isinstance(o, (list, tuple)) and o:
        o = o[0]
    return o


def _rel(s, t, eps=1e-6):
    t = t.detach()
    scale = (t.pow(2).mean() + eps).sqrt()
    return s / scale, t / scale


def _hook_acts(model, names):
    acts, handles, ns = {}, [], set(names)
    for n, m in model.named_modules():
        if n in ns:
            def mk(nm):
                def h(mod, i, o):
                    acts[nm] = o
                return h
            handles.append(m.register_forward_hook(mk(n)))
    return acts, handles


def find_out_tap(teacher, adapter, imgs, rtol=1e-4):
    teacher.eval()
    with torch.no_grad():
        out = _main_out(adapter.forward(teacher, imgs))
    if not torch.is_tensor(out) or out.dim() != 4:
        return None
    B, K, H, W = out.shape

    acts, handles = {}, []
    for n, m in teacher.named_modules():
        if not n:
            continue

        def mk(nm):
            def h(mod, i, o):
                o = _unwrap(o)
                if torch.is_tensor(o) and o.dim() == 4 and o.shape[:2] == (B, K) \
                        and (o.shape[2] < H or o.shape[3] < W):
                    acts[nm] = o.detach()
            return h
        handles.append(m.register_forward_hook(mk(n)))
    try:
        with torch.no_grad():
            _main_out(adapter.forward(teacher, imgs))
    finally:
        for h in handles:
            h.remove()

    ref, best = out.float(), None
    raspon = (ref.max() - ref.min()).abs().item()
    prag = max(rtol * raspon, torch.finfo(torch.float32).eps)
    for n, a in acts.items():
        up = nn.functional.interpolate(a.float(), size=(H, W), mode="bilinear", align_corners=False)
        if (up - ref).abs().max().item() <= prag:
            px = a.shape[2] * a.shape[3]
            if best is None or px < best[1]:
                best = (n, px)
    return best[0] if best else None


def teacher_signals(teacher, adapter, imgs, taps, out_tap=None):
    hook_names = list(taps) + ([out_tap] if out_tap and out_tap not in taps else [])
    t_acts, th = _hook_acts(teacher, hook_names)
    try:
        teacher.eval()
        with torch.no_grad():
            t_out = _main_out(adapter.forward(teacher, imgs))
    finally:
        for h in th:
            h.remove()
    feat = {n: t_acts[n].detach() for n in taps if n in t_acts}
    if out_tap and out_tap in t_acts:
        return {"feat": feat, "out": _unwrap(t_acts[out_tap]).detach(),
                "out_size": tuple(t_out.shape[-2:])}
    return {"feat": feat, "out": t_out.detach()}


def kd_terms(student, teacher, adapter, imgs, taps, kd_mode, out_kind, T=4.0, w_feat=1.0, w_out=1.0,
             teacher_sig=None):
    s_acts, sh = _hook_acts(student, taps)
    try:
        if teacher_sig is None:
            teacher_sig = teacher_signals(teacher, adapter, imgs, taps)
        t_acts, t_out = teacher_sig["feat"], teacher_sig["out"]
        s_out = _main_out(adapter.forward(student, imgs))
    finally:
        for h in sh:
            h.remove()

    terms = []
    if kd_mode == "feature+logit" and taps:
        common = [n for n in taps if n in s_acts and n in t_acts]
        if common:
            sd, td = {}, {}
            for n in common:
                sd[n], td[n] = _rel(s_acts[n], t_acts[n])
            terms.append({"group": "feat", "type": "feature", "student": sd, "teacher": td, "w": w_feat})
    if out_kind == "kl":
        s_l, t_l = s_out, t_out.detach()
        if s_l.dim() == 4:
            s_l = s_l.permute(0, 2, 3, 1).reshape(-1, s_l.shape[1])
            t_l = t_l.permute(0, 2, 3, 1).reshape(-1, t_l.shape[1])
        terms.append({"group": "out", "type": "logit", "student": s_l,
                      "teacher": t_l, "w": w_out, "kw": {"T": T}})
    else:
        s_n, t_n = _rel(s_out, t_out)
        terms.append({"group": "out", "type": "feature", "student": [s_n], "teacher": [t_n], "w": w_out})
    return terms


def kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=None, **w):
    return kd.kd_total(kd_terms(student, teacher, adapter, imgs, taps, kd_mode, out_kind,
                                teacher_sig=teacher_sig, **w))


def _bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def kd_importance(student, teacher, adapter, batches, taps, kd_mode, out_kind, prunable=None, loss_fn=None, **w):
    import settings as _CFG
    mode = getattr(_CFG, "PRUNE_IMPORTANCE", "grad")
    norm = getattr(_CFG, "PRUNE_IMP_NORM", "none")
    leaves = weighted_leaves(student)
    if prunable is not None:
        leaves = [(n, m, t, wt) for (n, m, t, wt) in leaves if n in prunable]
    acc = {n: torch.zeros(wt.shape[0]) for n, _, _, wt in leaves}
    gacc = {n: torch.zeros(wt.shape[0], wt.reshape(wt.shape[0], -1).shape[1]) for n, _, _, wt in leaves}
    for p in student.parameters():
        p.requires_grad_(True)
    _bn_eval(student)
    bn_snap = [(m, m.running_mean.clone(), m.running_var.clone(),
                None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
               for m in student.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None]
    nb = 0
    try:
        for imgs in batches:
            for p in student.parameters():
                p.grad = None
            loss = (loss_fn(student, imgs)[0] if loss_fn is not None
                    else kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, **w)[0])
            loss.backward()
            for n, m, _, _ in leaves:
                g = getattr(m.weight, "grad", None)
                if g is not None:
                    gd = g.detach().cpu()
                    pv = (gd * m.weight.detach().cpu()) if mode == "taylor" else gd
                    acc[n] += pv.abs().flatten(1).mean(1)
                    gacc[n] += gd.reshape(gd.shape[0], -1)
            nb += 1
    finally:
        for m, rm, rv, nbt in bn_snap:
            m.running_mean.copy_(rm); m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)
        for p in student.parameters():
            p.grad = None
    nb = max(nb, 1)
    imp = {n: acc[n] / nb for n in acc}
    if norm in ("mean", "max"):
        red = (lambda v: v.mean()) if norm == "mean" else (lambda v: v.max())
        imp = {n: v / (float(red(v.float())) + 1e-12) for n, v in imp.items()}
    return imp, {n: gacc[n] / nb for n in gacc}
