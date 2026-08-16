"""
loss.py — GENERICKI hook-KD (Faza 4.1), model-agnosticno.

Jezgra (bez decode-a, bez student_signals):
  feature-KD  — forward-hook na TAP slojevima (position ih daje); MSE student<->teacher (kanali poravnati
                jer position stiti tapove). Aktivan samo kad je kd_mode == "feature+logit".
  output-KD   — na FINALNOM izlazu modela (adapter.forward, _unwrap); tip gubitka bira TASK:
                "kl"  = KL@T na distribuciji klasa [B,K] (classification/multilabel),
                "mse" = MSE na sirovom/kontinuiranom izlazu (regression/segmentation/detection-glava).

Matematika ide kroz morphology `kd.kd_total` (reuse, izolacija). Teacher = smrznuta kopija ORIGINALA.
"""
import sys

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import kd                                                    # noqa: E402
from classify import _unwrap, weighted_leaves                # noqa: E402


def _main_out(o):
    """Reduciraj bilo kakav izlaz modela na PRIMARNI tenzor: HF ModelOutput -> .logits (via _unwrap);
    dict (torchvision seg {'out','aux'}) -> 'out'; ugnijezdjena lista/tuple -> prvi tenzor."""
    o = _unwrap(o)
    if isinstance(o, dict):
        o = o.get("out", next(iter(o.values())))
    while isinstance(o, (list, tuple)) and o:
        o = o[0]
    return o


def _rel(s, t, eps=1e-6):
    """Skala-invarijantno: podijeli studenta i teachera s teacher-RMS. Plain MSE potom = RELATIVNI MSE
    = MSE(s,t)/mean(teacher²) ~O(1), usporediv kroz modele/tapove. Nazivnik je od SMRZNUTOG teachera
    (konstanta) -> progress-signal ostaje (brojnik->0 kad student sustigne teachera)."""
    t = t.detach()
    scale = (t.pow(2).mean() + eps).sqrt()
    return s / scale, t / scale


def _hook_acts(model, names):
    """Registriraj forward-hookove na dane leaf-slojeve; vrati (acts_dict, handles)."""
    acts, handles, ns = {}, [], set(names)
    for n, m in model.named_modules():
        if n in ns:
            def mk(nm):
                def h(mod, i, o):
                    acts[nm] = o
                return h
            handles.append(m.register_forward_hook(mk(n)))
    return acts, handles


def teacher_signals(teacher, adapter, imgs, taps):
    """SMRZNUTI teacher -> KD-referenca za ovaj batch: {"feat": {tap: aktivacija}, "out": finalni izlaz}.
    Sve detached. Izdvojeno da ga (a) kd_terms koristi inline (teacher_sig=None) i (b) engine PREDRACUNA i
    CACHIRA (Faza 5.1) — jednom po batchu, reuse kroz epohe, bez ponovnog vrtenja teachera svaki FT korak."""
    t_acts, th = _hook_acts(teacher, taps)
    try:
        teacher.eval()
        with torch.no_grad():
            t_out = _main_out(adapter.forward(teacher, imgs))
    finally:
        for h in th:
            h.remove()
    feat = {n: t_acts[n].detach() for n in taps if n in t_acts}
    return {"feat": feat, "out": t_out.detach()}


def kd_terms(student, teacher, adapter, imgs, taps, kd_mode, out_kind, T=4.0, w_feat=1.0, w_out=1.0,
             teacher_sig=None):
    """Sastavi KD `terms` (za kd.kd_total): feature (tapovi) + output (finalni izlaz). Teacher je detached.
    `teacher_sig` (iz `teacher_signals`, predracunat/cachiran) preskace teacher-forward; inace se racuna inline."""
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
            for n in common:                                 # skala-invarijantno po tapu -> nijedan ne dominira
                sd[n], td[n] = _rel(s_acts[n], t_acts[n])
            terms.append({"group": "feat", "type": "feature", "student": sd, "teacher": td, "w": w_feat})
    if out_kind == "kl":
        s_l, t_l = s_out, t_out.detach()
        if s_l.dim() == 4:                                   # segmentacija [B,K,H,W] -> per-piksel [B·H·W, K]:
            s_l = s_l.permute(0, 2, 3, 1).reshape(-1, s_l.shape[1])   # KL po klasnoj distribuciji SVAKOG piksela
            t_l = t_l.permute(0, 2, 3, 1).reshape(-1, t_l.shape[1])   # (globalni MSE bi dominirala pozadina -> foreground IoU kolabira)
        terms.append({"group": "out", "type": "logit", "student": s_l,          # KL vec skala-invarijantan (softmax)
                      "teacher": t_l, "w": w_out, "kw": {"T": T}})
    else:                                                    # MSE na izlazu -> normaliziran teacher-skalom (rel. MSE)
        s_n, t_n = _rel(s_out, t_out)
        terms.append({"group": "out", "type": "feature", "student": [s_n], "teacher": [t_n], "w": w_out})
    return terms


def kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=None, **w):
    """Vrati (total, info) generickog KD-gubitka; info[group] = neponderirani doprinos po grupi.
    `teacher_sig` (predracunat) preskace teacher-forward — koristi ga engine s cachiranim signalima."""
    return kd.kd_total(kd_terms(student, teacher, adapter, imgs, taps, kd_mode, out_kind,
                                teacher_sig=teacher_sig, **w))


def _bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def kd_importance(student, teacher, adapter, batches, taps, kd_mode, out_kind, prunable=None, loss_fn=None, **w):
    """KD-grad VAZNOST iz backwarda KD-gubitka. `loss_fn(student, imgs)->(loss,info)` (npr. detekcijski enhancer-KD)
    ZAMJENJUJE genericki kd_loss za rangiranje -> prune ne rezuje task-kriticne kanale (enhaneri i u VAZNOSTI, ne
    samo u FT-u). Iz JEDNOG prolaza po batchu vrati DVA signala (bez dodatnih backwarda):
      imp[name]  = mean |d(KD)/dw| po IZLAZNOJ jedinici (PRUNE rang; mean-norm -> usporedivo medju slojevima),
      gavg[name] = prosjecna SIGNED grad matrica [O, in*k] (GROW / GradMax SVD -> grow_potential).
    BN u eval (bez korupcije running-statsa, bez dropout-suma). `prunable` (skup imena) filtrira slojeve.
    Napomena: pri student==teacher KD=0 -> grad=0; u praksi je student vec razidjen (Faza 1/2), pa je signal
    smislen (u smoke-u perturbiramo studenta da to simuliramo)."""
    leaves = weighted_leaves(student)
    if prunable is not None:
        leaves = [(n, m, t, wt) for (n, m, t, wt) in leaves if n in prunable]
    acc = {n: torch.zeros(wt.shape[0]) for n, _, _, wt in leaves}
    gacc = {n: torch.zeros(wt.shape[0], wt.reshape(wt.shape[0], -1).shape[1]) for n, _, _, wt in leaves}
    for p in student.parameters():
        p.requires_grad_(True)
    _bn_eval(student)                                        # fiksne running-stats + bez dropouta -> determinist. grad
    # BN SNAPSHOT: neki loss_fn (detekcija _dense_decode(train_bn=True)) toggle-a BN u train -> azurira running-stats
    # tijekom importance-prolaza. Snimi buffere pa vrati na kraju -> mjerenje je SIDE-EFFECT-FREE (model koji prunamo
    # ostaje netaknut; obrazac iz morphology grad_pass). [[bn-eval-detection-trainmode]]
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
                    acc[n] += gd.abs().flatten(1).mean(1)     # prune: mean|grad| po izlaznoj jedinici
                    gacc[n] += gd.reshape(gd.shape[0], -1)    # grow: signed grad matrica [O, in*k]
            nb += 1
    finally:
        for m, rm, rv, nbt in bn_snap:                        # restore BN (poništi eventualni train_bn drift)
            m.running_mean.copy_(rm); m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)
        for p in student.parameters():
            p.grad = None
    nb = max(nb, 1)
    return {n: acc[n] / nb for n in acc}, {n: gacc[n] / nb for n in gacc}
