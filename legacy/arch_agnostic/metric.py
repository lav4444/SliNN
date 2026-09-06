import glob
import os
import sys

_MORPH = "/home/tomi/code/dipl/morphology"
_AA = "/home/tomi/code/dipl/arch_agnostic"
for d in (_MORPH, _AA):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from loss import _main_out                                    # noqa: E402  (izvlaci primarni tenzor izlaza)


def _voc_root(path):
    for dp, _, _ in os.walk(path):
        if dp.replace("\\", "/").endswith("VOC2012"):
            return dp
    return path


def pairs_segmentation(path, split="val", n=64, size=256):
    from PIL import Image
    root = _voc_root(path)
    ids = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
    if not os.path.exists(ids):
        ids = os.path.join(root, "ImageSets", "Segmentation", "val.txt")
    stems = [s.strip() for s in open(ids) if s.strip()][:n]
    jpg = os.path.join(root, "JPEGImages"); seg = os.path.join(root, "SegmentationClass")
    out = []
    for st in stems:
        ip, mp = os.path.join(jpg, st + ".jpg"), os.path.join(seg, st + ".png")
        if not (os.path.exists(ip) and os.path.exists(mp)):
            continue
        im = Image.open(ip).convert("RGB").resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0).permute(2, 0, 1)
        mk = Image.open(mp).resize((size, size), Image.NEAREST)
        y = torch.from_numpy(np.asarray(mk, dtype="int64"))
        out.append((x, y))
    return out


def pairs_regression(path, adapter, split="val", n=None):
    for f in glob.glob(os.path.join(path, "**", "*.npz"), recursive=True):
        try:
            z = np.load(f, allow_pickle=True)
        except BaseException:
            continue
        xk, yk = f"X_{split}", f"y_{split}"
        if xk not in z.files or yk not in z.files:
            xk, yk = ("X", "y") if "X" in z.files and "y" in z.files else (None, None)
        if xk is None:
            continue
        X = np.asarray(z[xk], dtype="float32"); y = np.asarray(z[yk], dtype="float32")
        if X.ndim == 2 and X.shape[1] == adapter._in_ch:
            if n:
                X, y = X[:n], y[:n]
            return [(torch.from_numpy(X[i]), float(y[i])) for i in range(len(X))]
    return []


@torch.no_grad()
def eval_regression(model, adapter, pairs, device, bs=256):
    model.eval()
    preds, gts = [], []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        xb = [x.to(device) for x, _ in chunk]
        out = _main_out(adapter.forward(model, xb)).float().flatten().cpu()
        preds.append(out[:len(chunk)]); gts.append(torch.tensor([g for _, g in chunk]))
    p, g = torch.cat(preds), torch.cat(gts)
    rmse = float(((p - g) ** 2).mean().sqrt())
    ss = float(((g - g.mean()) ** 2).sum())
    r2 = 1.0 - float(((p - g) ** 2).sum()) / ss if ss > 0 else 0.0
    return {"r2": r2, "rmse": rmse}


@torch.no_grad()
def eval_segmentation(model, adapter, pairs, device, bs=8, ignore=255):
    model.eval()
    inter = union = None
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        xb = [x.to(device) for x, _ in chunk]
        out = _main_out(adapter.forward(model, xb)).float()
        K = out.shape[1]
        if inter is None:
            inter = torch.zeros(K); union = torch.zeros(K)
        for b, (_, gt) in enumerate(chunk):
            pr = out[b].argmax(0).cpu()
            gt = gt.cpu()
            if pr.shape != gt.shape:
                pr = F.interpolate(pr[None, None].float(), size=gt.shape, mode="nearest")[0, 0].long()
            valid = gt != ignore
            for c in range(K):
                pi = (pr == c) & valid; gi = (gt == c) & valid
                inter[c] += float((pi & gi).sum()); union[c] += float((pi | gi).sum())
    iou = inter / union.clamp(min=1)
    present = union > 0
    return {"mIoU": float(iou[present].mean()) if present.any() else 0.0}


@torch.no_grad()
def eval_classification(model, adapter, pairs, device, bs=64):
    model.eval()
    preds, gts = [], []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        xb = [x.to(device) for x, _ in chunk]
        out = _main_out(adapter.forward(model, xb)).float()
        preds.append(out.argmax(-1).cpu()); gts.append(torch.tensor([g for _, g in chunk]))
    p, g = torch.cat(preds), torch.cat(gts)
    acc = float((p == g).float().mean())
    f1s = []
    for c in torch.unique(g):
        tp = float(((p == c) & (g == c)).sum()); fp = float(((p == c) & (g != c)).sum())
        fn = float(((p != c) & (g == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"f1_macro": sum(f1s) / len(f1s) if f1s else 0.0, "accuracy": acc}


@torch.no_grad()
def teacher_agreement(student, teacher, adapter, inputs, device, kind="kl", bs=16):
    student.eval(); teacher.eval()
    num = den = 0.0
    sse = 0.0; tvals = []
    for i in range(0, len(inputs), bs):
        xb = [x.to(device) for x in inputs[i:i + bs]]
        s = _main_out(adapter.forward(student, xb)).float()
        t = _main_out(adapter.forward(teacher, xb)).float()
        if kind == "kl":
            ax = 1 if s.dim() >= 3 else -1
            num += float((s.argmax(ax) == t.argmax(ax)).sum()); den += s.argmax(ax).numel()
        else:
            sse += float(((s - t) ** 2).sum()); tvals.append(t.flatten().cpu())
    if kind == "kl":
        return {"agreement": num / max(den, 1)}
    tt = torch.cat(tvals); sst = float(((tt - tt.mean()) ** 2).sum())
    return {"agreement": (1.0 - sse / sst) if sst > 0 else 0.0}


_EVAL = {"regression": eval_regression, "segmentation": eval_segmentation, "classification": eval_classification}


def evaluate(model, adapter, task, pairs, device):
    fn = _EVAL.get(task)
    if fn is None:
        raise NotImplementedError(f"Nema generičkog evaluatora za task '{task}' (detekcija ide preko morphology scorera).")
    return fn(model, adapter, pairs, device)
