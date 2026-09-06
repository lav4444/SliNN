import glob
import json
import os
import sys

_AA = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
for d in (_AA,):
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


def _cap(n):
    import settings as CFG
    d = CFG.DEV_DATA_SUBSET
    if not d:
        return n
    return int(d) if n is None else min(int(n), int(d))


def pairs_segmentation(path, split="val", n=64, size=256):
    from PIL import Image
    root = _voc_root(path)
    ids = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
    if not os.path.exists(ids):
        ids = os.path.join(root, "ImageSets", "Segmentation", "val.txt")
    stems = [s.strip() for s in open(ids) if s.strip()][:_cap(n)]
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
            n = _cap(n)
            if n:
                X, y = X[:n], y[:n]
            return [(torch.from_numpy(X[i]), float(y[i])) for i in range(len(X))]
    return []


def pairs_depth(path, adapter, split="val", n=None):
    import dataset as _DS
    tabs = [f for f in _DS._tab_files(path) if f.endswith((".parquet", ".npz"))]
    tabs = [f for f in tabs if split in os.path.basename(f) or split in f.split(os.sep)] or tabs
    if not tabs:
        return []
    peek = _DS._peek_table(tabs[0]) or {}
    lc = (peek.get("label_cols") or [None])[0]
    ic = (peek.get("input_cols") or [None])[0]
    if not lc or not ic:
        return []
    cap = _cap(n)
    out = []
    for f in tabs:
        cells = None
        try:
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                t = pq.ParquetFile(f).read(columns=[ic, lc])
                cells = list(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))
            else:
                z = np.load(f, allow_pickle=True)
                cells = list(zip(np.asarray(z[ic]), np.asarray(z[lc])))
        except BaseException:
            continue
        for xv, yv in cells:
            try:
                x = _DS._decode_cell(xv)
                y = np.asarray(_DS._decode_cell(yv), dtype="float32")
            except BaseException:
                continue
            if y.ndim < 2:
                return []
            out.append((x, y))
            if cap and len(out) >= cap:
                return out
    return out


def _prep_image(x, adapter):
    import torch.nn.functional as F
    xt = x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))
    if xt.dim() == 3 and xt.shape[-1] in (3, 4):
        xt = xt.permute(2, 0, 1)
    xt = xt.float()
    if float(xt.max()) > 1.5:
        xt = xt / 255.0
    if xt.dim() == 3:
        xt = xt.unsqueeze(0)
    size = getattr(adapter, "size", None)
    if size:
        xt = F.interpolate(xt, size=tuple(size), mode="bicubic", align_corners=False)
    nrm = getattr(adapter, "norm", None)
    if nrm and nrm.get("mean") and nrm.get("std"):
        m = torch.tensor(nrm["mean"], dtype=xt.dtype).view(1, -1, 1, 1)
        s = torch.tensor(nrm["std"], dtype=xt.dtype).view(1, -1, 1, 1)
        xt = (xt - m) / s
    return xt


@torch.no_grad()
def eval_depth(model, adapter, pairs, device, cap_m=10.0):
    import torch.nn.functional as F
    model.eval()
    ar, rm, d1 = [], [], []
    for x, gt in pairs:
        xt = _prep_image(x, adapter)
        pred = _main_out(adapter.forward(model, xt.to(device))).float()
        while pred.dim() > 3:
            pred = pred[:, 0] if pred.shape[1] == 1 else pred[0].unsqueeze(0)
        pred = F.interpolate(pred.unsqueeze(1), size=gt.shape, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
        valid = gt > 1e-3
        if gt.shape == (480, 640):
            crop = np.zeros_like(valid)
            crop[45:471, 41:601] = True
            valid = valid & crop
        if not valid.any():
            continue
        p, tgt = pred[valid], 1.0 / gt[valid]
        s, t = np.linalg.lstsq(np.stack([p, np.ones_like(p)], 1), tgt, rcond=None)[0]
        pd = np.clip(1.0 / np.clip(s * pred + t, 1e-6, None), 1e-3, cap_m)
        g, e = gt[valid], pd[valid]
        ar.append(float(np.mean(np.abs(e - g) / g)))
        rm.append(float(np.sqrt(np.mean((e - g) ** 2))))
        d1.append(float(np.mean(np.maximum(e / g, g / e) < 1.25)))
    if not d1:
        return {"delta1": 0.0, "AbsRel": 1.0, "RMSE_m": 0.0}
    return {"delta1": sum(d1) / len(d1), "AbsRel": sum(ar) / len(ar), "RMSE_m": sum(rm) / len(rm)}


def _class_map(*roots):
    seen = set()
    for r in roots:
        d = os.path.abspath(str(r))
        for _ in range(4):
            if d in seen:
                break
            seen.add(d)
            f = os.path.join(d, "classes.json")
            if os.path.isfile(f):
                try:
                    m = json.load(open(f))
                    if isinstance(m, dict) and m:
                        return {str(k): v for k, v in m.items()}
                except BaseException:
                    pass
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    return {}


def pairs_classification(path, adapter, split="val", n=256, n_classes=None, model=None):
    import dataset as DS

    info = DS.detect_format(path)
    root = info.get("root", path)
    if info.get("format") == "hf_datasets":
        pr = DS.hf_pairs(root, adapter, model, split=split, n=_cap(n))
        if pr and n_classes is not None:
            k = len({c for _, c in pr})
            if k > n_classes:
                print("[metrika] hf: {} razreda u oznakama naspram {} izlaza modela — odbijam.".format(
                    k, n_classes))
                return []
        return pr
    if info.get("format") != "folder_per_class":
        return []
    mode = getattr(adapter, "_mode", "image")
    exts, dec = ((DS._IMG, DS._decode_image) if mode == "image" else
                 (DS._AUD, DS._decode_audio) if mode == "seq" else (None, None))
    if exts is None:
        return []

    by_class = {}
    for dp, _, fs in os.walk(root):
        hits = [os.path.join(dp, f) for f in sorted(fs) if f.lower().endswith(exts)]
        if hits:
            by_class.setdefault(os.path.basename(dp), []).extend(hits)
    if len(by_class) < 2:
        return []
    classes = sorted(by_class)

    cmap = _class_map(path, root)
    if cmap:
        star = cmap.get("*")
        idx = {c: int(cmap.get(c, star)) for c in classes if c in cmap or star is not None}
        classes = [c for c in classes if c in idx]
        k = len(set(idx.values()))
        print("[metrika] classes.json: {} foldera -> {} razreda.".format(len(classes), k))
        if n_classes is not None and k != n_classes:
            print("[metrika] ...ali model ima {} izlaza. Mapiranje ne odgovara modelu.".format(n_classes))
            return []
    else:
        if n_classes is not None and len(classes) != n_classes:
            print("[metrika] {} foldera naspram {} izlaza modela — ime foldera NIJE oznaka razreda. "
                  "Stavi `classes.json` u korijen dataseta (mapa folder->indeks, \"*\" = catch-all) "
                  "ili se mjeri slaganje s uciteljem.".format(len(classes), n_classes))
            return []
        idx = {c: i for i, c in enumerate(classes)}

    pools = {}
    for c in classes:
        fs = by_class[c]
        if split:
            insplit = [f for f in fs if split.lower() in f.replace("\\", "/").lower().split("/")]
            fs = insplit or fs
        pools.setdefault(idx[c], []).append(fs)
    per = max(1, _cap(n) // max(len(pools), 1))
    _sr = getattr(adapter, "sr", None)
    out = []
    for k, groups in sorted(pools.items()):
        picked, gi = [], 0
        while len(picked) < per and any(gi < len(g) for g in groups):
            for g in groups:
                if gi < len(g) and len(picked) < per:
                    picked.append(g[gi])
            gi += 1
        for f in picked:
            try:
                out.append(((dec(f, adapter._in_ch, adapter.imgsz, _sr) if _sr is not None
                             else dec(f, adapter._in_ch, adapter.imgsz)), k))
            except BaseException:
                continue
    return out


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


_EVAL = {"regression": eval_regression, "segmentation": eval_segmentation,
         "classification": eval_classification, "depth": eval_depth}


def evaluate(model, adapter, task, pairs, device):
    fn = _EVAL.get(task)
    if fn is None:
        raise NotImplementedError(f"Nema generičkog evaluatora za task '{task}' (detekcija ide preko morphology scorera).")
    return fn(model, adapter, pairs, device)
