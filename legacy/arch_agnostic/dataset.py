import glob
import json
import os

FORMATS_PATH = os.path.join(os.path.dirname(__file__), "SUPPORTED_DATASET_FORMATS.json")

_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_AUD = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
_VID = (".mp4", ".avi", ".mov", ".mkv")
_TAB = (".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".npy", ".npz")
_TXT = (".txt", ".json", ".jsonl", ".xml", ".md")
_MEDIA = _IMG + _AUD
_SPLITS = ("train", "val", "valid", "validation", "test")
_CATS = {"image": _IMG, "audio": _AUD, "video": _VID, "tabular": _TAB, "text": _TXT}
_INPUT_CATS = ("image", "audio", "video", "tabular")


def load_formats(path=FORMATS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dirs(p):
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))) if os.path.isdir(p) else []


def _has_ext(p, exts, depth=2):
    for d in range(depth + 1):
        if glob.glob(os.path.join(p, *(["*"] * d), "*")):
            for f in glob.glob(os.path.join(p, *(["*"] * d), "*")):
                if f.lower().endswith(exts):
                    return f
    return None


def _first_file(p, exts):
    for root, _, files in os.walk(p):
        for f in sorted(files):
            if f.lower().endswith(exts):
                return os.path.join(root, f)
    return None


def _descend(p):
    for _ in range(5):
        subs = _dirs(p)
        if len(subs) == 1 and not _has_ext(p, _MEDIA, depth=1):
            p = os.path.join(p, subs[0])
        else:
            break
    return p


def _find_splits(p):
    found = []
    for loc in (p, os.path.join(p, "images"), os.path.join(p, "JPEGImages")):
        for d in _dirs(loc):
            if d.lower() in _SPLITS and d.lower() not in found:
                found.append(d.lower())
    return found


def _is_hf_datasets(p):
    info = None
    for base in (os.path.join(p, "datasets"), p):
        hits = glob.glob(os.path.join(base, "**", "dataset_info.json"), recursive=True)
        if hits:
            info = hits[0]
            break
    if not info:
        return None
    try:
        feats = (json.load(open(info, encoding="utf-8")).get("features") or {})
    except BaseException:
        feats = {}
    ct = lambda v: v.get("_type") if isinstance(v, dict) else None                      # noqa: E731
    if any(ct(v) == "ClassLabel" for v in feats.values()):
        return "classification", "hf datasets cache (ClassLabel feature)"
    if any(ct(v) == "Sequence" and ct((v or {}).get("feature")) == "ClassLabel" for v in feats.values()):
        return "unknown", "hf datasets cache (per-token ClassLabel -> token_classification, nepodrzano)"
    return "unknown", "hf datasets cache (nepoznata label-shema)"


def _is_coco(p):
    for jf in glob.glob(os.path.join(p, "*.json")) + glob.glob(os.path.join(p, "**", "*.json"), recursive=True)[:5]:
        try:
            if os.path.getsize(jf) > 200 * 1024 * 1024:
                continue
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
                seg = any("segmentation" in a for a in d["annotations"][:20])
                return "segmentation" if seg else "detection", f"coco json '{os.path.basename(jf)}'"
        except BaseException:
            continue
    return None


def _is_voc(path, p):
    top = set(_dirs(path))
    subs = set(_dirs(p))
    if "VOCdevkit" in top or "JPEGImages" in subs:
        if "SegmentationClass" in subs:
            return "segmentation", "VOC: JPEGImages + SegmentationClass"
        if "Annotations" in subs:
            return "detection", "VOC: JPEGImages + Annotations (xml)"
    return None


def _peek_yolo_label(txt):
    try:
        for line in open(txt, encoding="utf-8").read().splitlines():
            parts = line.split()
            if parts:
                [float(x) for x in parts]
                return len(parts)
    except BaseException:
        pass
    return None


def _is_yolo(p):
    subs = {d.lower(): d for d in _dirs(p)}
    if "images" in subs and "labels" in subs:
        lbl = _first_file(os.path.join(p, subs["labels"]), (".txt",))
        n = _peek_yolo_label(lbl) if lbl else None
        if n == 5:
            return "detection", "images/ + labels/ ; label redak = 5 tokena (cls cx cy w h)"
        if n is not None:
            return "detection", f"images/ + labels/ ; label redak = {n} tokena (box-like)"
        return "detection", "images/ + labels/ (prazne/nedostupne oznake)"
    return None


def _is_seg_masks(p):
    subs = {d.lower(): d for d in _dirs(p)}
    if "images" in subs and ("masks" in subs or "annotations" in subs):
        m = _first_file(os.path.join(p, subs.get("masks", subs.get("annotations"))), _IMG)
        if m:
            return "segmentation", "images/ + masks/ (PNG)"
    return None


def _is_folder_per_class(p):
    subs = _dirs(p)
    classes = [d for d in subs if _first_file(os.path.join(p, d), _MEDIA)]
    non_split = [c for c in classes if c.lower() not in _SPLITS]
    if len(non_split) >= 2:
        return "classification", f"{len(non_split)} podfoldera-klasa s medijem"
    return None


def _is_tabular(p):
    f = (glob.glob(os.path.join(p, "*.csv")) or glob.glob(os.path.join(p, "*.parquet"))
         or glob.glob(os.path.join(p, "*.npz")) or glob.glob(os.path.join(p, "*.npy")))
    if f:
        return "regression", f"tabular '{os.path.basename(f[0])}'"
    return None


def _is_nlp(p):
    if glob.glob(os.path.join(p, "*.jsonl")) or glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)[:1]:
        return "classification", "*.jsonl (tekst+label)"
    return None


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def _cat_of(f):
    fl = f.lower()
    for cat, exts in _CATS.items():
        if fl.endswith(exts):
            return cat
    return None


def _walk_all(path, cap=20000):
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            out.append(os.path.join(root, f))
            if len(out) >= cap:
                return out
    return out


def _isnum(t):
    try:
        float(t)
        return True
    except BaseException:
        return False


def _img_role(path):
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(path)
        if im.mode == "P":
            return "mask"
        if im.mode in ("L", "I", "I;16", "F", "1"):
            a = np.asarray(im).ravel()
            u = np.unique(a[:: max(1, a.size // 4096)])
            return "mask" if u.size <= 64 else "photo"
        return "photo"
    except BaseException:
        return None


def _peek_text_kind(path):
    try:
        raw = open(path, "rb").read(4000)
    except BaseException:
        return None
    if b"\x00" in raw:
        return None
    printable = sum(1 for b in raw if 9 <= b <= 13 or 32 <= b <= 126)
    if raw and printable / len(raw) < 0.85:
        return None
    head = raw.decode("utf-8", "ignore")
    s = head.strip()
    if not s:
        return None
    if s[0] in "{[":
        return "coco" if ('"annotations"' in head or '"bbox"' in head) else "json"
    if "bndbox" in head.lower():
        return "xml_boxes"
    lines = [ln.split() for ln in s.splitlines()[:6] if ln.split()]
    if lines and all(all(_isnum(t) for t in ln) for ln in lines):
        w = len(lines[0])
        if w == 1:
            return "scalar"
        if 5 <= w <= 7 and float(lines[0][0]).is_integer():
            return "boxes"
        return "numeric"
    if lines and any(t == "O" or t.startswith(("B-", "I-")) for ln in lines for t in ln):
        return "tokens"
    return "text"


def _peek_table(path):
    LAB = ("y", "label", "target", "class", "depth", "mask", "gt", "seg")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npz":
            import numpy as np
            z = np.load(path, allow_pickle=True)
            keys = list(z.files)
            lab = [k for k in keys if any(t in k.lower() for t in LAB)] or \
                  [k for k in keys if getattr(z[k], "ndim", 9) == 1]
            lk = None
            if lab:
                lk = "regression" if np.issubdtype(z[lab[0]].dtype, np.floating) else "classification"
            return {"label": bool(lab), "label_kind": lk, "input_kind": "tabular"}
        if ext == ".parquet":
            import pyarrow.parquet as pq
            names = [n.lower() for n in pq.ParquetFile(path).schema_arrow.names]
            lab = [n for n in names if any(t in n for t in LAB)]
            img = any(("image" in n or "img" in n) for n in names)
            lk = "regression" if any(("depth" in n) for n in lab) else ("classification" if lab else None)
            return {"label": bool(lab), "label_kind": lk, "input_kind": "image" if img else "tabular"}
        if ext in (".csv", ".tsv"):
            names = [h.strip().lower() for h in open(path, encoding="utf-8", errors="ignore").readline().split(",")]
            lab = [n for n in names if any(t in n for t in LAB)]
            return {"label": True, "label_kind": None, "input_kind": "tabular"}
    except BaseException:
        pass
    return {"label": False, "label_kind": None, "input_kind": "tabular"}


def _sniff_labels(path, samples, files):
    from collections import Counter
    sset = set(samples)
    stems = [_stem(s) for s in samples]
    sstemset = set(stems)

    by_stem = {}
    for f in files:
        if f not in sset:
            by_stem.setdefault(_stem(f), []).append(f)
    matched = [st for st in sstemset if st in by_stem]
    if sstemset and len(matched) >= max(1, len(sstemset) // 2):
        exts = Counter(os.path.splitext(f)[1].lower() for st in matched[:80] for f in by_stem[st])
        top_ext = exts.most_common(1)[0][0]
        kinds, peeked = Counter(), 0
        for st in matched:
            if peeked >= 30:
                break
            for f in by_stem[st]:
                k = _peek_text_kind(f)
                if k:
                    kinds[k] += 1
                    peeked += 1
                    break
        kind = kinds.most_common(1)[0][0] if kinds else None
        hint = {"boxes": "detection", "xml_boxes": "detection", "coco": "detection", "polygons": "segmentation",
                "tokens": "unknown", "scalar": "regression"}.get(kind) \
            or {".png": "segmentation"}.get(top_ext, "unknown")
        return "stem_sidecar", hint, f"po-uzorku '{top_ext}' (sadrzaj={kind}) uz {len(matched)}/{len(sstemset)}"

    parents = {}
    for s in samples:
        parents.setdefault(os.path.basename(os.path.dirname(s)), set()).add(_stem(s))
    uparents = {p: st for p, st in parents.items() if p.lower() not in _SPLITS}
    if len(uparents) >= 2:
        keys = list(uparents)
        ov, pairs = 0.0, 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = uparents[keys[i]], uparents[keys[j]]
                if a and b:
                    ov += len(a & b) / max(1, min(len(a), len(b)))
                    pairs += 1
        mean_ov = ov / max(1, pairs)
        if mean_ov >= 0.85 and len(uparents) <= 4:
            return "parallel_dir", "unknown", f"{len(uparents)} dirova, overlap {mean_ov:.2f} -> paralelni view-i, ne klase"
        return "class_subfolder", "classification", f"{len(uparents)} folder-klasa (overlap {mean_ov:.2f})"

    if glob.glob(os.path.join(path, "*.csv")) or glob.glob(os.path.join(path, "*.json")):
        return "manifest", "unknown", "tablica na rootu (csv/json)"
    return "none", "unknown", "nema oznaka -> KD-only (samo ulazi)"


def _mk(modality, task_hint, method, samples, path, why):
    return {"format": "sniffed", "modality": modality, "task_hint": task_hint, "label_method": method,
            "n_samples": len(samples), "splits": _find_splits(path), "root": path, "_samples": samples[:500],
            "why": f"sniffer: ulaz={modality} x{len(samples)}, oznake={method} ({why})"}


def _sniff_image(path, files, imgs):
    from collections import Counter, defaultdict
    by_dir = defaultdict(list)
    for f in imgs:
        by_dir[os.path.dirname(f)].append(f)
    role = {}
    for d, fl in by_dir.items():
        sample = fl[:: max(1, len(fl) // 8)][:8]
        r = Counter(x for x in (_img_role(p) for p in sample) if x)
        role[d] = "mask" if r.get("mask", 0) > 0 and r.get("mask", 0) >= r.get("photo", 0) else "photo"
    photos = [f for d, fl in by_dir.items() if role[d] == "photo" for f in fl]
    masks = [f for d, fl in by_dir.items() if role[d] == "mask" for f in fl]
    inputs = photos or imgs
    if masks and inputs:
        mstems, pstems = {_stem(m) for m in masks}, {_stem(f) for f in inputs}
        if len(mstems & pstems) >= 0.5 * len(mstems):
            return _mk("image", "segmentation", "mask_content", inputs, path,
                       f"{len(masks)} maski (sadrzaj) poravnato s fotama")
    method, task_hint, why = _sniff_labels(path, inputs, files)
    return _mk("image", task_hint, method, inputs, path, why)


def _sniff_tabular(path, files, tabs):
    from collections import Counter
    peeks = [_peek_table(f) for f in tabs[:5]]
    has_label = any(p["label"] for p in peeks)
    lk = Counter(p["label_kind"] for p in peeks if p["label_kind"])
    ik = Counter(p["input_kind"] for p in peeks)
    task_hint = lk.most_common(1)[0][0] if lk else "unknown"
    modality = ik.most_common(1)[0][0] if ik else "tabular"
    method = "in_table" if has_label else "none"
    return _mk(modality, task_hint, method, tabs, path,
               f"tablica: label={'da' if has_label else 'ne'} ({task_hint}), input={modality}")


def agnostic_sniffer(path, cap=2_000_000):
    from collections import Counter
    files = _walk_all(path, cap)
    if not files:
        return None
    cats = Counter(c for c in (_cat_of(f) for f in files) if c)
    input_cat = next((c for c, _ in cats.most_common() if c in _INPUT_CATS), None)
    if input_cat is None:
        input_cat = "text" if cats.get("text") else None
    if input_cat is None:
        return None
    cat_files = [f for f in files if _cat_of(f) == input_cat]
    if input_cat == "image":
        return _sniff_image(path, files, cat_files)
    if input_cat == "tabular":
        return _sniff_tabular(path, files, cat_files)
    method, task_hint, why = _sniff_labels(path, cat_files, files)
    return _mk(input_cat, task_hint, method, cat_files, path, why)


def detect_format(path):
    if not os.path.exists(path):
        return {"format": "unknown_format", "task_hint": "unknown", "splits": [], "why": f"put ne postoji: {path}"}
    root = _descend(path)
    for name, fn, args in (
        ("hf_datasets", _is_hf_datasets, (root,)),
        ("coco", _is_coco, (root,)),
        ("voc", _is_voc, (path, root)),
        ("yolo", _is_yolo, (root,)),
        ("seg_masks", _is_seg_masks, (root,)),
        ("folder_per_class", _is_folder_per_class, (root,)),
        ("tabular", _is_tabular, (root,)),
        ("nlp", _is_nlp, (root,)),
    ):
        r = fn(*args)
        if r:
            task_hint, why = r
            return {"format": name, "task_hint": task_hint, "splits": _find_splits(root) or _find_splits(path),
                    "why": why, "root": root}
    sniff = agnostic_sniffer(root)
    if sniff:
        return sniff
    seen = f"dirs={_dirs(root)[:6]} media={'da' if _has_ext(root, _MEDIA) else 'ne'}"
    return {"format": "unknown_format", "task_hint": "unknown", "splits": [], "why": f"nista citljivo ({seen})", "root": root}


def label_sample(path, n=8):
    import torch

    info = detect_format(path)
    fmt, root = info["format"], info.get("root", path)

    if fmt == "unknown_format":
        raise ValueError(f"DatasetProbe: nista citljivo ({info['why']}). Ni sniffer ne nalazi ulaze. "
                         f"Dodaj recognizer ili koristi podrzani format (v. SUPPORTED_DATASET_FORMATS.json).")

    if fmt == "sniffed":
        if info.get("label_method") == "class_subfolder":
            samples = info.get("_samples", [])
            classes = sorted({os.path.basename(os.path.dirname(s)) for s in samples})
            ix = {c: i for i, c in enumerate(classes)}
            return torch.tensor([ix[os.path.basename(os.path.dirname(s))] for s in samples[:n]], dtype=torch.long)
        return None

    if fmt == "folder_per_class":
        classes = sorted(d for d in _dirs(root) if _first_file(os.path.join(root, d), _MEDIA)
                         and d.lower() not in _SPLITS)
        idx = {c: i for i, c in enumerate(classes)}
        labs = []
        for c in classes:
            if _first_file(os.path.join(root, c), _MEDIA):
                labs.append(idx[c])
            if len(labs) >= n:
                break
        return torch.tensor(labs, dtype=torch.long)

    if fmt == "yolo":
        subs = {d.lower(): d for d in _dirs(root)}
        out = []
        for txt in _walk_files(os.path.join(root, subs["labels"]), (".txt",), n):
            rows = [[float(x) for x in ln.split()] for ln in open(txt).read().splitlines() if ln.split()]
            out.append(torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(0, 5))
        return out

    if fmt == "voc":
        from PIL import Image
        import numpy as np
        segdir = os.path.join(root, "SegmentationClass")
        masks = []
        for m in _walk_files(segdir, _IMG, n):
            masks.append(torch.from_numpy(np.array(Image.open(m))).long())
        if masks:
            h = min(x.shape[0] for x in masks); w = min(x.shape[1] for x in masks)
            return torch.stack([x[:h, :w] for x in masks])
        return None

    if fmt == "tabular":
        return _tabular_target(root, n)

    if fmt == "hf_datasets":
        import datasets
        arrows = _walk_files(root, (".arrow",), 100)
        arrows.sort(key=lambda f: 0 if any(s in os.path.basename(f).lower() for s in ("valid", "test")) else 1)
        if not arrows:
            return None
        ds = datasets.Dataset.from_file(arrows[0])
        col = next((k for k, v in ds.features.items() if type(v).__name__ == "ClassLabel"), None)
        if col is None:
            col = next((k for k in ds.column_names if k.lower() in ("label", "labels", "target")), None)
        if col is None:
            return None
        return torch.tensor(ds[col][:n])

    raise NotImplementedError(f"format '{fmt}' prepoznat, ali reader jos nije implementiran (todo)")


def _walk_files(p, exts, n):
    out = []
    for root, _, files in os.walk(p):
        for f in sorted(files):
            if f.lower().endswith(exts):
                out.append(os.path.join(root, f))
                if len(out) >= n:
                    return out
    return out


def _survey(root, exts):
    total, per = 0, {}
    for dp, _, files in os.walk(root):
        parts = [p.lower() for p in dp.replace("\\", "/").split("/")]
        split = next((s for s in reversed(parts) if s in _SPLITS), None)
        c = sum(1 for f in files if f.lower().endswith(exts))
        if c:
            total += c
            if split:
                per[split] = per.get(split, 0) + c
    return total, per


def _survey_tabular(root):
    import numpy as np
    total, per = 0, {}
    for f in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)[:5]:
        try:
            z = np.load(f, allow_pickle=True)
            keys = [k for k in z.files if any(t in k.lower() for t in ("y", "label", "target"))] or list(z.files)
            for k in keys:
                a = z[k]
                if getattr(a, "ndim", 0) >= 1:
                    sp = next((s for s in _SPLITS if s in k.lower()), None)
                    total += int(a.shape[0])
                    if sp:
                        per[sp] = per.get(sp, 0) + int(a.shape[0])
        except BaseException:
            pass
    for f in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)[:50]:
        try:
            import pyarrow.parquet as pq
            n = pq.ParquetFile(f).metadata.num_rows
            sp = next((s for s in _SPLITS if s in os.path.basename(f).lower()), None)
            total += n
            if sp:
                per[sp] = per.get(sp, 0) + n
        except BaseException:
            pass
    for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)[:3] if total == 0 else []:
        try:
            total += max(0, sum(1 for _ in open(f, encoding="utf-8")) - 1)
        except BaseException:
            pass
    return total, per


def _survey_hf(root):
    total, per = 0, {}
    try:
        import datasets
        for a in _walk_files(root, (".arrow",), 30):
            sp = next((s for s in _SPLITS if s in os.path.basename(a).lower()), None)
            n = datasets.Dataset.from_file(a).num_rows
            total += n
            if sp:
                per[sp] = per.get(sp, 0) + n
    except BaseException:
        pass
    return total, per


def _survey_for(info):
    fmt, root = info["format"], info.get("root", ".")
    if fmt == "tabular":
        return _survey_tabular(root)
    if fmt == "hf_datasets":
        return _survey_hf(root)
    if fmt == "sniffed":
        mod = info.get("modality")
        if mod in ("image", "audio", "video"):
            t, per = _survey(root, _CATS.get(mod, _MEDIA))
            if t > 0:
                return t, per
        return _survey_tabular(root)
    exts = {"voc": _IMG, "coco": _IMG, "yolo": _IMG, "seg_masks": _IMG,
            "folder_per_class": _MEDIA}.get(fmt, _MEDIA)
    return _survey(root, exts)


def probe_dataset(path):
    info = detect_format(path)
    fmt = info["format"]
    samples_found = fmt != "unknown_format"
    labels_found, lwhy, labs = False, "", None
    if samples_found:
        try:
            labs = label_sample(path, n=8)
            labels_found = labs is not None and (not hasattr(labs, "__len__") or len(labs) > 0)
            if not labels_found:
                lwhy = "oznake nedostupne ovom metodom"
        except NotImplementedError:
            lwhy = f"oznake postoje ali reader nije gotov ({fmt})"
        except Exception as e:
            lwhy = f"label greska ({type(e).__name__})"
    n_total, per_split = _survey_for(info) if samples_found else (0, {})
    mode = "stop" if not samples_found else ("full" if labels_found else "core_kd_only")
    return {"format": fmt, "modality": info.get("modality"), "task_hint": info.get("task_hint"),
            "samples_found": samples_found, "labels_found": labels_found,
            "n_samples": n_total, "splits": per_split or None, "mode": mode,
            "_labels": labs,
            "why": info["why"] + (f" | {lwhy}" if lwhy else "")}


def _tabular_target(root, n):
    import torch
    import numpy as np
    npz = glob.glob(os.path.join(root, "*.npz"))
    if npz:
        z = np.load(npz[0], allow_pickle=True)
        key = next((k for k in ("y", "target", "label", "labels") if k in z), None)
        if key is None:
            key = next((k for k in z.files if z[k].ndim == 1), z.files[-1])
        y = np.asarray(z[key]).ravel()[:n]
        return torch.as_tensor(y)
    csv = glob.glob(os.path.join(root, "*.csv"))
    if csv:
        import csv as _c
        rows = list(_c.reader(open(csv[0])))
        header, data = rows[0], rows[1:]
        ci = next((i for i, h in enumerate(header) if h.lower() in ("y", "target", "label", "labels")), len(header) - 1)
        vals = [float(r[ci]) for r in data[:n]]
        return torch.tensor(vals)
    return None


def _decode_image(f, in_ch, size):
    import numpy as np
    import torch
    from PIL import Image
    im = Image.open(f).convert("RGB" if in_ch >= 3 else "L").resize((size, size))
    t = torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)
    t = t.unsqueeze(0) if t.dim() == 2 else t.permute(2, 0, 1)
    if t.shape[0] < in_ch:
        t = t.repeat(in_ch, 1, 1)
    return t[:in_ch]


def _decode_audio(f, in_ch, length):
    import torch
    import torchaudio
    wav, _ = torchaudio.load(f)
    if wav.shape[0] != in_ch:
        wav = wav.mean(0, keepdim=True).repeat(in_ch, 1) if in_ch > wav.shape[0] else wav[:in_ch]
    lg = wav.shape[1]
    return wav[:, :length] if lg >= length else torch.nn.functional.pad(wav, (0, length - lg))


def _tabular_matrix(root, in_ch):
    import numpy as np
    for f in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)[:5]:
        try:
            z = np.load(f, allow_pickle=True)
            for k in z.files:
                a = z[k]
                if getattr(a, "ndim", 0) == 2 and a.shape[1] == in_ch:
                    return np.asarray(a, dtype="float32")
        except BaseException:
            pass
    for f in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)[:3]:
        try:
            import csv as _c
            rows = list(_c.reader(open(f, encoding="utf-8")))[1:]
            X = np.array([[float(x) for x in r[:in_ch]] for r in rows], dtype="float32")
            if X.ndim == 2 and X.shape[1] == in_ch:
                return X
        except BaseException:
            pass
    return None


def _media_files(root, exts, split, cap, skip_masks=False):
    hit, rest = [], []
    for dp, _, files in os.walk(root):
        parts = dp.replace("\\", "/").lower().split("/")
        insplit = bool(split) and split.lower() in parts
        for f in sorted(files):
            if f.lower().endswith(exts):
                (hit if insplit else rest).append(os.path.join(dp, f))
    files = hit or rest
    if not skip_masks:
        return files[:cap]
    out = []
    for p in files:
        if _img_role(p) != "mask":
            out.append(p)
        if len(out) >= cap:
            break
    return out


def input_batch(path, adapter, device, split=None, n=8):
    import torch
    mode = getattr(adapter, "_mode", "image")
    in_ch, size = adapter._in_ch, adapter.imgsz
    root = detect_format(path).get("root", path)
    try:
        if mode == "image":
            fs = _media_files(root, _IMG, split, n * 3, skip_masks=True)
            if fs:
                return [_decode_image(f, in_ch, size).to(device) for f in fs[:n]], "image-files"
        elif mode == "seq":
            fs = _media_files(root, _AUD, split, n * 3)
            if fs:
                return [_decode_audio(f, in_ch, size).to(device) for f in fs[:n]], "audio-files"
        elif mode == "vector":
            X = _tabular_matrix(root, in_ch)
            if X is not None and len(X):
                return [torch.from_numpy(X[i % len(X)]).to(device) for i in range(n)], "tabular"
    except BaseException:
        pass
    return [adapter._one(device) for _ in range(n)], "fallback-random"


def resolve_splits(splits):
    s = set((splits or {}).keys()) if isinstance(splits, dict) else set(splits or [])
    pick = lambda *names: next((n for n in names if n in s), None)   # noqa: E731
    tr, va, te = pick("train"), pick("val", "valid", "validation"), pick("test")
    if tr and va and te:
        return {"train": tr, "val": va, "test": te, "method": "as-is", "note": "sva tri splita"}
    if tr and va:
        return {"train": tr, "val": va, "test": va, "method": "alias", "note": "nema test -> test=val"}
    if tr and te:
        return {"train": tr, "val": te, "test": te, "method": "alias", "note": "nema val -> val=test"}
    if tr:
        return {"train": tr, "val": "AUTO", "test": "AUTO", "method": "auto-from-train",
                "note": "samo train -> stratified carve val/test (15/15) iz train"}
    return {"train": "AUTO", "val": "AUTO", "test": "AUTO", "method": "auto-pool",
            "note": "bez train/splitova -> pool sve + stratified 70/15/15"}


def stratified_split(labels, ratios=(0.70, 0.15, 0.15), seed=0):
    import numpy as np
    if labels is None:
        keys = None
    else:
        import torch
        y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        if y.ndim > 1:
            keys = np.zeros(len(y), dtype=int)
        elif np.issubdtype(y.dtype, np.floating):
            edges = np.quantile(y, np.linspace(0, 1, 11))
            keys = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
        else:
            keys = y.astype(int)
    n = 0 if keys is None else len(keys)
    if keys is None or n == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        rng.shuffle(idx)
        n1 = int(round(ratios[0] * len(idx)))
        n2 = n1 + int(round(ratios[1] * len(idx)))
        tr += idx[:n1].tolist()
        va += idx[n1:n2].tolist()
        te += idx[n2:].tolist()
    return sorted(tr), sorted(va), sorted(te)
