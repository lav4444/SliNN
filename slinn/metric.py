"""
metric.py — FAZA 5.6: GENERIČKA prava task-metrika (scorer ravnina za NE-detekciju).

Quality-gate u petlji je KD-proxy (loss vs teacher, [[kd-only-no-gt]]); OVDJE je PRAVI izvještajni broj koji
dokazuje da komprimirani model zadržava kvalitetu na STVARNOM tasku (GT ULAZI SAMO u metriku, nikad u loss).
Metrike po tasku iz SUPPORTED_TASKS. Detekcija (mAP, decode/NMS) = REUSE morphology scorer (A.evaluate);
ovdje generic mIoU / r2·rmse / f1·accuracy jer morphology `_EVALUATORS` pokriva samo detection+classification.

Pareani (input, GT) val-reader je po formatu (housing npz X_val/y_val; voc JPEGImages+SegmentationClass+ImageSets).
"""
import glob
import json
import os
import sys

_AA = "/home/tomi/code/dipl/slinn"
for d in (_AA,):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from loss import _main_out                                    # noqa: E402  (izvlaci primarni tenzor izlaza)


# =========================== PAREANI VAL-READERI (input, GT) =========================== #
def _voc_root(path):
    for dp, _, _ in os.walk(path):
        if dp.replace("\\", "/").endswith("VOC2012"):
            return dp
    return path


def _cap(n):
    """DEV_DATA_SUBSET vrijedi i za MJERENJE metrike, ne samo za KD ulaze.

    (Do 6.10: cap je hvatao `engine._candidate_files` i detekcijski GT loader, ali `pairs_*` citaci
    su si sami listali datoteke i zaobilazili ga — pa je kod segmentacije i klasifikacije dev-podskup
    rezao trening, a metrika se i dalje mjerila na CIJELOM val skupu. Nekonzistentno i sporo.)
    Znacenje je kao u morphology: najvise N uzoraka PO SPLITU."""
    import settings as CFG
    d = CFG.DEV_DATA_SUBSET
    if not d:
        return n
    return int(d) if n is None else min(int(n), int(d))


def pairs_segmentation(path, split="val", n=64, size=256):
    """VOC: (image[3,size,size] float, mask[size,size] long). Split iz ImageSets/Segmentation/<split>.txt;
    slika iz JPEGImages, maska iz SegmentationClass (palette PNG -> indeksi 0..20, 255=ignore). Oboje na `size`
    (slika bilinear, maska NEAREST) -> mIoU na istom protokolu za original i komprimirani (pošten RELATIVNI broj)."""
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
        mk = Image.open(mp).resize((size, size), Image.NEAREST)          # palette -> indeksi ostaju
        y = torch.from_numpy(np.asarray(mk, dtype="int64"))
        out.append((x, y))
    return out


def pairs_regression(path, adapter, split="val", n=None):
    """Tabular npz: (X[in_ch] float, y float). Trazi X_<split>/y_<split>; fallback X/y ili zadnja 2 arraya."""
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


def _class_map(*roots):
    """Ucitaj `classes.json` (mapa ime_foldera -> indeks razreda) iz bilo koje od danih putanja ili
    njihovih roditelja. Vrati {} ako ga nema. Dataset time SAM deklarira svoj prostor oznaka kad ime
    foldera nije razred — kod ostaje agnostican."""
    seen = set()
    for r in roots:
        d = os.path.abspath(str(r))
        for _ in range(4):                                   # do 4 razine prema gore
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
    """(ulaz, class_idx) parovi za `folder_per_class` — razred JE ime nadfoldera.

    Pokriva slike i audio (isti obrazac, razlikuje se samo dekoder). Razredi se sortiraju
    abecedno -> indeks je stabilan i poklapa se s konvencijom `ImageFolder`.

    `n_classes` = SIRINA IZLAZA MODELA. Ako je zadana i NE poklapa se s brojem foldera, vraca [].
    Zasto je to nuzno: ime foldera nije uvijek oznaka. SpeechCommands ima 35 foldera, ali je M5
    treniran na 12 razreda (`yes/no/.../unknown/silence`) — 25 foldera pada u `unknown`,
    `_background_noise_` u `silence`. Bez ove provjere dobila bi se uvjerljiva ali POTPUNO KRIVA
    metrika (izmjereno: f1 0.011, tj. razina slucajnog pogadjanja).

    Formati bez citljivog para (npr. `hf_datasets` — treba tokenizer) takodjer vracaju [] ->
    pozivatelj degradira na teacher-agreement UMJESTO da izmislja metriku.
    """
    import dataset as DS

    info = DS.detect_format(path)
    root = info.get("root", path)
    if info.get("format") == "hf_datasets":                  # tekst: oznaka iz ClassLabel, ulaz iz tokenizera
        pr = DS.hf_pairs(root, adapter, model, split=split, n=_cap(n))
        if pr and n_classes is not None:
            k = len({c for _, c in pr})
            if k > n_classes:                                # vise razreda nego izlaza -> metrika bi lagala
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

    # Razredi se popisuju iz SAME STRUKTURE (svi podfolderi s medijem), ne iz uzorka datoteka.
    # (BUG do 6.7: indeks se gradio iz uzorka, a uzorak je bio prvih N sortiranih datoteka = SVE iz
    #  jednog razreda -> oznake 0..0 naspram 35-razrednog izlaza modela -> f1 = 0.0000.)
    by_class = {}
    for dp, _, fs in os.walk(root):
        hits = [os.path.join(dp, f) for f in sorted(fs) if f.lower().endswith(exts)]
        if hits:
            by_class.setdefault(os.path.basename(dp), []).extend(hits)
    if len(by_class) < 2:
        return []
    classes = sorted(by_class)

    # KONVENCIJA `classes.json`: kad ime foldera NIJE oznaka razreda, dataset to smije DEKLARIRATI.
    #   {"yes": 0, "no": 1, ..., "_background_noise_": 11, "*": 10}
    # Kljuc "*" je catch-all za sve neimenovane foldere — bez njega se npr. Wardenova konvencija
    # (SpeechCommands: 10 naredbi + 'unknown' za ostalih 25 rijeci + 'silence') ne da izraziti.
    # Kod time NE zna nista o modelu; dataset nosi vlastito mapiranje.
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

    # Kvota ide PO RAZREDU, ne po folderu. Kad `classes.json` vise foldera slije u jedan razred
    # (SpeechCommands: 25 rijeci -> 'unknown'), kvota-po-folderu tom razredu da 25x vise uzoraka:
    # izmjereno 275 od 391 para, tj. 70% skupa -> macro-f1 pao 0.73 umjesto ~0.87. Foldere unutar
    # razreda uzimamo NAIZMJENICE (round-robin) da razred ne bude sav iz jedne rijeci.
    pools = {}
    for c in classes:
        fs = by_class[c]
        if split:                                            # ako postoji split u putanji, preferiraj ga
            insplit = [f for f in fs if split.lower() in f.replace("\\", "/").lower().split("/")]
            fs = insplit or fs
        pools.setdefault(idx[c], []).append(fs)
    per = max(1, _cap(n) // max(len(pools), 1))
    _sr = getattr(adapter, "sr", None)
    out = []
    for k, groups in sorted(pools.items()):
        picked, gi = [], 0
        while len(picked) < per and any(gi < len(g) for g in groups):
            for g in groups:                                 # round-robin po folderima istog razreda
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


# =========================== EVALUATORI PO TASKU =========================== #
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
        out = _main_out(adapter.forward(model, xb)).float()               # [B,K,h,w]
        K = out.shape[1]
        if inter is None:
            inter = torch.zeros(K); union = torch.zeros(K)
        for b, (_, gt) in enumerate(chunk):
            pr = out[b].argmax(0).cpu()                                   # [h,w]
            gt = gt.cpu()
            if pr.shape != gt.shape:                                      # poravnaj na masku (NEAREST)
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
    """(input, class_idx). f1_macro (primarno) + accuracy."""
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
    """LABEL-FREE gate (task nepoznat / bez oznaka): koliko se student SLAŽE s TEACHEROM (referenca = smrznuti
    original, ne GT). Ograničeno i interpretabilno -> kalibrabilno kao prava metrika ('zadrži ≥X% slaganja').
      kind 'kl'  (klasne distribucije, argmax): udio elemenata s istim argmaxom (top-1; per-piksel za densno)
      kind 'mse' (kontinuirano/raw): R² student-izlaza vs TEACHER-izlaz (teacher='istina'; 1.0 = identično)
    `inputs` = lista sirovih uzoraka (bez oznaka). Vrati {'agreement': [0,1]}."""
    student.eval(); teacher.eval()
    num = den = 0.0
    sse = 0.0; tvals = []
    for i in range(0, len(inputs), bs):
        xb = [x.to(device) for x in inputs[i:i + bs]]
        s = _main_out(adapter.forward(student, xb)).float()
        t = _main_out(adapter.forward(teacher, xb)).float()
        if kind == "kl":
            ax = 1 if s.dim() >= 3 else -1                    # klasna os: [B,K]-> -1, [B,K,H,W]-> 1
            num += float((s.argmax(ax) == t.argmax(ax)).sum()); den += s.argmax(ax).numel()
        else:
            sse += float(((s - t) ** 2).sum()); tvals.append(t.flatten().cpu())
    if kind == "kl":
        return {"agreement": num / max(den, 1)}
    tt = torch.cat(tvals); sst = float(((tt - tt.mean()) ** 2).sum())
    return {"agreement": (1.0 - sse / sst) if sst > 0 else 0.0}   # R² vs teacher


_EVAL = {"regression": eval_regression, "segmentation": eval_segmentation, "classification": eval_classification}


def evaluate(model, adapter, task, pairs, device):
    """Generička prava metrika za NE-detekciju (detekcija -> morphology A.evaluate). Vrati metric-dict."""
    fn = _EVAL.get(task)
    if fn is None:
        raise NotImplementedError(f"Nema generičkog evaluatora za task '{task}' (detekcija ide preko morphology scorera).")
    return fn(model, adapter, pairs, device)
