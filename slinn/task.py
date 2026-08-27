"""
task.py — automatska detekcija taska, model-agnosticno (pred-Faza 4).

Dva signala, krizno (hibrid), po §3 plana:
  A. OZNAKE (presudno)  — _label_signature: duck-typing DEKODIRANOG in-memory oblika oznaka.
                          data.py parsira YOLO/COCO/VOC/PNG/BIO s diska; ovdje gledamo samo tenzor/listu.
  B. ARHITEKTURA (potvrda + fallback bez oznaka) — _arch_signature: strukturni markeri (Detect/AnchorGenerator
                          /RoIHeads = detection) i oblik probe-izlaza ([B,K,H,W]=seg, [B,1,H,W] float=depth).
                          FORMAT-NEOVISNO (cita model, ne oznake) -> backstop za cudan/nepostojeci label-format.

Ljestvica povjerenja: override > A∩B (slazu) > A (vjeruj oznakama) > B (label-less) > unknown (KD-only).
Katalog taskova + metrike: SUPPORTED_TASKS.json (single source of truth). Nikad ne baca -> neprepoznato = unknown.
"""
import json
import os

import torch

TASKS_PATH = os.path.join(os.path.dirname(__file__), "SUPPORTED_TASKS.json")

# strukturni markeri detekcije (substring u imenu TIPA modula, lowercase) — genericko, ne per-model
_DET_MARKERS = ("detect", "anchorgenerator", "roihead", "roialign", "rpnhead", "regionproposal")


def load_tasks(path=TASKS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# =========================== POMOCNO =========================== #
def _as_tensor(x):
    """Best-effort u torch.Tensor; None ako ne ide (npr. lista raznolikih dictova)."""
    if isinstance(x, torch.Tensor):
        return x
    try:
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor) and x[0].dim() == 0:
            return torch.stack(list(x))                     # lista skalara
        t = torch.as_tensor(x)
        return t if t.dtype != torch.object else None
    except BaseException:
        return None


def _is_boxes(x):
    """Je li x jedna 'kutija-slicna' anotacija: [N,>=4] tensor/niz promjenjivog N."""
    t = _as_tensor(x)
    return t is not None and t.dim() == 2 and t.shape[-1] >= 4 and t.is_floating_point()


def _first_tensor(o):
    """Prvi tensor u ugnijezdjenom izlazu (dict/list/tuple/ModelOutput)."""
    if isinstance(o, torch.Tensor):
        return o
    if hasattr(o, "logits"):
        return o.logits
    if isinstance(o, dict):
        for v in o.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(o, (list, tuple)):
        for v in o:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


# =========================== SIGNAL A: OZNAKE =========================== #
def _label_signature(labels):
    """Vrati (task|None, why) iz DEKODIRANOG in-memory oblika oznaka. Neprepoznato -> (None, ...)."""
    if labels is None:
        return None, "nema oznaka"

    # detection: lista po slici (dict{boxes} ili [N,>=4] okviri promjenjivog N)
    if isinstance(labels, (list, tuple)) and len(labels):
        it = labels[0]
        if isinstance(it, dict) and any(k in it for k in ("boxes", "bbox", "bboxes")):
            return "detection", "oznake: lista dict{boxes} po slici"
        if _is_boxes(it):
            return "detection", "oznake: lista [N,>=4] okvira po slici"

    t = _as_tensor(labels)
    if t is None:
        return None, "neprepoznat oblik oznaka"
    is_int = not t.is_floating_point()

    if t.dim() >= 3:                                        # dense per-piksel
        return ("segmentation" if is_int else "regression"), \
            f"oznake: dense {tuple(t.shape)} {'int(klase)' if is_int else 'float(depth)'}"
    if t.dim() == 2:
        _, K = t.shape
        # binarnost se provjerava PRIJE dtype-a: multi-hot zna doci kao float (0.0/1.0), npr. schoolcnn
        binary = bool((((t == 0) | (t == 1)).all()).item())
        if binary:
            rowsum = t.sum(1)
            if bool((rowsum == 1).all().item()):
                return "classification", "oznake: [B,K] one-hot"
            return "multilabel", "oznake: [B,K] multi-hot (>1 aktivan)"
        if not is_int:
            return "regression", f"oznake: [B,{K}] float ne-binarno (vektor)"
        return None, f"oznake: [B,{K}] int ne-binarno (per-token/NER?) -> unknown"
    if t.dim() == 1:
        return ("classification" if is_int else "regression"), \
            f"oznake: [B] {'int(klasa)' if is_int else 'float(skalar)'}"
    return None, f"oznake: neprepoznat oblik {tuple(t.shape)}"


# =========================== SIGNAL B: ARHITEKTURA =========================== #
def _arch_signature(model, adapter, device):
    """Vrati (task|None, shape, why) iz STRUKTURE modela + oblika probe-izlaza. Format-neovisno.
    shape = klasa izlaza: 'detection' | 'spatial' [B,K,H,W] | 'dense1' [B,1,H,W]/dim3 | 'flat' [B,K]/[B] | None.
    shape sluzi da B OGRANICI A (strukturirani task je nemoguc na flat izlazu)."""
    for _, m in model.named_modules():
        tn = type(m).__name__.lower()
        if any(k in tn for k in _DET_MARKERS):
            return "detection", "detection", f"arh: detekcijski modul {type(m).__name__}"
    try:
        model.eval()
        with torch.no_grad():
            out = adapter.forward(model, adapter.forward_example(device))
        t = _first_tensor(out)
    except BaseException:
        t = None
    if t is None:
        return None, None, "arh: izlaz neuhvatljiv"
    if t.dim() == 4:
        if t.shape[1] > 1:
            return "segmentation", "spatial", f"arh: izlaz [B,{t.shape[1]},H,W] (per-piksel klase)"
        return "regression", "dense1", "arh: izlaz [B,1,H,W] (dense/depth)"
    if t.dim() == 3 and t.is_floating_point():
        return "regression", "dense1", f"arh: izlaz {tuple(t.shape)} float (dense/depth)"
    if t.dim() in (1, 2):
        return None, "flat", f"arh: flat izlaz {tuple(t.shape)} (cls/multilabel/reg — treba oznake)"
    return None, None, "arh: nedistinktivan izlaz"


def _recast_flat(labels):
    """Strukturirane oznake (detection/seg) na modelu s FLAT izlazom -> reinterpretiraj kao klasifikacijsku
    obitelj (model to ne moze proizvesti drugacije). Boxevi -> per-uzorak prisutnost klasa: >1 klasa = multilabel."""
    try:
        if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
            maxc = max((int(t[:, 0].unique().numel()) if t.numel() else 0) for t in labels)
            return "multilabel" if maxc > 1 else "classification"
    except BaseException:
        pass
    return "classification"


# =========================== DETEKCIJA (ljestvica) =========================== #
def detect_task(model, adapter, device, probe=None, labels=None, override=None, tasks=None):
    """Hibridna detekcija po ljestvici. Signal A = oznake (iz `probe` ako je dan, inace `labels`) + probe.task_hint;
    signal B = arhitektura (+ oblik izlaza koji OGRANICAVA A). Vrati dict: task, source, why, metrics, kd_core, ...

    `probe` = rezultat dataset.probe_dataset (nosi `_labels` uzorak + `task_hint`) -> jedan izvor oznaka, bez glue-a.
    """
    tasks = tasks if tasks is not None else load_tasks()
    T = tasks["tasks"]
    data_hint = None
    if probe is not None:                                      # dataset_probe je vec izvukao oznake + data-hint
        labels = probe.get("_labels", labels)
        data_hint = probe.get("task_hint")

    if override:
        task, source, why = override, "override", f"rucni override = {override}"
    else:
        a, wa = _label_signature(labels)
        if a is None and data_hint not in (None, "unknown", "?"):
            a, wa = data_hint, f"probe.task_hint={data_hint}"   # ojacaj A format-hintom iz probe-a
        b, shape, wb = _arch_signature(model, adapter, device)

        # B OGRANICAVA A: strukturirani task (detection/seg) je NEMOGUC na flat izlazu -> reinterpretiraj
        if a in ("detection", "segmentation") and shape == "flat":
            a2 = _recast_flat(labels)
            task, source, why = a2, "A|B-shape", f"A={a} nemoguc na flat izlazu -> {a2} | {wa} | {wb}"
        elif a and b:
            if a == b:
                task, source, why = a, "A∩B", f"{wa} | {wb}"
            else:
                task, source, why = a, "A", f"A={a} vs B={b} (vjeruj oznakama) | {wa} | {wb}"
        elif a:
            task, source, why = a, "A", f"{wa} | {wb}"
        elif b:
            task, source, why = b, "B", wb
        else:
            task, source, why = "unknown", "-", f"ni A ni B | {wa} | {wb}"

    spec = T.get(task, T["unknown"])
    return {"task": task, "source": source, "why": why,
            "metrics": spec["metrics"], "kd_core": spec["kd_core"],
            "enhancers": spec["enhancers"], "decode": spec["decode"]}
