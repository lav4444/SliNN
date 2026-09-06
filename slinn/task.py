import json
import os

import torch

TASKS_PATH = os.path.join(os.path.dirname(__file__), "SUPPORTED_TASKS.json")

_DET_MARKERS = ("detect", "anchorgenerator", "roihead", "roialign", "rpnhead", "regionproposal")


def load_tasks(path=TASKS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor) and x[0].dim() == 0:
            return torch.stack(list(x))
        t = torch.as_tensor(x)
        return t if t.dtype != torch.object else None
    except BaseException:
        return None


def _is_boxes(x):
    t = _as_tensor(x)
    return t is not None and t.dim() == 2 and t.shape[-1] >= 4 and t.is_floating_point()


def _first_tensor(o):
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


def _label_signature(labels):
    if labels is None:
        return None, "nema oznaka"

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

    if t.dim() >= 3:
        return ("segmentation" if is_int else "regression"), \
            f"oznake: dense {tuple(t.shape)} {'int(klase)' if is_int else 'float(depth)'}"
    if t.dim() == 2:
        _, K = t.shape
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


def _arch_signature(model, adapter, device):
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
    try:
        if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
            maxc = max((int(t[:, 0].unique().numel()) if t.numel() else 0) for t in labels)
            return "multilabel" if maxc > 1 else "classification"
    except BaseException:
        pass
    return "classification"


def detect_task(model, adapter, device, probe=None, labels=None, override=None, tasks=None):
    tasks = tasks if tasks is not None else load_tasks()
    T = tasks["tasks"]
    data_hint = None
    if probe is not None:
        labels = probe.get("_labels", labels)
        data_hint = probe.get("task_hint")

    if override:
        task, source, why = override, "override", f"rucni override = {override}"
    else:
        a, wa = _label_signature(labels)
        if a is None and data_hint not in (None, "unknown", "?"):
            a, wa = data_hint, f"probe.task_hint={data_hint}"
        b, shape, wb = _arch_signature(model, adapter, device)

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
