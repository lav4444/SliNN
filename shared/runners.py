# -*- coding: utf-8 -*-
"""runners.py — omotac oko ONNX Runtimea koji se izvana ponasa kao torch modul.

ZASTO OMOTAC, A NE NOVE EVAL SKRIPTE: mjeri se razlika izmedju dva RUNTIMEA, a ne izmedju
dva mjerna aparata. Ako se uz runtime promijeni i ucitavanje podataka, predobrada, prag ili
NMS, izmjerena razlika vise nije pripisiva. Zato eval skripta u BASELINE_OPTIM ostaje ista
kao u BASELINE_RAW, uz jednu zamijenjenu liniju: umjesto `torch.load(...)` stoji
`load_optim(...)`. Sve ostalo je bajt u bajt isto.

Omotac zato mora podnijeti sve nacine na koje ga postojece skripte zovu:
    model(x)              housing, m5, midas, yolo   -> tenzor
    model(x)["out"]       voc  (DeepLabV3 vraca dict)
    model(**enc).logits   distilbert (HF vraca objekt)
    model.eval()          svugdje u _median_ms / _per_sample
    model.to(dev)         yolo benchmark
Stil izlaza NE pogadja se ovdje — cita se iz model_onnx.json koji zapise export.py.

NITI: namjerno se NE postavlja intra_op_num_threads. ORT po zadanom uzme sve jezgre, kao
i eager torch u BASELINE_RAW. Ogranicavanje bi mjerilo raspored, ne runtime.

Isti modul koristi i SLINN_OPTIM — zato je u shared/, a ne u mjernoj mapi.
"""
import json
import os

import numpy as np
import torch

_SESS_OPTS = None


def _opts():
    global _SESS_OPTS
    if _SESS_OPTS is None:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.log_severity_level = 3
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESS_OPTS = o
    return _SESS_OPTS


class _Logits(object):
    """Minimalni nadomjestak za HF izlaz: eval koristi samo `.logits`."""

    __slots__ = ("logits",)

    def __init__(self, t):
        self.logits = t


class OrtShim(object):
    """ONNX Runtime sesija koja se poziva kao torch modul."""

    def __init__(self, onnx_path, meta):
        import onnxruntime as ort
        self.path = onnx_path
        self.in_names = meta["input_names"]
        self.style = meta["out_style"]
        self.sess = ort.InferenceSession(onnx_path, _opts(),
                                         providers=["CPUExecutionProvider"])
        # Stvarna imena iz grafa su izvor istine; json je samo redoslijed.
        graph = [i.name for i in self.sess.get_inputs()]
        if sorted(graph) != sorted(self.in_names):
            raise RuntimeError(f"imena ulaza se ne slazu: json {self.in_names} vs graf {graph}")

    # --- suceljem prema torchu ------------------------------------------------
    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def to(self, *_a, **_k):
        return self          # ORT sesija je vezana uz CPU; premjestanje nema smisla

    def parameters(self):
        return iter(())      # da `next(model.parameters()).device` ne pukne tiho

    # --- poziv ----------------------------------------------------------------
    def __call__(self, *args, **kw):
        if kw:
            feed = {n: _np(kw[n]) for n in self.in_names}
        else:
            feed = {n: _np(t) for n, t in zip(self.in_names, args)}
        out = torch.from_numpy(self.sess.run(None, feed)[0])
        if self.style == "dict_out":
            return {"out": out}
        if self.style == "logits":
            return _Logits(out)
        return out

    def __repr__(self):
        return f"OrtShim({os.path.basename(os.path.dirname(self.path))}, {self.style})"


def _np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def load_optim(model_dir, dev=None):
    """Ucitaj optimizirani model iz <model_dir>/model.onnx.

    `dev` se prima samo zato da zamjena u eval skripti bude jednoredna; ORT ovdje
    ide na CPU jer je to jedini uredjaj na Pi-ju.
    """
    onnx_path = os.path.join(model_dir, "model.onnx")
    meta_path = os.path.join(model_dir, "model_onnx.json")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"nema {onnx_path} — izvoz nije prosao. Pokreni shared/export.py i pogledaj "
            f"export_log_*.txt. (Namjerno se ne pada natrag na .pt: tiho eager mjerenje "
            f"u OPTIM mapi bilo bi gore od pada.)")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return OrtShim(onnx_path, meta)


def runtime_name():
    """Vrijednost za CSV stupac `runtime`."""
    return "onnxruntime"
