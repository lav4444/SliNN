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

# Ime engine datoteke i trenutno aktivan runtime (postavlja ih load_optim).
ENGINE_NAME = "model_fp16.engine"
_ACTIVE_RUNTIME = "onnxruntime"


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


# =========================================================================== TensorRT
# Na Jetsonu optimizirani runtime je TensorRT, ne ONNX Runtime: ORT bi ondje isao na CPU
# i mjerio bi nesto sto s akceleratorom nema veze. Sucelje je isto kao kod OrtShima —
# eval skripta se ne mijenja, mijenja se samo koga `load_optim` vrati.
#
# STREAM: namjerno se koristi TEKUCI (default) CUDA stream, ne zaseban. Eval skripte oko
# mjerenja zovu `torch.cuda.synchronize()`, sto sinkronizira default stream; da engine ide
# svojim streamom, ta bi sinkronizacija mjerila predaju posla umjesto izvrsavanja. Uz to
# `__call__` i sam sinkronizira prije nego vrati rezultat, pa je poziv blokirajuci i
# izmjereno vrijeme je stvarno vrijeme izvrsavanja.
#
# BATCH: engine se gradi za batch 1 i veci se batch obradjuje petljom. Nije stednja nego
# vjernost — latencija se u BASELINE_RAW mjeri na batch=1, a tocnost o velicini batcha ne
# ovisi. Time batch os nigdje ne mora biti dinamicna.


def _trt_dt(dt):
    import tensorrt as trt
    return {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8, trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64, trt.DataType.BOOL: torch.bool}.get(dt, torch.float32)


class TrtShim(object):
    """TensorRT engine koji se izvana ponasa kao torch modul."""

    def __init__(self, engine_path, meta):
        import tensorrt as trt
        if not torch.cuda.is_available():
            raise RuntimeError("TrtShim trazi CUDA-u, a torch.cuda.is_available() je False")
        self.trt = trt
        self.path = engine_path
        self.style = meta["out_style"]
        self.in_names = list(meta["input_names"])

        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        with open(engine_path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine se ne moze deserijalizirati: {engine_path}")
        self.ctx = self.engine.create_execution_context()

        self.inputs, self.output = [], None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inputs.append(n)
            elif self.output is None:
                self.output = n
        if self.output is None:
            raise RuntimeError("engine nema izlazni tenzor")
        # Graf je izvor istine; json samo daje redoslijed kojim eval predaje pozicijske args.
        if sorted(self.inputs) != sorted(self.in_names):
            raise RuntimeError(f"imena ulaza se ne slazu: json {self.in_names} vs engine {self.inputs}")

        self.in_dt = {n: _trt_dt(self.engine.get_tensor_dtype(n)) for n in self.inputs}
        self.out_dt = _trt_dt(self.engine.get_tensor_dtype(self.output))

    # --- suceljem prema torchu ------------------------------------------------
    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def to(self, *_a, **_k):
        # Engine je vezan uz GPU na kojem je gradjen. Premjestanje se prihvaca i ignorira
        # da eval skripta ostane nepromijenjena; CPU mjerenje kroz ovaj omotac NE postoji
        # i ne smije se tumaciti kao CPU brojka.
        return self

    def parameters(self):
        return iter(())

    # --- poziv ----------------------------------------------------------------
    def __call__(self, *args, **kw):
        if kw:
            feed = {n: kw[n] for n in self.in_names}
        else:
            feed = {n: t for n, t in zip(self.in_names, args)}

        first = feed[self.in_names[0]]
        batch = int(first.shape[0])
        if batch == 0:
            # Eager torch na praznom batchu vrati prazan izlaz umjesto da pukne, pa isto radi
            # i omotac — inace bi se runtime razlikovao i u rubnom slucaju, a ne samo u brzini.
            # Javlja se u mini-testu: `_per_sample` uzima 10 rezova iz batcha koji je
            # EVAL_LIMIT-om skracen na 5, pa zadnjih pet rezova ima nula redaka.
            empty = torch.empty((0,), dtype=torch.float32, device="cuda")
            if self.style == "dict_out":
                return {"out": empty}
            if self.style == "logits":
                return _Logits(empty)
            return empty
        outs = []
        for i in range(batch):
            hold = []                       # drzi reference dok engine radi
            for n in self.inputs:
                t = feed[n][i:i + 1]
                if not torch.is_tensor(t):
                    t = torch.as_tensor(t)
                t = t.to(device="cuda", dtype=self.in_dt[n]).contiguous()
                hold.append(t)
                self.ctx.set_input_shape(n, tuple(t.shape))
                self.ctx.set_tensor_address(n, int(t.data_ptr()))
            # Oblik izlaza je poznat tek kad su SVI ulazni oblici postavljeni.
            out = torch.empty(tuple(self.ctx.get_tensor_shape(self.output)),
                              dtype=self.out_dt, device="cuda")
            self.ctx.set_tensor_address(self.output, int(out.data_ptr()))
            if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError("execute_async_v3 nije uspio")
            torch.cuda.synchronize()
            outs.append(out.float())
            del hold

        y = outs[0] if batch == 1 else torch.cat(outs, 0)
        if self.style == "dict_out":
            return {"out": y}
        if self.style == "logits":
            return _Logits(y)
        return y

    def __repr__(self):
        return f"TrtShim({os.path.basename(os.path.dirname(self.path))}, {self.style})"


def load_optim(model_dir, dev=None):
    """Ucitaj optimizirani model: prvo TensorRT engine, ako ga nema onda ONNX Runtime.

    Redoslijed nije proizvoljan. Postoji li engine, on JE ono sto se u toj mjernoj celiji
    mjeri; ORT je put za uredjaje bez akceleratora (Pi). Pada natrag na eager `.pt` NEMA
    ni u jednoj grani — tiho eager mjerenje u OPTIM mapi bilo bi gore od glasnog pada.

    `dev` se prima samo da zamjena u eval skripti bude jednoredna.
    """
    global _ACTIVE_RUNTIME
    meta_path = os.path.join(model_dir, "model_onnx.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"nema {meta_path} — izvoz nije prosao. Pokreni shared/export.py.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    engine_path = os.path.join(model_dir, ENGINE_NAME)
    if os.path.isfile(engine_path):
        _ACTIVE_RUNTIME = "tensorrt"
        return TrtShim(engine_path, meta)

    onnx_path = os.path.join(model_dir, "model.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"nema ni {ENGINE_NAME} ni model.onnx u {model_dir} — izvoz/gradnja nisu prosli. "
            f"Pokreni shared/export.py pa shared/build_engines.sh i pogledaj logove.")
    _ACTIVE_RUNTIME = "onnxruntime"
    return OrtShim(onnx_path, meta)


def runtime_name():
    """Vrijednost za CSV stupac `runtime` — postavlja je `load_optim` prema tome sto je stvarno
    ucitano, a ne prema uredjaju. Prije prvog `load_optim` vraca zadano."""
    return _ACTIVE_RUNTIME
