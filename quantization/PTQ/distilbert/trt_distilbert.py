"""
trt_distilbert.py — TensorRT za DistilBERT SST-2, s EKSPLICITNOM kvantizacijom i PROVJEROM.

ZAŠTO JE PRVA VERZIJA BILA KRIVA (i zašto ovo izgleda ovako):
  Prva izvedba je samo postavila `BuilderFlag.FP16` / `BuilderFlag.INT8` i predala kalibrator. U TRT-u su
  ti flagovi DOPUŠTENJA, ne naredbe — builder po vlastitom mjerenju bira preciznost po sloju, i za male
  matrice (batch 1, 64 tokena) je ostao na FP32. Tri traga su to odala:
      engine 255.9 MB (= FP32)   |   INT8 sporiji od FP16   |   acc identična FP32 u 5. decimali
  Mjereno ubrzanje 3.10 -> 1.05 ms bilo je STVARNO, ali od fuzije kernela, ne od precizije.

ISPRAVAK — preciznost se nosi U GRAFU, ne u zastavici:
  FP32-TRT : referenca koja izolira koliko donosi SAMA graf-optimizacija (bez ikakve promjene precizije)
  FP16-TRT : ONNX izvezen iz .half() modela na CUDA -> težine su fp16 već u grafu
  INT8-TRT : QDQ graf (QuantizeLinear/DequantizeLinear) preko onnxruntime `quantize_static`
             s `QuantFormat.QDQ` + simetričnim int8 aktivacijama. TRT Q/DQ čvorove poštuje kao
             OBVEZUJUĆE. To je ujedno put na koji upućuje upozorenje "Superseded by explicit quantization".

PROVJERA (novo, i najvažnije):
  Svaki engine se nakon gradnje ispituje `EngineInspector`-om (profiling verbosity = DETAILED) i u
  izvještaj ide HISTOGRAM PRECIZNOSTI PO SLOJEVIMA + veličina enginea. Ako engine tvrdi INT8 a histogram
  kaže FP32, to se VIDI u tablici umjesto da se provuče kao lažni rezultat.

Duljina je fiksna (padding='max_length'), ista kao u ptq_distilbert -> latencije izravno usporedive.
Device-buferi su torch CUDA tenzori (data_ptr) -> bez cuda-python.
Pokrenuti NAKON ptq_distilbert.py. BEZ CLI argumenata.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ptq_distilbert as PT                     # reuse: load_fp32, D, Q, eval_panel, CSV_COLS, OUT_*, konstante

Q = PT.Q
D = PT.D
LEN = D.MAX_LEN

ONNX_FP32   = os.path.join(_HERE, "distilbert_trt_fp32.onnx")
ONNX_FP16   = os.path.join(_HERE, "distilbert_trt_fp16.onnx")
ONNX_PRE    = os.path.join(_HERE, "distilbert_trt_pre.onnx")
ONNX_QDQ    = os.path.join(_HERE, "distilbert_trt_qdq.onnx")
ENGINE      = {"FP32-TRT": os.path.join(_HERE, "distilbert_fp32.engine"),
               "FP16-TRT": os.path.join(_HERE, "distilbert_fp16.engine"),
               "INT8-TRT": os.path.join(_HERE, "distilbert_int8.engine")}
TRT_CALIB_SENTENCES = 256
TRT_WORKSPACE_GB = 4

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class LogitsOnly(nn.Module):
    """HF vraća SequenceClassifierOutput; ONNX/TRT put treba JEDAN tenzor i pozicijske argumente."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(path, half=False):
    """half=True -> model se prebaci na CUDA i .half() PRIJE izvoza, pa su tezine fp16 U GRAFU.
    (CPU nema fp16 kernele za vecinu operacija, zato CUDA.)"""
    dev = "cuda" if half else "cpu"
    m = LogitsOnly(PT.load_fp32()).eval().to(dev)
    if half:
        m = m.half()
    ids = torch.ones(1, LEN, dtype=torch.long, device=dev)
    att = torch.ones(1, LEN, dtype=torch.long, device=dev)
    torch.onnx.export(m, (ids, att), path, opset_version=17,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes=None)                       # FIKSNI oblici -> TRT bez profila
    del m
    torch.cuda.empty_cache()
    print(f"  [onnx{'-fp16' if half else ''}] {path}  ({os.path.getsize(path)/1024**2:.2f} MB)")


def calib_pairs(n):
    """n rečenica iz train splita, nadopunjeno na FIKSNU duljinu LEN -> [(ids, att), ...] int64 np."""
    out = []
    for enc, _y in D.loader("train", 32, limit=n):
        ids, att = enc["input_ids"], enc["attention_mask"]
        pad = LEN - ids.shape[1]
        if pad > 0:
            ids = torch.nn.functional.pad(ids, (0, pad), value=0)
            att = torch.nn.functional.pad(att, (0, pad), value=0)
        else:
            ids, att = ids[:, :LEN], att[:, :LEN]
        for i in range(ids.shape[0]):
            out.append((ids[i:i + 1].numpy().astype(np.int64),
                        att[i:i + 1].numpy().astype(np.int64)))
            if len(out) >= n:
                return out
    return out


def make_qdq(pairs):
    """FP32 ONNX -> QDQ ONNX (eksplicitna kvantizacija). TRT Q/DQ cvorove tretira kao OBVEZUJUCE.
    Simetricne int8 aktivacije: TRT ne podrzava asimetricne (uint8 zero-point) na svom int8 putu."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, ps):
            self.it = iter([{"input_ids": i, "attention_mask": a} for i, a in ps])

        def get_next(self):
            return next(self.it, None)

    # per_channel=False i SAMO MatMul — izmjereno zasto:
    #   per_channel=True je kvantizirao i 0-dimenzionalnu konstantu (skalar 1/sqrt(d_k) u paznji)
    #   po osi 1, pa je TRT parser pao:
    #     "DequantizeLinear (convertAxis): Axis must be in the range [0, nbDims (0)]. Provided axis is: 1"
    #     node: layer.4/attention/Constant_4_output_0_DequantizeLinear
    #   Ogranicenje na MatMul drzi Q/DQ ondje gdje int8 uopce donosi brzinu (BERT je sav MatMul) i
    #   ne dira skalarne konstante; per-tensor skale su usto ono sto TRT-ov int8 put najbolje podnosi.
    quant_pre_process(ONNX_FP32, ONNX_PRE, skip_symbolic_shape=True)
    quantize_static(ONNX_PRE, ONNX_QDQ, Reader(pairs),
                    quant_format=QuantFormat.QDQ, per_channel=False,
                    op_types_to_quantize=["MatMul"],
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    print(f"  [qdq] {ONNX_QDQ}  ({os.path.getsize(ONNX_QDQ)/1024**2:.2f} MB)")


def build_engine(onnx_path, engine_path, fp16=False, int8=False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed: " + " | ".join(errs[:3]))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TRT_WORKSPACE_GB * (1 << 30))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED   # nuzno za inspektora precizosti
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print(f"   [fp16] eksplicitna preciznost na {force_fp16(network)} slojeva")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)                      # fallback za ne-int8 slojeve
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network vratio None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  [engine] {engine_path}  ({os.path.getsize(engine_path)/1024**2:.2f} MB)")


def precision_histogram(engine):
    """Sto je TRT STVARNO izabrao. Broji dtype IZLAZA svakog sloja.

    Inspector NEMA kljuc "Precision" (prva verzija ga je trazila i dobivala {'?': 64}). Slojevi imaju
    ['Name','LayerType','Inputs','Outputs','TacticName','StreamId','Metadata'], a stvarni format je u
    Outputs[i]["Format/Datatype"] ("Float" | "Half" | "Int8" | "Int64" | ...). TRT 10 usto fuzionira
    slojeve u 'kgen' (myelin) kernele, pa je granulacija gruba — ali za pitanje "je li ista postalo
    fp16/int8" je dovoljna."""
    try:
        insp = engine.create_engine_inspector()
        info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
        hist = {}
        for l in info.get("Layers", []):
            if not isinstance(l, dict):
                continue
            for o in l.get("Outputs", []) or []:
                dt = (o or {}).get("Format/Datatype", "?")
                dt = str(dt).split("(")[0].strip() or "?"
                hist[dt] = hist.get(dt, 0) + 1
        return hist or {"?": 0}
    except Exception as e:
        return {"inspector_failed": f"{type(e).__name__}"}


def force_fp16(network):
    """TRT bira preciznost po VLASTITOM mjerenju vremena; ako fp16 kernel nije brzi, ostane fp32 —
    tocno se to dogodilo (engine 255.8 MB = FP32, izlaz prvog sloja 'Float'). Zato se preciznost
    trazi EKSPLICITNO po sloju, uz PREFER_PRECISION_CONSTRAINTS (blazi od OBEY: ako sloj nema fp16
    izvedbu, build ne pukne nego se vrati na fp32 — a histogram ce to pokazati).
    Diraju se SAMO slojevi ciji su svi ulazi i izlazi float (ne shape/int put)."""
    n = 0
    for i in range(network.num_layers):
        l = network.get_layer(i)
        try:
            ins = [l.get_input(j) for j in range(l.num_inputs)]
            outs = [l.get_output(j) for j in range(l.num_outputs)]
            if not outs or any(t is None for t in outs):
                continue
            if any(t is not None and t.dtype not in (trt.float32, trt.float16) for t in ins):
                continue
            if any(t.dtype not in (trt.float32, trt.float16) for t in outs):
                continue
            l.precision = trt.float16
            for j in range(l.num_outputs):
                l.set_output_type(j, trt.float16)
            n += 1
        except Exception:
            continue
    return n


def _trt_dtype_to_torch(dt):
    return {trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8}.get(dt, torch.float32)


class TRTRunner:
    """Batch-1, dva cjelobrojna ulaza + jedan izlaz; torch CUDA buferi + NE-DEFAULT stream."""
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.torch_stream = torch.cuda.Stream()                 # TRT upozorava na default stream
        self.stream = self.torch_stream.cuda_stream
        self.inp = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(n))
            dt = _trt_dtype_to_torch(self.engine.get_tensor_dtype(n))
            t = torch.empty(shp, dtype=dt, device="cuda")
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp[n] = t
            else:
                self.out_name, self.out_t = n, t
            self.ctx.set_tensor_address(n, int(t.data_ptr()))

    def _enqueue(self):
        self.ctx.execute_async_v3(self.stream)

    def infer(self, enc):                                        # dict [B,LEN] -> [B,2] cpu float
        ids, att = enc["input_ids"], enc["attention_mask"]
        outs = []
        for i in range(ids.shape[0]):
            self.inp["input_ids"].copy_(ids[i:i + 1].to(self.inp["input_ids"].dtype))
            self.inp["attention_mask"].copy_(att[i:i + 1].to(self.inp["attention_mask"].dtype))
            self._enqueue()
            self.torch_stream.synchronize()
            outs.append(self.out_t.detach().float().cpu().clone())
        return torch.cat(outs, 0)

    def preload_for_latency(self, enc):
        self.inp["input_ids"].copy_(enc["input_ids"][:1].to(self.inp["input_ids"].dtype))
        self.inp["attention_mask"].copy_(enc["attention_mask"][:1].to(self.inp["attention_mask"].dtype))
        torch.cuda.synchronize()


def pad_loader(split, batch, limit=None):
    """Isti loader, ali svaki batch nadopunjen na FIKSNU duljinu LEN (engine ima fiksni oblik)."""
    for enc, y in D.loader(split, batch, limit=limit):
        ids, att = enc["input_ids"], enc["attention_mask"]
        pad = LEN - ids.shape[1]
        if pad > 0:
            ids = torch.nn.functional.pad(ids, (0, pad), value=0)
            att = torch.nn.functional.pad(att, (0, pad), value=0)
        else:
            ids, att = ids[:, :LEN], att[:, :LEN]
        yield {"input_ids": ids, "attention_mask": att}, y


def main():
    if not os.path.exists(PT.OUT_JSON):
        print(f"Nema {PT.OUT_JSON} — prvo pokreni ptq_distilbert.py"); return
    if not torch.cuda.is_available():
        print("Nema CUDA — TensorRT nije moguć."); return
    Q.set_cpu_threads(PT.CPU_THREADS)

    names = D.classes()
    enc = PT.fixed_encoding()
    print(f"\n########## TensorRT — DistilBERT SST-2 (TRT {trt.__version__}, ulaz 1x{LEN}) ##########")

    export_onnx(ONNX_FP32, half=False)
    export_onnx(ONNX_FP16, half=True)

    pairs = calib_pairs(TRT_CALIB_SENTENCES)
    print(f"  [calib] {len(pairs)} rečenica")
    qdq_ok, qdq_err = True, None
    try:
        make_qdq(pairs)
    except Exception as e:
        qdq_ok, qdq_err = False, f"{type(e).__name__}: {str(e)[:90]}"
        print(f"  [qdq] PAO -> {qdq_err}")

    plan = [("FP32-TRT", ONNX_FP32, dict()),                     # izolira dobitak od SAME fuzije
            ("FP16-TRT", ONNX_FP16, dict(fp16=True)),
            ("INT8-TRT", ONNX_QDQ if qdq_ok else None, dict(int8=True))]

    results = {}
    for tag, onnx_path, kw in plan:
        if onnx_path is None:
            results[tag] = {"error": f"QDQ graf nije izgrađen — {qdq_err}"}
            print(f"  [{tag}] preskočen -> {results[tag]['error']}")
            continue
        try:
            build_engine(onnx_path, ENGINE[tag], **kw)
            r = TRTRunner(ENGINE[tag])
            hist = precision_histogram(r.engine)
            panel = PT.eval_panel(r.infer, pad_loader(PT.EVAL_SPLIT, PT.EVAL_BATCH, PT.EVAL_LIMIT), names)
            r.preload_for_latency(enc)
            latr = Q.benchmark(r._enqueue, "cuda", PT.LAT_WARMUP, PT.LAT_ITERS)
            results[tag] = {"panel": panel, "lat": latr, "size_mb": Q.file_size_mb(ENGINE[tag]),
                            "precision_layers": hist}
            print(f"  [{tag}] acc={panel['acc']:.4f} f1M={panel['f1_macro']:.4f} | "
                  f"{results[tag]['size_mb']:.2f} MB | GPU {latr['median_ms']:.4f} ms")
            print(f"           preciznost po sloju: {hist}")
            del r
            torch.cuda.empty_cache()
        except Exception as e:
            results[tag] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
            print(f"  [{tag}] PAO -> {results[tag]['error']}")

    # ---- PROVJERA: je li se precizija stvarno spustila? ----
    base = results.get("FP32-TRT", {}).get("size_mb")
    print("\n  --- provjera (engine MB naspram FP32-TRT) ---")
    for tag in ("FP16-TRT", "INT8-TRT"):
        r = results.get(tag, {})
        if "size_mb" in r and base:
            ratio = base / r["size_mb"]
            ok = "OK" if ratio > 1.3 else "SUMNJIVO (tezine vjerojatno ostale FP32)"
            print(f"    {tag}: {r['size_mb']:.2f} MB vs {base:.2f} MB -> {ratio:.2f}x  {ok}")

    # ---- spoji u postojeći izvještaj ----
    rep = json.load(open(PT.OUT_JSON))
    rows = [row for row in rep["rows"] if row.get("format") not in results]
    for tag, _o, _k in plan:
        r = results.get(tag)
        if r is None:
            continue
        if "error" in r:
            rows.append({"format": tag, "backend": "TensorRT",
                         "cpu_ms": Q.na("TensorRT je GPU-only"), "gpu_ms": Q.na(r["error"]),
                         "size_mb": Q.na("—"), **{k: Q.na("—") for k in PT.CSV_COLS[5:]}})
            continue
        p = r["panel"]
        row = {"format": tag, "backend": "TensorRT", "cpu_ms": Q.na("TensorRT je GPU-only"),
               "gpu_ms": round(r["lat"]["median_ms"], 4), "size_mb": round(r["size_mb"], 4)}
        row.update({k: round(p[k], 5) for k in PT.CSV_COLS[5:]
                    if isinstance(p.get(k), float) and p[k] == p[k]})
        row["per_class_recall"] = p.get("per_class_recall")
        row["precision_layers"] = r["precision_layers"]
        rows.append(row)
        rep.setdefault("meta", {}).setdefault("lat_full", {})[tag] = r["lat"]

    rep["meta"]["tensorrt"] = {
        "version": trt.__version__, "calib_sentences": TRT_CALIB_SENTENCES, "input": f"1x{LEN}",
        "fp16_path": "ONNX izvezen iz .half() modela (tezine fp16 u grafu)",
        "int8_path": "QDQ graf (onnxruntime quantize_static, QuantFormat.QDQ, simetricni int8)",
        "why": ("TRT flagovi FP16/INT8 su DOPUSTENJA, ne naredbe — prva verzija je ostala na FP32 "
                "(engine 255.9 MB, INT8 sporiji od FP16, acc identicna FP32). Preciznost se zato nosi "
                "u GRAFU, a rezultat se provjerava histogramom precizosti po sloju + velicinom enginea."),
        "fp32_trt_row": "referenca koja izolira dobitak od SAME graf-optimizacije (bez promjene precizije)",
        "note": "ORT u ovoj okolini nema CUDA provider (CPU-only build), pa je TRT jedini GPU int8 put.",
    }
    Q.write_report(rows, PT.CSV_COLS, PT.OUT_CSV, PT.OUT_JSON, rep["meta"])
    print(f"\n########## TRT GOTOVO — tablica ažurirana ({len(rows)} formata) ##########")


if __name__ == "__main__":
    main()
