"""
ptq_distilbert.py — PTQ za DistilBERT SST-2 (sentiment, 2 klase). PRVI NE-KONVOLUCIJSKI model u setu.

Formati × uređaj:
  FP32               (PyTorch, CPU + GPU)
  FP16               (PyTorch, GPU pravo; CPU samo demonstrativno — x86 nema fp16 compute put)
  INT8-PT dynamic    (PyTorch, CPU) — KANONSKI BERT put: kvantizira nn.Linear po batchu, bez kalibracije
  INT8-PT static FX  (PyTorch, CPU) — pokušaj; HF modeli nisu torch.fx traceable (treba transformers.utils.fx)
  INT8-ORT dynamic   (ONNX Runtime, CPU) — drugi int8 put; opcionalno (TRY_ORT)

ZAŠTO OVAJ MODEL: fasterrcnn je pokazao da INT8-PT-dynamic zna biti SPORIJI od FP32 (24.17 vs 20.98 ms)
jer ondje `Linear` čine tek ROI/RPN glave. DistilBERT je gotovo isključivo `Linear`, pa ista tehnika
mora dati suprotan ishod. Time se frcnn-ov negativan rezultat pretvara iz rupe u tablici u pravilo:
ISPLATIVOST DINAMIČKE KVANTIZACIJE ODREĐUJE UDIO `Linear` SLOJEVA. Usput, SliNN u ovom modelu nalazi
NULA rezivih slojeva (svi spregnuti s feature-KD tapovima) -> kvantizacija mu je jedini put kompresije.

Kvaliteta: accuracy + macro/micro F1 + precision/recall + AUROC + cross-entropy, na PUNOM validation
splitu (test oznake su skrivene, -1). Referenca (eval_result.txt): acc 0.9106, macro-F1 0.9104.
Latencija: quant_common.benchmark (warmup + medijan + p90), batch=1, FIKSNA duljina MAX_LEN.
  -> `padding="max_length"`, ne `padding=True`: inače batch=1 daje duljinu te rečenice pa formati
     ne rade isti posao i latencije nisu usporedive.
Reuse: baseline_models/sst2_distilbert (data.py) + quant_common. Ništa se izvan ovog foldera ne mijenja.
BEZ CLI argumenata — konstante niže; __main__ pokreće sve.
"""

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(os.path.dirname(_HERE))                       # .../quantization
_MODEL_DIR = os.path.join(os.path.dirname(_QROOT), "baseline_models", "sst2_distilbert")
sys.path.insert(0, _QROOT)
sys.path.insert(0, _MODEL_DIR)

import quant_common as Q                                              # noqa: E402
import data as D                                                      # noqa: E402  (sst2: loader/tokenizer/classes)

# ============================ POSTAVKE ============================ #
MODEL_PT    = os.path.join(_MODEL_DIR, "model.pt")
QENGINE     = "x86"           # CPU int8 backend (oneDNN/fbgemm; VNNI ako ga CPU ima)
CPU_THREADS = 8               # FIKSNO radi usporedivosti CPU latencije
LAT_BATCH   = 1               # edge scenarij: jedna rečenica
LAT_WARMUP  = 15
LAT_ITERS   = 100
LAT_WARMUP_SLOW = 2           # za NAMJERNO spore ćelije (CPU fp16): 100 iteracija bi trajalo desecima minuta
LAT_ITERS_SLOW  = 5
EVAL_BATCH  = 64
EVAL_SPLIT  = "validation"    # test oznake su skrivene (-1)
EVAL_LIMIT  = None            # None = pun split (872)
TRY_ORT     = True            # pokušaj i ONNX Runtime dynamic INT8

OUT_CSV  = os.path.join(_HERE, "distilbert_ptq_report.csv")
OUT_JSON = os.path.join(_HERE, "distilbert_ptq_report.json")
ONNX_FP32 = os.path.join(_HERE, "distilbert_fp32.onnx")
ONNX_INT8 = os.path.join(_HERE, "distilbert_int8_ort.onnx")

CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb",
            "acc", "f1_macro", "f1_micro", "prec_macro", "recall_macro", "auroc", "ce"]


def load_fp32():
    """Puni eager modul (spremljen torch.save(model)); uvijek s CPU-a pa se kopira po potrebi."""
    return torch.load(MODEL_PT, map_location="cpu", weights_only=False).eval()


def fixed_encoding(device=None, batch=LAT_BATCH):
    """FIKSNA duljina MAX_LEN -> svi formati rade identičan posao (usporediva latencija)."""
    tok = D.tokenizer()
    text = ["a fixed calibration sentence for latency measurement ."] * batch
    enc = tok(text, padding="max_length", truncation=True, max_length=D.MAX_LEN, return_tensors="pt")
    enc = dict(enc)
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


@torch.no_grad()
def eval_panel(infer, loader, names):
    """infer(enc_cpu) -> logiti [B,K] (bilo koji uređaj/dtype). Akumulira pa računa panel na CPU/fp32."""
    logits, tgts = [], []
    for enc, y in loader:
        out = infer(enc)
        logits.append(out.float().cpu())
        tgts.append(y)
    L = torch.cat(logits); T = torch.cat(tgts)
    P = torch.softmax(L, dim=1).numpy()
    yt = T.numpy(); yp = P.argmax(1)
    panel = {
        "acc":          float(accuracy_score(yt, yp)),
        "f1_macro":     float(f1_score(yt, yp, average="macro")),
        "f1_micro":     float(f1_score(yt, yp, average="micro")),
        "prec_macro":   float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "auroc":        float(roc_auc_score(yt, P[:, 1])) if len(set(yt.tolist())) > 1 else float("nan"),
        "ce":           float(log_loss(yt, P, labels=list(range(len(names))))),
    }
    panel["per_class_recall"] = {
        nm: (float((yp[yt == k] == k).mean()) if (yt == k).any() else float("nan"))
        for k, nm in enumerate(names)}
    return panel


def make_infer(model, device, half=False):
    def infer(enc):
        e = {k: v.to(device) for k, v in enc.items()}
        out = model(**e)                                   # HF SequenceClassifierOutput
        return out.logits if hasattr(out, "logits") else out
    _ = half                                               # ulazi su ID-jevi (long) -> nikad se ne castaju
    return infer


def lat(model, device, enc, warmup=LAT_WARMUP, iters=LAT_ITERS):
    e = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return Q.benchmark(lambda: model(**e), device, warmup, iters)


def round_ms(r):
    return round(r["median_ms"], 4) if isinstance(r, dict) else r


def nan_panel(names):
    p = {k: float("nan") for k in CSV_COLS[5:]}
    p["per_class_recall"] = {nm: float("nan") for nm in names}
    return p


# --------------------------------------------------------------------------- ONNX Runtime dynamic
def ort_dynamic(model, enc, loader, names):
    """Izvezi FP32 ONNX -> ORT quantize_dynamic (QInt8 na MatMul/Gemm) -> mAP panel + latencija.
    Vrati (panel, size_mb, lat_dict) ili podigne iznimku (pozivatelj je pretvara u N/A s razlogom)."""
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    m = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(
        m, (enc["input_ids"], enc["attention_mask"]), ONNX_FP32, opset_version=17,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "logits": {0: "b"}})
    quantize_dynamic(ONNX_FP32, ONNX_INT8, weight_type=QuantType.QInt8)

    so = ort.SessionOptions()
    so.intra_op_num_threads = CPU_THREADS
    sess = ort.InferenceSession(ONNX_INT8, so, providers=["CPUExecutionProvider"])

    def infer(e):
        out = sess.run(["logits"], {"input_ids": e["input_ids"].cpu().numpy(),
                                    "attention_mask": e["attention_mask"].cpu().numpy()})[0]
        return torch.from_numpy(out)

    panel = eval_panel(infer, loader, names)
    feed = {"input_ids": enc["input_ids"].cpu().numpy(),
            "attention_mask": enc["attention_mask"].cpu().numpy()}
    lt = Q.benchmark(lambda: sess.run(["logits"], feed), "cpu", LAT_WARMUP, LAT_ITERS)
    return panel, Q.file_size_mb(ONNX_INT8), lt


def main():
    Q.set_cpu_threads(CPU_THREADS)
    torch.backends.quantized.engine = QENGINE
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu"); gpu = torch.device("cuda") if has_cuda else None

    names = D.classes()
    loader = D.loader(EVAL_SPLIT, EVAL_BATCH, limit=EVAL_LIMIT)
    enc = fixed_encoding()                                  # CPU, fiksna duljina MAX_LEN
    model_fp32 = load_fp32()
    rows, lat_full = [], {}

    print(f"\n########## PTQ — DistilBERT SST-2 (engine={QENGINE}, CPU niti={CPU_THREADS}, "
          f"lat batch={LAT_BATCH}, len={D.MAX_LEN}) ##########")

    def add(fmt, backend, panel, size_mb, cpu_lat, gpu_lat):
        row = {"format": fmt, "backend": backend,
               "size_mb": round(size_mb, 4) if isinstance(size_mb, (int, float)) else size_mb,
               "cpu_ms": round_ms(cpu_lat), "gpu_ms": round_ms(gpu_lat)}
        for k in CSV_COLS[5:]:
            v = panel.get(k, float("nan"))
            row[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        row["per_class_recall"] = panel.get("per_class_recall")
        rows.append(row)
        lat_full[fmt] = {"cpu": cpu_lat, "gpu": gpu_lat}
        print(f"  [{fmt:18}] acc={row['acc']} f1M={row['f1_macro']} | {row['size_mb']} MB | "
              f"CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    # -------- FP32 (CPU + GPU) --------
    m32 = copy.deepcopy(model_fp32).to(cpu).eval()
    panel = eval_panel(make_infer(m32, cpu), loader, names)
    cpu_lat = lat(m32, cpu, enc)
    gpu_lat = Q.na("nema CUDA")
    if has_cuda:
        m32g = copy.deepcopy(model_fp32).to(gpu).eval()
        gpu_lat = lat(m32g, gpu, enc)
        del m32g; torch.cuda.empty_cache()
    add("FP32", "PyTorch", panel, Q.model_size_mb(m32), cpu_lat, gpu_lat)

    # -------- FP16 (GPU pravo; CPU demonstrativno) --------
    if has_cuda:
        m16 = copy.deepcopy(model_fp32).to(gpu).half().eval()
        panel16 = eval_panel(make_infer(m16, gpu), loader, names)
        gpu_lat = lat(m16, gpu, enc)
        del m16; torch.cuda.empty_cache()
    else:
        panel16, gpu_lat = nan_panel(names), Q.na("nema CUDA")
    try:                                                    # CPU fp16: namjerno — pokazuje nedostatak HW puta
        m16c = copy.deepcopy(model_fp32).to(cpu).half().eval()
        cpu_lat = lat(m16c, cpu, enc, LAT_WARMUP_SLOW, LAT_ITERS_SLOW)
        del m16c
    except (RuntimeError, NotImplementedError) as e:
        cpu_lat = Q.na(f"CPU half: {type(e).__name__}")
    add("FP16", "PyTorch", panel16, Q.model_size_mb(copy.deepcopy(model_fp32).half()), cpu_lat, gpu_lat)

    # -------- INT8-PT dynamic (CPU) — GLAVNI REZULTAT --------
    try:
        m8d = tq.quantize_dynamic(copy.deepcopy(model_fp32).to(cpu).eval(), {nn.Linear}, dtype=torch.qint8)
        p8d = eval_panel(make_infer(m8d, cpu), loader, names)
        add("INT8-PT-dynamic", f"PyTorch {QENGINE}", p8d, Q.model_size_mb(m8d),
            lat(m8d, cpu, enc), Q.na("PyTorch quant je CPU-only"))
        del m8d
    except Exception as e:
        add("INT8-PT-dynamic", f"PyTorch {QENGINE}", nan_panel(names), Q.na("—"),
            Q.na(f"{type(e).__name__}: {str(e)[:40]}"), Q.na("CPU-only"))

    # -------- INT8-PT static FX (CPU) — očekivana granica --------
    # HF modeli nisu torch.fx symbolic-traceable (dinamični control flow, kwargs, dict izlazi);
    # standardni put je `transformers.utils.fx.symbolic_trace`, koji prepare_fx ne prihvaća izravno.
    try:
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        qmap = tq.get_default_qconfig_mapping(QENGINE)
        prep = prepare_fx(copy.deepcopy(model_fp32).to(cpu).eval(), qmap,
                          example_inputs=(enc["input_ids"], enc["attention_mask"]))
        with torch.no_grad():
            for e_, _ in D.loader("train", EVAL_BATCH, limit=512):
                prep(e_["input_ids"], e_["attention_mask"])
        m8s = convert_fx(prep).to(cpu).eval()
        p8s = eval_panel(lambda e_: m8s(e_["input_ids"], e_["attention_mask"]), loader, names)
        add("INT8-PT-static", f"PyTorch {QENGINE}", p8s, Q.model_size_mb(m8s),
            lat(m8s, cpu, enc), Q.na("PyTorch quant je CPU-only"))
    except Exception as e:
        add("INT8-PT-static", f"PyTorch {QENGINE}", nan_panel(names), Q.na("—"),
            Q.na(f"HF nije fx-traceable: {type(e).__name__}"), Q.na("CPU-only"))

    # -------- INT8-ORT dynamic (CPU) --------
    if TRY_ORT:
        try:
            p_ort, sz_ort, lt_ort = ort_dynamic(model_fp32, enc, loader, names)
            add("INT8-ORT-dynamic", "ONNX Runtime", p_ort, sz_ort, lt_ort, Q.na("CPUExecutionProvider"))
        except Exception as e:
            add("INT8-ORT-dynamic", "ONNX Runtime", nan_panel(names), Q.na("—"),
                Q.na(f"{type(e).__name__}: {str(e)[:40]}"), Q.na("CPU"))
    else:
        add("INT8-ORT-dynamic", "ONNX Runtime", nan_panel(names), Q.na("—"),
            Q.na("TRY_ORT=False"), Q.na("CPU"))

    # -------- INT8-TRT (GPU) — granica, po uzoru na frcnn --------
    rows.append({"format": "INT8-TRT", "backend": "TensorRT",
                 "cpu_ms": Q.na("TRT je GPU-only"),
                 "gpu_ms": Q.na("BERT->TRT traži zaseban trt_ skript (kao trt_schoolcnn.py)"),
                 "size_mb": Q.na("—"), **{k: Q.na("—") for k in CSV_COLS[5:]}})
    print("  [INT8-TRT          ] (zaseban TRT skript, po uzoru na trt_schoolcnn.py)")

    meta = {
        "model": "distilbert-base-uncased-finetuned-sst-2-english", "task": "text-classification (SST-2)",
        "weights": MODEL_PT, "num_classes": len(names), "max_len": D.MAX_LEN,
        "eval_split": EVAL_SPLIT, "eval_images": "full" if EVAL_LIMIT is None else EVAL_LIMIT,
        "quant_engine": QENGINE,
        "reference_eval_result": {"acc": 0.9106, "f1_macro": 0.9104,
                                  "gpu_ms": 3.007, "cpu_ms": 18.054},
        "conditions": {"cpu_threads": CPU_THREADS, "lat_batch": LAT_BATCH,
                       "lat_warmup": LAT_WARMUP, "lat_iters": LAT_ITERS,
                       "lat_slow_iters": LAT_ITERS_SLOW, "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                       "metric_note": "latencija = medijan; p90/min u 'lat_full'",
                       "padding": "max_length (fiksna duljina -> usporediva latencija)"},
        "note": ("Prvi ne-konvolucijski model u setu. INT8-PT-dynamic je ovdje kanonski put (model je "
                 "gotovo sav Linear) — suprotno fasterrcnn-u, gdje su Linear samo glave pa je dynamic bio sporiji. "
                 "CPU fp16 mjeren s manje iteracija jer je namjerno spor."),
        "lat_full": lat_full,
    }
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)
    print(f"\n########## DISTILBERT PTQ GOTOVO — {len(rows)} formata ##########")


if __name__ == "__main__":
    main()
