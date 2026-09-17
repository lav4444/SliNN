
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_QAT = os.path.dirname(_HERE)
_QROOT = os.path.dirname(_QAT)
sys.path.insert(0, _QAT)
sys.path.insert(0, _QROOT)
sys.path.insert(0, os.path.join(_QROOT, "PTQ", "schoolcnn"))

import qat_common as QC
import quant_common as Q
import ptq_schoolcnn as PT

C2 = PT.C2
NC = PT.NUM_CLASSES

CALIB_IMAGES = 512
QAT_STEPS    = 800
QAT_LR       = 1e-4
BATCH        = 32
EVAL_BATCH   = 32
EVAL_MAX     = None
LAT_WARMUP, LAT_ITERS = 15, 100
LAT_WARMUP_SLOW, LAT_ITERS_SLOW = 2, 5

OUT_DIR  = _HERE
ONNX_DIR = os.path.join(_HERE, "onnx")
OUT_CSV  = os.path.join(_HERE, "schoolcnn_qat_report.csv")
OUT_JSON = os.path.join(_HERE, "schoolcnn_qat_report.json")
CSV_COLS = PT.CSV_COLS


def main():
    Q.set_cpu_threads(PT.CPU_THREADS)
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if has_cuda else cpu
    t_all = time.time()

    val_loader = C2.make_loader("val", EVAL_BATCH, shuffle=False, num_workers=4, max_images=EVAL_MAX)
    calib_loader = C2.make_loader("train", BATCH, shuffle=False, num_workers=4, max_images=CALIB_IMAGES)
    train_loader = C2.make_loader("train", BATCH, shuffle=True, num_workers=4)
    example = next(iter(val_loader))[0][:1].clone()

    print(f"\n########## QAT — SchoolCNN (koraka {QAT_STEPS}, LR {QAT_LR}, kalib {CALIB_IMAGES}) ##########")

    teacher = PT.load_fp32().to(gpu).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = PT.load_fp32()
    n_wrapped = QC.wrap_model(student, skip=())
    print(f"  [wrap] {n_wrapped} modula omotano | fake-quant tocaka: {QC.n_fakequant(student)}")

    nb = QC.calibrate(student.to(gpu), calib_loader, gpu, forward_fn=lambda m, b: m(b[0]))
    print(f"  [calib] {nb} batcheva")

    def kd_logit(m, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        with torch.no_grad():
            t = torch.sigmoid(teacher(x))
        return F.binary_cross_entropy_with_logits(m(x), t)

    loss, steps = QC.qat_finetune(student, train_loader, kd_logit, steps=QAT_STEPS, lr=QAT_LR,
                                  device=gpu, on_step=lambda k, n, l: print(f"      {k}/{n}  kd={l:.4f}", flush=True))
    print(f"  [qat] {steps} koraka, prosjecni KD {loss:.4f}")

    student = student.cpu().eval()
    paths = QC.export_all(student, example, ONNX_DIR, "schoolcnn")
    ver = {}
    for mode, p in paths.items():
        if p:
            ver[mode] = QC.verify(p)
            print(f"  [onnx {mode:4}] {ver[mode]['mb']:.2f} MB | QDQ {ver[mode]['qdq']}/{ver[mode]['nodes']}"
                  f" | domene {ver[mode]['domains'] or 'std'}")

    rows, lat_full = [], {}

    def add(fmt, backend, panel, size_mb, cpu_lat, gpu_lat, extra=None):
        row = {"format": fmt, "backend": backend,
               "size_mb": round(size_mb, 4) if isinstance(size_mb, (int, float)) else size_mb,
               "cpu_ms": round(cpu_lat["median_ms"], 4) if isinstance(cpu_lat, dict) else cpu_lat,
               "gpu_ms": round(gpu_lat["median_ms"], 4) if isinstance(gpu_lat, dict) else gpu_lat}
        for k in CSV_COLS[5:]:
            v = (panel or {}).get(k)
            row[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        if panel:
            row["per_class_ap"] = panel.get("per_class_ap")
        if extra:
            row.update(extra)
        rows.append(row)
        lat_full[fmt] = {"cpu": cpu_lat, "gpu": gpu_lat}
        print(f"  [{fmt:16}] mAP={row.get('map_macro')} F1={row.get('f1_macro')} | "
              f"{row['size_mb']} MB | CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    ptq = {}
    if os.path.exists(PT.OUT_JSON):
        for r in json.load(open(PT.OUT_JSON))["rows"]:
            ptq[r["format"]] = r
    for tag in ("FP32", "INT8-PT-static"):
        if tag in ptq:
            r = dict(ptq[tag]); r["format"] = f"[PTQ] {tag}"
            rows.append(r)
            print(f"  [{r['format']:16}] mAP={r.get('map_macro')} (iz PTQ izvjestaja)")

    plain = QC.unwrap_model(student).eval()
    p32 = PT.eval_panel(PT.make_infer(plain.to(cpu), cpu), val_loader, NC)
    c32 = PT.lat(plain.to(cpu), cpu, example)
    g32 = PT.lat(plain.to(gpu), gpu, example) if has_cuda else Q.na("nema CUDA")
    add("QAT-FP32", "PyTorch", p32, Q.model_size_mb(plain.cpu()), c32, g32)

    if has_cuda:
        h = QC.unwrap_model(student).to(gpu).half().eval()
        p16 = PT.eval_panel(PT.make_infer(h, gpu, half=True), val_loader, NC)
        g16 = PT.lat(h, gpu, example, half=True)
        del h
        torch.cuda.empty_cache()
    else:
        p16, g16 = None, Q.na("nema CUDA")
    try:
        hc = QC.unwrap_model(student).to(cpu).half().eval()
        with torch.no_grad():
            c16 = Q.benchmark(lambda: hc(example.half()), cpu,
                              LAT_WARMUP_SLOW, LAT_ITERS_SLOW)
        del hc
    except Exception as e:
        c16 = Q.na(f"CPU half: {type(e).__name__}")
    add("QAT-FP16", "PyTorch", p16, Q.model_size_mb(QC.unwrap_model(student).half()), c16, g16)

    if paths.get("qdq"):
        try:
            ov = QC.OVRunner(paths["qdq"], threads=PT.CPU_THREADS)
            pov = PT.eval_panel(ov.infer, val_loader, NC)
            lov = Q.benchmark(ov.bench_call(example), "cpu", LAT_WARMUP, LAT_ITERS)
            add("QAT-INT8-OV", "OpenVINO CPU", pov, ver["qdq"]["mb"], lov, Q.na("OpenVINO CPU"))
        except Exception as e:
            add("QAT-INT8-OV", "OpenVINO CPU", None, Q.na("—"),
                Q.na(f"{type(e).__name__}: {str(e)[:50]}"), Q.na("—"))

    if has_cuda:
        for tag, src, kw in (("QAT-FP16-TRT", "fp16", dict(fp16=True)),
                             ("QAT-INT8-TRT", "qdq", dict(int8=True))):
            if not paths.get(src):
                continue
            try:
                eng = os.path.join(ONNX_DIR, f"schoolcnn_{tag.lower()}.engine")
                QC.trt_build(paths[src], eng, **kw)
                r = QC.TRTRunner(eng)
                hist = r.precision_histogram()
                pt_ = PT.eval_panel(r.infer, val_loader, NC)
                r.preload(example)
                lt = Q.benchmark(r.enqueue, "cuda", LAT_WARMUP, LAT_ITERS)
                add(tag, "TensorRT", pt_, Q.file_size_mb(eng), Q.na("TensorRT je GPU-only"), lt,
                    extra={"precision_layers": hist})
                print(f"        preciznost po sloju: {hist}")
                del r
                torch.cuda.empty_cache()
            except Exception as e:
                add(tag, "TensorRT", None, Q.na("—"), Q.na("TRT GPU-only"),
                    Q.na(f"{type(e).__name__}: {str(e)[:60]}"))

    meta = {
        "model": "SchoolCNN", "method": "QAT (rucni fake-quant -> QDQ ONNX)",
        "loss": "KD-logit prema zamrznutom originalu (bez GT)",
        "qat": {"steps": steps, "lr": QAT_LR, "batch": BATCH, "calib_images": CALIB_IMAGES,
                "wrapped_modules": n_wrapped, "device": str(gpu)},
        "onnx_verify": ver,
        "eval": "pun val skup" if EVAL_MAX is None else EVAL_MAX,
        "why_qat_fp32_row": ("kontrola koja izolira efekt SAMOG fine-tunea; usporedba s PTQ ide "
                             "razmak-protiv-razmaka (PTQ: orig->PTQ-INT8, QAT: QAT-FP32->QAT-INT8)"),
        "conditions": {"cpu_threads": PT.CPU_THREADS, "lat_batch": PT.LAT_BATCH,
                       "lat_warmup": LAT_WARMUP, "lat_iters": LAT_ITERS,
                       "lat_slow_iters": LAT_ITERS_SLOW, "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if has_cuda else None},
        "lat_full": lat_full,
        "minutes": round((time.time() - t_all) / 60.0, 2),
    }
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)

    def g(fmt, key="map_macro"):
        for r in rows:
            if r["format"] == fmt and isinstance(r.get(key), (int, float)):
                return r[key]
        return None

    o32, optq = g("[PTQ] FP32"), g("[PTQ] INT8-PT-static")
    q32, qint = g("QAT-FP32"), (g("QAT-INT8-TRT") or g("QAT-INT8-OV"))
    print("\n=== RAZMAK PROTIV RAZMAKA (macro mAP) ===")
    if o32 and optq:
        print(f"  PTQ:  {o32:.5f} -> {optq:.5f}   ({optq - o32:+.5f})")
    if q32 and qint:
        print(f"  QAT:  {q32:.5f} -> {qint:.5f}   ({qint - q32:+.5f})")
    if o32 and q32:
        print(f"  fine-tune sam po sebi: {o32:.5f} -> {q32:.5f} ({q32 - o32:+.5f})")
    print(f"\n########## QAT GOTOVO — {len(rows)} redaka, {meta['minutes']} min ##########")


if __name__ == "__main__":
    main()
