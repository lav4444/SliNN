
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_QAT = os.path.dirname(_HERE)
_QROOT = os.path.dirname(_QAT)
_ROOT = os.path.dirname(_QROOT)
_SLINN = os.path.join(_ROOT, "slinn")
for _p in (_QAT, _QROOT, _SLINN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qat_common as QC
import quant_common as Q
from plugins.detection import adapters as A

WEIGHTS      = os.path.join(_ROOT, "baseline_models", "yolo26n", "yolo26n.pt")
IMGSZ        = 640
CALIB_BATCH  = 4
CALIB_STEPS  = 32
QAT_STEPS    = 300
QAT_LR       = 1e-5
TRAIN_BATCH  = 4
EVAL_BATCH   = 4
EVAL_MAX     = None
CPU_THREADS  = 8
LAT_WARMUP, LAT_ITERS = 10, 50
LAT_WARMUP_SLOW, LAT_ITERS_SLOW = 2, 3

ONNX_DIR = os.path.join(_HERE, "onnx")
OUT_CSV  = os.path.join(_HERE, "yolo26n_qat_report.csv")
OUT_JSON = os.path.join(_HERE, "yolo26n_qat_report.json")
PTQ_JSON = os.path.join(_QROOT, "PTQ", "yolo26n", "yolo26n_ptq_report.json")
CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb", "map", "map50", "map75", "mar100"]


def load_fp32():
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)
    m = YOLO(WEIGHTS).model.float().eval()
    list(m.model)[-1].end2end = False
    return m


class _Shim(nn.Module):

    def __init__(self, runner, nc):
        super().__init__()
        head = nn.Module()
        head.nc = nc
        head.end2end = False
        self.model = nn.ModuleList([head])
        self._r = runner

    def forward(self, batch):
        return self._r.infer(batch).to(batch.device)


def _head_nc(model):
    return int(list(model.model)[-1].nc)


def eval_map(model, loader, device, adapter=A.YoloAdapter):
    m, _ = A.eval_map(model, adapter, loader, device, max_images=EVAL_MAX)
    f = lambda k, d=float("nan"): float(m.get(k, d))
    return {"map": f("map"), "map50": f("map_50", f("map50")),
            "map75": f("map_75", f("map75")), "mar100": f("mar_100")}


def main():
    Q.set_cpu_threads(CPU_THREADS)
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if has_cuda else cpu
    t_all = time.time()

    val_loader = A.make_gt_loader("val", bs=EVAL_BATCH)
    train_loader = A.make_gt_loader("train", bs=TRAIN_BATCH)
    example = torch.randn(1, 3, IMGSZ, IMGSZ)

    print(f"\n########## QAT — yolo26n (koraka {QAT_STEPS}, LR {QAT_LR}, imgsz {IMGSZ}) ##########")

    teacher = load_fp32().to(gpu).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    nc = _head_nc(teacher)

    rows, lat_full = [], {}

    def add(fmt, backend, panel, size_mb, cpu_lat, gpu_lat, extra=None):
        row = {"format": fmt, "backend": backend,
               "size_mb": round(size_mb, 4) if isinstance(size_mb, (int, float)) else size_mb,
               "cpu_ms": round(cpu_lat["median_ms"], 4) if isinstance(cpu_lat, dict) else cpu_lat,
               "gpu_ms": round(gpu_lat["median_ms"], 4) if isinstance(gpu_lat, dict) else gpu_lat}
        for k in CSV_COLS[5:]:
            v = (panel or {}).get(k)
            row[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        if extra:
            row.update(extra)
        rows.append(row)
        lat_full[fmt] = {"cpu": cpu_lat, "gpu": gpu_lat}
        print(f"  [{fmt:16}] mAP={row['map']} mAP50={row['map50']} | {row['size_mb']} MB | "
              f"CPU {row['cpu_ms']} | GPU {row['gpu_ms']}", flush=True)

    orig = load_fp32().to(gpu)
    p_orig = eval_map(orig, val_loader, gpu)
    g_orig = Q.benchmark(lambda: orig(example.to(gpu)), gpu, LAT_WARMUP, LAT_ITERS) if has_cuda else Q.na("nema CUDA")
    o_cpu = load_fp32().to(cpu)
    c_orig = Q.benchmark(lambda: o_cpu(example), cpu, 5, 10)
    add("[ORIG] FP32", "PyTorch", p_orig, Q.model_size_mb(o_cpu), c_orig, g_orig)
    del orig, o_cpu
    torch.cuda.empty_cache()

    student = load_fp32()
    last = f"model.{len(student.model) - 1}"
    n_wrapped = QC.wrap_model(student, skip=(f"{last}.",))
    print(f"  [wrap] {n_wrapped} modula omotano | fake-quant tocaka: {QC.n_fakequant(student)}")
    student = student.to(gpu)

    def _lb(imgs):
        return torch.stack([A.YoloAdapter._letterbox(im)[0] for im in imgs]).to(gpu)

    def _dense(o):
        if isinstance(o, dict):
            o = o.get("one2many", o.get("preds", next(iter(o.values()))))
        return o[0] if isinstance(o, (tuple, list)) else o

    calib_iter = []
    for i, (imgs, _t) in enumerate(train_loader):
        if i >= CALIB_STEPS:
            break
        calib_iter.append(_lb(imgs).cpu())
    QC.calibrate(student, calib_iter, gpu)
    print(f"  [calib] {len(calib_iter)} batcheva")

    def kd_logit(m, batch):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        lb = _lb(imgs) if isinstance(imgs, list) else imgs.to(gpu)
        m.eval()
        with torch.no_grad():
            t = _dense(teacher(lb)).detach()
        return F.mse_loss(_dense(m(lb)), t)

    loss, steps = QC.qat_finetune(student, train_loader, kd_logit, steps=QAT_STEPS, lr=QAT_LR,
                                  device=gpu, on_step=lambda k, n, l: print(f"      {k}/{n}  kd={l:.5f}", flush=True))
    print(f"  [qat] {steps} koraka, prosjecni KD {loss:.5f}")

    student = student.cpu().eval()
    list(student.model)[-1].end2end = False
    paths = QC.export_all(student, example, ONNX_DIR, "yolo26n")
    ver = {}
    for mode, p in paths.items():
        if p:
            ver[mode] = QC.verify(p)
            print(f"  [onnx {mode:4}] {ver[mode]['mb']:.2f} MB | QDQ {ver[mode]['qdq']}/{ver[mode]['nodes']}"
                  f" | domene {ver[mode]['domains'] or 'std'}")

    plain = QC.unwrap_model(student).eval()
    list(plain.model)[-1].end2end = False
    p32 = eval_map(plain.to(gpu), val_loader, gpu)
    g32 = Q.benchmark(lambda: plain(example.to(gpu)), gpu, LAT_WARMUP, LAT_ITERS) if has_cuda else Q.na("nema CUDA")
    pc = QC.unwrap_model(student).to(cpu).eval()
    list(pc.model)[-1].end2end = False
    c32 = Q.benchmark(lambda: pc(example), cpu, 5, 10)
    add("QAT-FP32", "PyTorch", p32, Q.model_size_mb(pc), c32, g32)
    del plain, pc
    torch.cuda.empty_cache()

    if has_cuda:
        h = QC.unwrap_model(student).to(gpu).half().eval()
        list(h.model)[-1].end2end = False

        class _Half(nn.Module):
            def __init__(self, m):
                super().__init__(); self.model = m.model; self.m = m
            def forward(self, b):
                return self.m(b.half())
        p16 = eval_map(_Half(h), val_loader, gpu)
        g16 = Q.benchmark(lambda: h(example.to(gpu).half()), gpu, LAT_WARMUP, LAT_ITERS)
        sz16 = Q.model_size_mb(QC.unwrap_model(student).half())
        del h
        torch.cuda.empty_cache()
    else:
        p16, g16, sz16 = None, Q.na("nema CUDA"), Q.na("—")
    add("QAT-FP16", "PyTorch", p16, sz16, Q.na("CPU fp16 bez HW puta"), g16)

    if paths.get("fp32"):
        try:
            ov0 = QC.OVRunner(paths["fp32"], threads=CPU_THREADS)
            add("QAT-FP32-OV", "OpenVINO CPU", eval_map(_Shim(ov0, nc), val_loader, cpu),
                Q.na("ONNX = opis grafa"),
                Q.benchmark(ov0.bench_call(example), "cpu", 5, 10), Q.na("OpenVINO CPU"))
        except Exception as e:
            add("QAT-FP32-OV", "OpenVINO CPU", None, Q.na("—"),
                Q.na(f"{type(e).__name__}: {str(e)[:50]}"), Q.na("—"))

    if paths.get("qdq"):
        try:
            ov = QC.OVRunner(paths["qdq"], threads=CPU_THREADS)
            pov = eval_map(_Shim(ov, nc), val_loader, cpu)
            lov = Q.benchmark(ov.bench_call(example), "cpu", 5, 20)
            add("QAT-INT8-OV", "OpenVINO CPU", pov, Q.na("QDQ ONNX = opis grafa, ne tezine"),
                lov, Q.na("OpenVINO CPU"))
        except Exception as e:
            add("QAT-INT8-OV", "OpenVINO CPU", None, Q.na("—"),
                Q.na(f"{type(e).__name__}: {str(e)[:50]}"), Q.na("—"))

    if has_cuda:
        for tag, src, kw in (("QAT-FP16-TRT", "fp32", dict(fp16=True)),
                             ("QAT-INT8-TRT", "qdq", dict(int8=True))):
            if not paths.get(src):
                continue
            try:
                eng = os.path.join(ONNX_DIR, f"yolo26n_{tag.lower()}.engine")
                QC.trt_build(paths[src], eng, **kw)
                r = QC.TRTRunner(eng)
                hist = r.precision_histogram()
                pt_ = eval_map(_Shim(r, nc), val_loader, gpu)
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

    ptq = {}
    if os.path.exists(PTQ_JSON):
        for r in json.load(open(PTQ_JSON))["rows"]:
            ptq[r["format"]] = r.get("map")
        rows.append({"format": "[PTQ] ultralytics val", "backend": "informativno",
                     "cpu_ms": Q.na("drugi mjerni put"), "gpu_ms": Q.na("drugi mjerni put"),
                     "size_mb": Q.na("—"),
                     "map": json.dumps(ptq), "map50": Q.na("—"),
                     "map75": Q.na("—"), "mar100": Q.na("—")})

    meta = {
        "model": "yolo26n", "method": "QAT (rucni fake-quant -> QDQ ONNX)",
        "loss": "KD-logit (MSE na gustom izlazu) prema zamrznutom originalu, bez GT",
        "qat": {"steps": steps, "lr": QAT_LR, "batch": TRAIN_BATCH, "calib_batches": CALIB_STEPS,
                "wrapped_modules": n_wrapped, "skip": f"{last}.dfl", "imgsz": IMGSZ},
        "onnx_verify": ver,
        "eval": {"path": "plugins.detection (letterbox+NMS+COCO->6, torchmetrics)",
                 "images": "pun val" if EVAL_MAX is None else EVAL_MAX,
                 "note": ("PTQ tablica je mjerena ultralytics val-om -> APSOLUTNE vrijednosti nisu "
                          "izravno usporedive; usporedjuju se RAZMACI unutar svake tablice. Zato je "
                          "[ORIG] FP32 izmjeren OVIM putem.")},
        "ptq_reference_map": ptq,
        "conditions": {"cpu_threads": CPU_THREADS, "lat_warmup": LAT_WARMUP, "lat_iters": LAT_ITERS,
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if has_cuda else None},
        "lat_full": lat_full,
        "minutes": round((time.time() - t_all) / 60.0, 2),
    }
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)

    def g(fmt):
        for r in rows:
            if r["format"] == fmt and isinstance(r.get("map"), (int, float)):
                return r["map"]
        return None

    o, q32, qint = g("[ORIG] FP32"), g("QAT-FP32"), (g("QAT-INT8-TRT") or g("QAT-INT8-OV"))
    print("\n=== RAZMAK PROTIV RAZMAKA (mAP@50:95, nas mjerni put) ===")
    if o and q32:
        print(f"  fine-tune sam po sebi: {o:.5f} -> {q32:.5f} ({q32 - o:+.5f})")
    if q32 and qint:
        print(f"  QAT kvantizacija:      {q32:.5f} -> {qint:.5f} ({qint - q32:+.5f})")
    if ptq.get("FP32") and ptq.get("INT8-TRT"):
        d = ptq["INT8-TRT"] - ptq["FP32"]
        print(f"  PTQ (ultralytics val): {ptq['FP32']:.5f} -> {ptq['INT8-TRT']:.5f} ({d:+.5f})")
    print(f"\n########## QAT yolo26n GOTOVO — {len(rows)} redaka, {meta['minutes']} min ##########")


if __name__ == "__main__":
    main()
