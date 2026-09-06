
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(os.path.dirname(_HERE))
_MODEL_DIR = os.path.join(os.path.dirname(_QROOT), "baseline_models", "voc_deeplabv3")
sys.path.insert(0, _QROOT)
sys.path.insert(0, _MODEL_DIR)

import quant_common as Q                                              # noqa: E402
import data as D                                                      # noqa: E402  (voc: voc/transform/classes)
import eval_baseline as EB                                            # noqa: E402  (metrics(conf) — provjeren racun)

MODEL_PT    = os.path.join(_MODEL_DIR, "model.pt")
QENGINE     = "x86"
CPU_THREADS = 8
LAT_SIZE    = 520
LAT_BATCH   = 1
LAT_WARMUP  = 10
LAT_ITERS   = 50
LAT_WARMUP_SLOW = 2
LAT_ITERS_SLOW  = 5
EVAL_SPLIT  = "val"
EVAL_LIMIT  = None
CALIB_IMAGES = 128

OUT_CSV  = os.path.join(_HERE, "deeplabv3_ptq_report.csv")
OUT_JSON = os.path.join(_HERE, "deeplabv3_ptq_report.json")

CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb", "miou", "pixel_acc"]


def load_fp32():
    return torch.load(MODEL_PT, map_location="cpu", weights_only=False).eval()


def fixed_input(device=None, half=False):
    x = torch.rand(LAT_BATCH, 3, LAT_SIZE, LAT_SIZE)
    if device is not None:
        x = x.to(device)
    return x.half() if half else x


@torch.no_grad()
def confusion(infer):
    ds = D.voc(EVAL_SPLIT)
    tf = D.transform()
    n = len(ds) if EVAL_LIMIT is None else min(EVAL_LIMIT, len(ds))
    C = D.NUM_CLASSES
    conf = torch.zeros(C, C, dtype=torch.long)
    for i in range(n):
        img, mask = ds[i]
        out = infer(tf(img).unsqueeze(0)).float().cpu()
        m = torch.as_tensor(np.array(mask), dtype=torch.long)
        out = F.interpolate(out, size=m.shape, mode="bilinear", align_corners=False)
        pred = out.argmax(1)[0]
        valid = (m != D.IGNORE) & (m < C)
        conf += torch.bincount(C * m[valid] + pred[valid], minlength=C * C).reshape(C, C)
    return conf


def panel_from(infer, names):
    conf = confusion(infer)
    miou, pacc, iou = EB.metrics(conf)
    return {"miou": float(miou), "pixel_acc": float(pacc),
            "per_class_iou": {names[i]: float(iou[i]) for i in range(len(names))}}


def make_infer(model, device, half=False):
    def infer(x):
        xx = x.to(device)
        if half:
            xx = xx.half()
        out = model(xx)
        return out["out"] if isinstance(out, dict) else (out[0] if isinstance(out, (tuple, list)) else out)
    return infer


def lat(model, device, half=False, warmup=LAT_WARMUP, iters=LAT_ITERS):
    x = fixed_input(device, half)
    with torch.no_grad():
        return Q.benchmark(lambda: model(x), device, warmup, iters)


def round_ms(r):
    return round(r["median_ms"], 4) if isinstance(r, dict) else r


def nan_panel(names):
    return {"miou": float("nan"), "pixel_acc": float("nan"),
            "per_class_iou": {nm: float("nan") for nm in names}}


def main():
    Q.set_cpu_threads(CPU_THREADS)
    torch.backends.quantized.engine = QENGINE
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu"); gpu = torch.device("cuda") if has_cuda else None

    names = D.classes()
    model_fp32 = load_fp32()
    rows, lat_full = [], {}

    print(f"\n########## PTQ — DeepLabV3-MobileNetV3 / VOC2012 (engine={QENGINE}, "
          f"CPU niti={CPU_THREADS}, lat {LAT_BATCH}x3x{LAT_SIZE}x{LAT_SIZE}) ##########")

    def add(fmt, backend, panel, size_mb, cpu_lat, gpu_lat):
        row = {"format": fmt, "backend": backend,
               "size_mb": round(size_mb, 4) if isinstance(size_mb, (int, float)) else size_mb,
               "cpu_ms": round_ms(cpu_lat), "gpu_ms": round_ms(gpu_lat)}
        for k in ("miou", "pixel_acc"):
            v = panel.get(k, float("nan"))
            row[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        row["per_class_iou"] = panel.get("per_class_iou")
        rows.append(row)
        lat_full[fmt] = {"cpu": cpu_lat, "gpu": gpu_lat}
        print(f"  [{fmt:17}] mIoU={row['miou']} pixAcc={row['pixel_acc']} | {row['size_mb']} MB | "
              f"CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    dev_eval = gpu if has_cuda else cpu
    m32 = copy.deepcopy(model_fp32).to(dev_eval).eval()
    panel = panel_from(make_infer(m32, dev_eval), names)
    gpu_lat = lat(m32, dev_eval) if has_cuda else Q.na("nema CUDA")
    m32c = copy.deepcopy(model_fp32).to(cpu).eval()
    cpu_lat = lat(m32c, cpu, warmup=5, iters=max(10, LAT_ITERS // 3))
    add("FP32", "PyTorch", panel, Q.model_size_mb(m32c), cpu_lat, gpu_lat)
    del m32; torch.cuda.empty_cache() if has_cuda else None

    if has_cuda:
        m16 = copy.deepcopy(model_fp32).to(gpu).half().eval()
        panel16 = panel_from(make_infer(m16, gpu, half=True), names)
        gpu_lat = lat(m16, gpu, half=True)
        del m16; torch.cuda.empty_cache()
    else:
        panel16, gpu_lat = nan_panel(names), Q.na("nema CUDA")
    try:
        m16c = copy.deepcopy(model_fp32).to(cpu).half().eval()
        cpu_lat = lat(m16c, cpu, half=True, warmup=LAT_WARMUP_SLOW, iters=LAT_ITERS_SLOW)
        del m16c
    except (RuntimeError, NotImplementedError) as e:
        cpu_lat = Q.na(f"CPU half: {type(e).__name__}")
    add("FP16", "PyTorch", panel16, Q.model_size_mb(copy.deepcopy(model_fp32).half()), cpu_lat, gpu_lat)

    try:
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
        qmap = tq.get_default_qconfig_mapping(QENGINE)
        prep = prepare_fx(copy.deepcopy(model_fp32).to(cpu).eval(), qmap,
                          example_inputs=(fixed_input(),))
        ds, tf = D.voc("train"), D.transform()
        with torch.no_grad():
            for i in range(min(CALIB_IMAGES, len(ds))):
                prep(tf(ds[i][0]).unsqueeze(0))
        m8s = convert_fx(prep).to(cpu).eval()
        p8s = panel_from(make_infer(m8s, cpu), names)
        add("INT8-PT-static", f"PyTorch {QENGINE}", p8s, Q.model_size_mb(m8s),
            lat(m8s, cpu, warmup=5, iters=max(10, LAT_ITERS // 3)), Q.na("PyTorch quant je CPU-only"))
        del m8s
    except Exception as e:
        add("INT8-PT-static", f"PyTorch {QENGINE}", nan_panel(names), Q.na("—"),
            Q.na(f"{type(e).__name__}: {str(e)[:44]}"), Q.na("CPU-only"))

    n_lin = sum(1 for m in model_fp32.modules() if isinstance(m, nn.Linear))
    try:
        m8d = tq.quantize_dynamic(copy.deepcopy(model_fp32).to(cpu).eval(), {nn.Linear}, dtype=torch.qint8)
        p8d = panel_from(make_infer(m8d, cpu), names)
        add("INT8-PT-dynamic", f"PyTorch {QENGINE} ({n_lin} Linear)", p8d, Q.model_size_mb(m8d),
            lat(m8d, cpu, warmup=5, iters=max(10, LAT_ITERS // 3)), Q.na("PyTorch quant je CPU-only"))
        del m8d
    except Exception as e:
        add("INT8-PT-dynamic", f"PyTorch {QENGINE}", nan_panel(names), Q.na("—"),
            Q.na(f"{type(e).__name__}"), Q.na("CPU-only"))

    rows.append({"format": "INT8-TRT", "backend": "TensorRT",
                 "cpu_ms": Q.na("TRT je GPU-only"),
                 "gpu_ms": Q.na("zaseban trt_deeplabv3.py (uzor: trt_schoolcnn.py)"),
                 "size_mb": Q.na("—"), "miou": Q.na("—"), "pixel_acc": Q.na("—")})
    print("  [INT8-TRT         ] (zaseban TRT skript, po uzoru na trt_schoolcnn.py)")

    meta = {
        "model": "deeplabv3_mobilenet_v3_large", "task": "semantic-segmentation (VOC2012, 21 kl.)",
        "weights": MODEL_PT, "num_classes": D.NUM_CLASSES, "ignore_index": D.IGNORE,
        "eval_split": EVAL_SPLIT, "eval_images": "full val (1449)" if EVAL_LIMIT is None else EVAL_LIMIT,
        "calib_images": CALIB_IMAGES, "quant_engine": QENGINE,
        "n_linear_layers": n_lin,
        "reference_eval_result": {"miou": 0.7269, "gpu_ms": 6.55, "cpu_ms": 94.32},
        "conditions": {"cpu_threads": CPU_THREADS, "lat_batch": LAT_BATCH, "lat_size": LAT_SIZE,
                       "lat_warmup": LAT_WARMUP, "lat_iters": LAT_ITERS,
                       "lat_slow_iters": LAT_ITERS_SLOW, "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                       "metric_note": "latencija = medijan; p90/min u 'lat_full'",
                       "eval_note": "FP32/FP16 panel na GPU; int8 panel na CPU (PyTorch quant je CPU-only)"},
        "note": ("Jedini gusti zadatak u setu. Isti backbone kao fasterrcnn (mobilenet_v3_large) ali bez "
                 "RPN/NMS/dinamičnih oblika -> provjerava je li frcnn pao zbog obitelji ili zbog detekcijskih "
                 "operacija. INT8-PT-dynamic je kontrola: model gotovo nema Linear slojeva."),
        "lat_full": lat_full,
    }
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)
    print(f"\n########## DEEPLABV3 PTQ GOTOVO — {len(rows)} formata ##########")


if __name__ == "__main__":
    main()
