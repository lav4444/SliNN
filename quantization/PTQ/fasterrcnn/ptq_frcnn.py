"""
ptq_frcnn.py — PTQ za Faster R-CNN (torchvision fasterrcnn_mobilenet_v3_large_320_fpn, COCO->6 klasa).

Formati × uređaj:
  FP32             (PyTorch, GPU + CPU) — mAP + latencija
  FP16             (PyTorch, GPU) — pokušaj (.half()); torchvision detekcija često ne podržava -> N/A
  INT8-PT dynamic  (PyTorch, CPU) — kvantizira Linear slojeve (ROI/RPN glave); static/FX nije izvedivo (NMS, dyn. control flow)
  INT8-TRT         (N/A) — frcnn ONNX->TensorRT je notorno težak (RPN/NMS/dyn. shapes); dokumentirano kao granica

Kvaliteta = morphology A.evaluate (detekcijski mAP, isti put kao ostatak projekta).
Latencija = morphology A.bench_speed (ms/img, warmup + median; imgsz=320).
Reuse morphology (analysis) + quant_common. Ne mijenja morphology (DEV_DATA_SUBSET override u runtime-u). BEZ CLI argumenata.
"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.ao.quantization as tq

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(os.path.dirname(_HERE))
_MORPH = os.path.join(os.path.dirname(_QROOT), "morphology")
sys.path.insert(0, _QROOT)
sys.path.insert(0, _MORPH)

import quant_common as Q
import analysis as A

A.DEV_DATA_SUBSET = None                       # pun val skup (override morphology dev-cap)

EVAL_BATCH = 4
LAT_REPS = 50
LAT_WARMUP = 10
IMGSZ = 320

OUT_CSV  = os.path.join(_HERE, "frcnn_ptq_report.csv")
OUT_JSON = os.path.join(_HERE, "frcnn_ptq_report.json")
CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb", "map", "map_50", "map_75", "mar_100"]


def panel(model, adapter, loader, device):
    m, _ = A.evaluate(model, adapter, loader, device)
    return {"map": float(m.get("map", float("nan"))), "map_50": float(m.get("map_50", m.get("map50", float("nan")))),
            "map_75": float(m.get("map_75", m.get("map75", float("nan")))),
            "mar_100": float(m.get("mar_100", float("nan")))}


def main():
    has_cuda = torch.cuda.is_available()
    cpu = "cpu"; gpu = "cuda" if has_cuda else "cpu"
    val_loader = A.make_gt_loader("val", bs=EVAL_BATCH)
    rows = []
    print("\n########## PTQ — Faster R-CNN (detekcija) ##########")

    def add(fmt, backend, p, size_mb, cpu_ms, gpu_ms):
        row = {"format": fmt, "backend": backend,
               "cpu_ms": cpu_ms if isinstance(cpu_ms, str) else round(cpu_ms, 4),
               "gpu_ms": gpu_ms if isinstance(gpu_ms, str) else round(gpu_ms, 4),
               "size_mb": size_mb if isinstance(size_mb, str) else round(size_mb, 4)}
        row.update({k: (round(p[k], 5) if k in p and p[k] == p[k] else Q.na("—")) for k in ("map", "map_50", "map_75", "mar_100")})
        rows.append(row)
        print(f"  [{fmt:16}] map={row['map']} mar100={row['mar_100']} | {row['size_mb']} MB | CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    # ---- FP32 ----
    model = A.build_fasterrcnn().to(gpu).eval()
    adapter = A.pick_adapter(model)
    p32 = panel(model, adapter, val_loader, gpu)
    gpu_ms = A.bench_speed(model, adapter, gpu, reps=LAT_REPS, warmup=LAT_WARMUP, imgsz=IMGSZ) if has_cuda else Q.na("nema CUDA")
    cpu_ms = A.bench_speed(model, adapter, cpu, reps=max(10, LAT_REPS // 3), warmup=5, imgsz=IMGSZ)
    add("FP32", "PyTorch", p32, Q.model_size_mb(model.cpu()), cpu_ms, gpu_ms)
    model.to(gpu)

    # ---- FP16 (GPU) — pokušaj ----
    if has_cuda:
        try:
            m16 = copy.deepcopy(model).to(gpu).half().eval()
            p16 = panel(m16, adapter, val_loader, gpu)
            g16 = A.bench_speed(m16, adapter, gpu, reps=LAT_REPS, warmup=LAT_WARMUP, imgsz=IMGSZ)
            add("FP16", "PyTorch", p16, Q.model_size_mb(copy.deepcopy(model).half().cpu()), Q.na("CPU FP16 bez koristi"), g16)
            del m16
        except Exception as e:
            add("FP16", "PyTorch", {}, Q.na("—"), Q.na("CPU FP16 bez koristi"), Q.na(f"{type(e).__name__}"))
    else:
        add("FP16", "PyTorch", {}, Q.na("—"), Q.na("nema CUDA"), Q.na("nema CUDA"))

    # ---- INT8-PT dynamic (CPU) ----
    try:
        m8 = tq.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
        p8 = panel(m8, adapter, val_loader, cpu)
        c8 = A.bench_speed(m8, adapter, cpu, reps=max(10, LAT_REPS // 3), warmup=5, imgsz=IMGSZ)
        add("INT8-PT-dynamic", "PyTorch", p8, Q.model_size_mb(m8), c8, Q.na("PyTorch quant je CPU-only"))
    except Exception as e:
        add("INT8-PT-dynamic", "PyTorch", {}, Q.na("—"), Q.na(f"{type(e).__name__}"), Q.na("CPU-only"))

    # ---- INT8-TRT (granica metode za složene detektore) ----
    rows.append({"format": "INT8-TRT", "backend": "TensorRT",
                 "cpu_ms": Q.na("TRT GPU-only"), "gpu_ms": Q.na("frcnn ONNX->TRT netrivijalan (RPN/NMS/dyn)"),
                 "size_mb": Q.na("—"), **{k: Q.na("—") for k in ("map", "map_50", "map_75", "mar_100")}})

    meta = {"model": "fasterrcnn_mobilenet_v3_large_320_fpn", "task": "detection",
            "classes": "COCO->6 (Person,Car,Truck,Bus,Motorcycle,Bicycle)", "imgsz": IMGSZ,
            "eval": "full val", "lat_method": "A.bench_speed (median, warmup, drop-worst)",
            "note": "FP16/INT8-TRT su granice za torchvision detektor; INT8-PT dynamic kvantizira samo Linear glave."}
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)
    print(f"\n########## FRCNN PTQ GOTOVO ##########")


if __name__ == "__main__":
    main()
