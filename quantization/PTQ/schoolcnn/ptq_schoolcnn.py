
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torchmetrics.functional.classification import (
    multilabel_average_precision, multilabel_f1_score, multilabel_precision,
    multilabel_recall, multilabel_exact_match, multilabel_accuracy, multilabel_auroc)

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(os.path.dirname(_HERE))
_EXP2 = os.path.join(os.path.dirname(_QROOT), "pruning", "critereum_experiment2")
sys.path.insert(0, _QROOT)
sys.path.insert(0, _EXP2)

import quant_common as Q
import common as C2
from model_cnn import SchoolCNN, INPUT_SIZE, NUM_CLASSES, CLASS_NAMES

CKPT_PATH   = os.path.join(_EXP2, "checkpoints", "best.pt")
QENGINE     = "x86"
CPU_THREADS = 8
LAT_BATCH   = 1
LAT_WARMUP  = 15
LAT_ITERS   = 100
EVAL_BATCH  = 32
CALIB_IMAGES = 512
EVAL_MAX    = None

OUT_CSV  = os.path.join(_HERE, "schoolcnn_ptq_report.csv")
OUT_JSON = os.path.join(_HERE, "schoolcnn_ptq_report.json")

CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb",
            "map_macro", "map_micro", "f1_macro", "f1_micro",
            "prec_macro", "recall_macro", "acc_macro", "subset_acc", "auroc_macro", "bce"]


def load_fp32():
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    m = SchoolCNN(num_classes=NUM_CLASSES).eval()
    m.load_state_dict(state)
    return m


@torch.no_grad()
def eval_panel(infer, loader, nc):
    logits, tgts = [], []
    for x, y in loader:
        out = infer(x)
        logits.append(out.float().cpu())
        tgts.append(y.int())
    L = torch.cat(logits); T = torch.cat(tgts); P = torch.sigmoid(L)
    ap_pc = multilabel_average_precision(P, T, num_labels=nc, average=None)
    return {
        "map_macro":   float(multilabel_average_precision(P, T, num_labels=nc, average="macro")),
        "map_micro":   float(multilabel_average_precision(P, T, num_labels=nc, average="micro")),
        "f1_macro":    float(multilabel_f1_score(P, T, num_labels=nc, average="macro")),
        "f1_micro":    float(multilabel_f1_score(P, T, num_labels=nc, average="micro")),
        "prec_macro":  float(multilabel_precision(P, T, num_labels=nc, average="macro")),
        "recall_macro":float(multilabel_recall(P, T, num_labels=nc, average="macro")),
        "acc_macro":   float(multilabel_accuracy(P, T, num_labels=nc, average="macro")),
        "subset_acc":  float(multilabel_exact_match(P, T, num_labels=nc)),
        "auroc_macro": float(multilabel_auroc(P, T, num_labels=nc, average="macro")),
        "bce":         float(F.binary_cross_entropy(P.clamp(1e-7, 1 - 1e-7), T.float())),
        "per_class_ap": {CLASS_NAMES[i]: float(ap_pc[i]) for i in range(nc)},
    }


def make_infer(model, device, half=False):
    def infer(x):
        x = x.to(device)
        if half:
            x = x.half()
        return model(x)
    return infer


def lat(model, device, example, half=False):
    x = example.to(device)
    if half:
        x = x.half()
    with torch.no_grad():
        return Q.benchmark(lambda: model(x), device, LAT_WARMUP, LAT_ITERS)


def round_ms(latres):
    return round(latres["median_ms"], 4) if isinstance(latres, dict) else latres


def main():
    Q.set_cpu_threads(CPU_THREADS)
    torch.backends.quantized.engine = QENGINE
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu"); gpu = torch.device("cuda") if has_cuda else None

    val_loader = C2.make_loader("val", EVAL_BATCH, shuffle=False, num_workers=4, max_images=EVAL_MAX)
    calib_loader = C2.make_loader("train", EVAL_BATCH, shuffle=False, num_workers=4, max_images=CALIB_IMAGES)
    example = next(iter(val_loader))[0][:LAT_BATCH].clone()

    model_fp32 = load_fp32()
    nc = NUM_CLASSES
    rows, lat_full = [], {}

    print(f"\n########## PTQ — SchoolCNN (engine={QENGINE}, CPU niti={CPU_THREADS}, lat batch={LAT_BATCH}) ##########")

    def add(fmt, backend, panel, size_mb, cpu_lat, gpu_lat):
        row = {"format": fmt, "backend": backend, "size_mb": round(size_mb, 4) if size_mb else size_mb,
               "cpu_ms": round_ms(cpu_lat), "gpu_ms": round_ms(gpu_lat)}
        row.update({k: round(panel[k], 5) for k in panel if k != "per_class_ap"})
        row["per_class_ap"] = panel.get("per_class_ap")
        rows.append(row)
        lat_full[fmt] = {"cpu": cpu_lat, "gpu": gpu_lat}
        print(f"  [{fmt:14}] mAP={row['map_macro']:.4f} acc={row['acc_macro']:.4f} F1={row['f1_macro']:.4f} | "
              f"{size_mb:.2f} MB | CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    m32_cpu = copy.deepcopy(model_fp32).to(cpu).eval()
    panel = eval_panel(make_infer(m32_cpu, cpu), val_loader, nc)
    cpu_lat = lat(m32_cpu, cpu, example)
    gpu_lat = Q.na("nema CUDA")
    if has_cuda:
        m32_gpu = copy.deepcopy(model_fp32).to(gpu).eval()
        gpu_lat = lat(m32_gpu, gpu, example)
    add("FP32", "PyTorch", panel, Q.model_size_mb(m32_cpu), cpu_lat, gpu_lat)

    if has_cuda:
        m16 = copy.deepcopy(model_fp32).to(gpu).half().eval()
        panel16 = eval_panel(make_infer(m16, gpu, half=True), val_loader, nc)
        gpu_lat = lat(m16, gpu, example, half=True)
    else:
        panel16 = {k: float("nan") for k in CSV_COLS if k not in ("format", "backend", "cpu_ms", "gpu_ms", "size_mb")}
        panel16["per_class_ap"] = {}
        gpu_lat = Q.na("nema CUDA")
    try:
        m16_cpu = copy.deepcopy(model_fp32).to(cpu).half().eval()
        cpu_lat = lat(m16_cpu, cpu, example, half=True)
    except (RuntimeError, NotImplementedError) as e:
        cpu_lat = Q.na(f"CPU half: {type(e).__name__}")
    add("FP16", "PyTorch", panel16, Q.model_size_mb(copy.deepcopy(model_fp32).half()), cpu_lat, gpu_lat)

    qmap = tq.get_default_qconfig_mapping(QENGINE)
    prep = prepare_fx(copy.deepcopy(model_fp32).to(cpu).eval(), qmap, example_inputs=(example.to(cpu),))
    with torch.no_grad():
        for x, _ in calib_loader:
            prep(x)
    m8 = convert_fx(prep).to(cpu).eval()
    panel8 = eval_panel(make_infer(m8, cpu), val_loader, nc)
    add("INT8-PT-static", f"PyTorch {QENGINE}", panel8, Q.model_size_mb(m8),
        lat(m8, cpu, example), Q.na("PyTorch quant je CPU-only"))

    m8d = tq.quantize_dynamic(copy.deepcopy(model_fp32).to(cpu).eval(), {nn.Linear}, dtype=torch.qint8)
    panel8d = eval_panel(make_infer(m8d, cpu), val_loader, nc)
    add("INT8-PT-dynamic", f"PyTorch {QENGINE}", panel8d, Q.model_size_mb(m8d),
        lat(m8d, cpu, example), Q.na("PyTorch quant je CPU-only"))

    rows.append({"format": "INT8-TRT", "backend": "TensorRT",
                 "cpu_ms": Q.na("TensorRT je GPU-only"), "gpu_ms": Q.na("treba TRT setup"),
                 "size_mb": Q.na("—"), **{k: Q.na("—") for k in CSV_COLS[5:]}})
    print("  [INT8-TRT      ] (čeka TensorRT setup)")

    meta = {
        "model": "SchoolCNN", "task": "multilabel-classification", "ckpt": CKPT_PATH,
        "num_classes": nc, "img_size": INPUT_SIZE, "eval_images": "full val" if EVAL_MAX is None else EVAL_MAX,
        "calib_images": CALIB_IMAGES, "quant_engine": QENGINE,
        "conditions": {"cpu_threads": CPU_THREADS, "lat_batch": LAT_BATCH, "lat_warmup": LAT_WARMUP,
                       "lat_iters": LAT_ITERS, "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                       "metric_note": "latencija = medijan; p90/min u 'lat_full'"},
        "lat_full": lat_full,
    }
    Q.write_report(rows, CSV_COLS, OUT_CSV, OUT_JSON, meta)
    print(f"\n########## GOTOVO — {len(rows)} formata ##########")


if __name__ == "__main__":
    main()
