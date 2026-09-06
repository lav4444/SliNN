
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import quant_common as Q
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging
LOGGER.setLevel(logging.ERROR)

YAML = os.path.join(_HERE, "yolo_coco_data", "yolo_coco.yaml")
IMGSZ = 640
CONF = 0.001
CPU_THREADS = 8

MODELS = [
    ("yolo26n", "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt", os.path.join(_HERE, "PTQ", "yolo26n")),
    ("yolo26l", "/home/tomi/code/dipl/baseline_models/yolo26l/yolo26l.pt", os.path.join(_HERE, "PTQ", "yolo26l")),
]
CSV_COLS = ["format", "cpu_ms", "gpu_ms", "size_mb", "map", "map50", "map75", "mp", "mr"]


def dir_mb(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1024 ** 2


def val_quality(weights_or_model, device, half=False):
    m = weights_or_model if isinstance(weights_or_model, YOLO) else YOLO(weights_or_model, task="detect")
    r = m.val(data=YAML, split="val", imgsz=IMGSZ, conf=CONF, device=device, batch=1,
              half=half, verbose=False, plots=False, save_json=False)
    panel = {"map": float(r.box.map), "map50": float(r.box.map50), "map75": float(r.box.map75),
             "mp": float(r.box.mp), "mr": float(r.box.mr)}
    return panel, float(r.speed.get("inference", float("nan")))


def raw_lat(torch_model, device, half, n_warmup, n_iter):
    m = copy.deepcopy(torch_model).to(device).eval()
    if half:
        m = m.half()
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device=device)
    if half:
        x = x.half()
    with torch.no_grad():
        return Q.benchmark(lambda: m(x), device, n_warmup, n_iter)["median_ms"]


def run_model(tag, weights, outdir):
    os.makedirs(outdir, exist_ok=True)
    pt_mb = os.path.getsize(weights) / 1024 ** 2
    rows = []
    print(f"\n########## PTQ (pojednostavljeno) — {tag} ##########")

    def add(fmt, cpu_ms, gpu_ms, size_mb, panel):
        row = {"format": fmt,
               "cpu_ms": cpu_ms if isinstance(cpu_ms, str) else round(cpu_ms, 4),
               "gpu_ms": gpu_ms if isinstance(gpu_ms, str) else round(gpu_ms, 4),
               "size_mb": size_mb if isinstance(size_mb, str) else round(size_mb, 4)}
        row.update({k: round(panel.get(k, float("nan")), 5) for k in ("map", "map50", "map75", "mp", "mr")})
        rows.append(row)
        print(f"  [{fmt:12}] map={row['map']} | {row['size_mb']} MB | CPU {row['cpu_ms']} | GPU {row['gpu_ms']}")

    torch_model = YOLO(weights).model.float().eval()

    q32, _ = val_quality(weights, 0)
    cpu32 = raw_lat(torch_model, "cpu", False, 3, 15)
    gpu32 = raw_lat(torch_model, "cuda", False, 10, 50)
    add("FP32", cpu32, gpu32, pt_mb, q32)

    q16, _ = val_quality(weights, 0, half=True)
    try:
        cpu16 = raw_lat(torch_model, "cpu", True, 2, 5)
    except Exception as e:
        cpu16 = Q.na(type(e).__name__)
    gpu16 = raw_lat(torch_model, "cuda", True, 10, 50)
    add("FP16", cpu16, gpu16, pt_mb / 2, q16)

    try:
        ov = YOLO(weights).export(format="openvino", int8=True, data=YAML, imgsz=IMGSZ, batch=1, verbose=False)
        qov, ov_ms = val_quality(YOLO(ov, task="detect"), "cpu")
        add("INT8-static", ov_ms, Q.na("static je CPU"), dir_mb(ov), qov)
    except Exception as e:
        add("INT8-static", Q.na(f"{type(e).__name__}"), Q.na("static je CPU"), Q.na("—"), {})

    try:
        eng = YOLO(weights).export(format="engine", int8=True, data=YAML, imgsz=IMGSZ, batch=1, device=0, verbose=False)
        qtrt, trt_ms = val_quality(YOLO(eng, task="detect"), 0)
        add("INT8-TRT", Q.na("TRT je GPU"), trt_ms, os.path.getsize(eng) / 1024 ** 2, qtrt)
    except Exception as e:
        add("INT8-TRT", Q.na("TRT je GPU"), Q.na(f"{type(e).__name__}"), Q.na("—"), {})

    meta = {"model": tag, "task": "detection", "weights": weights, "imgsz": IMGSZ, "conf": CONF,
            "pt_size_mb": round(pt_mb, 4), "cpu_threads": CPU_THREADS,
            "lat_method": "PyTorch FP32/FP16 = sirovi forward (median, batch1); INT8 = native val speed['inference']",
            "note": "FP16-CPU = sirovi .half() (spor, demonstrira nedostatak HW podrške); veličine TRT/OV nisu samo težine."}
    Q.write_report(rows, CSV_COLS, os.path.join(outdir, f"{tag}_ptq_report.csv"),
                   os.path.join(outdir, f"{tag}_ptq_report.json"), meta)
    print(f"  -> {outdir}/{tag}_ptq_report.csv")


if __name__ == "__main__":
    if not os.path.exists(YAML):
        print("Nema yolo_coco yaml — prvo pokreni prep_yolo_coco.py"); sys.exit(1)
    Q.set_cpu_threads(CPU_THREADS)
    for tag, w, out in MODELS:
        run_model(tag, w, out)
    print("\n########## YOLO PTQ (pojednostavljeno) GOTOVO ##########")
