
import copy
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_QROOT)
_SLINN = os.path.join(_ROOT, "slinn")
for _p in (_HERE, _QROOT, _SLINN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qat_common as QC
import quant_common as Q
from plugins.detection import adapters as A

MODELS = os.environ.get("QAT_YOLO", "yolo26n,yolo26l").split(",")
YAML = os.path.join(_QROOT, "yolo_coco_data", "yolo_coco.yaml")
IMGSZ, CONF = 640, 0.001
QAT_STEPS, QAT_LR = 300, 1e-5
CALIB_BATCHES = 16
CSV_COLS = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb", "map", "map50", "map75"]


def dir_mb(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1024 ** 2


def qat_weights(tag, dev):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)

    w = os.path.join(_ROOT, "baseline_models", tag, f"{tag}.pt")
    teacher = YOLO(w).model.float().eval().to(dev)
    list(teacher.model)[-1].end2end = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = YOLO(w).model.float().eval()
    list(student.model)[-1].end2end = False
    last = len(student.model) - 1
    n = QC.wrap_model(student, skip=(f"model.{last}.",))
    student = student.to(dev)
    print(f"  [wrap] {n} modula (glava model.{last} izuzeta)", flush=True)

    tr = A.make_gt_loader("train", bs=4, workers=0)
    lb = lambda imgs: torch.stack([A.YoloAdapter._letterbox(im)[0] for im in imgs]).to(dev)
    dense = lambda o: (o[0] if isinstance(o, (tuple, list)) else o)

    cal = []
    for i, b in enumerate(tr):
        if i >= CALIB_BATCHES:
            break
        cal.append(lb(b[0]).cpu())
    QC.calibrate(student, cal, dev)
    print(f"  [calib] {len(cal)} batcheva", flush=True)
    del cal

    def kd(m, b):
        x = lb(b[0]) if isinstance(b, (list, tuple)) else b.to(dev)
        m.eval()
        with torch.no_grad():
            t = dense(teacher(x)).detach()
        return F.mse_loss(dense(m(x)), t)

    loss, steps = QC.qat_finetune(student, tr, kd, steps=QAT_STEPS, lr=QAT_LR, device=dev,
                                  on_step=lambda k, n_, l: print(f"      {k}/{n_} kd={l:.5f}", flush=True))
    print(f"  [qat] {steps} koraka, KD {loss:.5f}", flush=True)
    del teacher
    torch.cuda.empty_cache()
    return QC.unwrap_model(student.cpu()).eval(), n


def save_ultralytics_ckpt(tag, qat_model, out_pt):
    ck = torch.load(os.path.join(_ROOT, "baseline_models", tag, f"{tag}.pt"),
                    map_location="cpu", weights_only=False)
    base = ck["model"]
    base.load_state_dict(qat_model.state_dict(), strict=True)
    ck["model"] = base.float()
    ck["ema"] = None
    torch.save(ck, out_pt)
    return out_pt


def val(path, device, task="detect"):
    from ultralytics import YOLO
    r = YOLO(path, task=task).val(data=YAML, split="val", imgsz=IMGSZ, conf=CONF,
                                  device=device, batch=1, verbose=False, plots=False)
    return ({"map": float(r.box.map), "map50": float(r.box.map50), "map75": float(r.box.map75)},
            float(r.speed.get("inference", float("nan"))))


def run(tag):
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(_HERE, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 92}\nQAT + ULTRALYTICS IZVOZ — {tag}\n{'=' * 92}", flush=True)

    rows = []

    def add(fmt, backend, panel, size, cpu_ms, gpu_ms):
        r = {"format": fmt, "backend": backend,
             "size_mb": round(size, 4) if isinstance(size, (int, float)) else size,
             "cpu_ms": round(cpu_ms, 4) if isinstance(cpu_ms, (int, float)) else cpu_ms,
             "gpu_ms": round(gpu_ms, 4) if isinstance(gpu_ms, (int, float)) else gpu_ms}
        for k in ("map", "map50", "map75"):
            v = (panel or {}).get(k)
            r[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        rows.append(r)
        print(f"  [{fmt:18}] map={r['map']} map50={r['map50']} | {r['size_mb']} MB | "
              f"CPU {r['cpu_ms']} | GPU {r['gpu_ms']}", flush=True)

    qat_model, n_w = qat_weights(tag, dev)
    qat_pt = save_ultralytics_ckpt(tag, qat_model, os.path.join(out_dir, f"{tag}_qat.pt"))
    print(f"  [ckpt] {qat_pt}", flush=True)

    p, ms = val(qat_pt, 0 if dev.type == "cuda" else "cpu")
    add("QAT-FP32", "PyTorch", p, os.path.getsize(qat_pt) / 1024 ** 2, Q.na("—"), ms)

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)

    try:
        ov = YOLO(qat_pt).export(format="openvino", int8=True, data=YAML, imgsz=IMGSZ,
                                 batch=1, verbose=False)
        p, ms = val(ov, "cpu")
        add("QAT-INT8-OV", "OpenVINO CPU", p, dir_mb(ov), ms, Q.na("OpenVINO je CPU"))
    except Exception as e:
        add("QAT-INT8-OV", "OpenVINO CPU", None, Q.na("—"),
            Q.na(f"{type(e).__name__}: {str(e)[:44]}"), Q.na("—"))

    if dev.type == "cuda":
        try:
            eng = YOLO(qat_pt).export(format="engine", int8=True, data=YAML, imgsz=IMGSZ,
                                      batch=1, device=0, verbose=False)
            p, ms = val(eng, 0)
            add("QAT-INT8-TRT", "TensorRT", p, os.path.getsize(eng) / 1024 ** 2,
                Q.na("TRT je GPU"), ms)
        except Exception as e:
            add("QAT-INT8-TRT", "TensorRT", None, Q.na("—"), Q.na("TRT je GPU"),
                Q.na(f"{type(e).__name__}: {str(e)[:44]}"))

    ptq_json = os.path.join(_QROOT, "PTQ", tag, f"{tag}_ptq_report.json")
    ptq = {}
    if os.path.exists(ptq_json):
        for r in json.load(open(ptq_json))["rows"]:
            ptq[r["format"]] = r
        for t in ("FP32", "INT8-static", "INT8-TRT"):
            if t in ptq:
                rr = dict(ptq[t]); rr["format"] = f"[PTQ] {t}"
                rows.append(rr)
                print(f"  [{rr['format']:18}] map={rr.get('map')}  (iz PTQ izvjestaja)", flush=True)

    meta = {"model": tag, "method": "QAT (nas fake-quant, KD bez GT) + ULTRALYTICS izvoz",
            "why": ("Nas rucni QDQ izvoz je za yolo davao 0.006 (OV) / 0.264 (TRT) iako je fake-quant "
                    "model u torchu davao 0.419 — dakle skale su dobre, a OV/TRT propagiraju kvantizaciju "
                    "u dekodiranje okvira. Ultralytics izvoznik to izuzima (nncf.IgnoredScope na "
                    "Add/Sub/Mul/Div glave + sve Sigmoid), pa se koristi on."),
            "qat": {"steps": QAT_STEPS, "lr": QAT_LR, "wrapped": n_w, "calib_batches": CALIB_BATCHES},
            "eval": "ultralytics val (ISTI put kao PTQ tablica -> redci su izravno usporedivi)",
            "minutes": round((time.time() - t0) / 60.0, 2)}
    Q.write_report(rows, CSV_COLS, os.path.join(out_dir, f"{tag}_qat_ultra_report.csv"),
                   os.path.join(out_dir, f"{tag}_qat_ultra_report.json"), meta)

    g = lambda f: next((r["map"] for r in rows if r["format"] == f and isinstance(r.get("map"), float)), None)
    print(f"\n  --- {tag}: QAT protiv PTQ, isti mjerni put ---")
    for a, b, lbl in (("QAT-FP32", "QAT-INT8-OV", "OpenVINO"), ("QAT-FP32", "QAT-INT8-TRT", "TensorRT")):
        x, y = g(a), g(b)
        if x and y:
            print(f"    QAT {lbl:9}: {x:.5f} -> {y:.5f} ({y - x:+.5f})")
    if ptq.get("FP32"):
        for t, lbl in (("INT8-static", "OpenVINO"), ("INT8-TRT", "TensorRT")):
            if ptq.get(t) and isinstance(ptq[t].get("map"), float):
                d = ptq[t]["map"] - ptq["FP32"]["map"]
                print(f"    PTQ {lbl:9}: {ptq['FP32']['map']:.5f} -> {ptq[t]['map']:.5f} ({d:+.5f})")
    print(f"  --- {meta['minutes']} min ---", flush=True)


if __name__ == "__main__":
    for t in MODELS:
        try:
            run(t.strip())
        except BaseException:
            import traceback
            print(f"\n!!! {t} PAO:\n{traceback.format_exc()[-1000:]}", flush=True)
        torch.cuda.empty_cache()
