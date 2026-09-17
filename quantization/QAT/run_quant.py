
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_QROOT)
for _p in (_HERE, _QROOT, os.path.join(_ROOT, "slinn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qat_common as QC
import quant_common as Q
import run_qat as RQ

MODE = os.environ.get("MODE", "both")
MODELS = os.environ.get("MODELS", "schoolcnn,deeplabv3,distilbert,yolo26n,yolo26l").split(",")
OUT_ROOT = os.path.join(_HERE, "results2")
CPU_THREADS, LAT_WARMUP, LAT_ITERS = 8, 10, 50
ULTRA = {"yolo26n", "yolo26l"}


def bench(fn, dev, slow=False):
    try:
        return Q.benchmark(fn, dev, 2 if slow else LAT_WARMUP, 5 if slow else LAT_ITERS)
    except Exception as e:
        return Q.na(type(e).__name__)


def ms(v):
    return round(v["median_ms"], 4) if isinstance(v, dict) else v


def free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ultra_rows(tag, qat_model, out_dir, keys):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import logging
    LOGGER.setLevel(logging.ERROR)

    yaml_p = os.path.join(_QROOT, "yolo_coco_data", "yolo_coco.yaml")
    src = os.path.join(_ROOT, "baseline_models", tag, f"{tag}.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if qat_model is not None:
        ck["model"].load_state_dict(qat_model.state_dict(), strict=True)
    ck["model"] = ck["model"].float()
    ck["ema"] = None
    pt = os.path.join(out_dir, f"{tag}.pt")
    torch.save(ck, pt)
    size = Q.model_size_mb(ck["model"])

    def val(path, dev, task="detect"):
        r = YOLO(path, task=task).val(data=yaml_p, split="val", imgsz=640, conf=0.001,
                                      device=dev, batch=1, verbose=False, plots=False)
        return ({"map": float(r.box.map), "map50": float(r.box.map50),
                 "map75": float(r.box.map75), "mar100": float(r.box.mr)},
                float(r.speed.get("inference", float("nan"))))

    del ck
    free_gpu()

    rows = []
    has_cuda = torch.cuda.is_available()
    p, t = val(pt, 0 if has_cuda else "cpu")
    rows.append(("FP32", "PyTorch", p, size, Q.na("—"), t))
    free_gpu()
    try:
        ov = YOLO(pt).export(format="openvino", int8=True, data=yaml_p, imgsz=640,
                             batch=1, verbose=False)
        free_gpu()
        p, t = val(ov, "cpu")
        rows.append(("INT8-OV", "OpenVINO CPU", p, size / 4, t, Q.na("OpenVINO je CPU")))
    except Exception as e:
        rows.append(("INT8-OV", "OpenVINO CPU", None, Q.na("—"),
                     Q.na(f"{type(e).__name__}: {str(e)[:40]}"), Q.na("—")))
    free_gpu()
    if has_cuda:
        for fmt, kw in (("FP16-TRT", dict(half=True)), ("INT8-TRT", dict(int8=True))):
            try:
                free_gpu()
                _pre = torch.cuda.mem_get_info()[0] / 1024 ** 3
                print(f"      [{fmt}] slobodno na GPU prije builda: {_pre:.2f} GB", flush=True)
                eng = YOLO(pt).export(format="engine", data=yaml_p, imgsz=640, batch=1,
                                      device=0, verbose=False, **kw)
                free_gpu()
                p, t = val(eng, 0)
                rows.append((fmt, "TensorRT", p, os.path.getsize(eng) / 1024 ** 2,
                             Q.na("TRT je GPU"), t))
            except Exception as e:
                rows.append((fmt, "TensorRT", None, Q.na("—"), Q.na("TRT je GPU"),
                             Q.na(f"{type(e).__name__}: {str(e)[:40]}")))
            free_gpu()
    return rows


def qdq_rows(pf, student, example, out_dir, name, cpu, gpu, has_cuda):
    paths = QC.export_all(student, example, os.path.join(out_dir, "onnx"), name,
                          pick=pf.get("pick"), input_names=pf.get("input_names", ("input",)))
    ver = {m_: QC.verify(p) for m_, p in paths.items() if p}
    ev_be = pf.get("evaluate_backend") or pf["evaluate"]
    rows, extra = [], {}

    plain = QC.unwrap_model(student).eval()
    rows.append(("FP32", "PyTorch", pf["evaluate"](plain.to(gpu), gpu), Q.model_size_mb(plain.cpu()),
                 Q.na("—"), bench(lambda: _call(pf, plain.to(gpu), example, gpu), gpu)
                 if has_cuda else Q.na("nema CUDA")))
    if has_cuda:
        try:
            h = RQ.HalfWrap(QC.unwrap_model(student).to(gpu).half().eval())
            rows.append(("FP16", "PyTorch", pf["evaluate"](h, gpu),
                         Q.model_size_mb(QC.unwrap_model(student).half()),
                         Q.na("CPU fp16 bez HW puta"), Q.na("mjereno u TRT retku")))
            del h
            torch.cuda.empty_cache()
        except Exception as e:
            rows.append(("FP16", "PyTorch", None, Q.na("—"), Q.na("—"), Q.na(type(e).__name__)))

    if pf.get("evaluate_backend"):
        pl = QC.unwrap_model(student).to(gpu).eval()
        rows.append(("FP32-BE", "PyTorch (backend put)", ev_be(pl, gpu), Q.na("kontrola"),
                     Q.na("—"), Q.na("—")))
        del pl
        torch.cuda.empty_cache()

    feed = (example if not isinstance(example, tuple)
            else {"input_ids": example[0], "attention_mask": example[1]})
    if paths.get("qdq"):
        try:
            ov = QC.OVRunner(paths["qdq"], threads=CPU_THREADS)
            rows.append(("INT8-OV", "OpenVINO CPU", ev_be(RQ.Shim(ov, **pf.get("shim_kw", {})), cpu),
                         Q.na("QDQ ONNX = opis grafa"), bench(ov.bench_call(feed), "cpu"),
                         Q.na("OpenVINO je CPU")))
        except Exception as e:
            rows.append(("INT8-OV", "OpenVINO CPU", None, Q.na("—"),
                         Q.na(f"{type(e).__name__}: {str(e)[:40]}"), Q.na("—")))
    if has_cuda:
        for fmt, src, kw in (("FP16-TRT", "fp32", dict(fp16=True)),
                             ("INT8-TRT", "qdq", dict(int8=True))):
            if not paths.get(src):
                continue
            try:
                eng = os.path.join(out_dir, "onnx", f"{name}_{fmt.lower()}.engine")
                QC.trt_build(paths[src], eng, fp32_layers=pf.get("fp32_layers", ()), **kw)
                r = QC.TRTRunner(eng)
                extra[fmt] = {"precision_layers": r.precision_histogram()}
                pnl = ev_be(RQ.Shim(r, **pf.get("shim_kw", {})), gpu)
                r.preload(feed)
                rows.append((fmt, "TensorRT", pnl, Q.file_size_mb(eng), Q.na("TRT je GPU"),
                             bench(r.enqueue, "cuda")))
                del r
                torch.cuda.empty_cache()
            except Exception as e:
                rows.append((fmt, "TensorRT", None, Q.na("—"), Q.na("TRT je GPU"),
                             Q.na(f"{type(e).__name__}: {str(e)[:40]}")))
    return rows, ver, extra


def _call(pf, m, x, dev):
    if pf.get("kwargs_forward"):
        return m(input_ids=x[0].to(dev), attention_mask=x[1].to(dev))
    x = tuple(t.to(dev) for t in x) if isinstance(x, tuple) else x.to(dev)
    return m(*x) if isinstance(x, tuple) else m(x)


def run(key, modes):
    t0 = time.time()
    pf = RQ.PROFILES[key]()
    name = pf["name"]
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    has_cuda = torch.cuda.is_available()
    cpu, gpu = torch.device("cpu"), torch.device("cuda" if has_cuda else "cpu")
    keys = pf["keys"]
    cols = ["postupak", "format", "backend", "cpu_ms", "gpu_ms", "size_mb"] + list(keys)
    is_ultra = name in ULTRA
    all_rows = []

    print(f"\n{'=' * 100}\n{name}   (izvoz: {'ultralytics' if is_ultra else 'nas QDQ'})\n{'=' * 100}",
          flush=True)

    def emit(proc, tup, extra=None):
        fmt, backend, panel, size, c, g = tup
        r = {"postupak": proc, "format": fmt, "backend": backend,
             "size_mb": round(size, 4) if isinstance(size, (int, float)) else size,
             "cpu_ms": ms(c), "gpu_ms": ms(g)}
        for k in keys:
            v = (panel or {}).get(k)
            r[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        if extra and fmt in extra:
            r.update(extra[fmt])
        all_rows.append(r)
        print(f"  [{proc:4}] {fmt:10} " + "  ".join(f"{k}={r[k]}" for k in keys) +
              f" | {r['size_mb']} MB | CPU {r['cpu_ms']} | GPU {r['gpu_ms']}", flush=True)

    orig = pf["load"]().to(gpu).eval()
    p_o = pf["evaluate"](orig, gpu)
    oc = pf["load"]().to(cpu).eval()
    emit("ORIG", ("FP32", "PyTorch", p_o, Q.model_size_mb(oc),
                  bench(lambda: _call(pf, oc, pf["example"], cpu), cpu, slow=True),
                  bench(lambda: _call(pf, orig, pf["example"], gpu), gpu) if has_cuda
                  else Q.na("nema CUDA")))
    del oc
    torch.cuda.empty_cache()

    meta_extra = {}
    for proc in modes:
        student = pf["load"]()
        n_fold = 0 if is_ultra else QC.fold_bn(student)
        n_w = QC.wrap_model(student, skip=pf["skip"])
        print(f"  [{proc}] foldano Conv+BN: {n_fold} | omotano: {n_w}", flush=True)
        student = student.to(gpu)

        cal = []
        for i, b in enumerate(pf["calib"]):
            if i >= pf.get("calib_max", 16):
                break
            cal.append(b)
        QC.calibrate(student, cal, gpu, forward_fn=pf.get("calib_fwd"))
        del cal

        steps = 0
        if proc == "QAT":
            orig = orig.to(gpu)
            QC.start_lsq(student)
            loss, steps = QC.qat_finetune(
                student, pf["train"], lambda m, b: pf["loss"](m, b, orig),
                steps=pf["steps"], lr=pf["lr"], device=gpu,
                on_step=lambda k, n_, l: print(f"      {k}/{n_} kd={l:.5f}", flush=True))
            print(f"  [QAT] {steps} koraka, KD {loss:.5f}", flush=True)

        student = student.cpu().eval()
        orig = orig.cpu()
        free_gpu()
        if is_ultra:
            for tup in ultra_rows(name, QC.unwrap_model(student), out_dir, keys):
                emit(proc, tup)
            meta_extra[proc] = {"steps": steps, "folded": n_fold, "wrapped": n_w,
                                "export": "ultralytics"}
        else:
            rows, ver, extra = qdq_rows(pf, student, pf["example"], out_dir,
                                        f"{name}_{proc.lower()}", cpu, gpu, has_cuda)
            for tup in rows:
                emit(proc, tup, extra)
            meta_extra[proc] = {"steps": steps, "folded": n_fold, "wrapped": n_w,
                                "export": "qdq", "onnx_verify": ver}
        del student
        free_gpu()
    del orig
    free_gpu()

    meta = {"model": name, "modes": modes, "export": "ultralytics" if is_ultra else "qdq",
            "note": ("[ORIG] FP32 mjeren JEDNOM i dijeljen -> PTQ2 i QAT2 redci su izravno usporedivi. "
                     "Velicina je iz FP32 state_dicta, ne iz datoteke (ultralytics sprema fp16)."),
            "detalji": meta_extra, "minutes": round((time.time() - t0) / 60.0, 2)}
    Q.write_report(all_rows, cols, os.path.join(out_dir, f"{name}_report.csv"),
                   os.path.join(out_dir, f"{name}_report.json"), meta)

    k0 = keys[0]
    g = lambda pr, f: next((r[k0] for r in all_rows if r["postupak"] == pr and r["format"] == f
                            and isinstance(r.get(k0), (int, float))), None)
    o = g("ORIG", "FP32")
    print(f"\n  --- {name}: {k0}, razmak protiv razmaka ---")
    for pr in modes:
        b = g(pr, "FP32") or o
        for f in ("INT8-OV", "INT8-TRT"):
            v = g(pr, f)
            if b and v:
                print(f"    {pr:4} {f:9}: {b:.5f} -> {v:.5f} ({v - b:+.5f})")
    print(f"  --- {meta['minutes']} min ---", flush=True)


def main():
    modes = {"ptq": ["PTQ"], "qat": ["QAT"], "both": ["PTQ", "QAT"]}[MODE]
    os.makedirs(OUT_ROOT, exist_ok=True)
    for k in MODELS:
        k = k.strip()
        if k not in RQ.PROFILES:
            print(f"[preskacem] {k}")
            continue
        try:
            run(k, modes)
        except BaseException:
            print(f"\n!!! {k} PAO:\n{traceback.format_exc()[-1200:]}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    Q.set_cpu_threads(CPU_THREADS)
    main()
