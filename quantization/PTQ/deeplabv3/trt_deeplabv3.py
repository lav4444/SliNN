
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ptq_deeplabv3 as PT

Q = PT.Q
D = PT.D
EB = PT.EB
IMG = PT.LAT_SIZE

ONNX_PATH   = os.path.join(_HERE, "deeplabv3.onnx")
ENGINE_INT8 = os.path.join(_HERE, "deeplabv3_int8.engine")
ENGINE_FP16 = os.path.join(_HERE, "deeplabv3_fp16.engine")
CALIB_CACHE = os.path.join(_HERE, "deeplabv3_int8_calib.cache")
TRT_CALIB_IMAGES = 128
TRT_WORKSPACE_GB = 4

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class OutOnly(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


def _preset():
    tf = D.transform()
    mean = list(getattr(tf, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(tf, "std", [0.229, 0.224, 0.225]))
    return mean, std


def square_tensor(img, mean, std):
    from torchvision.transforms import functional as TF
    t = TF.to_tensor(TF.resize(img, [IMG, IMG]))
    return TF.normalize(t, mean, std).unsqueeze(0)


def export_onnx(path):
    model = OutOnly(PT.load_fp32()).eval().cpu()
    x = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(model, x, path, input_names=["input"], output_names=["out"],
                      opset_version=17, dynamic_axes=None)
    print(f"  [onnx] {path}  ({os.path.getsize(path)/1024**2:.2f} MB)")


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images_np, cache_path):
        super().__init__()
        self.images = images_np; self.idx = 0; self.cache_path = cache_path
        self.dev = torch.empty(tuple(images_np[0].shape), dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.images):
            return None
        self.dev.copy_(torch.from_numpy(np.ascontiguousarray(self.images[self.idx], np.float32)))
        self.idx += 1
        return [int(self.dev.data_ptr())]

    def read_calibration_cache(self):
        return open(self.cache_path, "rb").read() if os.path.exists(self.cache_path) else None

    def write_calibration_cache(self, cache):
        open(self.cache_path, "wb").write(cache)


def build_engine(onnx_path, engine_path, int8=False, fp16=False, calib=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("   ONNX parse err:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, TRT_WORKSPACE_GB * (1 << 30))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calib
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network vratio None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  [engine] {engine_path}  ({os.path.getsize(engine_path)/1024**2:.2f} MB)")


class TRTRunner:
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n; self.in_shape = tuple(self.engine.get_tensor_shape(n))
            else:
                self.out_name = n; self.out_shape = tuple(self.engine.get_tensor_shape(n))
        self.in_t = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        self.out_t = torch.empty(self.out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.in_name, int(self.in_t.data_ptr()))
        self.ctx.set_tensor_address(self.out_name, int(self.out_t.data_ptr()))

    def _enqueue(self):
        self.ctx.execute_async_v3(self.stream)

    def infer(self, x):
        self.in_t.copy_(x.to("cuda", torch.float32))
        self._enqueue(); torch.cuda.synchronize()
        return self.out_t.detach().cpu().clone()

    def preload_for_latency(self, x):
        self.in_t.copy_(x.to("cuda", torch.float32))
        torch.cuda.synchronize()


@torch.no_grad()
def confusion_square(infer, mean, std):
    ds = D.voc(PT.EVAL_SPLIT)
    n = len(ds) if PT.EVAL_LIMIT is None else min(PT.EVAL_LIMIT, len(ds))
    C = D.NUM_CLASSES
    conf = torch.zeros(C, C, dtype=torch.long)
    for i in range(n):
        img, mask = ds[i]
        out = infer(square_tensor(img, mean, std)).float()
        m = torch.as_tensor(np.array(mask), dtype=torch.long)
        out = F.interpolate(out, size=m.shape, mode="bilinear", align_corners=False)
        pred = out.argmax(1)[0]
        valid = (m != D.IGNORE) & (m < C)
        conf += torch.bincount(C * m[valid] + pred[valid], minlength=C * C).reshape(C, C)
    return conf


def panel_square(infer, mean, std, names):
    miou, pacc, iou = EB.metrics(confusion_square(infer, mean, std))
    return {"miou": float(miou), "pixel_acc": float(pacc),
            "per_class_iou": {names[i]: float(iou[i]) for i in range(len(names))}}


def main():
    if not os.path.exists(PT.OUT_JSON):
        print(f"Nema {PT.OUT_JSON} — prvo pokreni ptq_deeplabv3.py"); return
    if not torch.cuda.is_available():
        print("Nema CUDA — TensorRT nije moguć."); return
    Q.set_cpu_threads(PT.CPU_THREADS)

    names = D.classes()
    mean, std = _preset()
    print(f"\n########## TensorRT — DeepLabV3-MobileNetV3 (TRT {trt.__version__}, ulaz 1x3x{IMG}x{IMG}) ##########")

    results = {}

    m32 = PT.load_fp32().to("cuda").eval()
    ref_infer = lambda x: m32(x.to("cuda"))["out"].cpu()            # noqa: E731
    p_ref = panel_square(ref_infer, mean, std, names)
    x_ex = torch.randn(1, 3, IMG, IMG, device="cuda")
    with torch.no_grad():
        lat_ref = Q.benchmark(lambda: m32(x_ex), "cuda", PT.LAT_WARMUP, PT.LAT_ITERS)
    results["FP32-sq520"] = {"panel": p_ref, "lat": lat_ref,
                             "size_mb": Q.model_size_mb(PT.load_fp32()), "backend": "PyTorch (TRT referenca)"}
    print(f"  [FP32-sq520] mIoU={p_ref['miou']:.4f} pixAcc={p_ref['pixel_acc']:.4f} | "
          f"GPU {lat_ref['median_ms']:.4f} ms")
    del m32; torch.cuda.empty_cache()

    export_onnx(ONNX_PATH)

    ds_tr = D.voc("train")
    calib_imgs = [square_tensor(ds_tr[i][0], mean, std).numpy().astype(np.float32)
                  for i in range(min(TRT_CALIB_IMAGES, len(ds_tr)))]
    print(f"  [calib] {len(calib_imgs)} slika")

    for tag, kw, engine_path in [("FP16-TRT", dict(fp16=True), ENGINE_FP16),
                                 ("INT8-TRT", dict(int8=True), ENGINE_INT8)]:
        try:
            calib = Calibrator(calib_imgs, CALIB_CACHE) if kw.get("int8") else None
            build_engine(ONNX_PATH, engine_path, calib=calib, **kw)
            r = TRTRunner(engine_path)
            panel = panel_square(r.infer, mean, std, names)
            r.preload_for_latency(torch.randn(1, 3, IMG, IMG))
            latr = Q.benchmark(r._enqueue, "cuda", PT.LAT_WARMUP, PT.LAT_ITERS)
            results[tag] = {"panel": panel, "lat": latr,
                            "size_mb": Q.file_size_mb(engine_path), "backend": "TensorRT"}
            print(f"  [{tag}] mIoU={panel['miou']:.4f} pixAcc={panel['pixel_acc']:.4f} | "
                  f"{results[tag]['size_mb']:.2f} MB | GPU {latr['median_ms']:.4f} ms")
            del r; torch.cuda.empty_cache()
        except Exception as e:
            results[tag] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
            print(f"  [{tag}] PAO -> {results[tag]['error']}")

    rep = json.load(open(PT.OUT_JSON))
    new_tags = set(results)
    rows = [row for row in rep["rows"] if row.get("format") not in new_tags]
    for tag in ("FP32-sq520", "FP16-TRT", "INT8-TRT"):
        r = results.get(tag)
        if r is None:
            continue
        if "error" in r:
            rows.append({"format": tag, "backend": "TensorRT",
                         "cpu_ms": Q.na("TensorRT je GPU-only"), "gpu_ms": Q.na(r["error"]),
                         "size_mb": Q.na("—"), "miou": Q.na("—"), "pixel_acc": Q.na("—")})
            continue
        p = r["panel"]
        rows.append({"format": tag, "backend": r["backend"],
                     "cpu_ms": Q.na("TensorRT je GPU-only") if "TRT" in tag else Q.na("mjereno u ptq_ skriptu"),
                     "gpu_ms": round(r["lat"]["median_ms"], 4), "size_mb": round(r["size_mb"], 4),
                     "miou": round(p["miou"], 5), "pixel_acc": round(p["pixel_acc"], 5),
                     "per_class_iou": p["per_class_iou"]})
        rep.setdefault("meta", {}).setdefault("lat_full", {})[tag] = r["lat"]

    rep["meta"]["tensorrt"] = {
        "version": trt.__version__, "calib_images": TRT_CALIB_IMAGES, "input": f"1x3x{IMG}x{IMG}",
        "note": ("TRT engine ima FIKSNI kvadratni ulaz, a torchvision preset radi resize kraćeg ruba -> "
                 "TRT redci se uspoređuju s FP32-sq520 (isti model, isti kvadratni put), ne s FP32 retkom."),
    }
    Q.write_report(rows, PT.CSV_COLS, PT.OUT_CSV, PT.OUT_JSON, rep["meta"])
    print(f"\n########## TRT GOTOVO — tablica ažurirana ({len(rows)} formata) ##########")


if __name__ == "__main__":
    main()
