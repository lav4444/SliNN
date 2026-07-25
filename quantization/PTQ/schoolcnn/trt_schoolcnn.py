"""
trt_schoolcnn.py — TensorRT INT8 (+ FP16) za SchoolCNN; popunjava INT8-TRT redak cross-backend tablice.

Tok: SchoolCNN -> ONNX (batch 1) -> TensorRT engine
  * INT8-TRT: vlastiti IInt8EntropyCalibrator2 (kalibracija na našem podskupu),
  * FP16-TRT: bonus redak (TRT half).
Mjeri ISTI bogati panel kvalitete (reuse ptq_schoolcnn.eval_panel) + GPU latenciju (Q.benchmark, ulaz već na uređaju)
+ veličinu enginea, pa UČITA postojeći schoolcnn_ptq_report i prepiše ga s TRT redovima (jedna ujedinjena tablica).

Device-buferi su TORCH CUDA tenzori (TRT-u treba samo data_ptr() + stream) -> bez cuda-python (izbjegnut CUDA mismatch).
Pokrenuti NAKON ptq_schoolcnn.py. BEZ CLI argumenata.
"""

import json
import os
import sys

import numpy as np
import torch
import tensorrt as trt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ptq_schoolcnn as PT                     # reuse: eval_panel, load_fp32, C2, Q, CSV_COLS, OUT_*, konstante

Q = PT.Q
C2 = PT.C2
NC = PT.NUM_CLASSES
IMG = PT.INPUT_SIZE

ONNX_PATH   = os.path.join(_HERE, "schoolcnn.onnx")
ENGINE_INT8 = os.path.join(_HERE, "schoolcnn_int8.engine")
ENGINE_FP16 = os.path.join(_HERE, "schoolcnn_fp16.engine")
CALIB_CACHE = os.path.join(_HERE, "schoolcnn_int8_calib.cache")
TRT_CALIB_IMAGES = 256                          # slika za INT8 kalibraciju (batch 1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def export_onnx(path):
    model = PT.load_fp32().eval().cpu()
    x = torch.randn(1, 3, IMG, IMG)
    torch.onnx.export(model, x, path, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes=None)
    print(f"  [onnx] {path}")


class Calibrator(trt.IInt8EntropyCalibrator2):
    """Hrani INT8 kalibraciju batch po batch (batch=1). Device-buffer = torch CUDA tenzor (data_ptr)."""
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
    network = builder.create_network(0)          # TRT10: explicit batch (default)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("   ONNX parse err:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)    # fp16 fallback za ne-int8 slojeve
        config.int8_calibrator = calib
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network vratio None")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  [engine] {engine_path}  ({os.path.getsize(engine_path)/1024**2:.2f} MB)")


class TRTRunner:
    """Batch-1 izvođenje preko torch CUDA bufera (data_ptr) + torch stream."""
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream
        self.in_name = self.out_name = None
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

    def run_single(self, arr):                   # arr [1,3,IMG,IMG] -> [1,NC] np
        self.in_t.copy_(torch.from_numpy(np.ascontiguousarray(arr, np.float32)))
        self._enqueue(); torch.cuda.synchronize()
        return self.out_t.detach().cpu().numpy().copy()

    def infer_batch(self, x):                    # torch [B,3,IMG,IMG] -> torch [B,NC]
        xn = x.detach().cpu().numpy().astype(np.float32)
        outs = [self.run_single(xn[i:i + 1]) for i in range(xn.shape[0])]
        return torch.from_numpy(np.concatenate(outs, 0))

    def preload_for_latency(self, arr):          # ulaz već na uređaju -> mjeri se samo execute (kao PyTorch GPU)
        self.in_t.copy_(torch.from_numpy(np.ascontiguousarray(arr, np.float32)))
        torch.cuda.synchronize()


def main():
    if not os.path.exists(PT.OUT_JSON):
        print(f"Nema {PT.OUT_JSON} — prvo pokreni ptq_schoolcnn.py"); return
    if not torch.cuda.is_available():
        print("Nema CUDA — TensorRT nije moguć."); return
    Q.set_cpu_threads(PT.CPU_THREADS)

    val_loader = C2.make_loader("val", PT.EVAL_BATCH, shuffle=False, num_workers=4, max_images=PT.EVAL_MAX)
    calib_loader = C2.make_loader("train", PT.EVAL_BATCH, shuffle=False, num_workers=4, max_images=TRT_CALIB_IMAGES)
    example = next(iter(val_loader))[0][:1].clone().numpy().astype(np.float32)

    print("\n########## TensorRT — SchoolCNN ##########")
    export_onnx(ONNX_PATH)

    calib_imgs = []
    for x, _ in calib_loader:
        for i in range(x.shape[0]):
            calib_imgs.append(x[i:i + 1].numpy().astype(np.float32))
            if len(calib_imgs) >= TRT_CALIB_IMAGES:
                break
        if len(calib_imgs) >= TRT_CALIB_IMAGES:
            break

    results = {}
    for tag, kw, engine_path in [
        ("FP16-TRT", dict(fp16=True), ENGINE_FP16),
        ("INT8-TRT", dict(int8=True), ENGINE_INT8),
    ]:
        calib = Calibrator(calib_imgs, CALIB_CACHE) if kw.get("int8") else None
        build_engine(ONNX_PATH, engine_path, calib=calib, **kw)
        r = TRTRunner(engine_path)
        panel = PT.eval_panel(r.infer_batch, val_loader, NC)
        r.preload_for_latency(example)
        latr = Q.benchmark(r._enqueue, "cuda", PT.LAT_WARMUP, PT.LAT_ITERS)
        results[tag] = {"panel": panel, "lat": latr, "size_mb": Q.file_size_mb(engine_path)}
        print(f"  [{tag}] mAP={panel['map_macro']:.4f} acc={panel['acc_macro']:.4f} "
              f"F1={panel['f1_macro']:.4f} | {results[tag]['size_mb']:.2f} MB | GPU {latr['median_ms']:.4f} ms")

    rep = json.load(open(PT.OUT_JSON))
    rows = [row for row in rep["rows"] if row.get("format") not in ("INT8-TRT", "FP16-TRT")]
    for tag in ("FP16-TRT", "INT8-TRT"):
        p = results[tag]["panel"]; latr = results[tag]["lat"]
        row = {"format": tag, "backend": "TensorRT", "cpu_ms": Q.na("TensorRT je GPU-only"),
               "gpu_ms": round(latr["median_ms"], 4), "size_mb": round(results[tag]["size_mb"], 4)}
        row.update({k: round(p[k], 5) for k in p if k != "per_class_ap"})
        row["per_class_ap"] = p["per_class_ap"]
        rows.append(row)
        rep.setdefault("meta", {}).setdefault("lat_full", {})[tag] = latr

    rep["meta"]["tensorrt"] = {"version": trt.__version__, "calib_images": TRT_CALIB_IMAGES}
    Q.write_report(rows, PT.CSV_COLS, PT.OUT_CSV, PT.OUT_JSON, rep["meta"])
    print(f"\n########## TRT GOTOVO — tablica ažurirana ({len(rows)} formata) ##########")


if __name__ == "__main__":
    main()
