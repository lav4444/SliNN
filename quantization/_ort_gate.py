import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER; import logging; LOGGER.setLevel(logging.ERROR)
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx

W = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
YAML = "/home/tomi/code/dipl/quantization/yolo_coco_data/yolo_coco.yaml"
IMG = 640
HERE = "/home/tomi/code/dipl/quantization"

lb = LetterBox((IMG, IMG), auto=False, stride=32)
def prep(p):
    im = lb(image=cv2.imread(p))
    im = im[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im[None], np.float32) / 255.0

class Reader(CalibrationDataReader):
    def __init__(self, paths, name):
        self.data = iter([{name: prep(p)} for p in paths]);
    def get_next(self):
        return next(self.data, None)

onnx_fp = YOLO(W).export(format="onnx", imgsz=IMG, batch=1, verbose=False)
print("onnx:", onnx_fp)
in_name = onnx.load(onnx_fp).graph.input[0].name
print("input:", in_name)
calib = sorted(glob.glob("/home/tomi/code/dipl/quantization/yolo_coco_data/images/train/*"))[:48]
pre_fp = os.path.join(HERE, "yolo26n_pre.onnx")
quant_pre_process(onnx_fp, pre_fp, skip_symbolic_shape=True)
out_fp = os.path.join(HERE, "yolo26n_int8_ort.onnx")
quantize_static(pre_fp, out_fp, Reader(calib, in_name), per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                calibrate_method=CalibrationMethod.Entropy)
print("quantized:", out_fp, round(os.path.getsize(out_fp)/1024**2, 2), "MB")

r = YOLO(out_fp, task="detect").val(data=YAML, split="val", imgsz=IMG, conf=0.001,
                                    device="cpu", batch=1, verbose=False, plots=False)
print("RESULT ORT-INT8 map50-95=%.4f map50=%.4f speed_inf=%.2fms" % (r.box.map, r.box.map50, r.speed.get("inference", -1)))
