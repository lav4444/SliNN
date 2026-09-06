import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
from ultralytics.utils import LOGGER; import logging; LOGGER.setLevel(logging.ERROR)

W = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
YAML = "/home/tomi/code/dipl/quantization/yolo_coco_data/yolo_coco.yaml"
IMG = 640

print(">>> export TFLite INT8 (auto-install TF/onnx2tf ako treba)...")
tfl = YOLO(W).export(format="tflite", int8=True, data=YAML, imgsz=IMG, batch=1, verbose=False)
print("tflite:", tfl)
r = YOLO(tfl, task="detect").val(data=YAML, split="val", imgsz=IMG, conf=0.001,
                                 device="cpu", batch=1, verbose=False, plots=False)
print("RESULT TFLITE-INT8 map50-95=%.4f map50=%.4f speed_inf=%.2fms" % (r.box.map, r.box.map50, r.speed.get("inference", -1)))
