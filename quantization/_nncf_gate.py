import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER; import logging; LOGGER.setLevel(logging.ERROR)
import nncf

W = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
IMG = 640
lb = LetterBox((IMG, IMG), auto=False, stride=32)
def prep(p):
    im = lb(image=cv2.imread(p))[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im[None], np.float32) / 255.0

model = YOLO(W).model.float().eval()
print("model loaded:", type(model).__name__)

paths = sorted(glob.glob("/home/tomi/code/dipl/quantization/yolo_coco_data/images/train/*"))[:32]
calib = [prep(p) for p in paths]
ds = nncf.Dataset(calib, lambda x: torch.from_numpy(x))

print(">>> nncf.quantize ...")
q = nncf.quantize(model, ds, subset_size=32)
print("quantize OK:", type(q).__name__)
with torch.no_grad():
    y = q(torch.randn(1, 3, IMG, IMG))
print("forward OK ->", (tuple(y.shape) if torch.is_tensor(y) else [type(y).__name__, len(y)]))

onnx_fp = "/home/tomi/code/dipl/quantization/yolo26n_qdq.onnx"
torch.onnx.export(q, torch.randn(1, 3, IMG, IMG), onnx_fp, opset_version=17,
                  input_names=["images"], output_names=["out"])
print("ONNX QDQ export OK:", round(os.path.getsize(onnx_fp) / 1024 ** 2, 2), "MB")
print("GATE_OK")
