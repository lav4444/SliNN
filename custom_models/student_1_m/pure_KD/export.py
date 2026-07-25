"""
Export trained StudentYOLO to ONNX.

Output: student.onnx with input "images" [B, 3, 640, 640] and output "raw" [B, 10, 8400].
Decoding (anchor-free, sigmoid for class) must be done in the runtime that loads the model.

For NCNN: onnx -> onnx2ncnn student.onnx student.param student.bin
For TensorRT: trtexec --onnx=student.onnx --saveEngine=student.engine --fp16

Run:
    python export.py
"""

from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_arch import StudentYOLO


SCRIPT_DIR = Path(__file__).parent
CKPT_PATH = SCRIPT_DIR / "checkpoints" / "best.pt"
ONNX_PATH = SCRIPT_DIR / "student.onnx"
OPSET = 14


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device("cpu")  # export on CPU for portability
    model = StudentYOLO(num_classes=6, input_size=640).to(device).eval()
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val mAP@50:95 = {ckpt.get('val_map', float('nan')):.4f})")

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(ONNX_PATH),
            input_names=["images"],
            output_names=["raw"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes={
                "images": {0: "batch"},
                "raw": {0: "batch"},
            },
        )
    print(f"Exported: {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.2f} MB)")
    print("\nNext steps:")
    print("  NCNN:     onnx2ncnn student.onnx student.param student.bin")
    print("  TensorRT: trtexec --onnx=student.onnx --saveEngine=student.engine --fp16")
    print("\nNote: ONNX export contains the raw output [B, 10, 8400].")
    print("Decoding (boxes from anchors, sigmoid for class probs) is done in your")
    print("runtime code — see StudentYOLO.decode() and the anchor_xy / anchor_stride buffers.")


if __name__ == "__main__":
    main()
