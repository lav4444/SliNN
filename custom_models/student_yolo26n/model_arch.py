"""
Student that uses the actual ultralytics YOLO26n architecture out-of-the-box.

Methodology comparison purpose: takes the exact same architecture as the
shipped `yolo26n.pt` (2.5M params) but with random initialization (no pretrained
weights) and only 6 output classes (instead of COCO's 80). It is then trained
through our KD pipeline like all other student variants. Ablation we get:

    yolo26n.pt              (Ultralytics pretrained + Ultralytics training on COCO)
    student_yolo26n (this)  (Random init + OUR KD training on our 6-class subset)

If the KD training is decent, these two should land in the same ballpark on
the same test set — that validates the training methodology.

Output format mirrors all other student variants:
    forward(x):  raw tensor [B, 4+nc, 8400]
    decode(raw): boxes_xywh [B, 8400, 4] in 640x640 letterbox px
                 class_probs [B, 8400, nc] in [0, 1]

BUT internally the box channels here are *decoded xywh in pixel coords*
(unlike our custom students whose box channels are anchor-relative raw).
This is because yolo26n applies its own dist2bbox decode in the Detect head.
To signal this, we set BOX_OUTPUT_FORMAT = "decoded" — train_kd.py reads
this attribute to switch between raw-space and decoded-space box loss.
Our custom students don't set it, so they default to "raw" — no breakage.

Class channels here are LOGITS (pre-sigmoid). yolo26n applies sigmoid in its
eval-mode head; we reverse it with `logit(p) = log(p / (1-p))` so the
focal-loss formulation in train_kd.py keeps working unchanged.

Architecture details (from ultralytics/cfg/models/26/yolo26.yaml, scale 'n'):
    - depth multiplier 0.5, width multiplier 0.25, max channels 1024
    - Backbone: Conv stem + C3k2 blocks + SPPF + C2PSA
    - Head: Standard ultralytics Detect, reg_max=1 (DFL is Identity)
    - end2end mode is disabled in our wrapper (we want dense pre-NMS output for KD)

All operators in the architecture are export-friendly (ONNX opset 14+, NCNN,
TensorRT) since ultralytics designed yolo26 for deployment from day one.
"""

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


class StudentYOLO(nn.Module):
    BOX_OUTPUT_FORMAT = "decoded"   # ← signals train_kd.py to use decoded-space box loss
    NUM_CLASSES = 6
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 640
    CFG = "yolo26n.yaml"

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE):
        super().__init__()
        self.nc = num_classes
        self.no = 4 + num_classes
        self.input_size = input_size

        # Build yolo26n architecture with random init, override nc to 6
        self.yolo = DetectionModel(cfg=self.CFG, ch=3, nc=num_classes, verbose=False)

        # Locate Detect head and disable end2end so it returns dense pre-NMS output
        self.detect_head = self._find_detect_head()
        if getattr(self.detect_head, "end2end", False):
            self.detect_head.end2end = False

        # Build OUR anchor grid (in pixel coords, P3/P4/P5 stride 8/16/32)
        # — kept consistent with sibling students so train_kd.py / evaluate_student.py
        # can use the same anchor_xy / anchor_stride buffers downstream.
        self._build_anchor_grid(input_size)

    def _find_detect_head(self) -> nn.Module:
        candidates = [m for m in self.yolo.modules()
                      if m.__class__.__name__ in ("Detect", "v10Detect", "DetectV2")]
        if not candidates:
            raise RuntimeError("No Detect head found in yolo26n architecture")
        return candidates[-1]

    def _build_anchor_grid(self, input_size: int):
        anchor_xy_list = []
        stride_list = []
        for s in self.STRIDES:
            g = input_size // s
            yv, xv = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
            grid = torch.stack([xv, yv], dim=-1).float() + 0.5
            grid = grid * s
            anchor_xy_list.append(grid.reshape(-1, 2))
            stride_list.append(torch.full((g * g,), float(s)))
        self.register_buffer("anchor_xy", torch.cat(anchor_xy_list, dim=0))
        self.register_buffer("anchor_stride", torch.cat(stride_list, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 4+nc, 8400] with decoded xywh boxes (px) and class LOGITS."""
        # Trick: temporarily put the Detect head in eval mode so it returns the
        # post-dist2bbox decoded output (single tensor) instead of the training-mode
        # list of raw per-scale tensors. Other modules (BN etc.) stay in whatever
        # training state they were in — only the head's behavior flips.
        head_was_training = self.detect_head.training
        self.detect_head.training = False
        try:
            out = self.yolo(x)
        finally:
            self.detect_head.training = head_was_training

        # eval-mode head returns y if export else (y, x). Pick the y tensor.
        if isinstance(out, (tuple, list)):
            out = out[0]

        # out is [B, 4+nc, 8400] with:
        #   [:, :4, :] = decoded xywh in 640x640 letterbox pixel coords
        #   [:, 4:, :] = sigmoid'd class probabilities
        boxes = out[:, :4, :]
        probs = out[:, 4:, :].clamp(min=1e-7, max=1.0 - 1e-7)
        # Reverse the sigmoid to recover logits for focal-loss compatibility.
        # Mathematically identical to logits-before-sigmoid in gradient terms
        # (the dlogit/dsigmoid * dsigmoid/dlogit cancels to identity); just costs
        # an extra log+clamp per forward. Single-sigmoid in our custom students
        # is more numerically stable but functionally equivalent here.
        cls_logits = torch.log(probs / (1.0 - probs))
        return torch.cat([boxes, cls_logits], dim=1)

    def decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode our forward() output to (boxes_xywh, class_probs) — matching sibling API.

        Since forward() already returns DECODED boxes (in our case), this is just
        a permute + sigmoid on the class part. The signature still matches what
        evaluate_student.py / gui_test.py expect.
        """
        boxes_xywh = raw[:, :4, :].permute(0, 2, 1)        # already decoded
        cls_logits = raw[:, 4:, :].permute(0, 2, 1)
        class_probs = torch.sigmoid(cls_logits)
        return boxes_xywh, class_probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = StudentYOLO(num_classes=6).eval()
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}  ({n_params / 1e6:.3f} M)")
    print(f"BOX_OUTPUT_FORMAT = {model.BOX_OUTPUT_FORMAT}")
    print(f"Detect head: nc={model.detect_head.nc}, reg_max={model.detect_head.reg_max}, "
          f"end2end={getattr(model.detect_head, 'end2end', None)}")

    print("\n--- Forward sanity check ---")
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        raw = model(x)
        boxes, probs = model.decode(raw)
    print(f"input              {tuple(x.shape)}")
    print(f"raw output         {tuple(raw.shape)}    expected (2, 10, 8400)")
    print(f"decoded boxes_xywh {tuple(boxes.shape)} expected (2, 8400, 4)")
    print(f"decoded class_probs{tuple(probs.shape)} expected (2, 8400, 6)")
    print(f"boxes range:  cx∈[{boxes[..., 0].min().item():.1f}, {boxes[..., 0].max().item():.1f}], "
          f"w∈[{boxes[..., 2].min().item():.1f}, {boxes[..., 2].max().item():.1f}]  (expected ~0..640)")
    print(f"prob range:        [{probs.min().item():.4f}, {probs.max().item():.4f}]")
