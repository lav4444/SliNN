
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


class StudentYOLO(nn.Module):
    BOX_OUTPUT_FORMAT = "decoded"
    NUM_CLASSES = 6
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 640
    CFG = "yolo26n.yaml"

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE):
        super().__init__()
        self.nc = num_classes
        self.no = 4 + num_classes
        self.input_size = input_size

        self.yolo = DetectionModel(cfg=self.CFG, ch=3, nc=num_classes, verbose=False)

        self.detect_head = self._find_detect_head()
        if getattr(self.detect_head, "end2end", False):
            self.detect_head.end2end = False

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
        head_was_training = self.detect_head.training
        self.detect_head.training = False
        try:
            out = self.yolo(x)
        finally:
            self.detect_head.training = head_was_training

        if isinstance(out, (tuple, list)):
            out = out[0]

        boxes = out[:, :4, :]
        probs = out[:, 4:, :].clamp(min=1e-7, max=1.0 - 1e-7)
        cls_logits = torch.log(probs / (1.0 - probs))
        return torch.cat([boxes, cls_logits], dim=1)

    def decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        boxes_xywh = raw[:, :4, :].permute(0, 2, 1)
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
