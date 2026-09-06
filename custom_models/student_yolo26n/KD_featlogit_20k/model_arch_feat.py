
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


TEACHER_TAP_CH = (256, 512, 512)
STUDENT_TAP_CH = (64, 128, 256)


class StudentYOLOFeat(nn.Module):
    BOX_OUTPUT_FORMAT = "decoded"
    NUM_CLASSES = 6
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 640
    CFG = "yolo26n.yaml"

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE,
                 teacher_tap_ch: tuple = TEACHER_TAP_CH):
        super().__init__()
        self.nc = num_classes
        self.no = 4 + num_classes
        self.input_size = input_size

        self.yolo = DetectionModel(cfg=self.CFG, ch=3, nc=num_classes, verbose=False)
        self.detect_head = self._find_detect_head()
        if getattr(self.detect_head, "end2end", False):
            self.detect_head.end2end = False

        self.feat_proj = nn.ModuleList([
            nn.Conv2d(sc, tc, kernel_size=1, bias=False)
            for sc, tc in zip(STUDENT_TAP_CH, teacher_tap_ch)
        ])
        for m in self.feat_proj:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        self._grab = {}
        self.detect_head.register_forward_pre_hook(self._capture_neck)

        self._build_anchor_grid(input_size)

    def _capture_neck(self, module, inp):
        self._grab["f"] = list(inp[0])

    def _find_detect_head(self) -> nn.Module:
        cands = [m for m in self.yolo.modules()
                 if m.__class__.__name__ in ("Detect", "v10Detect", "DetectV2")]
        if not cands:
            raise RuntimeError("No Detect head found in yolo26n architecture")
        return cands[-1]

    def _build_anchor_grid(self, input_size: int):
        anchor_xy_list, stride_list = [], []
        for s in self.STRIDES:
            g = input_size // s
            yv, xv = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
            grid = torch.stack([xv, yv], dim=-1).float() + 0.5
            grid = grid * s
            anchor_xy_list.append(grid.reshape(-1, 2))
            stride_list.append(torch.full((g * g,), float(s)))
        self.register_buffer("anchor_xy", torch.cat(anchor_xy_list, dim=0))
        self.register_buffer("anchor_stride", torch.cat(stride_list, dim=0))

    def forward(self, x: torch.Tensor, return_feats: bool = False):
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
        raw = torch.cat([boxes, cls_logits], dim=1)

        if return_feats:
            neck = self._grab["f"]
            feats = [self.feat_proj[i](neck[i]) for i in range(len(neck))]
            return raw, feats
        return raw

    def decode(self, raw: torch.Tensor):
        boxes_xywh = raw[:, :4, :].permute(0, 2, 1)
        cls_logits = raw[:, 4:, :].permute(0, 2, 1)
        class_probs = torch.sigmoid(cls_logits)
        return boxes_xywh, class_probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = StudentYOLOFeat(num_classes=6).eval()
    total = count_parameters(model)
    proj = sum(p.numel() for p in model.feat_proj.parameters())
    print(f"Total params:   {total:,}  ({total / 1e6:.3f} M)")
    print(f"  yolo26n (deploy):  {total - proj:,}  ({(total - proj) / 1e6:.3f} M)")
    print(f"  FGD adapteri (trening-only): {proj:,}")

    print("\n--- Forward sanity ---")
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        raw, feats = model(x, return_feats=True)
        boxes, probs = model.decode(raw)
    print(f"raw   {tuple(raw.shape)}   expected (2, 10, 8400)")
    print(f"feats {[tuple(f.shape) for f in feats]}   expected [(2,256,80,80),(2,512,40,40),(2,512,20,20)]")
    print(f"boxes {tuple(boxes.shape)}  probs {tuple(probs.shape)}  prob∈[{probs.min():.3f},{probs.max():.3f}]")
