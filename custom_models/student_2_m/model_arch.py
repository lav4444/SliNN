"""
Larger student detector (~2.07M params) with PAN neck and SPPF block.

Architectural step up from the width-scaled `student_1_m` (~1.01M params),
following modern YOLO design conventions (YOLOv5/v8/v10/v11 style):

    * SPPF (Spatial Pyramid Pooling - Fast) at end of backbone:
        Adds multi-scale receptive field to P5 features via a chain of 5x5
        maxpools. ~93k params, well-documented +0.5..1 mAP in YOLO literature.

    * PAN (Path Aggregation Network) neck:
        Top-down (FPN) + bottom-up paths so localization information from the
        high-resolution P3 stage propagates back up to P4 and P5. Critical for
        accurate big-object boxes (trucks, buses). Modern YOLO uses PAN; the
        plain top-down FPN is a YOLOv3-era design.

Backbone identical to the discarded plain-FPN 2M variant (width-only scaled
from 1M):
    Channels:  32 / 64 / 96 / 128 / 192
    Depth:     1 / 2 / 4 / 4 / 2 blocks per stage

Trade-off vs the discarded plain-FPN 2M: same param budget, but slimmer neck
(neck_ch 96 -> 64) buys the bottom-up PAN path + SPPF.

Output format and anchor layout are identical to all student variants —
drop-in replacement in any KD training/eval script:
    forward(x):  raw tensor [B, 4+nc, 8400]
    decode(raw): boxes_xywh [B, 8400, 4] in pixel coords on 640x640 letterbox
                 class_probs [B, 8400, nc] in [0, 1]
    Anchors:     P3(80x80) + P4(40x40) + P5(20x20) at strides 8/16/32

Operators used (all export-friendly): Conv2d, BatchNorm2d, SiLU, MaxPool2d,
Upsample(nearest), Concat, Reshape, Sigmoid.
ONNX (opset 14+), NCNN, TensorRT all supported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5+ design).

    Equivalent receptive field to classic SPP (with parallel 5/9/13 kernels)
    but ~3x faster by chaining three identical 5x5 maxpools. Adds multi-scale
    spatial context to the deepest feature map without changing its size.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, k=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv2 = ConvBNAct(hidden * 4, out_ch, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class StudentYOLO(nn.Module):
    NUM_CLASSES = 6
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 640

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE):
        super().__init__()
        self.nc = num_classes
        self.no = 4 + num_classes
        self.input_size = input_size

        # ---- Backbone ----
        self.stem = ConvBNAct(3, 32, k=3, s=2)
        self.dark2 = nn.Sequential(
            ConvBNAct(32, 64, k=3, s=2),
            ConvBNAct(64, 64, k=3, s=1),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(64, 96, k=3, s=2),
            ConvBNAct(96, 96, k=3, s=1),
            ConvBNAct(96, 96, k=3, s=1),
            ConvBNAct(96, 96, k=3, s=1),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(96, 128, k=3, s=2),
            ConvBNAct(128, 128, k=3, s=1),
            ConvBNAct(128, 128, k=3, s=1),
            ConvBNAct(128, 128, k=3, s=1),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(128, 192, k=3, s=2),
            ConvBNAct(192, 192, k=3, s=1),
        )

        # ---- SPPF on P5 (multi-scale RF) ----
        self.sppf = SPPF(192, 192, k=5)

        # ---- PAN neck (top-down FPN + bottom-up path aggregation) ----
        neck_ch = 64
        # Laterals: project backbone outputs to common neck width
        self.lat_p3 = ConvBNAct(96, neck_ch, k=1, p=0)
        self.lat_p4 = ConvBNAct(128, neck_ch, k=1, p=0)
        self.lat_p5 = ConvBNAct(192, neck_ch, k=1, p=0)
        # Top-down fuses (P5 -> P4 -> P3): concat(lateral, upsampled-deeper) -> 3x3
        self.fuse_td_p4 = ConvBNAct(neck_ch * 2, neck_ch, k=3)
        self.fuse_td_p3 = ConvBNAct(neck_ch * 2, neck_ch, k=3)
        # Bottom-up path (P3 -> P4 -> P5): downsample + concat + 3x3
        self.down_p3 = ConvBNAct(neck_ch, neck_ch, k=3, s=2)
        self.fuse_bu_p4 = ConvBNAct(neck_ch * 2, neck_ch, k=3)
        self.down_p4 = ConvBNAct(neck_ch, neck_ch, k=3, s=2)
        self.fuse_bu_p5 = ConvBNAct(neck_ch * 2, neck_ch, k=3)

        # ---- Heads (one per scale, coupled) ----
        self.head_p3 = self._make_head(neck_ch)
        self.head_p4 = self._make_head(neck_ch)
        self.head_p5 = self._make_head(neck_ch)

        self._init_weights()
        self._build_anchor_grid(input_size)

    def _make_head(self, in_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(in_ch, in_ch, k=3),
            nn.Conv2d(in_ch, self.no, 1, 1, 0),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias final 1x1 class outputs so initial sigmoid prob ~ 0.01.
        prior = 0.01
        bias_init = -float(torch.log(torch.tensor((1 - prior) / prior)))
        for head in (self.head_p3, self.head_p4, self.head_p5):
            final: nn.Conv2d = head[-1]
            with torch.no_grad():
                final.bias[4:].fill_(bias_init)

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
        # Backbone
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        p5 = self.sppf(p5)

        # Laterals
        l3 = self.lat_p3(p3)
        l4 = self.lat_p4(p4)
        l5 = self.lat_p5(p5)

        # Top-down: P5 -> P4 -> P3
        td_p4 = self.fuse_td_p4(torch.cat([l4, F.interpolate(l5, scale_factor=2, mode="nearest")], dim=1))
        td_p3 = self.fuse_td_p3(torch.cat([l3, F.interpolate(td_p4, scale_factor=2, mode="nearest")], dim=1))

        # Bottom-up: P3 -> P4 -> P5
        bu_p3 = td_p3
        bu_p4 = self.fuse_bu_p4(torch.cat([td_p4, self.down_p3(bu_p3)], dim=1))
        bu_p5 = self.fuse_bu_p5(torch.cat([l5, self.down_p4(bu_p4)], dim=1))

        # Heads (each output map: [B, no, H, W])
        o3 = self.head_p3(bu_p3)   # [B, no, 80, 80]
        o4 = self.head_p4(bu_p4)   # [B, no, 40, 40]
        o5 = self.head_p5(bu_p5)   # [B, no, 20, 20]

        b = x.shape[0]
        return torch.cat(
            [o3.reshape(b, self.no, -1), o4.reshape(b, self.no, -1), o5.reshape(b, self.no, -1)],
            dim=2,
        )  # [B, no, 8400] raw

    def decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode raw output to (boxes_xywh, class_probs) matching teacher format."""
        box_raw = raw[:, :4, :].permute(0, 2, 1)   # [B, 8400, 4]
        cls_raw = raw[:, 4:, :].permute(0, 2, 1)   # [B, 8400, nc]

        anchor_xy = self.anchor_xy.unsqueeze(0)                   # [1, 8400, 2]
        stride = self.anchor_stride.unsqueeze(0).unsqueeze(-1)    # [1, 8400, 1]

        cxcy = anchor_xy + box_raw[..., :2] * stride
        wh = torch.exp(box_raw[..., 2:].clamp(min=-8.0, max=8.0)) * stride
        boxes_xywh = torch.cat([cxcy, wh], dim=-1)                # [B, 8400, 4]
        class_probs = torch.sigmoid(cls_raw)                      # [B, 8400, nc]
        return boxes_xywh, class_probs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _component_breakdown(model: nn.Module):
    print(f"{'component':<20} {'params':>12}")
    print("-" * 34)
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n > 0:
            print(f"{name:<20} {n:>12,}")


if __name__ == "__main__":
    model = StudentYOLO(num_classes=6).eval()
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}  ({n_params / 1e6:.3f} M)\n")

    _component_breakdown(model)

    print("\n--- Forward sanity check ---")
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        raw = model(x)
        boxes, probs = model.decode(raw)
    print(f"input              {tuple(x.shape)}")
    print(f"raw output         {tuple(raw.shape)}    expected (2, 10, 8400)")
    print(f"decoded boxes_xywh {tuple(boxes.shape)} expected (2, 8400, 4)")
    print(f"decoded class_probs{tuple(probs.shape)} expected (2, 8400, 6)")
    print(f"prob range:        [{probs.min().item():.4f}, {probs.max().item():.4f}]  "
          f"(expected ~0.01 from biased init)")
