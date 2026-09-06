
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


TEACHER_TAP_CH = (256, 512, 512)


class StudentYOLOFeat(nn.Module):
    NUM_CLASSES = 6
    STRIDES = (8, 16, 32)
    INPUT_SIZE = 640

    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE,
                 tap_ch: tuple = TEACHER_TAP_CH):
        super().__init__()
        self.nc = num_classes
        self.no = 4 + num_classes
        self.input_size = input_size
        self.tap_ch = tuple(tap_ch)

        self.stem = ConvBNAct(3, 24, k=3, s=2)
        self.dark2 = nn.Sequential(
            ConvBNAct(24, 48, k=3, s=2),
            ConvBNAct(48, 48, k=3, s=1),
        )
        self.dark3 = nn.Sequential(
            ConvBNAct(48, 64, k=3, s=2),
            ConvBNAct(64, 64, k=3, s=1),
            ConvBNAct(64, 64, k=3, s=1),
        )
        self.dark4 = nn.Sequential(
            ConvBNAct(64, 96, k=3, s=2),
            ConvBNAct(96, 96, k=3, s=1),
            ConvBNAct(96, 96, k=3, s=1),
        )
        self.dark5 = nn.Sequential(
            ConvBNAct(96, 128, k=3, s=2),
            ConvBNAct(128, 128, k=3, s=1),
        )

        neck_ch = 64
        self.lat_p3 = ConvBNAct(64, neck_ch, k=1, p=0)
        self.lat_p4 = ConvBNAct(96, neck_ch, k=1, p=0)
        self.lat_p5 = ConvBNAct(128, neck_ch, k=1, p=0)
        self.fuse_p4 = ConvBNAct(neck_ch * 2, neck_ch, k=3)
        self.fuse_p3 = ConvBNAct(neck_ch * 2, neck_ch, k=3)

        c3, c4, c5 = self.tap_ch
        self.proj_p3 = ConvBNAct(neck_ch, c3, k=1, p=0)
        self.proj_p4 = ConvBNAct(neck_ch, c4, k=1, p=0)
        self.proj_p5 = ConvBNAct(neck_ch, c5, k=1, p=0)

        self.head_p3 = self._make_head(c3, neck_ch)
        self.head_p4 = self._make_head(c4, neck_ch)
        self.head_p5 = self._make_head(c5, neck_ch)

        self._init_weights()
        self._build_anchor_grid(input_size)

    def _make_head(self, in_ch: int, mid_ch: int) -> nn.Sequential:
        return nn.Sequential(
            ConvBNAct(in_ch, mid_ch, k=1, p=0),
            ConvBNAct(mid_ch, mid_ch, k=3),
            nn.Conv2d(mid_ch, self.no, 1, 1, 0),
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
        prior = 0.01
        bias_init = -float(torch.log(torch.tensor((1 - prior) / prior)))
        for head in (self.head_p3, self.head_p4, self.head_p5):
            final: nn.Conv2d = head[-1]
            with torch.no_grad():
                final.bias[4:].fill_(bias_init)

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
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)

        l3 = self.lat_p3(p3)
        l4 = self.lat_p4(p4)
        l5 = self.lat_p5(p5)
        f4 = self.fuse_p4(torch.cat([l4, F.interpolate(l5, scale_factor=2, mode="nearest")], dim=1))
        f3 = self.fuse_p3(torch.cat([l3, F.interpolate(f4, scale_factor=2, mode="nearest")], dim=1))

        t3 = self.proj_p3(f3)
        t4 = self.proj_p4(f4)
        t5 = self.proj_p5(l5)

        o3 = self.head_p3(t3)
        o4 = self.head_p4(t4)
        o5 = self.head_p5(t5)

        b = x.shape[0]
        raw = torch.cat(
            [o3.reshape(b, self.no, -1), o4.reshape(b, self.no, -1), o5.reshape(b, self.no, -1)],
            dim=2,
        )
        if return_feats:
            return raw, [t3, t4, t5]
        return raw

    def decode(self, raw: torch.Tensor):
        box_raw = raw[:, :4, :].permute(0, 2, 1)
        cls_raw = raw[:, 4:, :].permute(0, 2, 1)
        anchor_xy = self.anchor_xy.unsqueeze(0)
        stride = self.anchor_stride.unsqueeze(0).unsqueeze(-1)
        cxcy = anchor_xy + box_raw[..., :2] * stride
        wh = torch.exp(box_raw[..., 2:].clamp(min=-8.0, max=8.0)) * stride
        boxes_xywh = torch.cat([cxcy, wh], dim=-1)
        class_probs = torch.sigmoid(cls_raw)
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
    model = StudentYOLOFeat(num_classes=6).eval()
    n = count_parameters(model)
    print(f"Parameters: {n:,}  ({n / 1e6:.3f} M)\n")
    _component_breakdown(model)

    print("\n--- Forward sanity ---")
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        raw, feats = model(x, return_feats=True)
        boxes, probs = model.decode(raw)
    print(f"raw   {tuple(raw.shape)}   expected (2, 10, 8400)")
    print(f"feats {[tuple(f.shape) for f in feats]}   expected [(2,256,80,80),(2,512,40,40),(2,512,20,20)]")
    print(f"boxes {tuple(boxes.shape)}  probs {tuple(probs.shape)}")
