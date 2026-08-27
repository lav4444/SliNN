"""
model_arch_feat.py — yolo26n student (random init) + FGD feature-KD adapteri, za KD_featlogit.

Student je EGZAKTNA ultralytics yolo26n arhitektura (random init, nc=6, ~2.506M) — kao
student_yolo26n/model_arch.py. Teacher je yolo26l (tap-širine 256/512/512).

RAZLIKA vs hand-rolled studenti: yolo26n neck-izlazi su uži (64/128/256) i NE smijemo mijenjati
arhitekturu (svrha studenta = validacija da random yolo26n + naš KD ≈ pretrenirani yolo26n).
Zato feature-KD koristi UČEĆE FGD-projekcije (1×1 conv: 64→256, 128→512, 256→512) koje
poravnaju kanale na teacherove SAMO za MSE-član. Projekcije:
  - NE hrane glavu (glava vidi native yolo26n neck-izlaze),
  - su TRENING-ONLY: na inferenciji (forward bez return_feats) se ne pozivaju → isporučeni
    model je čisti yolo26n (2.506M; ~213K adaptera je odbačeni overhead).

forward(x)                     -> raw [B, 4+nc, 8400]  (dekodiran xywh px + cls LOGITI)
forward(x, return_feats=True)  -> (raw, [t3, t4, t5])  (projicirani tap featuri, 256/512/512)
decode(raw)                    -> (boxes_xywh, class_probs)
"""

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


TEACHER_TAP_CH = (256, 512, 512)   # yolo26l Detect-head ulazi (P3,P4,P5)
STUDENT_TAP_CH = (64, 128, 256)    # yolo26n Detect-head ulazi (probirano)


class StudentYOLOFeat(nn.Module):
    BOX_OUTPUT_FORMAT = "decoded"   # yolo26n glava daje dekodiran izlaz (kao model_arch.py)
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

        # yolo26n arhitektura, random init, nc=6
        self.yolo = DetectionModel(cfg=self.CFG, ch=3, nc=num_classes, verbose=False)
        self.detect_head = self._find_detect_head()
        if getattr(self.detect_head, "end2end", False):
            self.detect_head.end2end = False

        # FGD trening-only projekcije: yolo26n neck (64/128/256) -> teacher (256/512/512)
        self.feat_proj = nn.ModuleList([
            nn.Conv2d(sc, tc, kernel_size=1, bias=False)
            for sc, tc in zip(STUDENT_TAP_CH, teacher_tap_ch)
        ])
        for m in self.feat_proj:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # pre-hook na glavu: hvata ulaze glave (= neck izlazi) pri svakom forwardu
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
        # Glava u eval-modu -> dekodiran izlaz (jedan tensor); BN itd. ostaju kako jesu.
        head_was_training = self.detect_head.training
        self.detect_head.training = False
        try:
            out = self.yolo(x)                      # pre-hook usput puni self._grab["f"]
        finally:
            self.detect_head.training = head_was_training

        if isinstance(out, (tuple, list)):
            out = out[0]
        boxes = out[:, :4, :]                        # dekodiran xywh (px, 640 letterbox)
        probs = out[:, 4:, :].clamp(min=1e-7, max=1.0 - 1e-7)
        cls_logits = torch.log(probs / (1.0 - probs))  # natrag u logite (focal kompatibilnost)
        raw = torch.cat([boxes, cls_logits], dim=1)     # [B, 4+nc, 8400]

        if return_feats:
            neck = self._grab["f"]                   # [64x80x80, 128x40x40, 256x20x20]
            feats = [self.feat_proj[i](neck[i]) for i in range(len(neck))]  # -> 256/512/512
            return raw, feats
        return raw

    def decode(self, raw: torch.Tensor):
        boxes_xywh = raw[:, :4, :].permute(0, 2, 1)   # već dekodiran
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
