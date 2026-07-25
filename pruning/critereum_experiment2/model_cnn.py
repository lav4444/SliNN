"""
model_cnn.py — jednostavan "skolski" CNN za MULTI-LABEL klasifikaciju (6 klasa).

Namjerno klasican dizajn: nekoliko (Conv-BN-ReLU + MaxPool) blokova -> flatten ->
dense (Linear) slojevi. Svrha: imati i CONV FILTERE i DENSE NEURONE za structured
pruning (za razliku od exp1 gdje je bilo samo filtera).

Ulaz: 320x320 RGB. Izlaz: 6 logita (po jedan po klasi, sigmoid + BCE).
Klase: Person, Car, Truck, Bus, Motorcycle, Bicycle (prisutnost u slici).

Backbone (5 stepenica, svaka /2):   320 ->160 ->80 ->40 ->20 ->10
Kanali:                              3 ->32 ->64 ->128 ->128 ->128
Flatten: 128 * 10 * 10 = 12800  ->  Dense 256 -> Dense 128 -> Dense 6
"""

import torch
import torch.nn as nn


NUM_CLASSES = 6
INPUT_SIZE = 320
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=1, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SchoolCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, input_size: int = INPUT_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # ---- Konvolucijski dio (filteri za rezati) ----
        self.conv1 = ConvBNReLU(3, 32)
        self.conv2 = ConvBNReLU(32, 64)
        self.conv3 = ConvBNReLU(64, 128)
        self.conv4 = ConvBNReLU(128, 128)
        self.conv5 = ConvBNReLU(128, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # odredi flatten dim i prostornu povrsinu na flatten rubu
        with torch.no_grad():
            d = torch.zeros(1, 3, input_size, input_size)
            f = self._features(d)
            self.feat_channels = f.shape[1]
            self.flatten_hw = f.shape[2] * f.shape[3]    # H*W (block size za conv->fc1)
            flat = f.numel()

        # ---- Dense dio (neuroni za rezati) ----
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)   # izlaz (ne rezati)
        self.act = nn.ReLU(inplace=True)

    def _features(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        return x

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)            # raw logiti [B, num_classes]
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = SchoolCNN().eval()
    n = count_parameters(m)
    print(f"SchoolCNN params: {n:,}  ({n/1e6:.3f} M)")
    print(f"feat_channels={m.feat_channels}  flatten_hw={m.flatten_hw}  "
          f"fc1.in={m.fc1.in_features}")
    print("\ncomponent params:")
    for name, child in m.named_children():
        c = sum(p.numel() for p in child.parameters())
        if c:
            print(f"  {name:<8} {c:>12,}")
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        y = m(x)
    print(f"\ninput {tuple(x.shape)} -> output {tuple(y.shape)} (ocek. (2,6))")
