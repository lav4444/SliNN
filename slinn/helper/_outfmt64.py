"""_outfmt64.py -- provjera prepoznavanja FORMATA detekcijskog izlaza (6.4).

Cilj koji se testira: model koji NIKAD nismo vidjeli mora proci. Sve mreze ovdje su izmisljene
(nemaju veze s ultralytics/torchvision) -- jedino po cemu se mogu razvrstati je OBLIK izlaza.

OUT: ../REPORTS/outfmt64.txt
"""

import os
import sys

import torch
import torch.nn as nn

_SLINN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SLINN not in sys.path:
    sys.path.insert(0, _SLINN)

from plugins.detection import adapters as AD                  # noqa: E402
from plugins.detection import outfmt                          # noqa: E402

OUT = os.path.join(_SLINN, "REPORTS", "outfmt64.txt")
L = []


def say(s=""):
    print(s)
    L.append(s)


# =========================== izmisljeni modeli =========================== #
class DenseNC(nn.Module):                                     # [B, 4+K, N]  (yolov8-stil)
    def __init__(self, k=6, n=8400):
        super().__init__()
        self.k, self.n = k, n
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.stack(list(x))
        b = x.shape[0]
        self.stem(x)
        return torch.rand(b, 4 + self.k, self.n, device=x.device)


class DenseCN(nn.Module):                                     # [B, N, 4+K]  (yolov5-stil poredak)
    def __init__(self, k=6, n=6300):
        super().__init__()
        self.k, self.n = k, n
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.stack(list(x))
        b = x.shape[0]
        self.stem(x)
        return torch.rand(b, self.n, 4 + self.k, device=x.device)


class SetPred(nn.Module):                                     # DETR-stil
    def __init__(self, k=6, q=100):
        super().__init__()
        self.k, self.q = k, q
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.stack(list(x))
        b = x.shape[0]
        self.stem(x)
        return {"pred_logits": torch.rand(b, self.q, self.k + 1, device=x.device),
                "pred_boxes": torch.rand(b, self.q, 4, device=x.device)}


class BoxesDicts(nn.Module):                                  # torchvision-stil
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        xs = list(x) if not torch.is_tensor(x) else list(x)
        self.stem(torch.stack([t if t.dim() == 3 else t[0] for t in xs]))
        return [{"boxes": torch.rand(5, 4) * 100, "labels": torch.randint(0, 6, (5,)),
                 "scores": torch.rand(5)} for _ in xs]


class Multilevel(nn.Module):                                  # sirove glave po FPN razini
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.stack(list(x))
        f = self.stem(x)
        return [f, f[:, :, ::2, ::2], f[:, :, ::4, ::4]]


class Nonsense(nn.Module):                                    # nista slicno detekciji
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 64 * 64, 7)

    def forward(self, x):
        x = x if torch.is_tensor(x) else torch.stack(list(x))
        return self.fc(x.flatten(1))


CASES = [("DenseNC (yolov8-oblik)", DenseNC(), "dense_nc", "YoloAdapter"),
         ("DenseCN (yolov5-oblik)", DenseCN(), "dense_cn", "YoloAdapter"),
         ("SetPred (DETR-oblik)", SetPred(), "set_pred", None),
         ("BoxesDicts (torchvision)", BoxesDicts(), "boxes_dicts", "FrcnnAdapter"),
         ("Multilevel (sirove glave)", Multilevel(), "multilevel", None),
         ("Nonsense (nije detekcija)", Nonsense(), "unknown", None)]


def main():
    dev = torch.device("cpu")
    sample = torch.rand(1, 3, 64, 64, device=dev)

    say("===== 6.4 PREPOZNAVANJE FORMATA IZLAZA =====")
    say("Nijedan model ispod nije iz ultralytics/torchvision -> ime NE pomaze, samo oblik.")
    say("")
    say("{:<28} {:<14} {:<14} {}".format("model", "prepoznato", "decode", "ocekivano"))
    say("-" * 84)
    ok = True
    for name, m, exp_fmt, exp_ad in CASES:
        rep = outfmt.describe(m.to(dev), sample)
        good = rep["format"] == exp_fmt and rep.get("adapter") == exp_ad
        ok = ok and good
        say("{:<28} {:<14} {:<14} {}".format(
            name, rep["format"], str(rep.get("adapter")), "OK" if good else "!! " + exp_fmt))
        say("      {}".format(rep["why"][:96]))
    say("")

    say("--- pick_adapter na nepoznatom modelu (puni put) ---")
    got = AD.pick_adapter(DenseNC().to(dev), sample_input=sample)
    say("   DenseNC -> {}".format(getattr(got, "__name__", got)))
    say("")
    say("--- pick_adapter kad NEMA decode-a (mora degradirati, ne pasti) ---")
    got2 = AD.pick_adapter(Nonsense().to(dev), sample_input=sample)
    say("   Nonsense -> {}  (None = KD-only, proces se NASTAVLJA)".format(got2))
    degraded = got2 is None
    say("")

    say("--- probe_box_layout: konvencija okvira MJERENJEM ---")
    H, W = 640, 640

    def to_cxcywh(xyxy):
        return torch.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                            xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], -1)

    # (a) okviri koji DODIRUJU rub -> razlika xywh/cxcywh je MJERLJIVA
    edge = torch.tensor([[0., 0., 120., 120.], [520., 500., 640., 640.], [0., 300., 90., 420.],
                         [560., 10., 640., 130.], [200., 0., 330., 100.], [10., 520., 140., 640.]])
    # (b) okviri SVI DUBOKO UNUTRA -> razlika je NEMJERLJIVA, mora priznati nesigurnost
    inner = torch.tensor([[200., 200., 300., 320.], [300., 300., 400., 500.], [150., 260., 260., 360.]])

    lay_ok = True
    for tag, b, exp_lay, exp_norm in [
            ("xyxy/px  (rub)", edge, "xyxy", False),
            ("cxcywh/px (rub)", to_cxcywh(edge), "cxcywh", False),
            ("cxcywh/norm (rub)", to_cxcywh(edge) / torch.tensor([W, H, W, H]), "cxcywh", True)]:
        r = outfmt.probe_box_layout(b, (H, W))
        good = (r["layout"] == exp_lay and r["normalized"] == exp_norm and r["confident"])
        lay_ok = lay_ok and good
        say("   {:<20} -> {}/{}  score={:.2f} pouzdano={}  {}".format(
            tag, r["layout"], "norm" if r["normalized"] else "px", r["score"], r["confident"],
            "OK" if good else "!! ocekivano " + exp_lay))

    r = outfmt.probe_box_layout(to_cxcywh(inner), (H, W))
    honest = (not r["confident"]) and len(r["ambiguous"]) > 1
    lay_ok = lay_ok and honest
    say("   {:<20} -> pouzdano={} ambiguous={}  {}".format(
        "cxcywh (sve unutra)", r["confident"], r["ambiguous"],
        "OK (priznaje nesigurnost)" if honest else "!! tiho je pogodio"))
    say("")
    say("VERDIKT: formati={}  degradacija={}  konvencija-okvira={}  ->  {}".format(
        "DA" if ok else "NE", "DA" if degraded else "NE", "DA" if lay_ok else "NE",
        "PROLAZI" if (ok and degraded and lay_ok) else "PADA"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n[zapisano] -> " + OUT)


if __name__ == "__main__":
    main()
