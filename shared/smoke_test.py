# -*- coding: utf-8 -*-
"""Provjera da se svaki baseline model UCITAVA i odradi JEDAN prolaz — prije evaluacije.

Zasto zasebno: full-eager `torch.save(model)` sprema referencu na klasu, ne kod. Model se
ucitava samo ako je modul koji tu klasu definira uvoziv. Izmjereno iz samih pickle-ova:

    housing_mlp         model_mlp                  <- lokalna datoteka uz .pt
    speechcommands_m5   model_m5                   <- lokalna datoteka uz .pt
    voc_deeplabv3       (samo torch/torchvision)
    sst2_distilbert     transformers.*
    yolo26n / yolo26l   ultralytics.*
    midas_depth         geffnet.*, midas.*         <- torch.hub kes (salje se, 4 MB)

Ova skripta zato ne pada na prvom problemu nego prijavi SVE, s modulom koji nedostaje.

UPORABA:  python smoke_test.py        (pokrenuti iz edge/ mape na uredjaju)
"""
import os
import platform
import sys
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Skripta je u shared/, mjereni modeli u BASELINE_RAW/, podaci u shared/datasets/.
BM = os.path.join(ROOT, "BASELINE_RAW")

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load(path, extra_sys_path=None):
    """Ucitaj full-eager .pt; `extra_sys_path` je mapa s lokalnim model_*.py."""
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- po modelu
def housing():
    d = os.path.join(BM, "housing_mlp")
    m = _load(os.path.join(d, "model.pt"), d).eval()
    return m, torch.randn(4, 8)


def m5():
    d = os.path.join(BM, "speechcommands_m5")
    m = _load(os.path.join(d, "model.pt"), d).eval()
    return m, torch.randn(2, 1, 8000)


def deeplab():
    d = os.path.join(BM, "voc_deeplabv3")
    m = _load(os.path.join(d, "model.pt")).eval()
    return m, torch.randn(1, 3, 520, 520)


def distilbert():
    d = os.path.join(BM, "sst2_distilbert")
    m = _load(os.path.join(d, "model.pt")).eval()
    return m, {"input_ids": torch.randint(0, 30000, (2, 32)),
               "attention_mask": torch.ones(2, 32, dtype=torch.long)}


def midas():
    """Trazi module `midas` i `geffnet`, kojih NEMA na PyPI-ju — dolaze iz torch.hub kesa
    (intel-isl/MiDaS 3.5 MB + rwightman/gen-efficientnet 0.5 MB). `data.py` ih vec zna
    pod HUB_DIRS; treba ih samo staviti na sys.path prije `torch.load`."""
    d = os.path.join(BM, "midas_depth")
    sys.path.insert(0, d)
    import data as MD                                  # noqa: E402
    for h in MD.HUB_DIRS:
        if os.path.isdir(h) and h not in sys.path:
            sys.path.insert(0, h)
        elif not os.path.isdir(h):
            raise ModuleNotFoundError(
                f"nema torch.hub kesa {h}. Na uredjaju: "
                f"python -c \"import torch; torch.hub.load('intel-isl/MiDaS','MiDaS_small')\"")
    m = _load(os.path.join(d, "model.pt")).eval()
    return m, torch.randn(1, 3, 256, 256)


def _yolo(tag):
    def f():
        from ultralytics import YOLO
        m = YOLO(os.path.join(BM, tag, tag + ".pt")).model.float().eval()
        return m, torch.randn(1, 3, 640, 640)
    return f


CASES = [
    ("housing_mlp", housing),
    ("speechcommands_m5", m5),
    ("voc_deeplabv3", deeplab),
    ("sst2_distilbert", distilbert),
    ("yolo26n", _yolo("yolo26n")),
    ("yolo26l", _yolo("yolo26l")),
    ("midas_depth", midas),          # zadnji: zna pasti, ne zeli se rusiti ostale
]


# ---------------------------------------------------------------- podaci
DATA = [
    ("housing_mlp/data/california_housing.npz", "datoteka"),
    ("speechcommands_m5/data/SpeechCommands/speech_commands_v0.02", "mapa"),
    ("sst2_distilbert/data/hf", "mapa"),
    ("voc_deeplabv3/data/VOCdevkit/VOC2012/JPEGImages", "mapa"),
    ("midas_depth/data/val", "mapa"),      # 254; train ne ide na edge
]
YOLO_DATA = os.path.join(ROOT, "shared", "datasets", "mini_set",
                         "sub10k_open_images_v7", "images", "val")


def main():
    print("=" * 74)
    print(f"  {platform.machine()}  |  python {sys.version.split()[0]}  |  torch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
          + f"  |  racunam na: {DEV}")
    print("=" * 74)

    print("\n--- PODACI ---")
    for rel, kind in DATA:
        p = os.path.join(BM, rel)
        ok = os.path.isdir(p) if kind == "mapa" else os.path.isfile(p)
        print(f"  [{'OK ' if ok else 'NEMA'}] {rel}")
    ok = os.path.isdir(YOLO_DATA)
    n = len(os.listdir(YOLO_DATA)) if ok else 0
    print(f"  [{'OK ' if ok else 'NEMA'}] yolo val slike"
          + (f"  ({n} — ocekivano 837)" if ok else ""))

    print("\n--- MODELI ---")
    bad = []
    for name, fn in CASES:
        t0 = time.time()
        try:
            m, x = fn()
            m = m.to(DEV)
            with torch.no_grad():
                y = m(**{k: v.to(DEV) for k, v in x.items()}) if isinstance(x, dict) \
                    else m(x.to(DEV))
            shape = getattr(y, "shape", None)
            if shape is None:
                shape = type(y).__name__
            print(f"  [OK ] {name:20} {time.time() - t0:6.2f} s   izlaz {shape}")
        except Exception as e:
            first = str(e).split("\n")[0][:88]
            print(f"  [PAD] {name:20} {type(e).__name__}: {first}")
            bad.append((name, e))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    if not bad:
        print("  Svi modeli ucitani i prosli jedan prolaz.")
    else:
        print(f"  PALO: {len(bad)} / {len(CASES)}")
        for name, e in bad:
            print(f"\n  ----- {name}")
            traceback.print_exception(type(e), e, e.__traceback__, limit=3)
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
