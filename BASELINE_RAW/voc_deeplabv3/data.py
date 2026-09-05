"""data.py — Pascal VOC 2012 (segmentacija) + torchvision transforms, lokalno u ./data/.

voc(split) -> VOCSegmentation (daje (PIL slika, PIL maska)); transform() = weights preset (resize+normalize).
classes() -> 21 imena, NUM_CLASSES=21, IGNORE=255 (void). Skida se u ./data/VOCdevkit prvi put.
"""

import os

from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights as _W

HERE = os.path.dirname(os.path.abspath(__file__))
# Podaci su ULAZ, dijele ih sve mjerne mape -> shared/datasets/.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "shared", "datasets", "voc2012")

WEIGHTS = _W.DEFAULT
NUM_CLASSES = 21
IGNORE = 255
CLASSES = list(WEIGHTS.meta["categories"])


def classes():
    return list(CLASSES)


def transform():
    return WEIGHTS.transforms()                     # resize (smaller edge 520) + normalize (ImageNet)


def _downloaded():
    return os.path.isdir(os.path.join(DATA_DIR, "VOCdevkit", "VOC2012"))


def voc(split):
    """split: 'train' | 'val' | 'trainval'."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return VOCSegmentation(root=DATA_DIR, year="2012", image_set=split, download=not _downloaded())
