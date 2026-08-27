"""slinn/plugins/detection/dsconfig.py — konstante KONKRETNOG detekcijskog dataseta.

Ove NE pripadaju u slinn/settings.py: razredi, COCO remap i putanja dataseta vrijede samo za
ovaj detekcijski zadatak. Jezgra ih nikad ne cita.
"""

import os
import sys
from pathlib import Path

_SLINN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SLINN not in sys.path:
    sys.path.insert(0, _SLINN)

DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")  # root sub10k Open Images (6 kl)
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]            # redoslijed = label idx 0..5
IMG_EXTS = (".jpg", ".jpeg", ".png")                                                # podrzane ekstenzije slika

# JEDAN prekidac za dev-podskup, iz jezgre. (BUG do 6.8: ekstrakcija pluga je ovamo kopirala VLASTITU
# vrijednost 200, pa je gasenje `settings.DEV_DATA_SUBSET` tiho ostavljalo GT loader kapiran na 200 —
# mAP gate je davao ISTI rezultat za n_gate 200/400/837.)
from settings import DEV_DATA_SUBSET                                                # noqa: E402,F401
NUM_CLASSES = len(CLASS_NAMES) + 1   # +1 background (torchvision konvencija za fasterrcnn glavu)
COCO_IDS = {"Person": 1, "Car": 3, "Truck": 8, "Bus": 6, "Motorcycle": 4, "Bicycle": 2}      # nasih 6 -> torchvision COCO-91
COCO_YOLO_IDS = {"Person": 0, "Car": 2, "Truck": 7, "Bus": 5, "Motorcycle": 3, "Bicycle": 1}  # nasih 6 -> ultralytics COCO-80
USE_PROFILE_ADAPTERS = True   # True = novi per-komponenta profili (profiles.py); False = stari monolitni adapteri (sigurnost).
GRAD_BATCH = 4           # batch za gradijentni prolaz u analizi. Veci = brze ali vise VRAM-a.
EVAL_BATCH = 8     # batch za eval (mAP/brzina, bez grada). Veci = brze; sigurnije na 8GB nego trening batch.
