
from pathlib import Path

DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
IMG_EXTS = (".jpg", ".jpeg", ".png")

DEV_DATA_SUBSET = 200

NUM_CLASSES = len(CLASS_NAMES) + 1
COCO_IDS = {"Person": 1, "Car": 3, "Truck": 8, "Bus": 6, "Motorcycle": 4, "Bicycle": 2}
COCO_YOLO_IDS = {"Person": 0, "Car": 2, "Truck": 7, "Bus": 5, "Motorcycle": 3, "Bicycle": 1}
YOLO_PATH = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
YOLO26L_PATH = "/home/tomi/code/dipl/baseline_models/yolo26l/yolo26l.pt"

MODEL_SPEC = YOLO_PATH

USE_PROFILE_ADAPTERS = True

GRAD_MAX_IMAGES = 1000
GRAD_BATCH = 4
SHOW_LAYER_TABLE = True

EVAL_MAX = None
VAL_CAP = 1000
EVAL_BATCH = 8

TRAIN_BATCH = 8
CENSUS_MAX = 1000
FT_PATIENCE = 5
FT_MAX_EPOCHS = 20
FT_METRICS = ["map"]
FT_RECOVERY_FRAC = 0.75

PHASE2_STOP_METRIC = "gflops"
PHASE2_STOP_FRAC = 0.25

ALIGN_M = 32
ALIGN_BETA = 1.0
ALIGN_POW = 1
PHASE2_PRUNE_STEP_FRAC = 0.015

PHASE2_PRUNE_LAYER_CAP = 0.15
PHASE2_MIN_ALIVE_FRAC = 0.00001
PHASE2_MIN_ALIVE = ALIGN_M // 2
PHASE2_COST_FLOPS_W = 0.60
PHASE2_PRUNE_PATIENCE = 5
PHASE2_MAX_STEPS = 200
PHASE2_REINVEST_FRAC = 0.30
PHASE2_GROW_DOM = 4.0
PHASE2_GROW_MAX_LAYERS = 3
PHASE2_CHURN_COOLDOWN = 2

_HERE = Path(__file__).parent
TMP_ROOT = str(_HERE / "tmp")
MODELS_DIR = str(_HERE / "models")
