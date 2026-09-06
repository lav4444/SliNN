
import os
import sys
from pathlib import Path

_SLINN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SLINN not in sys.path:
    sys.path.insert(0, _SLINN)

from settings import DEV_DATA_SUBSET                                                # noqa: E402,F401

IMG_EXTS = (".jpg", ".jpeg", ".png")
GRAD_BATCH = 4
EVAL_BATCH = 8
USE_PROFILE_ADAPTERS = True

DATASET_ROOT = None
CLASS_NAMES = None
NUM_CLASSES = None
COCO_YOLO_IDS = None
COCO_IDS = None
_CONFIGURED = None

_TV_COCO91 = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5, "bus": 6, "train": 7,
    "truck": 8, "boat": 9, "traffic light": 10, "fire hydrant": 11, "stop sign": 13,
    "parking meter": 14, "bench": 15, "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20,
    "cow": 21, "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27,
    "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34, "skis": 35,
    "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39, "baseball glove": 40,
    "skateboard": 41, "surfboard": 42, "tennis racket": 43, "bottle": 44, "wine glass": 46,
    "cup": 47, "fork": 48, "knife": 49, "spoon": 50, "bowl": 51, "banana": 52, "apple": 53,
    "sandwich": 54, "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59,
    "donut": 60, "cake": 61, "chair": 62, "couch": 63, "potted plant": 64, "bed": 65,
    "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73, "mouse": 74, "remote": 75,
    "keyboard": 76, "cell phone": 77, "microwave": 78, "oven": 79, "toaster": 80, "sink": 81,
    "refrigerator": 82, "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88,
    "hair drier": 89, "toothbrush": 90,
}


def _norm(s):
    return str(s).strip().lower().replace("_", " ")


def _yaml_names(root):
    for nm in ("dataset.yaml", "data.yaml"):
        p = Path(root) / nm
        if not p.exists():
            continue
        try:
            import yaml
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            n = d.get("names")
            if isinstance(n, dict):
                return [n[k] for k in sorted(n, key=int)], str(p)
            if isinstance(n, list) and n:
                return list(n), str(p)
        except ImportError:
            pass
        out, inside = [], False
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("names:"):
                inside = True
                continue
            if inside:
                if ln[:1] not in (" ", "\t", "-") or not ln.strip():
                    break
                t = ln.strip()
                out.append(t[1:].strip() if t.startswith("-") else t.split(":", 1)[-1].strip())
        if out:
            return out, str(p)
    return None, None


def _model_names(model):
    for obj in (model, getattr(model, "model", None)):
        n = getattr(obj, "names", None)
        if isinstance(n, dict) and n:
            return {int(k): v for k, v in n.items()}
        if isinstance(n, (list, tuple)) and n:
            return dict(enumerate(n))
    return None


def configure(dataset_path, model=None, strict=True):
    global DATASET_ROOT, CLASS_NAMES, NUM_CLASSES, COCO_YOLO_IDS, COCO_IDS, _CONFIGURED
    root = Path(dataset_path)
    names, src = _yaml_names(root)
    if not names:
        raise RuntimeError(
            "[detekcija] nema imena razreda: ocekujem `names:` u {}/dataset.yaml (ili data.yaml). "
            "Bez njih se ne zna sto koji indeks u oznakama znaci.".format(root))

    mnames = _model_names(model) if model is not None else None
    yolo_map, miss = {}, []
    if mnames:
        inv = {_norm(v): k for k, v in mnames.items()}
        for n in names:
            if _norm(n) in inv:
                yolo_map[n] = inv[_norm(n)]
            else:
                miss.append(n)
        if miss and strict:
            raise RuntimeError(
                "[detekcija] razredi dataseta koje model ne poznaje: {}\n"
                "  imena dataseta ({}): {}\n"
                "  imena modela ({}): {}...\n"
                "  Preslikavanje se radi usporedbom imena; preimenuj u dataset.yaml ili "
                "upotrijebi model treniran na tim razredima.".format(
                    miss, src, names, len(mnames), list(mnames.values())[:12]))

    tv_map, tv_miss = {}, [n for n in names if _norm(n) not in _TV_COCO91]
    if not tv_miss:
        tv_map = {n: _TV_COCO91[_norm(n)] for n in names}

    DATASET_ROOT = root
    CLASS_NAMES = list(names)
    NUM_CLASSES = len(CLASS_NAMES) + 1
    COCO_YOLO_IDS = yolo_map or None
    COCO_IDS = tv_map or None
    _CONFIGURED = (str(root), tuple(CLASS_NAMES))
    print("[detekcija] dataset={} · razreda={} ({})".format(root, len(CLASS_NAMES),
                                                            ", ".join(map(str, CLASS_NAMES[:8]))))
    print("[detekcija] izvor imena: {}".format(src))
    if yolo_map:
        print("[detekcija] preslikavanje u model.names: {}".format(
            {n: yolo_map[n] for n in CLASS_NAMES[:8]}))
    elif model is not None:
        print("[detekcija] model ne nosi `names` — yolo grana ne moze preslikati oznake.")
    if tv_miss:
        print("[detekcija] izvan COCO-91: {} -> fasterrcnn grana nedostupna.".format(tv_miss))
    return _CONFIGURED


def require():
    if DATASET_ROOT is None:
        raise RuntimeError(
            "[detekcija] plug nije konfiguriran za ovaj run. `dsconfig.configure(dataset_path, model)` "
            "mora proci prije bilo kojeg pristupa podacima (zove ga backend.load_ctx).")
    return DATASET_ROOT
