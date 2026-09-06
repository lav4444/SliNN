
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLINN = os.path.dirname(HERE)
MORPH = "/home/tomi/code/dipl/legacy/morphology"

if not os.path.isdir(MORPH):
    print("[_extractplug64] `legacy/morphology` ne postoji — preseljenje je vec obavljeno (6.4).")
    print("[_extractplug64] Ovaj alat je POVIJESNI: dokumentira KAKO je kod presao u slinn/.")
    raise SystemExit(0)
PLUG = os.path.join(SLINN, "plugins", "detection")

ADAPTER_SYMS = ["_img_dir", "_lbl_dir", "list_splits", "scan_split", "dev_subset_note",
                "set_bn_eval", "build_fasterrcnn", "ModelAdapter", "FrcnnAdapter",
                "YoloAdapter", "ADAPTERS", "pick_adapter", "model_num_classes",
                "_DetDataset", "make_gt_loader", "eval_map"]


def extract(path, syms):
    src = open(path, errors="replace").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    found = []
    for n in tree.body:
        name = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        if name in syms:
            end = getattr(n, "end_lineno", n.lineno)
            start = n.lineno - 1
            while start > 0 and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            found.append((name, "\n".join(lines[start:end])))
    return found


os.makedirs(PLUG, exist_ok=True)
open(os.path.join(SLINN, "plugins", "__init__.py"), "w").write(
    '"""slinn/plugins — izolirani per-obitelj dodaci. Jezgra radi i bez njih."""\n')

cfg_src = open(os.path.join(MORPH, "config.py"), errors="replace").read()
keep = ["DATASET_ROOT", "CLASS_NAMES", "IMG_EXTS", "DEV_DATA_SUBSET", "NUM_CLASSES",
        "COCO_IDS", "COCO_YOLO_IDS", "GRAD_BATCH", "USE_PROFILE_ADAPTERS", "EVAL_BATCH"]
tree = ast.parse(cfg_src)
clines = cfg_src.splitlines()
rows = []
for n in tree.body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
            and n.targets[0].id in keep:
        end = getattr(n, "end_lineno", n.lineno)
        rows.append("\n".join(clines[n.lineno - 1:end]))
open(os.path.join(PLUG, "dsconfig.py"), "w").write(
    '"""slinn/plugins/detection/dsconfig.py — konstante KONKRETNOG detekcijskog dataseta.\n\n'
    'Ove NE pripadaju u slinn/settings.py: razredi, COCO remap i putanja dataseta vrijede samo za\n'
    'ovaj detekcijski zadatak. Jezgra ih nikad ne cita.\n"""\n\n'
    "from pathlib import Path\n\n" + "\n".join(rows) + "\n")
print("dsconfig.py  <- {} konstanti".format(len(rows)))

prof = open(os.path.join(MORPH, "profiles.py"), errors="replace").read()
prof = prof.replace("import analysis as A", "from . import adapters as A", 1)
prof = prof.replace("import kd\n", "import kdterms as kd\n", 1)
open(os.path.join(PLUG, "profiles.py"), "w").write(prof)
print("profiles.py  <- cijeli modul")

found = extract(os.path.join(MORPH, "analysis.py"), set(ADAPTER_SYMS))
got = [n for n, _ in found]
missing = [s for s in ADAPTER_SYMS if s not in got]
print("adapters.py  <- {}/{} simbola{}".format(
    len(got), len(ADAPTER_SYMS), "  !! NEDOSTAJE: " + ", ".join(missing) if missing else ""))
head = ('"""slinn/plugins/detection/adapters.py — detekcijski decode plug (iz morphology/analysis.py).\n\n'
        'JEDINI priznati per-obitelj dio: dekodiranje izlaza (yolo DFL/sidra, frcnn ROI), NMS i mAP.\n'
        'To se NE moze izvesti mjerenjem, pa je ogradeno ovdje. Jezgra ovo doseze samo kroz\n'
        'backend.build_metric_fn; bez ovog foldera jezgra pada na teacher-agreement gate i radi dalje.\n"""\n\n'
        "import math\n"
        "import os\n"
        "import sys\n\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
        "_SLINN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
        "if _SLINN not in sys.path:\n"
        "    sys.path.insert(0, _SLINN)\n\n"
        "from introspect import unfreeze_bn                            # noqa: E402  (jezgra)\n"
        "from .dsconfig import (DATASET_ROOT, CLASS_NAMES, IMG_EXTS, DEV_DATA_SUBSET, NUM_CLASSES,\n"
        "                       COCO_IDS, COCO_YOLO_IDS, GRAD_BATCH, USE_PROFILE_ADAPTERS)\n\n\n")
open(os.path.join(PLUG, "adapters.py"), "w").write(head + "\n\n\n".join(s for _, s in found) + "\n")

open(os.path.join(PLUG, "__init__.py"), "w").write(
    '"""slinn/plugins/detection — decode + NMS + mAP za detekciju.\n\n'
    'JAVNI OTVOR (jedino sto jezgra smije zvati):\n'
    '    pick_adapter(model)   -> per-obitelj adapter (auto po arhitekturi)\n'
    '    make_gt_loader(split) -> GT loader za mAP\n'
    '    eval_map(model, adapter, loader, device) -> (metrike, n)\n"""\n\n'
    "from .adapters import pick_adapter, make_gt_loader, eval_map, set_bn_eval   # noqa: F401\n\n"
    '__all__ = ["pick_adapter", "make_gt_loader", "eval_map", "set_bn_eval"]\n')
print("__init__.py  <- javni otvor (4 simbola)")
print("-> " + PLUG)
