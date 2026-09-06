
import os
import shutil
import yaml
from ultralytics import YOLO

SRC = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"
DST = "/home/tomi/code/dipl/quantization/yolo_coco_data"
SPLITS = ["train", "val", "test"]
DS2COCO = {0: 0, 1: 2, 2: 7, 3: 5, 4: 3, 5: 1}
YAML_PATH = os.path.join(DST, "yolo_coco.yaml")


def remap_labels():
    n_files = n_boxes = 0
    for sp in SPLITS:
        src_img = os.path.join(SRC, "images", sp)
        src_lbl = os.path.join(SRC, "labels", sp)
        dst_img = os.path.join(DST, "images", sp)
        dst_lbl = os.path.join(DST, "labels", sp)
        if os.path.islink(dst_img):
            os.unlink(dst_img)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        if os.path.isdir(src_img):
            for fn in os.listdir(src_img):
                dl = os.path.join(dst_img, fn)
                if not os.path.exists(dl):
                    try:
                        os.link(os.path.join(src_img, fn), dl)
                    except OSError:
                        shutil.copy2(os.path.join(src_img, fn), dl)
        if not os.path.isdir(src_lbl):
            continue
        for fn in os.listdir(src_lbl):
            if not fn.endswith(".txt"):
                continue
            out_lines = []
            for line in open(os.path.join(src_lbl, fn)).read().splitlines():
                p = line.split()
                if len(p) == 5:
                    c = int(p[0])
                    if c in DS2COCO:
                        out_lines.append(f"{DS2COCO[c]} {p[1]} {p[2]} {p[3]} {p[4]}")
                        n_boxes += 1
            open(os.path.join(dst_lbl, fn), "w").write("\n".join(out_lines))
            n_files += 1
    return n_files, n_boxes


def write_yaml():
    names = YOLO("/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt").names
    data = {"path": DST, "train": "./images/train", "val": "./images/val", "test": "./images/test",
            "names": {int(k): v for k, v in names.items()}}
    os.makedirs(DST, exist_ok=True)
    yaml.safe_dump(data, open(YAML_PATH, "w"), sort_keys=False, allow_unicode=True)
    return YAML_PATH, len(names)


if __name__ == "__main__":
    nf, nb = remap_labels()
    yp, nn = write_yaml()
    print(f"remapirano: {nf} label datoteka, {nb} okvira | yaml: {yp} ({nn} COCO imena)")
