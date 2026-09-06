
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_NAME = "yolo26l.pt"
MODEL_PATH = "/home/tomi/code/dipl/baseline_models/yolo26l/yolo26l.pt"
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7_part2")
PRED_ROOT = DATASET_ROOT / "yolo26l"
SPLIT = "train"
IMG_SIZE = 640
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED = [(256, 80, 80), (512, 40, 40), (512, 20, 20)]


def letterbox_image(img_bgr, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (IMG_SIZE - new_unpad[0]) / 2.0
    dh = (IMG_SIZE - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_bgr = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_bgr


def preprocess_image(img_path: Path) -> torch.Tensor:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {img_path}")
    img_lb = letterbox_image(img_bgr)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw).float() / 255.0


def find_detect_head(model: YOLO):
    cands = [m for m in model.model.modules()
             if m.__class__.__name__ in ("Detect", "v10Detect", "DetectV2")]
    if not cands:
        raise RuntimeError("Could not locate a Detect head on the model.")
    return cands[-1]


def list_images(img_dir: Path):
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


@torch.no_grad()
def main():
    print(f"Device: {DEVICE}  Model: {MODEL_NAME}  split: {SPLIT}  batch: {BATCH}")
    img_dir = DATASET_ROOT / "images" / SPLIT
    feat_dir = PRED_ROOT / SPLIT / "feat"
    feat_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = list_images(img_dir)
    todo = [p for p in all_imgs if not (feat_dir / f"{p.stem}.pt").exists()]
    print(f"[{SPLIT}] {len(all_imgs)} slika ukupno, {len(todo)} za izračunati "
          f"({len(all_imgs) - len(todo)} već u cacheu)")
    if not todo:
        print("Sve već precomputano — gotovo.")
        return

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    head = find_detect_head(model)
    if getattr(head, "end2end", False):
        head.end2end = False
        print("[setup] end2end OFF (treba pre-NMS dense + neck featuri)")
    model.model.eval()

    grab = {}
    h = head.register_forward_pre_hook(lambda m, inp: grab.__setitem__("f", list(inp[0])))

    t0 = time.time()
    done = 0
    try:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            batch = torch.stack([preprocess_image(p) for p in chunk]).to(DEVICE, non_blocking=True)
            grab.clear()
            model.model(batch)
            feats = grab["f"]

            if done == 0:
                got = [tuple(f.shape[1:]) for f in feats]
                assert got == EXPECTED, f"tap shapes {got} != expected {EXPECTED}"
                print(f"  tap shapes OK: {got}")

            for j, p in enumerate(chunk):
                per = [f[j].detach().cpu().half().contiguous() for f in feats]
                torch.save({"feat": per}, feat_dir / f"{p.stem}.pt")
            done += len(chunk)

            if done % 400 < BATCH:
                dt = time.time() - t0
                print(f"  [{SPLIT}] {done}/{len(todo)}  ({dt:.0f}s, {done / max(dt,1e-9):.1f} img/s)")
    finally:
        h.remove()

    print(f"[{SPLIT}] gotovo: {done} slika u {feat_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
