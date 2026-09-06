
import importlib.util
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO         = Path("/home/tomi/code/dipl")
DATASET_ROOT = REPO / "datasets/mini_set/sub10k_open_images_v7"
SPLIT        = "train"
IMG_SIZE     = 640

IMAGE_STEMS  = None
SEED         = 7
N_IMAGES     = 3

TAP          = 2
CHANNEL      = None
MODE         = "channel"
SHARED_SCALE = True
CMAP         = "inferno"

OUT_PNG      = REPO / "help" / "feat_kd.png"

STUDENTS = [
    ("STUDENT 0.5M", REPO / "custom_models/student_0_5_m/KD_featlogit"),
    ("STUDENT 1M",   REPO / "custom_models/student_1_m/KD_featlogit"),
    ("STUDENT 2M",   REPO / "custom_models/student_2_m/KD_featlogit"),
]
TAP_NAMES = ["P3", "P4", "P5"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return cv2.copyMakeBorder(img_bgr, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(img_path: Path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError("cv2 ne moze procitati " + str(img_path))
    img_lb = letterbox_image(img_bgr)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw).float() / 255.0, img_rgb


def pick_stems():
    if IMAGE_STEMS:
        return list(IMAGE_STEMS)
    img_dir = DATASET_ROOT / "images" / SPLIT
    feat_dir = DATASET_ROOT / "yolo26l" / SPLIT / "feat"
    stems = [p.stem for p in sorted(img_dir.iterdir())
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")
             and (feat_dir / (p.stem + ".pt")).exists()]
    if len(stems) < N_IMAGES:
        raise RuntimeError("Samo " + str(len(stems)) + " slika ima feat cache.")
    return random.Random(SEED).sample(stems, N_IMAGES)


def find_image(stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = DATASET_ROOT / "images" / SPLIT / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def load_student(exp_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "arch_" + exp_dir.parent.name, exp_dir / "model_arch_feat.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.StudentYOLOFeat(num_classes=6, input_size=IMG_SIZE).to(DEVICE).eval()
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, ckpt.get("epoch", "?"), ckpt.get("val_map", float("nan"))


def teacher_tap(stem):
    d = torch.load(DATASET_ROOT / "yolo26l" / SPLIT / "feat" / (stem + ".pt"),
                   map_location="cpu", weights_only=False)
    t = d["feat"][TAP].float()
    return t[0] if t.dim() == 4 else t


@torch.no_grad()
def student_tap(model, x):
    _, feats = model(x.unsqueeze(0).to(DEVICE), return_feats=True)
    return feats[TAP][0].float().cpu()


def to_map(t, ch):
    if MODE == "channel":
        return t[ch].numpy()
    if MODE == "magnitude":
        return t.pow(2).sum(0).sqrt().numpy()
    raise ValueError("MODE=" + repr(MODE))


def main():
    stems = pick_stems()
    print("tap      : " + TAP_NAMES[TAP] + "   mode: " + MODE + "   device: " + DEVICE)
    print("slike    : " + str(stems))
    print()

    students = []
    for name, exp_dir in STUDENTS:
        m, n_params, ep, vmap = load_student(exp_dir)
        print("{:<14} {:>9,} params   epoch={}  val_mAP={:.4f}".format(name, n_params, ep, vmap))
        students.append((name, m))
    print()

    rows = []
    for stem in stems:
        x, img_rgb = preprocess(find_image(stem))
        rows.append({
            "stem": stem,
            "img": img_rgb,
            "teacher": teacher_tap(stem),
            "students": [(name, student_tap(m, x)) for name, m in students],
        })

    ch = CHANNEL
    if MODE == "channel" and ch is None:
        per_ch = torch.stack([r["teacher"].abs().mean(dim=(1, 2)) for r in rows]).mean(0)
        ch = int(per_ch.argmax())
        print("auto kanal: {} / {}  (najveca prosjecna aktivacija kod teachera)".format(
            ch, rows[0]["teacher"].shape[0]))
        print()

    print("{:<20} {:<14} {:>9} {:>9} {:>16}".format("slika", "model", "mean", "std", "MSE vs teacher"))
    for r in rows:
        T = r["teacher"]
        print("{:<20} {:<14} {:>9.4f} {:>9.4f} {:>16}".format(
            r["stem"], "TEACHER", T.mean().item(), T.std().item(), "-"))
        for name, S in r["students"]:
            mse = torch.nn.functional.mse_loss(S, T).item()
            print("{:<20} {:<14} {:>9.4f} {:>9.4f} {:>16.4f}".format(
                "", name, S.mean().item(), S.std().item(), mse))

    ncols = 2 + len(students)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.0 * ncols, 3.0 * len(rows)))
    axes = np.atleast_2d(axes)

    col_titles = ["", "TEACHER (yolo26l)"] + [n for n, _ in students]
    for i, r in enumerate(rows):
        maps = [to_map(r["teacher"], ch)] + [to_map(t, ch) for _, t in r["students"]]
        if SHARED_SCALE:
            allv = np.concatenate([m.ravel() for m in maps])
            vmin, vmax = float(allv.min()), float(allv.max())
        else:
            vmin = vmax = None

        axes[i, 0].imshow(r["img"])
        axes[i, 0].axis("off")
        for j, m in enumerate(maps, start=1):
            axes[i, j].imshow(m, cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[i, j].axis("off")

    for j, t in enumerate(col_titles):
        if t:
            axes[0, j].set_title(t, fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print()
    print("spremljeno -> " + str(OUT_PNG))


if __name__ == "__main__":
    main()
