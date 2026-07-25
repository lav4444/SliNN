"""
Visual sanity check tool for the trained student detector.

Loads a checkpoint, picks a random image from a chosen split (train/val/test),
runs preprocessing + forward + NMS, and displays the letterboxed image with
predicted boxes (class + confidence) drawn on top.

Run:
    python gui_test.py                               # uses checkpoints/best.pt
    python gui_test.py --ckpt checkpoints/last.pt    # explicit checkpoint
    python gui_test.py --conf 0.10                   # lower confidence threshold

Requires WSLg or X server for the Tk window.
"""

import argparse
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from KD_first import StudentYOLO
from train_kd import IMG_SIZE, postprocess_for_eval, preprocess_image


SCRIPT_DIR = Path(__file__).parent
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints" / "best.pt"

CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def load_model(ckpt_path: Path, device: torch.device):
    model = StudentYOLO(num_classes=6, input_size=IMG_SIZE).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


@torch.no_grad()
def predict_image(model, img_path: Path, device: torch.device,
                  conf_thresh: float, iou_thresh: float, max_det: int):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read {img_path}")

    img_t, _, _ = preprocess_image(img_bgr)
    img_lb_rgb = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    inp = img_t.unsqueeze(0).to(device)
    raw = model(inp)
    boxes, probs = model.decode(raw)

    boxes_xyxy, scores, classes = postprocess_for_eval(
        boxes[0].cpu(), probs[0].cpu(),
        conf_thresh=conf_thresh, iou_thresh=iou_thresh, max_det=max_det,
    )
    return img_lb_rgb, boxes_xyxy, scores, classes


class GUITester:
    def __init__(self, root, model, ckpt_info, args):
        self.root = root
        self.model = model
        self.device = next(model.parameters()).device
        self.conf = args.conf
        self.iou = args.iou
        self.max_det = args.max_det

        root.title("Student detector — visual test")
        root.geometry("960x900")

        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        tk.Label(ctrl, text="Split:").pack(side=tk.LEFT)
        self.split_var = tk.StringVar(value=args.split)
        split_dropdown = ttk.Combobox(
            ctrl, textvariable=self.split_var,
            values=["train", "val", "test"], state="readonly", width=8,
        )
        split_dropdown.pack(side=tk.LEFT, padx=4)
        split_dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_random())

        tk.Button(ctrl, text="Random image", command=self.show_random,
                  font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=10)

        tk.Label(ctrl, text="Conf:").pack(side=tk.LEFT, padx=(20, 2))
        self.conf_var = tk.DoubleVar(value=args.conf)
        conf_scale = tk.Scale(ctrl, variable=self.conf_var, from_=0.01, to=0.95,
                              resolution=0.01, orient=tk.HORIZONTAL, length=180,
                              command=lambda v: self._on_conf_change())
        conf_scale.pack(side=tk.LEFT)

        info = (f"Checkpoint: {Path(args.ckpt).name}  |  "
                f"epoch {ckpt_info.get('epoch', '?')}  |  "
                f"val_kd_loss={ckpt_info.get('val_kd_loss', float('nan')):.4f}  |  "
                f"val_mAP={ckpt_info.get('val_map', float('nan')):.4f}")
        tk.Label(ctrl, text=info, fg="gray").pack(side=tk.RIGHT)

        self.fig = Figure(figsize=(9, 8.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Click 'Random image' to start")
        tk.Label(root, textvariable=self.status_var, anchor=tk.W,
                 fg="gray").pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)

        # Cache last image so changing conf doesn't re-pick a random image
        self._last_img_path = None
        self._cached_raw_dets = None  # (img_lb, full_boxes_xyxy, full_scores, full_classes)

        self.show_random()

    def _on_conf_change(self):
        self.conf = float(self.conf_var.get())
        if self._cached_raw_dets is not None:
            self._render_cached()

    def show_random(self):
        split = self.split_var.get()
        img_dir = DATASET_ROOT / "images" / split
        all_imgs = sorted(p for p in img_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not all_imgs:
            self.status_var.set(f"No images in {img_dir}")
            return
        img_path = random.choice(all_imgs)

        try:
            img_lb, boxes, scores, classes = predict_image(
                self.model, img_path, self.device,
                conf_thresh=0.001,  # cache low-conf, filter at draw time
                iou_thresh=self.iou, max_det=self.max_det,
            )
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            return

        self._last_img_path = img_path
        self._cached_raw_dets = (img_lb, boxes, scores, classes)
        self._render_cached()

    def _render_cached(self):
        img_lb, boxes, scores, classes = self._cached_raw_dets

        # Filter by current conf threshold
        keep = scores > self.conf
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        self.ax.clear()
        self.ax.imshow(img_lb)
        for (x1, y1, x2, y2), score, cls in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
            color = CLASS_COLORS[cls]
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            self.ax.add_patch(rect)
            self.ax.text(
                x1 + 2, max(y1 - 4, 0),
                f"{CLASS_NAMES[cls]} {score:.2f}",
                color="white", fontsize=9, weight="bold",
                bbox=dict(facecolor=color, edgecolor=color, alpha=0.9, boxstyle="round,pad=0.25"),
            )

        title = (f"{self._last_img_path.parent.name} / {self._last_img_path.name}  —  "
                 f"{len(scores)} detections (conf > {self.conf:.2f})")
        self.ax.set_title(title, fontsize=11)
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set(f"Showed: {self._last_img_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="Path to model checkpoint .pt")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Default split shown on launch")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Initial confidence threshold")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300,
                   help="Max detections after NMS")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.ckpt.exists():
        print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.ckpt} on {device}...")
    model, ckpt_info = load_model(args.ckpt, device)
    print(f"Loaded epoch {ckpt_info.get('epoch')}, "
          f"val_kd_loss={ckpt_info.get('val_kd_loss', float('nan')):.4f}, "
          f"val_mAP={ckpt_info.get('val_map', float('nan')):.4f}")

    root = tk.Tk()
    GUITester(root, model, ckpt_info, args)
    root.mainloop()


if __name__ == "__main__":
    main()
