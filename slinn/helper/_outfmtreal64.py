"""_outfmtreal64.py -- prepoznavanje + DECODE formata izlaza na STVARNIM mrezama (6.4).

Za razliku od _outfmt64.py (izmisljene mreze, provjera logike), ovdje idu prave tezine:
  yolos-tiny (HF, DETR-obitelj set-prediction, 6.5M)   -> set_pred
  yolo26n eval  (end2end NMS)                          -> nms_out
  yolo26n train (razdvojeni box/score)                 -> dense_split
  yolo26n train ['feats']                              -> feat_pyramid (znacajke, NE predikcije)
  fasterrcnn (torchvision)                             -> boxes_dicts

yolos-tiny se skida s HF-a; obrisati nakon prolaza (v. ispis na kraju).
OUT: ../REPORTS/outfmt_real64.txt
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import torch                                                  # noqa: E402

_SLINN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SLINN not in sys.path:
    sys.path.insert(0, _SLINN)

import introspect                                             # noqa: E402
from plugins.detection import outfmt                          # noqa: E402

OUT = os.path.join(_SLINN, "REPORTS", "outfmt_real64.txt")
YOLO = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
HW = (640, 640)
L = []


def say(s=""):
    print(s)
    L.append(s)


def check(tag, out, exp_fmt, img_hw=HW, want_dets=True):
    rep = outfmt.classify_output(out)
    ok_fmt = rep["format"] == exp_fmt
    say("{:<32} -> {:<13} {}".format(tag, rep["format"], "OK" if ok_fmt else "!! ocekivano " + exp_fmt))
    say("     {}".format(rep["why"][:104]))
    dets = outfmt.decode(out, img_hw, conf=0.25, rep=rep)
    if not want_dets:
        ok_dec = dets is None
        say("     decode -> None  {}".format("OK (ispravno odbija)" if ok_dec else "!! ne bi smio dekodirati"))
        return ok_fmt and ok_dec
    if dets is None:
        say("     decode -> None  !! ocekivan decode")
        return False
    d = dets[0]
    n = int(d["boxes"].shape[0])
    if n:
        b = d["boxes"]
        inb = float((((b[:, 0] >= -2) & (b[:, 1] >= -2) & (b[:, 2] <= img_hw[1] + 2)
                      & (b[:, 3] <= img_hw[0] + 2) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]))).float().mean())
        say("     decode -> {} okvira · valjanih {:.0%} · scores [{:.3f}, {:.3f}] · razreda {}".format(
            n, inb, float(d["scores"].min()), float(d["scores"].max()), len(set(d["labels"].tolist()))))
        return ok_fmt and inb > 0.95
    say("     decode -> 0 okvira (prag) — struktura OK")
    return ok_fmt


IMG = ("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7/images/val/"
       "002bf2fbe7a55727.jpg")


def real_image(size):
    """STVARNA slika, ne sum: na sumu model daje smece pa provjera valjanosti okvira nema smisla."""
    import numpy as np
    from PIL import Image
    im = Image.open(IMG).convert("RGB").resize((size, size))
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255).unsqueeze(0)


def main():
    dev = torch.device("cpu")
    x = real_image(640)
    results = []

    say("===== 6.4 FORMAT + DECODE NA STVARNIM MREZAMA =====")
    say("")

    # --- yolo26n: eval (end2end NMS) i train (razdvojeni box/score + znacajke) ---
    m = introspect.load_any(YOLO, dev)
    m.eval()
    with torch.no_grad():
        o_eval = m(x)
    results.append(check("yolo26n eval (end2end)", o_eval, "nms_out"))

    m.train()
    with torch.no_grad():
        o_tr = m(x)
    d = o_tr[1] if isinstance(o_tr, (list, tuple)) else o_tr
    results.append(check("yolo26n train one2one", d["one2one"], "dense_split"))
    results.append(check("yolo26n train ['feats']", d["one2one"]["feats"], "feat_pyramid", want_dets=False))
    say("")

    # --- yolos-tiny (DETR obitelj) ---
    try:
        from transformers import AutoModelForObjectDetection
        y = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny").eval()
        say("yolos-tiny ucitan ({:,} params)".format(sum(p.numel() for p in y.parameters())))
        with torch.no_grad():
            o_y = y(pixel_values=real_image(512))
        results.append(check("yolos-tiny (set-prediction)", o_y, "set_pred", img_hw=(512, 512)))
    except BaseException as e:
        say("yolos-tiny PRESKOCEN: {}: {}".format(type(e).__name__, str(e)[:90]))
        results.append(False)
    say("")

    # --- fasterrcnn (torchvision) ---
    try:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        f = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT").eval()
        with torch.no_grad():
            o_f = f([real_image(320)[0]])
        results.append(check("fasterrcnn (torchvision)", o_f, "boxes_dicts", img_hw=(320, 320)))
    except BaseException as e:
        say("fasterrcnn PRESKOCEN: {}: {}".format(type(e).__name__, str(e)[:90]))
        results.append(False)

    say("")
    say("VERDIKT: {}/{} -> {}".format(sum(results), len(results),
                                      "PROLAZI" if all(results) else "PADA"))
    say("")
    say("CISCENJE (yolos-tiny vise ne treba nakon prolaza):")
    say("  rm -rf ~/.cache/huggingface/hub/models--hustvl--yolos-tiny")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n[zapisano] -> " + OUT)


if __name__ == "__main__":
    main()
