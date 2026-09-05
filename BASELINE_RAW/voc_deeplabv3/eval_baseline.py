"""eval_baseline.py — baseline evaluacija voc_deeplabv3, format srodan yolo eval_result.txt.

Po splitu (train podskup / val): brzina (inference batch=1), mIoU + pixel-acc, per-class IoU.
+ CPU vs GPU latency benchmark (fiksni 520-ulaz). Zapisuje eval_result.txt uz model. Self-contained.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D                                                # noqa: E402

# --- CSV uz .txt (shared/emit.py) --------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "shared"))
import emit as EMIT                                             # noqa: E402
# -----------------------------------------------------------------------------


# --- MINI-TEST kocnica (edge/add_eval_limit.py) ------------------------------
# EVAL_LIMIT prazan/0 -> postojece ponasanje. EVAL_LIMIT=5 -> 5 uzoraka po splitu.
def _env_int(name, default=0):
    """Cijeli broj iz okoline; prazno ili besmisleno -> default, negativno -> 0."""
    v = os.environ.get(name, "").strip()
    if not (v.lstrip("+-").isdigit()):
        return default
    return max(0, int(v))


def _env_flag(name, default=True):
    """Zastavica iz okoline; postuje 0/false/no/off (i velika slova)."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


EVAL_LIMIT = _env_int("EVAL_LIMIT") or None

def _env_splits(name="EVAL_SPLITS"):
    """Skup trazenih splitova iz okoline, ili None (= svi). 'val' i 'validation' su isto."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return None
    out = set()
    for s in v.replace(";", ",").split(","):
        s = s.strip()
        if s:
            out.add(s)
            if s in ("val", "validation"):
                out |= {"val", "validation"}
    return out or None


EVAL_SPLITS = _env_splits()


def _want(split):
    return EVAL_SPLITS is None or split.lower() in EVAL_SPLITS



def _cap(n, lim=EVAL_LIMIT):
    """Manje od zadanog i stvarnog broja; `None` znaci bez kapice."""
    return n if lim is None else min(lim, n)
# -----------------------------------------------------------------------------


MODEL_PT = os.path.join(HERE, "model.pt")
# Jedan pecat po runu (run_evals.sh ga izveze); rucno pokretanje si ga izracuna samo.
RUN_STAMP = os.environ.get("RUN_STAMP") or __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(HERE, f"eval_result_{RUN_STAMP}.txt")
if EVAL_LIMIT:            # mini-test u zasebnu datoteku, kanonske brojke ostaju
    OUT = os.path.join(HERE, f"eval_result_mini_{RUN_STAMP}.txt")
TRAIN_LIMIT = 300


@torch.no_grad()
def confusion(model, split, dev, transform, limit=None):
    ds = D.voc(split)
    n = len(ds) if limit is None else min(limit, len(ds))
    C = D.NUM_CLASSES
    conf = torch.zeros(C, C, dtype=torch.long)
    model.eval()
    for i in range(n):
        img, mask = ds[i]
        out = model(transform(img).unsqueeze(0).to(dev))["out"]
        m = torch.as_tensor(np.array(mask), dtype=torch.long)
        out = F.interpolate(out, size=m.shape, mode="bilinear", align_corners=False)
        pred = out.argmax(1)[0].cpu()
        valid = (m != D.IGNORE) & (m < C)
        conf += torch.bincount(C * m[valid] + pred[valid], minlength=C * C).reshape(C, C)
    return conf


def metrics(conf):
    conf = conf.double()
    diag = conf.diag()
    iou = diag / (conf.sum(1) + conf.sum(0) - diag).clamp(min=1)
    present = conf.sum(1) > 0
    return float(iou[present].mean()), float(diag.sum() / conf.sum().clamp(min=1)), iou


def _median_ms(model, x, dev, warmup=5, iters=30):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _per_sample(model, x, dev, n=10):
    model.eval()
    out = []
    with torch.no_grad():
        for _ in range(n):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
    return out


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()
    tf = D.transform()
    names = D.classes()

    L = [f"Model: model.pt (deeplabv3_mobilenet_v3_large)",
         f"Task: semantic segmentation ({D.NUM_CLASSES} classes, void=255 ignored)",
         f"Input: RGB image (resize smaller-edge 520 + ImageNet norm)",
         f"Metric: mIoU (mean IoU) + pixel accuracy",
         f"Device: {dev}", ""]

    for split, lim in [t for t in (("train", TRAIN_LIMIT), ("val", None))
                       if _want(t[0])]:
        lim = EVAL_LIMIT or lim
        conf = confusion(model, split, dev, tf, limit=lim)
        miou, pix, iou = metrics(conf)
        img0, _ = D.voc(split)[0]
        ms = _median_ms(model, tf(img0).unsqueeze(0).to(dev), dev)
        n = lim or len(D.voc(split))
        EMIT.write(HERE, "voc_deeplabv3", "VOC2012_segmentation", split, n,
                   {"mIoU": miou, "pixel_acc": pix}, latency_ms=ms)
        tag = f" (prvih {lim})" if lim else ""
        L += [f"=========== {split.upper()} ({n} images{tag}) ===========", "",
              "Speed:",
              f"  Inference (batch=1):   {ms:.2f} ms/image  ({1000 / ms:,.1f} images/s)", "",
              "Segmentation metrics:",
              f"  mIoU        = {miou:.4f}",
              f"  pixel-acc   = {pix:.4f}", "",
              "Per-class IoU:"]
        for c in range(D.NUM_CLASSES):
            L.append(f"  {names[c]:<14} {float(iou[c]):.4f}")
        L.append("")

    # --- CPU vs GPU latency benchmark (fiksni 1x3x520x520 ulaz) ---
    xb = torch.randn(1, 3, 520, 520)
    cpu = torch.device("cpu")
    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()
    cpu_t = _per_sample(m_cpu, xb, cpu)
    gpu_t = _per_sample(model, xb.to(dev), dev) if dev.type == "cuda" else None
    mean_fast = lambda ts: float(np.mean(sorted(ts)[:8]))              # noqa: E731

    L += ["================================================================",
          "CPU vs GPU latency benchmark (fixed 1x3x520x520 input, batch=1)",
          "================================================================", "",
          f"CPU per-run times (ms): [{', '.join(f'{t:7.1f}' for t in cpu_t)}]"]
    if gpu_t:
        L.append(f"GPU per-run times (ms): [{', '.join(f'{t:7.1f}' for t in gpu_t)}]")
    cpu_m = mean_fast(cpu_t)
    L += ["", "Mean of 8 fastest (2 slowest discarded as warmup):",
          f"  CPU:  {cpu_m:8.2f} ms/run  ({1000 / cpu_m:,.1f} runs/s)"]
    if gpu_t:
        gpu_m = mean_fast(gpu_t)
        L.append(f"  GPU:  {gpu_m:8.2f} ms/run  ({1000 / gpu_m:,.1f} runs/s)")
        L.append(f"  Speedup (GPU vs CPU):  {cpu_m / gpu_m:.1f}x")

    text = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[save] -> {OUT}")


if __name__ == "__main__":
    main()
