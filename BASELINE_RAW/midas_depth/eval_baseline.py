"""eval_baseline.py — baseline evaluacija midas_depth, format srodan yolo eval_result.txt.

NYU val (podjela: 254; v. data.py): brzina (inference batch=1), AbsRel / RMSE / delta1 (scale-invariant). + CPU vs GPU benchmark.
Self-contained; model.pt reload treba MiDaS hub kod na sys.path (data.HUB_DIR).
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


def _cap(n, lim=EVAL_LIMIT):
    """Manje od zadanog i stvarnog broja; `None` znaci bez kapice."""
    return n if lim is None else min(lim, n)
# -----------------------------------------------------------------------------


for _d in D.HUB_DIRS:                                           # MiDaS + geffnet klase za torch.load
    sys.path.insert(0, _d)
MODEL_PT = os.path.join(HERE, "model.pt")
# Jedan pecat po runu (run_evals.sh ga izveze); rucno pokretanje si ga izracuna samo.
RUN_STAMP = os.environ.get("RUN_STAMP") or __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(HERE, f"eval_result_{RUN_STAMP}.txt")
if EVAL_LIMIT:            # mini-test u zasebnu datoteku, kanonske brojke ostaju
    OUT = os.path.join(HERE, f"eval_result_mini_{RUN_STAMP}.txt")
DEPTH_CAP = 10.0


def evaluate(model, transform, dev, limit=None):
    ds = D.NYUVal(limit)
    ar, rm, d1 = [], [], []
    model.eval()
    for i in range(len(ds)):
        img, gt = ds[i]
        with torch.no_grad():
            pred = model(transform(np.asarray(img, dtype=np.float32) / 255.0).to(dev))
        pred = F.interpolate(pred.unsqueeze(1), size=gt.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        valid = gt > 1e-3
        if gt.shape == (480, 640):                      # standardni NYU Eigen crop
            crop = np.zeros_like(valid)
            crop[45:471, 41:601] = True
            valid = valid & crop
        p, tgt = pred[valid], 1.0 / gt[valid]
        s, t = np.linalg.lstsq(np.stack([p, np.ones_like(p)], 1), tgt, rcond=None)[0]
        pd = np.clip(1.0 / np.clip(s * pred + t, 1e-6, None), 1e-3, DEPTH_CAP)
        g, e = gt[valid], pd[valid]
        ar.append(float(np.mean(np.abs(e - g) / g)))
        rm.append(float(np.sqrt(np.mean((e - g) ** 2))))
        d1.append(float(np.mean(np.maximum(e / g, g / e) < 1.25)))
    import statistics as st
    return st.mean(ar), st.mean(rm), st.mean(d1)


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


def _per_run(model, x, dev, n=10):
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
    tf = D.midas_transform()

    L = [f"Model: model.pt (MiDaS_small, intel-isl/MiDaS)",
         f"Task: monocular depth estimation (dense regression)",
         f"Input: RGB image (MiDaS small preset)",
         f"Metric: AbsRel / RMSE (m) / delta1 (scale-invariant, LS scale+shift)",
         f"Device: {dev}", ""]

    absrel, rmse, d1 = evaluate(model, tf, dev, limit=EVAL_LIMIT)
    # Broj se CITA, ne upisuje: val je od podjele 254 (ne vise 654), a uz EVAL_LIMIT
    # jos manje. Tvrdo upisan broj u naslovu lagao bi o tome sto je mjereno.
    n_eval = len(D.NYUVal(EVAL_LIMIT))
    img0, _ = D.NYUVal(1)[0]
    ms = _median_ms(model, tf(np.asarray(img0, dtype=np.float32) / 255.0).to(dev), dev)
    EMIT.write(HERE, "midas_depth", "NYU_Depth_V2", "val", n_eval,
               {"AbsRel": absrel, "RMSE_m": rmse, "delta1": d1}, latency_ms=ms)
    L += [f"=========== NYU VAL ({n_eval} images) ===========", "",
          "Speed:",
          f"  Inference (batch=1):   {ms:.2f} ms/image  ({1000 / ms:,.1f} images/s)", "",
          "Depth metrics (scale-invariant):",
          f"  AbsRel   = {absrel:.4f}   (lower better)",
          f"  RMSE     = {rmse:.4f} m",
          f"  delta1   = {d1:.4f}   (higher better)", ""]

    xb = torch.randn(1, 3, 256, 256)
    cpu = torch.device("cpu")
    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()
    cpu_t = _per_run(m_cpu, xb, cpu)
    gpu_t = _per_run(model, xb.to(dev), dev) if dev.type == "cuda" else None
    mean_fast = lambda ts: float(np.mean(sorted(ts)[:8]))              # noqa: E731

    L += ["================================================================",
          "CPU vs GPU latency benchmark (fixed 1x3x256x256 input, batch=1)",
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
