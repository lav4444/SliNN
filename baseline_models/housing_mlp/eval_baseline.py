
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D

def _env_int(name, default=0):
    v = os.environ.get(name, "").strip()
    if not (v.lstrip("+-").isdigit()):
        return default
    return max(0, int(v))


def _env_flag(name, default=True):
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


EVAL_LIMIT = _env_int("EVAL_LIMIT") or None

def _env_splits(name="EVAL_SPLITS"):
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
    return n if lim is None else min(lim, n)


MODEL_PT = os.path.join(HERE, "model.pt")
OUT = os.path.join(HERE, "eval_result.txt")
if EVAL_LIMIT:
    OUT = os.path.join(HERE, "eval_result_mini.txt")
SEED = 42


def _median_ms(model, x, dev, warmup=20, iters=200):
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


def _per_sample(model, sample, dev, n=10):
    model.eval()
    s = sample.to(dev)
    out = []
    with torch.no_grad():
        for i in range(n):
            xi = s[i:i + 1]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(xi)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()
    ymu, ysd = D.y_stats()
    Xtr_std, ytr = D.data("train")
    lin = LinearRegression().fit(Xtr_std.numpy(), ytr.numpy())

    L = []
    L += [f"Model: model.pt",
          f"Task: regression (tabular)",
          f"Input: 8 features (vector), StandardScaler on train",
          f"Target: median house value ($100k), range 0.15-5.0",
          f"Device: {dev}",
          ""]

    for split in [s for s in ("train", "val", "test") if _want(s)]:
        X, y = D.data(split)
        Xr, _ = D.data_raw(split)
        if EVAL_LIMIT:
            X, Xr, y = X[:EVAL_LIMIT], Xr[:EVAL_LIMIT], y[:EVAL_LIMIT]
        yn = y.numpy()
        with torch.no_grad():
            pred = model(Xr.to(dev)).squeeze(1).cpu().numpy()
        r2 = r2_score(yn, pred)
        rmse = float(mean_squared_error(yn, pred)) ** 0.5
        mae = float(mean_absolute_error(yn, pred))
        resid = np.abs(pred - yn)
        lin_r2 = r2_score(yn, lin.predict(X.numpy()))
        ms = _median_ms(model, Xr[:1].to(dev), dev)

        L += [f"=========== {split.upper()} ({len(y)} samples) ===========", "",
              "Speed:",
              f"  Inference (batch=1):   {ms:.4f} ms/sample  ({1000 / ms:,.0f} samples/s)", "",
              "Regression metrics (original scale, $100k):",
              f"  R2   = {r2:.4f}",
              f"  RMSE = {rmse:.4f}",
              f"  MAE  = {mae:.4f}", "",
              "Error breakdown (|residual|, $100k):",
              f"  MedAE = {np.median(resid):.4f}",
              f"  P90   = {np.percentile(resid, 90):.4f}",
              f"  Max   = {resid.max():.4f}", "",
              "Baseline (linear regression):",
              f"  lin_R2 = {lin_r2:.4f}   gap = {r2 - lin_r2:+.4f}", ""]

    idx = np.random.RandomState(SEED).choice(len(Xtr_std), 10, replace=False)
    sample = D.data_raw("train")[0][idx]
    cpu = torch.device("cpu")
    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()
    cpu_t = _per_sample(m_cpu, sample, cpu)
    gpu_t = _per_sample(model, sample, dev) if dev.type == "cuda" else None
    mean_fast = lambda ts: float(np.mean(sorted(ts)[:8]))

    L += ["================================================================",
          "CPU vs GPU latency benchmark (10 random train samples, seed=42, batch=1)",
          "================================================================", "",
          f"CPU per-sample times (ms): [{', '.join(f'{t:7.3f}' for t in cpu_t)}]"]
    if gpu_t:
        L.append(f"GPU per-sample times (ms): [{', '.join(f'{t:7.3f}' for t in gpu_t)}]")
    cpu_m = mean_fast(cpu_t)
    L += ["", "Mean of 8 fastest (2 slowest discarded as warmup):",
          f"  CPU:  {cpu_m:8.4f} ms/sample  ({1000 / cpu_m:,.0f} samples/s)"]
    if gpu_t:
        gpu_m = mean_fast(gpu_t)
        L.append(f"  GPU:  {gpu_m:8.4f} ms/sample  ({1000 / gpu_m:,.0f} samples/s)")
        L.append(f"  Speedup (GPU vs CPU):  {cpu_m / gpu_m:.1f}x   "
                 f"(napomena: za sitni MLP GPU launch overhead moze nadjacati -> speedup < 1 je ocekivan)")

    text = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[save] -> {OUT}")


if __name__ == "__main__":
    main()
