"""eval_baseline.py — baseline evaluacija M1 (regresija), format srodan yolo eval_result.txt.

Po splitu (train/val/test): brzina (inference batch=1), R2/RMSE/MAE, error percentili, linearni baseline.
+ CPU vs GPU latency benchmark (10 nasumicnih train uzoraka, seed 42). Zapisuje eval_result.txt uz model.
Self-contained: treba samo torch + sklearn + data.py (ne uvozi morphology).
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        X, y = D.data(split)                       # STANDARDIZIRANI — za linearni baseline
        Xr, _ = D.data_raw(split)                  # SIROVI — za model
        if EVAL_LIMIT:
            X, Xr, y = X[:EVAL_LIMIT], Xr[:EVAL_LIMIT], y[:EVAL_LIMIT]
        yn = y.numpy()
        # `model.pt` je NormalizedRegressor: standardizaciju ulaza i de-standardizaciju
        # izlaza nosi kao buffere, pa prima SIROVI X i vraca SIROVI y. Rucni `* ysd + ymu`
        # bio bi druga denormalizacija (izmjereno: R2 = -360 umjesto 0.7938).
        with torch.no_grad():
            pred = model(Xr.to(dev)).squeeze(1).cpu().numpy()
        r2 = r2_score(yn, pred)
        rmse = float(mean_squared_error(yn, pred)) ** 0.5
        mae = float(mean_absolute_error(yn, pred))
        resid = np.abs(pred - yn)
        lin_r2 = r2_score(yn, lin.predict(X.numpy()))
        ms = _median_ms(model, Xr[:1].to(dev), dev)
        EMIT.write(HERE, "housing_mlp", "california_housing", split, len(y),
                   {"R2": r2, "RMSE": rmse, "MAE": mae,
                    "MedAE": float(np.median(resid)),
                    "P90": float(np.percentile(resid, 90)),
                    "lin_R2": lin_r2}, latency_ms=ms)

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

    # --- CPU vs GPU latency benchmark ---
    idx = np.random.RandomState(SEED).choice(len(Xtr_std), 10, replace=False)
    sample = D.data_raw("train")[0][idx]   # model prima SIROVI X
    cpu = torch.device("cpu")
    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()
    cpu_t = _per_sample(m_cpu, sample, cpu)
    gpu_t = _per_sample(model, sample, dev) if dev.type == "cuda" else None
    mean_fast = lambda ts: float(np.mean(sorted(ts)[:8]))                   # noqa: E731

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
