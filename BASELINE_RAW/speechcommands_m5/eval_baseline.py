"""eval_baseline.py — baseline evaluacija speechcommands_m5, format srodan yolo eval_result.txt.

Po splitu (train/val/test): brzina (inference batch=1), accuracy + macro-F1, per-class accuracy.
+ CPU vs GPU latency benchmark (10 nasumicnih train uzoraka, seed 42). Zapisuje eval_result.txt uz model.
Self-contained: treba samo torch + sklearn + data.py.
"""

import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

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


@torch.no_grad()
def _collect(model, ld, dev):
    yt, yp = [], []
    for xb, yb in ld:
        yp.append(model(xb.to(dev)).argmax(1).cpu())
        yt.append(yb)
    return torch.cat(yt).numpy(), torch.cat(yp).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()
    names = D.classes()

    L = [f"Model: model.pt",
         f"Task: keyword spotting (audio classification, {len(names)} classes)",
         f"Input: raw waveform [1, 8000] (1 s @ 8 kHz)",
         f"Classes: {', '.join(names)}",
         f"Device: {dev}", ""]

    first_sample = None
    for split in [s for s in ("train", "val", "test") if _want(s)]:
        if EVAL_LIMIT:
            import torch.utils.data as _tud
            _ds = _tud.Subset(D.dataset(split), range(_cap(len(D.dataset(split)))))
            ld = _tud.DataLoader(_ds, batch_size=256, shuffle=False, num_workers=0)
        else:
            ld = D.loader(split, batch=256, shuffle=False, workers=4)
        yt, yp = _collect(model, ld, dev)
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        xb, _ = next(iter(D.loader(split, batch=1, shuffle=False, workers=0)))
        if first_sample is None and split == "train":
            first_sample = xb
        ms = _median_ms(model, xb.to(dev), dev)
        EMIT.write(HERE, "speechcommands_m5", "speech_commands_v0.02", split,
                   len(yt), {"accuracy": acc, "f1_macro": f1}, latency_ms=ms)

        L += [f"=========== {split.upper()} ({len(yt)} samples) ===========", "",
              "Speed:",
              f"  Inference (batch=1):   {ms:.4f} ms/sample  ({1000 / ms:,.0f} samples/s)", "",
              "Classification metrics:",
              f"  Accuracy  = {acc:.4f}",
              f"  macro-F1  = {f1:.4f}", "",
              "Per-class accuracy (recall):"]
        for k, nm in enumerate(names):
            mask = yt == k
            pc = float((yp[mask] == k).mean()) if mask.any() else float("nan")
            L.append(f"  {nm:<10} {pc:.4f}")
        L.append("")

    # --- CPU vs GPU latency benchmark ---
    if first_sample is None:
        first_sample = next(iter(D.loader("train", batch=1, shuffle=False, workers=0)))[0]
    bl = D.loader("train", batch=10, shuffle=True, workers=0)
    sample = next(iter(bl))[0]
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
        L.append(f"  Speedup (GPU vs CPU):  {cpu_m / gpu_m:.1f}x")

    text = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[save] -> {OUT}")


if __name__ == "__main__":
    main()
