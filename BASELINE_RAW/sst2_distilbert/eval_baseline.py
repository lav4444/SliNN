"""eval_baseline.py — baseline evaluacija sst2_distilbert, format srodan yolo eval_result.txt.

Po splitu (train podskup / validation): brzina (inference batch=1), accuracy + macro-F1, per-class recall.
+ CPU vs GPU latency benchmark. Zapisuje eval_result.txt uz model. (test split preskocen — oznake skrivene.)
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
OUT = os.path.join(HERE, "eval_result.txt")
if EVAL_LIMIT:            # mini-test u zasebnu datoteku, kanonske brojke ostaju
    OUT = os.path.join(HERE, "eval_result_mini.txt")
SEED = 42
TRAIN_LIMIT = 4000


def _median_ms(model, enc, dev, warmup=10, iters=50):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**enc)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(**enc)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def _per_sample(model, encs, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for enc in encs:
            enc = {k: v.to(dev) for k, v in enc.items()}
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**enc)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000)
    return out


@torch.no_grad()
def _collect(model, ld, dev):
    yt, yp = [], []
    for enc, y in ld:
        enc = {k: v.to(dev) for k, v in enc.items()}
        yp.append(model(**enc).logits.argmax(1).cpu())
        yt.append(y)
    return torch.cat(yt).numpy(), torch.cat(yp).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()
    names = D.classes()

    L = [f"Model: model.pt ({D.MODEL_ID})",
         f"Task: sentiment classification (text, {len(names)} classes)",
         f"Input: token ids + attention_mask (WordPiece, max_len {D.MAX_LEN})",
         f"Classes: {', '.join(names)}  (0=neg, 1=pos)",
         f"Device: {dev}", ""]

    for split, lim in [t for t in (("train", TRAIN_LIMIT), ("validation", None))
                       if _want(t[0])]:
        lim = EVAL_LIMIT or lim
        ld = D.loader(split, 64, limit=lim)
        yt, yp = _collect(model, ld, dev)
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro")
        one = next(iter(D.loader(split, 1)))[0]
        one = {k: v.to(dev) for k, v in one.items()}
        ms = _median_ms(model, one, dev)
        tag = f" (prvih {lim})" if lim else ""
        L += [f"=========== {split.upper()} ({len(yt)} samples{tag}) ===========", "",
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

    # --- CPU vs GPU latency benchmark (10 val recenica, batch=1) ---
    encs = []
    ld1 = D.loader("validation", 1, shuffle=True)
    for i, (enc, _) in enumerate(ld1):
        encs.append(enc)
        if len(encs) >= 10:
            break
    cpu = torch.device("cpu")
    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()
    cpu_t = _per_sample(m_cpu, encs, cpu)
    gpu_t = _per_sample(model, encs, dev) if dev.type == "cuda" else None
    mean_fast = lambda ts: float(np.mean(sorted(ts)[:8]))              # noqa: E731

    L += ["================================================================",
          "CPU vs GPU latency benchmark (10 validation sentences, batch=1)",
          "================================================================", "",
          f"CPU per-sample times (ms): [{', '.join(f'{t:7.2f}' for t in cpu_t)}]"]
    if gpu_t:
        L.append(f"GPU per-sample times (ms): [{', '.join(f'{t:7.2f}' for t in gpu_t)}]")
    cpu_m = mean_fast(cpu_t)
    L += ["", "Mean of 8 fastest (2 slowest discarded as warmup):",
          f"  CPU:  {cpu_m:8.3f} ms/sample  ({1000 / cpu_m:,.0f} samples/s)"]
    if gpu_t:
        gpu_m = mean_fast(gpu_t)
        L.append(f"  GPU:  {gpu_m:8.3f} ms/sample  ({1000 / gpu_m:,.0f} samples/s)")
        L.append(f"  Speedup (GPU vs CPU):  {cpu_m / gpu_m:.1f}x")

    text = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[save] -> {OUT}")


if __name__ == "__main__":
    main()
