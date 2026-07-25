"""
quant_common.py — dijeljeni alati za kvantizacijske eksperimente (model-agnostično):
  * fiksiranje uvjeta mjerenja (CPU niti),
  * veličina modela na disku (MB),
  * POŠTENA latencija (warmup + CUDA sync + medijan/p90),
  * pisanje izvještaja (CSV + JSON s uvjetima).

Task-specifična evaluacija (npr. multilabel panel) NIJE ovdje — živi uz pojedini model.
Mjerenje latencije prima `fn` (callable koji izvrši JEDAN forward) pa radi i za PyTorch i za TensorRT.
"""

import csv
import io
import json
import statistics
import time

import torch


def set_cpu_threads(n):
    """Fiksiraj broj CPU niti (usporedivost CPU latencije među formatima)."""
    torch.set_num_threads(int(n))
    return torch.get_num_threads()


def model_size_mb(model):
    """Veličina serijaliziranog state_dicta u MB (int8 težine su zbijene -> manje)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def file_size_mb(path):
    """Veličina datoteke na disku u MB (npr. TensorRT .engine)."""
    import os
    return os.path.getsize(path) / (1024 ** 2)


def benchmark(fn, device, n_warmup=15, n_iter=100):
    """POŠTENA latencija jednog forwarda. `fn` izvrši jednu inferenciju (zatvara nad modelom+ulazom).
    Kontrole: warmup se odbacuje; na CUDA-i se sinkronizira oko ŠTOPERICE (CUDA je asinkrona);
    izvještava medijan + p90 (robusno na throttle skokove). Vrati dict u ms."""
    is_cuda = torch.device(device).type == "cuda"

    for _ in range(n_warmup):                      # warmup: lazy init, cuDNN autotune, alokacije, ramp takta
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    p90 = times[min(len(times) - 1, int(round(0.9 * len(times))) - 1)]
    return {"median_ms": statistics.median(times), "p90_ms": p90,
            "mean_ms": sum(times) / len(times), "min_ms": times[0], "n": n_iter}


def na(reason):
    """Oznaka za ćeliju koja nije podržana (backend/hardver) — ide doslovno u izvještaj."""
    return f"N/A ({reason})"


def write_report(rows, cols, csv_path, json_path, meta):
    """Zapiši tablicu: CSV (redovi formata × stupci) + JSON (meta uvjeti + redovi)."""
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"meta": meta, "rows": rows}, open(json_path, "w"), indent=2, default=str)
    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")
