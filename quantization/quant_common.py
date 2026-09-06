
import csv
import io
import json
import statistics
import time

import torch


def set_cpu_threads(n):
    torch.set_num_threads(int(n))
    return torch.get_num_threads()


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def file_size_mb(path):
    import os
    return os.path.getsize(path) / (1024 ** 2)


def benchmark(fn, device, n_warmup=15, n_iter=100):
    is_cuda = torch.device(device).type == "cuda"

    for _ in range(n_warmup):
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
    return f"N/A ({reason})"


def write_report(rows, cols, csv_path, json_path, meta):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"meta": meta, "rows": rows}, open(json_path, "w"), indent=2, default=str)
    print(f"[save] {csv_path}")
    print(f"[save] {json_path}")
