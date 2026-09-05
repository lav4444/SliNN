# -*- coding: utf-8 -*-
"""Zajednicki upis rezultata u CSV, uz postojeci eval_result.txt.

ZASTO CSV UZ TXT: .txt je citljiv covjeku i razlicit po modelu (housing ima R2, yolo mAP
po klasama, deeplab IoU po klasama). Za tablicu prije/poslije treba nesto strojno citljivo
s istim stupcima za sve.

ZASTO DUGI FORMAT (jedna metrika po retku): modeli nemaju zajednicki skup metrika. Siroki
format bi trazio uniju svih stupaca i vecina celija bi bila prazna, a svaki novi model bi
mijenjao zaglavlje. Dugi format se lako pivotira u sirok kad zatreba.

Stupci:
    datum        ISO 8601, lokalno vrijeme
    uredjaj      RPi5 / Jetson / ime hosta — prepoznaje se, ne upisuje
    mjerenje     BASELINE_RAW / BASELINE_OPTIM / SLINN_RAW / SLINN_OPTIM (iz putanje)
    runtime      eager / onnxruntime / tensorrt
    model        housing_mlp, yolo26n, ...
    dataset      ime skupa podataka
    split        val / test / validation
    n_uzoraka    koliko je uzoraka uslo u metriku
    metrika      naziv
    vrijednost   broj
    latencija_ms ms po uzorku (isto za sve metrike istog splita)
    dev          cpu / cuda
    torch        verzija

Zapisuje u <mjerenje>/results.csv — jedna datoteka za sve modele te mape, dopisuje se.
"""
import csv
import datetime
import os

import torch

COLS = ["datum", "uredjaj", "mjerenje", "runtime", "model", "dataset", "split",
        "n_uzoraka", "metrika", "vrijednost", "latencija_ms", "dev", "torch"]


def device_name():
    """Ime uredjaja iz sustava — ista skripta radi na RPi5 i na Jetsonu."""
    if os.path.exists("/etc/nv_tegra_release") or os.path.isdir("/usr/lib/aarch64-linux-gnu/tegra"):
        return "Jetson"
    try:
        with open("/proc/device-tree/model", "rb") as f:
            m = f.read().decode("utf-8", "ignore").strip("\x00").strip()
        if "raspberry" in m.lower():
            return "RPi5" if " 5 " in m else m
        return m
    except OSError:
        return os.uname().nodename


def _roots(script_dir):
    """(mjerna_mapa, korijen) iz polozaja skripte: <korijen>/<MJERENJE>/<model>/skripta.py"""
    model_dir = os.path.abspath(script_dir)
    meas_dir = os.path.dirname(model_dir)
    return os.path.basename(meas_dir), meas_dir


def write(script_dir, model, dataset, split, n, metrics, latency_ms=None,
          runtime="eager", dev=None):
    """Dopisi jedan blok metrika. `metrics` je {naziv: vrijednost}.

    Poziva se JEDNOM PO SPLITU, s vrijednostima koje su vec izracunate za .txt —
    da CSV i .txt ne mogu razici."""
    mjerenje, meas_dir = _roots(script_dir)
    # Isti pecat kroz cijeli run -> svih 7 modela zapise u ISTI csv,
    # a stariji runovi ostaju netaknuti.
    stamp = os.environ.get("RUN_STAMP") or datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S")
    path = os.path.join(meas_dir, f"results_{stamp}.csv")
    new = not os.path.exists(path)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    d = dev or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and v != v):     # None ili NaN
                continue
            w.writerow([now, device_name(), mjerenje, runtime, model, dataset, split, n,
                        k, round(float(v), 6),
                        "" if latency_ms is None else round(float(latency_ms), 4),
                        d, torch.__version__])
    return path
