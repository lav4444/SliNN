"""data.py — uniformni loader za M1 (California Housing regresija), s LOKALNO spremljenim podacima.

Ugovor zoo-a:  data(split) -> (X_std [N,8] float32, y_raw [N] float32).
  X standardiziran (StandardScaler fit SAMO na train, bez curenja),
  y_raw u originalnim jedinicama ($100k) — za metriku i da task-detektor onjusi kontinuirani label.

Podaci se spremaju UZ model u ./data/california_housing.npz (RAW splitovi) i odatle se povlace.
Prvi put: fetch sa sklearn -> split 70/15/15 (seed 42) -> spremi npz. Poslije: cita lokalni npz.
Determinizam: split i scaler recomputaju se isto svaki put.
"""

import os

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
# Podaci su ULAZ, dijele ih sve mjerne mape -> shared/datasets/.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "shared", "datasets", "california_housing")
NPZ = os.path.join(DATA_DIR, "california_housing.npz")
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
_cache = {}


def _materialize():
    """Dohvati sa sklearn, podijeli 70/15/15 (seed 42), spremi RAW splitove u ./data/ uz model."""
    d = fetch_california_housing()
    X = d.data.astype("float32")
    y = d.target.astype("float32")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(NPZ, X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
             features=np.array(FEATURES))
    print(f"[data] materijalizirano -> {NPZ}")


def _prep():
    if _cache:
        return _cache
    if not os.path.exists(NPZ):
        _materialize()
    z = np.load(NPZ, allow_pickle=True)
    X_tr, y_tr = z["X_train"], z["y_train"]
    mu = X_tr.mean(0, keepdims=True)
    sd = X_tr.std(0, keepdims=True) + 1e-8
    std = lambda a: ((a - mu) / sd).astype("float32")                      # noqa: E731
    _cache.update(
        X={"train": std(X_tr), "val": std(z["X_val"]), "test": std(z["X_test"])},
        y={"train": y_tr, "val": z["y_val"], "test": z["y_test"]},
        ymu=float(y_tr.mean()), ysd=float(y_tr.std()) + 1e-8,
    )
    return _cache


def data(split):
    c = _prep()
    return torch.from_numpy(c["X"][split]), torch.from_numpy(c["y"][split])


def data_raw(split):
    """SIROVI (nestandardizirani) X + y.

    Treba modelu: `model.pt` je `NormalizedRegressor`, koji standardizaciju ulaza i
    de-standardizaciju izlaza nosi KAO BUFFERE — dakle ocekuje sirovi X i vraca sirovi y.
    `data()` vraca STANDARDIZIRANI X (za linearni baseline i slicno); hraniti njime
    NormalizedRegressor znaci dvostruku standardizaciju (izmjereno: R2 = -360)."""
    z = np.load(NPZ, allow_pickle=True)
    key = {"train": "X_train", "val": "X_val", "test": "X_test"}[split]
    ykey = {"train": "y_train", "val": "y_val", "test": "y_test"}[split]
    return (torch.from_numpy(z[key].astype("float32")),
            torch.from_numpy(z[ykey].astype("float32")))


def y_stats():
    c = _prep()
    return c["ymu"], c["ysd"]
