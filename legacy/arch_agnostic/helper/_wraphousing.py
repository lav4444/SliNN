"""_wraphousing.py — Faza 5.6 (odluka B): ubaci normalizaciju X/y U housing model (self-contained: sirovi X -> sirovi y).
Ucitaj postojeci HousingMLP (treniran na standardiziranom X, izlaz standardizirani y), izracunaj mu/sd iz npz-a
(X_train, y_train), umotaj u NormalizedRegressor, spremi kao model.pt (backup starog). Provjeri r2 na SIROVOM X_val.
-> REPORTS/wrap_housing.txt
"""
import os
import shutil
import sys

import numpy as np
import torch

_AA = "/home/tomi/code/dipl/arch_agnostic"
sys.path.insert(0, "/home/tomi/code/dipl/morphology")
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
HD = f"{BM}/housing_mlp"
sys.path.insert(0, HD)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "wrap_housing.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import metric as M                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from model_mlp import NormalizedRegressor                     # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


MP = f"{HD}/model.pt"
z = np.load(f"{HD}/data/california_housing.npz", allow_pickle=True)
Xtr, ytr = z["X_train"], z["y_train"]
mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-8
mu_y, sd_y = float(ytr.mean()), float(ytr.std()) + 1e-8

inner = A.load_any(MP, dev, code_dirs=[HD])
already = type(inner).__name__ == "NormalizedRegressor"
emit(f"ucitan model: {type(inner).__name__}  (vec umotan={already})")
if already:
    inner = inner.inner                                      # re-wrap iznova iz jezgre (idempotentno)

# provjera PRIJE: sirovi X -> (krivo, jer goli model ocekuje standardiziran) vs standardiziran
ad0 = probe_adapter(inner.to(dev), dev, verbose=False)
raw_pairs = M.pairs_regression(f"{HD}/data", ad0, split="val")   # (sirovi X_val, sirovi y_val)
r2_raw_bare = M.eval_regression(inner, ad0, raw_pairs, dev)["r2"]
emit(f"goli HousingMLP na SIROVOM X_val: r2={r2_raw_bare:.4f}  (ocekivano besmisleno -> treba standardizaciju)")

# umotaj + spremi (backup)
wrapped = NormalizedRegressor(inner, mu_x, sd_x, mu_y, sd_y).to(dev).eval()
if not os.path.exists(f"{HD}/model_prewrap.pt.bak"):
    shutil.copy(MP, f"{HD}/model_prewrap.pt.bak"); emit("backup -> model_prewrap.pt.bak")
torch.save(wrapped, MP)
emit(f"spremljen umotani self-contained model -> {MP}")

# reload + provjera POSLIJE: sirovi X -> sirovi y
w = A.load_any(MP, dev, code_dirs=[HD])
adw = probe_adapter(w, dev, verbose=False)
emit(f"reload OK: {type(w).__name__}  probe mode={adw._mode} in_ch={adw._in_ch} (vektor dim-8 ocuvan)")
r2_raw_wrap = M.eval_regression(w, adw, raw_pairs, dev)["r2"]
rmse = M.eval_regression(w, adw, raw_pairs, dev)["rmse"]
emit(f"umotani model na SIROVOM X_val: r2={r2_raw_wrap:.4f}  rmse={rmse:.4f}  (ocekivano ~0.79)")

ok = r2_raw_wrap > 0.7
emit(f"\nVERDIKT B (baked-norm housing): {'PROLAZI' if ok else 'PROVJERI'} (self-contained, sirovi X->y, r2 smislen)")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
