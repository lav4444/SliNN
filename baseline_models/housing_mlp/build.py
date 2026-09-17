
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data as D
from model_mlp import HousingMLP

SEED = 42
LR = 1e-3
BATCH = 256
PATIENCE = 7
MAX_EPOCHS = 500
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(HERE, "model.pt")
TRAIN_SUMMARY = os.path.join(HERE, "train_summary.txt")

ACCEPT = {"r2": 0.78, "rmse": 0.55, "mae": 0.39, "baseline_gap": 0.15}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ymu, ysd = D.y_stats()

    Xtr, ytr = D.data("train")
    Xva, yva = D.data("val")
    Xte, yte = D.data("test")
    print(f"[data] train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  (8 znacajki -> 1)")

    ytr_s = (ytr - ymu) / ysd
    yva_s = (yva - ymu) / ysd

    model = HousingMLP().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] HousingMLP  parametara={n_par:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr_s), batch_size=BATCH, shuffle=True)

    best = {"val": float("inf"), "state": None, "ep": 0}
    bad = 0
    history = []
    stop_reason = f"max_epochs ({MAX_EPOCHS})"
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vmse = float(nn.functional.mse_loss(model(Xva.to(dev)).squeeze(1).cpu(), yva_s))
        improved = vmse < best["val"] - 1e-6
        if improved:
            best = {"val": vmse, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "ep": ep}
            bad = 0
        else:
            bad += 1
        history.append((ep, vmse, improved))
        print(f"  [ep{ep:3d}] val_mse(std)={vmse:.4f}" + ("  *best" if improved else f"  (bad {bad}/{PATIENCE})"))
        if bad >= PATIENCE:
            stop_reason = f"early stop — {PATIENCE} epoha bez poboljsanja val_mse"
            print(f"  early stop @ ep{ep}  (best ep{best['ep']}, val_mse {best['val']:.4f})")
            break
    train_secs = time.time() - t0
    total_epochs = history[-1][0]

    model.load_state_dict(best["state"])
    model.eval()

    lin = LinearRegression().fit(Xtr.numpy(), ytr.numpy())

    def eval_split(Xs, ys):
        with torch.no_grad():
            p = model(Xs.to(dev)).squeeze(1).cpu().numpy() * ysd + ymu
        yv = ys.numpy()
        return (r2_score(yv, p), float(mean_squared_error(yv, p)) ** 0.5,
                float(mean_absolute_error(yv, p)), r2_score(yv, lin.predict(Xs.numpy())))

    res = {name: eval_split(Xs, ys) for name, (Xs, ys) in
           {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items()}

    header = f"  {'split':<6}{'R2':>9}{'RMSE':>9}{'MAE':>9}{'lin_R2':>9}{'gap':>9}"
    rows = [header] + [
        f"  {name:<6}{res[name][0]:>9.4f}{res[name][1]:>9.4f}{res[name][2]:>9.4f}{res[name][3]:>9.4f}{res[name][0] - res[name][3]:>+9.4f}"
        for name in ("train", "val", "test")]
    table = "\n".join(rows)

    r2, rmse, mae, r2_lin = res["test"]
    gap = r2 - r2_lin
    ok = r2 >= ACCEPT["r2"] and rmse <= ACCEPT["rmse"] and mae <= ACCEPT["mae"] and gap >= ACCEPT["baseline_gap"]

    print("\n=== METRIKE po splitu (originalna skala, $100k) ===")
    print(table)
    print(f"\n  prihvat (TEST): R2>={ACCEPT['r2']}, RMSE<={ACCEPT['rmse']}, MAE<={ACCEPT['mae']}, gap>={ACCEPT['baseline_gap']}")
    print(f"  => {'PRIHVACEN' if ok else 'NIJE PROSAO — dici sirinu/epohe'}")

    torch.save(model.eval().cpu(), MODEL_PT)
    print(f"\n[save] pun eager modul -> {MODEL_PT}")

    hist_lines = "\n".join(
        f"  ep{ep:3d}  val_mse(std)={v:.4f}" + ("  *best" if imp else "") for ep, v, imp in history
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"""housing_mlp — HousingMLP / California Housing (regresija) — train summary
================================================================
datum            : {now}
uredaj           : {dev}
model            : HousingMLP  (8 -> 256 -> 256 -> 128 -> 64 -> 1)  parametara={n_par:,}
podaci           : ukupno 20640  ->  train {len(Xtr)} / val {len(Xva)} / test {len(Xte)}  (70/15/15, seed {SEED})
hiperparametri   : Adam lr={LR}, batch={BATCH}, patience={PATIENCE}, max_epochs={MAX_EPOCHS}

TRENING
  trajalo epoha  : {total_epochs}
  best epoha     : {best['ep']}  (val_mse(std)={best['val']:.4f})
  razlog prekida : {stop_reason}
  trajanje       : {train_secs:.1f} s
  povijest (val_mse na standardiziranoj skali):
{hist_lines}

METRIKE po splitu (originalna skala, $100k)   [R2 = GLAVNA metrika; gap = R2 - linearni baseline]
{table}

prihvat (sudi se na TEST-u): R2 >= {ACCEPT['r2']}, RMSE <= {ACCEPT['rmse']}, MAE <= {ACCEPT['mae']}, gap >= {ACCEPT['baseline_gap']}
PRIHVAT: {'PRIHVACEN' if ok else 'NIJE PROSAO'}
"""
    with open(TRAIN_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[save] sazetak treninga -> {TRAIN_SUMMARY}")
    return ok


if __name__ == "__main__":
    main()
