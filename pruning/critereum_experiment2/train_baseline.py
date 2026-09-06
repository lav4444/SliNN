
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from model_cnn import SchoolCNN, INPUT_SIZE
import common


SCRIPT_DIR = Path(__file__).parent
CKPT_PATH = SCRIPT_DIR / "checkpoints" / "best.pt"

LR = 1e-3 / 3.0
BATCH_SIZE = 32
NUM_WORKERS = 4
PATIENCE = 3
BASELINE_MAX_EPOCHS = 100
SEED = 42


def train_loop(model, device, train_loader, val_loader, lr, max_epochs, patience, label=""):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics = {}
    no_improve = 0
    epoch_times = []
    t_start = time.time()
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t_ep = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_times.append(time.time() - t_ep)
        epochs_run = epoch

        val = common.evaluate(model, val_loader, device, criterion)
        print(f"    [{label} ep {epoch:3d}] val_loss={val['loss']:.4f} "
              f"mAP={val['map']:.4f} F1={val['f1']:.4f} acc={val['acc']:.4f} "
              f"({epoch_times[-1]:.1f}s)")

        if val["loss"] < best_val_loss - 1e-5:
            best_val_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{label}] early stop @ epoch {epoch} "
                      f"(val_loss bez poboljsanja {patience} epoha; best @ {best_epoch})")
                break

    stats = {
        "epochs_run": epochs_run,
        "total_time_s": time.time() - t_start,
        "sec_per_epoch": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_val_map": best_metrics.get("map", float("nan")),
        "best_val_f1": best_metrics.get("f1", float("nan")),
        "best_val_acc": best_metrics.get("acc", float("nan")),
    }
    return best_state, stats


def train_baseline_and_save(device, ckpt_path=CKPT_PATH):
    torch.manual_seed(SEED)
    print(">>> Treniram baseline SchoolCNN (full, do early-stopa)...")
    train_loader = common.make_loader("train", BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = common.make_loader("val", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = SchoolCNN().to(device)
    best_state, stats = train_loop(model, device, train_loader, val_loader,
                                   lr=LR, max_epochs=BASELINE_MAX_EPOCHS,
                                   patience=PATIENCE, label="baseline")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "stats": stats}, ckpt_path)
    print(f">>> Baseline spremljen: {ckpt_path}  "
          f"(epoha {stats['best_epoch']}, val_loss={stats['best_val_loss']:.4f}, "
          f"val_mAP={stats['best_val_map']:.4f})")
    return best_state, stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_baseline_and_save(device)


if __name__ == "__main__":
    main()
