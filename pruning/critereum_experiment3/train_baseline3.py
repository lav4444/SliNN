
import copy
import math
import time
from pathlib import Path

import torch

import common3 as C

SCRIPT_DIR = Path(__file__).parent
CKPT_PATH = SCRIPT_DIR / "checkpoints" / "best.pt"

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 4
NUM_WORKERS = 4
PATIENCE = 3
BASELINE_MAX_EPOCHS = 12
GRAD_CLIP = 5.0
WARMUP_ITERS = 500
SEED = 42


def train_loop(model, device, train_loader, val_loader, lr, max_epochs, patience, label=""):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    steps_per_epoch = max(1, len(train_loader))
    total_iters = max_epochs * steps_per_epoch
    warmup_iters = min(WARMUP_ITERS, max(1, steps_per_epoch - 1))

    def lr_factor(it):
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    best_map = -1.0
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
        for imgs, targets in train_loader:
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        epoch_times.append(time.time() - t_ep)
        epochs_run = epoch

        val = C.evaluate(model, val_loader, device)
        print(f"    [{label} ep {epoch:3d}] val_mAP={val['map']:.4f} "
              f"mAP50={val['map_50']:.4f} ({epoch_times[-1]:.1f}s)")

        if val["map"] > best_map:
            best_map = val["map"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    [{label}] early stop @ epoch {epoch} (best mAP {best_map:.4f} @ {best_epoch})")
                break

    stats = {
        "epochs_run": epochs_run,
        "total_time_s": time.time() - t_start,
        "sec_per_epoch": (sum(epoch_times) / len(epoch_times)) if epoch_times else 0.0,
        "best_val_map": best_map,
        "best_epoch": best_epoch,
        "best_val_map_50": best_metrics.get("map_50", float("nan")),
    }
    return best_state, stats


def train_baseline_and_save(device, ckpt_path=CKPT_PATH):
    torch.manual_seed(SEED)
    print(">>> Treniram baseline detektor (COCO-pretrained -> 6 klasa)...")
    train_loader = C.make_loader("train", BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, drop_empty=True)
    val_loader = C.make_loader("val", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = C.build_model(pretrained=True).to(device)
    best_state, stats = train_loop(model, device, train_loader, val_loader,
                                   lr=LR, max_epochs=BASELINE_MAX_EPOCHS,
                                   patience=PATIENCE, label="baseline")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "stats": stats}, ckpt_path)
    print(f">>> Baseline spremljen: {ckpt_path} (epoha {stats['best_epoch']}, "
          f"val_mAP={stats['best_val_map']:.4f})")
    return best_state, stats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_baseline_and_save(device)


if __name__ == "__main__":
    main()
