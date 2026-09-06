
import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED          = 42
MAX_EPOCHS    = 500
PATIENCE      = 3
BATCH_SIZE    = 128
LR            = 1e-3
SPLIT         = (0.6, 0.2, 0.2)
BENCH_BATCH   = 256
BENCH_WARMUP  = 2
BENCH_RUNS    = 3

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(HERE, "data")
OUT_DIR    = os.path.join(HERE, "results")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class SkipNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)

        self.fc_parallel = nn.Linear(169, 2)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2, padding=0)

        self.fc1 = nn.Linear(in_features=50, out_features=3, bias=True)

        self.out = nn.Linear(in_features=5, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        p = torch.flatten(x, start_dim=1)
        p = self.fc_parallel(p)
        p = F.relu(p)

        m = self.conv2(x)
        m = F.relu(m)
        m = torch.flatten(m, start_dim=1)
        m = self.fc1(m)
        m = F.relu(m)

        z = torch.cat([m, p], dim=1)

        out = self.out(z)
        return out


class NoSkipNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2, padding=0)
        self.fc1   = nn.Linear(in_features=50, out_features=3, bias=True)
        self.out   = nn.Linear(in_features=3, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)

        out = self.out(x)
        return out


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(seed):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_part = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=tfm)
    test_part = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=tfm)
    full = torch.utils.data.ConcatDataset([train_part, test_part])

    n = len(full)
    n_train = int(SPLIT[0] * n)
    n_val   = int(SPLIT[1] * n)
    n_test  = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full, [n_train, n_val, n_test], generator=gen)

    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, generator=loader_gen)
    val_loader   = DataLoader(val_set,  batch_size=512, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


def train_model(model, name, device):
    print(f"\n{'='*60}\nTreniram: {name}\n{'='*60}")
    set_seed(SEED)
    train_loader, val_loader, test_loader = make_loaders(SEED)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc  = -1.0
    best_epoch    = 0
    best_state    = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    epoch = 0
    while epoch < MAX_EPOCHS:
        epoch += 1
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen    += x.size(0)
        train_loss = running / seen

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            best_state    = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
              f"{'  *' if improved else f'  (no improve {epochs_no_improve}/{PATIENCE})'}")

        if epochs_no_improve >= PATIENCE:
            print(f"  >> early stop na epohi {epoch} "
                  f"(val loss ne napreduje {PATIENCE} epoha; najbolja: epoha {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"  >> best epoch={best_epoch} val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f} "
          f"| test_acc={test_acc:.4f} test_loss={test_loss:.4f}")
    history["test_loss"]     = test_loss
    history["test_acc"]      = test_acc
    history["best_val_acc"]  = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["best_epoch"]    = best_epoch
    history["epochs_run"]    = epoch
    return model, history


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_bytes(model):
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes + buffer_bytes


def benchmark_inference(model, device, batch=BENCH_BATCH,
                        warmup=BENCH_WARMUP, runs=BENCH_RUNS):
    model = model.to(device).eval()
    x = torch.randn(batch, 1, 28, 28, device=device)
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_s = sum(times) / len(times)
    return {
        "batch": batch,
        "avg_ms": avg_s * 1e3,
        "ms_per_img": avg_s * 1e3 / batch,
        "img_per_s": batch / avg_s,
    }


def measure_all(model, name, device):
    print(f"\n--- Mjerenja: {name} ---")
    total, trainable = count_params(model)
    size_b = model_size_bytes(model)

    gpu_bench = None
    if torch.cuda.is_available():
        gpu_bench = benchmark_inference(model, torch.device("cuda"))
    cpu_bench = benchmark_inference(model, torch.device("cpu"))

    m = {
        "params_total": total,
        "params_trainable": trainable,
        "size_bytes": size_b,
        "size_kb": size_b / 1024,
        "gpu": gpu_bench,
        "cpu": cpu_bench,
    }
    print(f"  params={total}  size={m['size_kb']:.2f} KB")
    if gpu_bench:
        print(f"  GPU infer: {gpu_bench['avg_ms']:.3f} ms/batch({BENCH_BATCH})  "
              f"{gpu_bench['img_per_s']:.0f} img/s")
    print(f"  CPU infer: {cpu_bench['avg_ms']:.3f} ms/batch({BENCH_BATCH})  "
          f"{cpu_bench['img_per_s']:.0f} img/s")
    return m


def plot_training_curves(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"SkipNet": "tab:blue", "NoSkipNet": "tab:red"}

    for name, r in results.items():
        c = colors[name]
        h = r["history"]
        ep = range(1, len(h["train_loss"]) + 1)
        be = h["best_epoch"]
        axes[0].plot(ep, h["train_loss"], color=c, label=name)
        axes[1].plot(ep, h["val_loss"],   color=c, label=name)
        axes[2].plot(ep, h["val_acc"],    color=c, label=name)
        axes[1].axvline(be, color=c, ls="--", alpha=0.5)

    axes[0].set_title("Train loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[1].set_title("Val loss");   axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[2].set_title("Val accuracy"); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("acc")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Head-to-head: trening (SkipNet vs NoSkipNet) — Fashion-MNIST", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  spremljeno: {path}")


def plot_metrics(results, path):
    names = list(results.keys())
    colors = ["tab:blue", "tab:red"]

    def vals(fn):
        return [fn(results[n]) for n in names]

    panels = [
        ("Test accuracy",        vals(lambda r: r["history"]["test_acc"]),          "acc"),
        ("Broj parametara",      vals(lambda r: r["metrics"]["params_total"]),      "params"),
        ("Velicina (KB)",        vals(lambda r: r["metrics"]["size_kb"]),           "KB"),
        ("GPU infer (ms/batch)", vals(lambda r: r["metrics"]["gpu"]["avg_ms"] if r["metrics"]["gpu"] else 0), "ms"),
        ("CPU infer (ms/batch)", vals(lambda r: r["metrics"]["cpu"]["avg_ms"]),     "ms"),
        ("Throughput GPU (img/s)", vals(lambda r: r["metrics"]["gpu"]["img_per_s"] if r["metrics"]["gpu"] else 0), "img/s"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, data, unit) in zip(axes.flat, panels):
        bars = ax.bar(names, data, color=colors)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y", alpha=0.3)
        for b, d in zip(bars, data):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{d:.4g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Head-to-head: metrike (SkipNet vs NoSkipNet)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  spremljeno: {path}")


def write_report(results, device_name, path):
    lines = []
    lines.append("=" * 72)
    lines.append("HEAD-TO-HEAD USPOREDBA:  SkipNet  vs  NoSkipNet")
    lines.append("Dataset: Fashion-MNIST  |  train/val/test split  |  isti trening pipeline")
    lines.append(f"GPU: {device_name}")
    lines.append(f"Split (train/val/test): {SPLIT} od ukupno 70 000 slika "
                 f"-> {int(SPLIT[0]*70000)}/{int(SPLIT[1]*70000)}/{int(SPLIT[2]*70000)}")
    lines.append(f"Hiperparametri: batch={BATCH_SIZE}, lr={LR}, seed={SEED}")
    lines.append(f"Trening: bez fiksnog broja epoha, early stop kad val loss "
                 f"ne napreduje {PATIENCE} uzastopne epohe (max kapica {MAX_EPOCHS})")
    lines.append(f"Inference benchmark: batch={BENCH_BATCH}, warmup={BENCH_WARMUP}, "
                 f"mjerenih={BENCH_RUNS} (prosjek)")
    lines.append("=" * 72)
    lines.append("")

    names = list(results.keys())

    def row(label, fn, fmt="{}"):
        cells = []
        for n in names:
            try:
                cells.append(fmt.format(fn(results[n])))
            except Exception:
                cells.append("n/a")
        return f"{label:<32}" + "".join(f"{c:>18}" for c in cells)

    lines.append(f"{'METRIKA':<32}" + "".join(f"{n:>18}" for n in names))
    lines.append("-" * 72)
    lines.append(row("Test accuracy",      lambda r: r["history"]["test_acc"],  "{:.4f}"))
    lines.append(row("Test loss",          lambda r: r["history"]["test_loss"], "{:.4f}"))
    lines.append(row("Best val accuracy",  lambda r: r["history"]["best_val_acc"], "{:.4f}"))
    lines.append(row("Best val loss",      lambda r: r["history"]["best_val_loss"], "{:.4f}"))
    lines.append(row("Epoha (odabrana)",   lambda r: r["history"]["best_epoch"], "{}"))
    lines.append(row("Epoha (ukupno)",     lambda r: r["history"]["epochs_run"], "{}"))
    lines.append(row("Parametri (total)",  lambda r: r["metrics"]["params_total"], "{}"))
    lines.append(row("Parametri (train.)", lambda r: r["metrics"]["params_trainable"], "{}"))
    lines.append(row("Velicina (KB)",      lambda r: r["metrics"]["size_kb"], "{:.3f}"))
    lines.append(row("GPU ms/batch",       lambda r: r["metrics"]["gpu"]["avg_ms"] if r["metrics"]["gpu"] else float("nan"), "{:.3f}"))
    lines.append(row("GPU ms/slika",       lambda r: r["metrics"]["gpu"]["ms_per_img"] if r["metrics"]["gpu"] else float("nan"), "{:.5f}"))
    lines.append(row("GPU img/s",          lambda r: r["metrics"]["gpu"]["img_per_s"] if r["metrics"]["gpu"] else float("nan"), "{:.0f}"))
    lines.append(row("CPU ms/batch",       lambda r: r["metrics"]["cpu"]["avg_ms"], "{:.3f}"))
    lines.append(row("CPU ms/slika",       lambda r: r["metrics"]["cpu"]["ms_per_img"], "{:.5f}"))
    lines.append(row("CPU img/s",          lambda r: r["metrics"]["cpu"]["img_per_s"], "{:.0f}"))
    lines.append("-" * 72)
    lines.append("")

    a = results[names[0]]; b = results[names[1]]
    lines.append("ZAKLJUCAK (pobjednik po metrici):")
    acc_win = names[0] if a["history"]["test_acc"] >= b["history"]["test_acc"] else names[1]
    par_win = names[0] if a["metrics"]["params_total"] <= b["metrics"]["params_total"] else names[1]
    gpu_win = names[0] if (a["metrics"]["gpu"]["avg_ms"] <= b["metrics"]["gpu"]["avg_ms"]) else names[1]
    cpu_win = names[0] if (a["metrics"]["cpu"]["avg_ms"] <= b["metrics"]["cpu"]["avg_ms"]) else names[1]
    lines.append(f"  - Najveca test accuracy : {acc_win}")
    lines.append(f"  - Najmanje parametara   : {par_win}")
    lines.append(f"  - Najbrza GPU inferencija: {gpu_win}")
    lines.append(f"  - Najbrza CPU inferencija: {cpu_win}")
    lines.append("")
    lines.append("Klase Fashion-MNIST (0-9): " + ", ".join(CLASS_NAMES))
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"  spremljeno: {path}")
    print("\n" + text)


def save_weights(model, name, history):
    sd_path   = os.path.join(OUT_DIR, f"{name}.pt")
    ckpt_path = os.path.join(OUT_DIR, f"{name}_ckpt.pt")

    torch.save(model.state_dict(), sd_path)

    torch.save({
        "model_class": type(model).__name__,
        "state_dict":  model.state_dict(),
        "test_acc":    history["test_acc"],
        "test_loss":   history["test_loss"],
        "best_val_acc":  history["best_val_acc"],
        "best_val_loss": history["best_val_loss"],
        "best_epoch":  history["best_epoch"],
        "epochs_run":  history["epochs_run"],
        "class_names": CLASS_NAMES,
        "seed": SEED, "batch_size": BATCH_SIZE, "lr": LR, "split": SPLIT,
    }, ckpt_path)

    kb = os.path.getsize(sd_path) / 1024
    print(f"  spremljene tezine: {sd_path} ({kb:.2f} KB)  +  {ckpt_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Uredaj za trening: {device} ({device_name})")

    model_specs = [("SkipNet", SkipNet), ("NoSkipNet", NoSkipNet)]
    results = {}

    for name, cls in model_specs:
        model, history = train_model(cls(), name, device)
        results[name] = {"model": model, "history": history}
        save_weights(model, name, history)

    for name, _ in model_specs:
        results[name]["metrics"] = measure_all(results[name]["model"], name, device)

    print("\n--- Spremam rezultate ---")
    plot_training_curves(results, os.path.join(OUT_DIR, "training_curves.png"))
    plot_metrics(results,         os.path.join(OUT_DIR, "head_to_head_metrics.png"))
    write_report(results, device_name, os.path.join(OUT_DIR, "comparison.txt"))
    print("\nGotovo.")


if __name__ == "__main__":
    main()
