
import json
from pathlib import Path

BASE = Path("/home/tomi/code/dipl/custom_models/student_0_5_m")
RUNS = {
    "part1 (5860)": BASE / "KD_featlogit" / "training_history.json",
    "20k   (14485)": BASE / "KD_featlogit_20k" / "training_history.json",
}


def load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    hist = {k: load(v) for k, v in RUNS.items()}
    for k, h in hist.items():
        if h is None:
            print("nema:", RUNS[k])
            return

    print("=== val mAP@50:95 po epohi ===")
    print("{:<8} {:>16} {:>16}".format("epoha", *RUNS.keys()))
    all_ep = sorted(set(hist["part1 (5860)"]["val_epoch"]) | set(hist["20k   (14485)"]["val_epoch"]))
    for ep in all_ep:
        row = []
        for k, h in hist.items():
            if ep in h["val_epoch"]:
                row.append("{:.4f}".format(h["val_map"][h["val_epoch"].index(ep)]))
            else:
                row.append("-")
        print("{:<8} {:>16} {:>16}".format(ep, *row))
    print("")

    print("=== Najbolje i zadnje ===")
    print("{:<26} {:>12} {:>12} {:>12} {:>12}".format(
        "run", "best mAP", "best epoha", "best mAP50", "val_resp"))
    for k, h in hist.items():
        i = h["val_map"].index(max(h["val_map"]))
        print("{:<26} {:>12.4f} {:>12} {:>12.4f} {:>12.4f}".format(
            k, h["val_map"][i], h["val_epoch"][i], h["val_map_50"][i], h["val_kd_loss"][i]))
    print("")

    b1 = max(hist["part1 (5860)"]["val_map"])
    b2 = max(hist["20k   (14485)"]["val_map"])
    print("apsolutno: {:+.4f}   relativno: {:.2f}x".format(b2 - b1, b2 / b1))
    print("")

    print("=== Trening loss u epohi najboljeg checkpointa ===")
    print("{:<26} {:>10} {:>10} {:>10} {:>10}".format("run", "loss", "feat", "cls", "box"))
    for k, h in hist.items():
        i = h["val_map"].index(max(h["val_map"]))
        ep = h["val_epoch"][i]
        j = h["epoch"].index(ep)
        print("{:<26} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            k, h["train_loss"][j], h["train_feat"][j], h["train_cls"][j], h["train_box"][j]))
    print("")

    print("=== Trajanje ===")
    for k, h in hist.items():
        print("{:<26} {} epoha odradeno, {} validacija".format(
            k, len(h["epoch"]), len(h["val_epoch"])))


if __name__ == "__main__":
    main()
