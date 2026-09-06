
import json
import re
from pathlib import Path

CUSTOM = Path("/home/tomi/code/dipl/custom_models")
STUDENTS = [
    ("0.5M", "student_0_5_m"),
    ("1M", "student_1_m"),
    ("2M", "student_2_m"),
    ("yolo26n", "student_yolo26n"),
]
SPLITS = ("TRAIN", "VAL", "TEST")


def eval_maps(f: Path) -> dict:
    if not f.exists():
        return {}
    txt = f.read_text()
    out = {}
    for s in SPLITS:
        m = re.search(r"=+ " + s + r" \([0-9]+ images\) =+(.*?)(?==+ [A-Z]+ \(|\Z)",
                      txt, re.S)
        if m:
            mm = re.search(r"mAP@50:95 = ([0-9.]+)", m.group(1))
            if mm:
                out[s] = float(mm.group(1))
    return out


def params(f: Path):
    if not f.exists():
        return None
    m = re.search(r"StudentYOLOFeat \(([\d,]+) params\)", f.read_text())
    return int(m.group(1).replace(",", "")) if m else None


def log_state(f: Path) -> str:
    if not f.exists():
        return "NEMA LOGA"
    txt = f.read_text()
    if "Saved:" not in txt:
        return "NIJE ZAVRSIO"
    if "Stopped early" in txt:
        m = re.search(r"Early stopping at epoch (\d+)", txt)
        return "early stop @ ep " + (m.group(1) if m else "?")
    return "max epochs"


def main():
    print("=== Provjera dovrsenosti ===")
    print("{:<9} {:<24} {:<10} {:<10} {:<10} {:<10}".format(
        "student", "trening", "ckpt", "history", "plots", "eval"))
    all_ok = True
    for label, sd in STUDENTS:
        d = CUSTOM / sd / "KD_featlogit_20k"
        state = log_state(d / "training_log.txt")
        marks = []
        for f in ("checkpoints/best.pt", "training_history.json",
                  "training_plots.png", "eval_result.txt"):
            ok = (d / f).exists()
            marks.append("OK" if ok else "NEMA")
            if not ok:
                all_ok = False
        print("{:<9} {:<24} {:<10} {:<10} {:<10} {:<10}".format(label, state, *marks))
    print("")

    print("=== KD_featlogit: part1 (5860) vs 20k (14485) ===")
    print("Validacija i eval su na part1 splitovima u oba runa -> direktno usporedivo.")
    print("")
    hdr = "{:<9} {:>10} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>6}"
    print(hdr.format("student", "params", "val p1", "val 20k", "test p1", "test 20k",
                     "tr p1", "tr 20k", "val x"))
    print("-" * 88)
    for label, sd in STUDENTS:
        e1 = eval_maps(CUSTOM / sd / "KD_featlogit" / "eval_result.txt")
        e2 = eval_maps(CUSTOM / sd / "KD_featlogit_20k" / "eval_result.txt")
        n = params(CUSTOM / sd / "KD_featlogit" / "eval_result.txt")
        if not e1 or not e2:
            print("{:<9} (nedostaje eval)".format(label))
            continue
        ratio = e2["VAL"] / e1["VAL"] if e1.get("VAL") else float("nan")
        print(hdr.format(
            label, "{:,}".format(n) if n else "?",
            "{:.4f}".format(e1.get("TRAIN", 0)) if False else "{:.4f}".format(e1["VAL"]),
            "{:.4f}".format(e2["VAL"]),
            "{:.4f}".format(e1["TEST"]), "{:.4f}".format(e2["TEST"]),
            "{:.4f}".format(e1["TRAIN"]), "{:.4f}".format(e2["TRAIN"]),
            "{:.2f}x".format(ratio)))
    print("")

    print("=== Trening detalji 20k runova ===")
    print("{:<9} {:>8} {:>12} {:>12} {:>10}".format(
        "student", "epoha", "best epoha", "best val mAP", "val_resp"))
    for label, sd in STUDENTS:
        h = CUSTOM / sd / "KD_featlogit_20k" / "training_history.json"
        if not h.exists():
            continue
        d = json.loads(h.read_text())
        i = d["val_map"].index(max(d["val_map"]))
        print("{:<9} {:>8} {:>12} {:>12.4f} {:>10.4f}".format(
            label, len(d["epoch"]), d["val_epoch"][i], d["val_map"][i], d["val_kd_loss"][i]))
    print("")
    print("SVE PRISUTNO" if all_ok else "NESTO NEDOSTAJE -- vidi gore")


if __name__ == "__main__":
    main()
