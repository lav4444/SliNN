"""
training_times.py -- trajanje treninga po runu, iz training_log.txt datoteka.

Parsira retke oblika:
    [epoch  28] train_loss=... (173.3s)
    [epoch  28] val_resp=...   (55.1s)
"""

import re
from pathlib import Path

CUSTOM = Path("/home/tomi/code/dipl/custom_models")

TRAIN_RE = re.compile(r"^\[epoch\s+(\d+)\].*?train_loss=.*?\(([\d.]+)s\)", re.M)
VAL_RE = re.compile(r"^\[epoch\s+(\d+)\].*?val_.*?\(([\d.]+)s\)", re.M)
IMGS_RE = re.compile(r"^Train:\s+(\d+)", re.M)


def hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h:
        return "{}h {:02d}m".format(h, m)
    if m:
        return "{}m {:02d}s".format(m, s)
    return "{}s".format(s)


rows = []
for log in sorted(CUSTOM.rglob("training_log.txt")):
    txt = log.read_text(errors="replace")
    tr = [(int(a), float(b)) for a, b in TRAIN_RE.findall(txt)]
    va = [(int(a), float(b)) for a, b in VAL_RE.findall(txt)]
    if not tr:
        continue
    m = IMGS_RE.search(txt)
    n_img = int(m.group(1)) if m else None
    t_train = sum(b for _, b in tr)
    t_val = sum(b for _, b in va)
    rows.append({
        "run": str(log.parent).replace(str(CUSTOM) + "/", ""),
        "epochs": len(tr),
        "n_img": n_img,
        "t_train": t_train,
        "t_val": t_val,
        "total": t_train + t_val,
        "s_ep": t_train / len(tr),
        "ms_img": 1000.0 * t_train / (len(tr) * n_img) if n_img else None,
    })

rows.sort(key=lambda r: r["run"])

print("{:<42} {:>5} {:>7} {:>9} {:>8} {:>9} {:>9}".format(
    "run", "eph", "slika", "s/epoha", "ms/img", "trening", "ukupno"))
print("-" * 96)
for r in rows:
    print("{:<42} {:>5} {:>7} {:>9.1f} {:>8} {:>9} {:>9}".format(
        r["run"], r["epochs"], r["n_img"] or 0, r["s_ep"],
        "{:.1f}".format(r["ms_img"]) if r["ms_img"] else "--",
        hms(r["t_train"]), hms(r["total"])))

print("")
print("=== KD vs GT, po studentu (samo part1 runovi, 5860 slika) ===")
print("{:<16} {:>12} {:>12} {:>12} {:>8}".format(
    "student", "GT s/epoha", "KDfeat s/e", "KDlogit s/e", "KD/GT"))
for st in ("student_0_5_m", "student_1_m", "student_2_m", "student_yolo26n"):
    get = lambda sub: next((r["s_ep"] for r in rows if r["run"] == st + "/" + sub), None)
    gt, kf, kl = get("GT_only"), get("KD_featlogit"), get("pure_KD")
    ratio = "{:.2f}x".format(kf / gt) if gt and kf else "--"
    print("{:<16} {:>12} {:>12} {:>12} {:>8}".format(
        st.replace("student_", ""),
        "{:.1f}".format(gt) if gt else "--",
        "{:.1f}".format(kf) if kf else "--",
        "{:.1f}".format(kl) if kl else "--",
        ratio))
