"""
compare_part1_part2.py -- je li part2 statisticki isti dataset kao part1?

Provjerava distribuciju klasa, broj objekata po slici, broj razreda po slici i
velicinu kutija. Vazno jer part2 nastaje iz preostalog pool-a NAKON izbacivanja
part1 slika, pa se sastav moze sustavno razlikovati.
"""

from collections import Counter
from pathlib import Path

MINI = Path("/home/tomi/code/dipl/datasets/mini_set")
PARTS = {
    "part1": MINI / "sub10k_open_images_v7",
    "part2": MINI / "sub10k_open_images_v7_part2",
}
SPLITS = ("train", "val", "test")
CLASSES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]


def scan(root: Path) -> dict:
    boxes = Counter()
    imgs_with_class = Counter()
    per_img_boxes = []
    per_img_classes = []
    areas = []
    n_img = 0
    for s in SPLITS:
        d = root / "labels" / s
        for p in sorted(d.iterdir()):
            if p.suffix.lower() != ".txt":
                continue
            n_img += 1
            cls_here = set()
            nb = 0
            for ln in p.read_text().splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                boxes[cid] += 1
                cls_here.add(cid)
                areas.append(w * h)
                nb += 1
            for c in cls_here:
                imgs_with_class[c] += 1
            per_img_boxes.append(nb)
            per_img_classes.append(len(cls_here))
    return {
        "n_img": n_img,
        "boxes": boxes,
        "imgs_with_class": imgs_with_class,
        "per_img_boxes": per_img_boxes,
        "per_img_classes": per_img_classes,
        "areas": areas,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    st = {k: scan(v) for k, v in PARTS.items()}

    print("=== Osnovno ===")
    print("{:<26} {:>12} {:>12}".format("", "part1", "part2"))
    print("{:<26} {:>12} {:>12}".format("slika", st["part1"]["n_img"], st["part2"]["n_img"]))
    for k in ("part1", "part2"):
        pass
    print("{:<26} {:>12} {:>12}".format(
        "kutija", sum(st["part1"]["boxes"].values()), sum(st["part2"]["boxes"].values())))
    print("{:<26} {:>12.2f} {:>12.2f}".format(
        "kutija / slika (mean)", mean(st["part1"]["per_img_boxes"]), mean(st["part2"]["per_img_boxes"])))
    print("{:<26} {:>12.1f} {:>12.1f}".format(
        "kutija / slika (median)", median(st["part1"]["per_img_boxes"]), median(st["part2"]["per_img_boxes"])))
    print("{:<26} {:>12.3f} {:>12.3f}".format(
        "razreda / slika (mean)", mean(st["part1"]["per_img_classes"]), mean(st["part2"]["per_img_classes"])))
    print("{:<26} {:>12.4f} {:>12.4f}".format(
        "rel. povrsina kutije (mean)", mean(st["part1"]["areas"]), mean(st["part2"]["areas"])))
    print("{:<26} {:>12.4f} {:>12.4f}".format(
        "rel. povrsina (median)", median(st["part1"]["areas"]), median(st["part2"]["areas"])))
    print("")

    print("=== Kutije po razredu (udio od ukupnog) ===")
    t1 = sum(st["part1"]["boxes"].values())
    t2 = sum(st["part2"]["boxes"].values())
    print("{:<12} {:>9} {:>7}   {:>9} {:>7}   {:>7}".format(
        "class", "p1 #", "p1 %", "p2 #", "p2 %", "delta"))
    for i, c in enumerate(CLASSES):
        b1, b2 = st["part1"]["boxes"][i], st["part2"]["boxes"][i]
        p1p, p2p = 100.0 * b1 / t1, 100.0 * b2 / t2
        print("{:<12} {:>9} {:>6.1f}%   {:>9} {:>6.1f}%   {:>+6.1f}pp".format(
            c, b1, p1p, b2, p2p, p2p - p1p))
    print("")

    print("=== Slike koje sadrze razred (udio od svih slika) ===")
    n1, n2 = st["part1"]["n_img"], st["part2"]["n_img"]
    print("{:<12} {:>9} {:>7}   {:>9} {:>7}   {:>7}".format(
        "class", "p1 img", "p1 %", "p2 img", "p2 %", "delta"))
    for i, c in enumerate(CLASSES):
        a1, a2 = st["part1"]["imgs_with_class"][i], st["part2"]["imgs_with_class"][i]
        q1, q2 = 100.0 * a1 / n1, 100.0 * a2 / n2
        print("{:<12} {:>9} {:>6.1f}%   {:>9} {:>6.1f}%   {:>+6.1f}pp".format(
            c, a1, q1, a2, q2, q2 - q1))
    print("")

    print("=== Histogram razreda po slici ===")
    print("{:<12} {:>10} {:>10}".format("#razreda", "part1 %", "part2 %"))
    h1 = Counter(st["part1"]["per_img_classes"])
    h2 = Counter(st["part2"]["per_img_classes"])
    for k in range(0, 7):
        if h1[k] or h2[k]:
            print("{:<12} {:>9.1f}% {:>9.1f}%".format(
                k, 100.0 * h1[k] / n1, 100.0 * h2[k] / n2))


if __name__ == "__main__":
    main()
