
from collections import Counter
from pathlib import Path

MINI = Path("/home/tomi/code/dipl/datasets/mini_set")
P1 = MINI / "sub10k_open_images_v7"
P2 = MINI / "sub10k_open_images_v7_part2"
SPLITS = ("train", "val", "test")
CLASSES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def stems(root: Path, split: str, sub: str) -> set:
    d = root / sub / split
    if not d.is_dir():
        return set()
    if sub == "images":
        return {p.stem for p in d.iterdir() if p.suffix.lower() in IMG_EXT}
    return {p.stem for p in d.iterdir() if p.suffix.lower() == ".txt"}


def main():
    print("=== Brojevi po splitu ===")
    print("{:<7} {:>8} {:>8} {:>8} {:>8}".format("split", "p1 img", "p2 img", "p2 lbl", "img-lbl"))
    p1_all, p2_all = set(), set()
    ok_pairs = True
    for s in SPLITS:
        i1 = stems(P1, s, "images")
        i2 = stems(P2, s, "images")
        l2 = stems(P2, s, "labels")
        diff = len(i2 ^ l2)
        if diff:
            ok_pairs = False
        print("{:<7} {:>8} {:>8} {:>8} {:>8}".format(s, len(i1), len(i2), len(l2), diff))
        p1_all |= i1
        p2_all |= i2
    print("{:<7} {:>8} {:>8}".format("UKUPNO", len(p1_all), len(p2_all)))
    print("")

    print("=== Disjunktnost ===")
    overlap = p1_all & p2_all
    print("part1                :", len(p1_all))
    print("part2                :", len(p2_all))
    print("preklop              :", len(overlap))
    if overlap:
        print("  PRIMJERI:", sorted(overlap)[:5])
    print("unija (novi trening) :", len(p1_all | p2_all))
    print("")

    print("=== Interna duplikacija unutar part2 splitova ===")
    tr, va, te = (stems(P2, s, "images") for s in SPLITS)
    print("train n val :", len(tr & va))
    print("train n test:", len(tr & te))
    print("val   n test:", len(va & te))
    print("")

    print("=== Format labela (part2) ===")
    bad_fields, empty, boxes = 0, 0, Counter()
    oob = 0
    for s in SPLITS:
        d = P2 / "labels" / s
        for p in d.iterdir():
            if p.suffix.lower() != ".txt":
                continue
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad_fields += 1
                    continue
                cid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                boxes[cid] += 1
                if cid < 0 or cid >= len(CLASSES):
                    bad_fields += 1
                if any(v < -1e-6 or v > 1.0 + 1e-6 for v in vals):
                    oob += 1
    print("praznih .txt        :", empty)
    print("loših linija        :", bad_fields)
    print("koordinata van [0,1]:", oob)
    print("")

    print("=== Distribucija klasa (part2, sve slike) ===")
    print("{:<12} {:>10}".format("class", "#bboxes"))
    for i, c in enumerate(CLASSES):
        print("{:<12} {:>10}".format(c, boxes[i]))
    print("{:<12} {:>10}".format("UKUPNO", sum(boxes.values())))
    print("")

    print("=== dataset.yaml ===")
    y = P2 / "dataset.yaml"
    print(y.read_text() if y.exists() else "NE POSTOJI")

    print("=== Teacher cache (part2) ===")
    for sub in ("soft", "labels", "feat"):
        for s in SPLITS:
            d = P2 / "yolo26l" / s / sub
            n = len(list(d.iterdir())) if d.is_dir() else 0
            if n or d.is_dir():
                print("  yolo26l/{}/{}: {}".format(s, sub, n))
    if not (P2 / "yolo26l").exists():
        print("  (nema -- treba evaluate_part2.py + precompute_feats_part2.py)")
    print("")

    verdict = (not overlap) and ok_pairs and bad_fields == 0 and oob == 0
    print("ZAKLJUCAK:", "OK" if verdict else "PROBLEM -- vidi gore")


if __name__ == "__main__":
    main()
