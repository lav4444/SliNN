"""
strip_p3_from_part2.py -- izbaci "p3_features" (512,80,80) iz part2 soft cachea.

Zasto: taj kljuc cita samo KD_GT_FGD. KD_featlogit koristi iskljucivo
boxes_xywh + class_probs iz soft fajla, a feature targete uzima iz zasebnog
yolo26l/train/feat/. p3_features je ~95% velicine soft fajla (6.25 MB od 6.57 MB).

Ucinak: ~80 GB -> ~4.2 GB.

JEDNOSMJERNO. Vracanje zahtijeva ponovno pokretanje evaluate_part2.py (sati).
Ako ikad zelis FGD nad 20k, umjesto ovoga pokreni
baseline_models/yolo26l/reduce_p3_to_pca_part2.py -- to ostavi 48-d verziju
(~11.7 GB) projiciranu kroz ISTU PCA bazu kao part1.

Idempotentno: fajlovi bez p3_features se preskacu.
Atomicni upis (tmp + rename) pa prekid ne moze ostaviti pola fajla.

DRY_RUN = True samo prijavi sto bi radio, bez diranja diska.
"""

import os
from pathlib import Path

import torch

PART2 = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7_part2")
SPLITS = ("train", "val", "test")
KEY = "p3_features"
DRY_RUN = False

# Guard: nikad ne smije gaziti part1
assert "part2" in PART2.name, "ova skripta smije dirati samo part2"


def du_mb(d: Path) -> float:
    return sum(p.stat().st_size for p in d.glob("*.pt")) / 1024 / 1024


def main():
    print("Cilj: " + str(PART2))
    print("DRY_RUN =", DRY_RUN)
    print("")

    total_before = 0.0
    total_after = 0.0
    for split in SPLITS:
        d = PART2 / "yolo26l" / split / "soft"
        if not d.is_dir():
            print("[{}] nema soft dira, preskacem".format(split))
            continue
        files = sorted(d.glob("*.pt"))
        before = du_mb(d)
        total_before += before
        print("[{}] {} fajlova, {:.1f} GB".format(split, len(files), before / 1024))

        n_stripped = 0
        n_skipped = 0
        for i, f in enumerate(files):
            data = torch.load(f, map_location="cpu", weights_only=False)
            if KEY not in data:
                n_skipped += 1
                continue
            if DRY_RUN:
                n_stripped += 1
                continue
            new = {k: v for k, v in data.items() if k != KEY}
            tmp = f.with_suffix(".pt.tmp")
            torch.save(new, tmp)
            os.replace(tmp, f)          # atomicno
            n_stripped += 1
            if (i + 1) % 1000 == 0:
                print("  {}/{}".format(i + 1, len(files)))

        after = du_mb(d)
        total_after += after
        print("[{}] izbaceno {}, preskoceno {} -> {:.1f} GB".format(
            split, n_stripped, n_skipped, after / 1024))
        print("")

    print("UKUPNO: {:.1f} GB -> {:.1f} GB  (ustedjeno {:.1f} GB)".format(
        total_before / 1024, total_after / 1024, (total_before - total_after) / 1024))

    # sanity: ostali kljucevi moraju biti netaknuti
    probe = sorted((PART2 / "yolo26l" / "train" / "soft").glob("*.pt"))
    if probe:
        d = torch.load(probe[0], map_location="cpu", weights_only=False)
        print("")
        print("Kljucevi u " + probe[0].name + ":")
        for k in sorted(d.keys()):
            v = d[k]
            shape = tuple(v.shape) if torch.is_tensor(v) else type(v).__name__
            print("  {} = {}".format(k, shape))
        need = ("boxes_xywh", "class_probs", "letterbox")
        missing = [k for k in need if k not in d]
        print("")
        print("KD_featlogit potrebni kljucevi:", "OK" if not missing else "NEDOSTAJE " + str(missing))


if __name__ == "__main__":
    main()
