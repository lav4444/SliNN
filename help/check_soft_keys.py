"""
check_soft_keys.py -- provjera jesu li part1 i part2 soft cache fajlovi
istog SADRZAJA (kljucevi, oblici, dtype), ne samo istog imena.

Motivacija: part2/yolo26l/train/soft je 56 GB naspram part1-ovih 5.3 GB.
"""

from pathlib import Path
import torch

MINI = Path("/home/tomi/code/dipl/datasets/mini_set")
PARTS = {
    "part1": MINI / "sub10k_open_images_v7",
    "part2": MINI / "sub10k_open_images_v7_part2",
}
SPLITS = ("train", "val", "test")


def describe(p: Path) -> str:
    d = torch.load(p, map_location="cpu", weights_only=False)
    out = []
    for k in sorted(d.keys()):
        v = d[k]
        if torch.is_tensor(v):
            out.append("{}={}{}".format(k, tuple(v.shape), str(v.dtype).replace("torch.", " ")))
        elif isinstance(v, dict):
            out.append("{}=dict({})".format(k, ",".join(sorted(v.keys()))))
        else:
            out.append("{}={}".format(k, type(v).__name__))
    return "  ".join(out)


def main():
    for name, root in PARTS.items():
        print("=== " + name + " ===")
        for s in SPLITS:
            d = root / "yolo26l" / s / "soft"
            if not d.is_dir():
                print("  {}: (nema)".format(s))
                continue
            files = sorted(d.glob("*.pt"))
            sample = files[0]
            mb = sample.stat().st_size / 1024 / 1024
            print("  {:<6} n={:<6} {:.2f} MB/file".format(s, len(files), mb))
            print("         " + describe(sample))
        print("")

    print("=== feat cache (samo train) ===")
    for name, root in PARTS.items():
        d = root / "yolo26l" / "train" / "feat"
        if not d.is_dir():
            print("  {}: (nema)".format(name))
            continue
        files = sorted(d.glob("*.pt"))
        sample = files[0]
        mb = sample.stat().st_size / 1024 / 1024
        dd = torch.load(sample, map_location="cpu", weights_only=False)
        shapes = [tuple(t.shape) for t in dd["feat"]]
        print("  {:<6} n={:<6} {:.2f} MB/file  feat={}  dtype={}".format(
            name, len(files), mb, shapes, dd["feat"][0].dtype))


if __name__ == "__main__":
    main()
