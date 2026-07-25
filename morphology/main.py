"""
main.py — morphology eksperiment runner (terminalni tok). BEZ CLI argumenata.
Sve postavke su u config.py; sva logika u compress.run_dead_ft (isti runner kao GUI).

FAZA 1, korak 1: baseline perf -> [analiza] -> precompute teacher -> dead+near-dead rez
-> FT recovery (pure-KD) -> finalni perf + spremi best-quality-compressed.
"""

import torch

import config
import compress as C


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C.run_dead_ft(config.MODEL_SPEC, device)   # default-i (eval_max/val_cap/batch/...) dolaze iz config.py


if __name__ == "__main__":
    main()
