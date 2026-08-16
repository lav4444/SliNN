"""Provjeri resolve_splits politiku na zoo-datasetima + stratified_split na sintetickim oznakama."""
import os
import sys

sys.path.insert(0, "/home/tomi/code/dipl/arch_agnostic")
import torch
import dataset as DS

BM = "/home/tomi/code/dipl/baseline_models"
PATHS = [
    ("sub10k", "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"),
    ("housing", f"{BM}/housing_mlp/data"),
    ("speechcommands", f"{BM}/speechcommands_m5/data"),
    ("voc", f"{BM}/voc_deeplabv3/data"),
    ("nyu", f"{BM}/midas_depth/data"),
    ("sst2", f"{BM}/sst2_distilbert/data"),
]

print("=== resolve_splits (na stvarnim datasetima) ===")
for name, p in PATHS:
    sp = DS.probe_dataset(p)["splits"]
    r = DS.resolve_splits(sp)
    print(f"  {name:16s} splits={str(sp):<28} -> {r['method']:<15} train={r['train']} val={r['val']} test={r['test']}")

print("\n=== stratified_split (sinteticke oznake) ===")
# klasifikacija: 300 uzoraka, 3 klase neuravnotezeno (150/100/50)
y = torch.tensor([0] * 150 + [1] * 100 + [2] * 50)
tr, va, te = DS.stratified_split(y)
import numpy as np
for nm, idx in (("train", tr), ("val", va), ("test", te)):
    c = np.bincount(y[idx].numpy(), minlength=3)
    frac = c / c.sum()
    print(f"  cls {nm:6s} n={len(idx):3d}  po klasi={c.tolist()}  udio={np.round(frac,2).tolist()}")
# regresija: 300 kontinuiranih
yr = torch.randn(300)
tr, va, te = DS.stratified_split(yr, seed=1)
print(f"  reg n(train/val/test)={len(tr)}/{len(va)}/{len(te)}  mean(train/val/test)="
      f"{yr[tr].mean():.2f}/{yr[va].mean():.2f}/{yr[te].mean():.2f}  (blizu -> dobra stratifikacija)")
# leakage: disjunktni?
allidx = set(tr) | set(va) | set(te)
print(f"  disjunktno & potpuno: {len(allidx)==300 and len(tr)+len(va)+len(te)==300}")
