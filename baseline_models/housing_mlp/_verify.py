import os
import sys

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
import introspect as A
import position as P
from classify import classify, probe_adapter

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = A.load_any(HERE + "/model.pt", dev, code_dirs=[HERE])
pr = probe_adapter(m, dev, verbose=True)
cls = classify(m, pr, dev)
print("leafova", len(cls),
      "| morph", sum(v["morph"] for v in cls.values()),
      "| akt", sum(v["is_activation"] for v in cls.values()),
      "| unknown", sum(v["is_unknown"] for v in cls.values()))

pos, meta = P.positional(m, pr, dev, cls=cls)
print("kd_mode:", meta["kd_mode"], "| tapova:", meta["taps"], "|", meta["tap_desc"])
print("morph_final:", meta["morph_final"], "| terminal:", meta["terminal"])
for n, v in cls.items():
    print("  {:18s} morph={:5} act={:5}  {}".format(n, str(v["morph"]), str(v["is_activation"]), v["why"]))
