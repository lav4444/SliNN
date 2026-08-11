import collections

import torch
import torch.nn as nn
from torchvision import models



MODEL_PATH = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"
layer_register_json = "/home/tomi/code/dipl/arch_agnostic/LAYER_REGISTER.json"

# A — Nositelji kapaciteta
# ima weight s dim in (2,4) i weight.shape[0] odgovara broju izlaznih jedinica

# B — Vezani normalizatori
# ima weight s dim == 1, ili ima running_mean (norm sloj)

# C — Bezparametarske točke mjerenja
# nema parametara, izlazni shape == ulazni

# D — Topološki operatori
# nema parametara, izlazni shape != ulazni ili prima više ulaza

# E — Task mehanika
# nema parametara, nije tensor→tensor

def load_model(path):
    """Cijeli eager modul; hvata i ultralytics ckpt dict['model']."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().float()
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().float()
    raise SystemExit(f"format nije podrzan: {type(obj).__name__}")


# print model layers, all
model = load_model(MODEL_PATH)

for i, (name, m) in enumerate(model.named_modules()):
    leaf = not list(m.children())
    #print(f"{i:4d}  {'L' if leaf else ' '}  {name or '<root>':55s}  {type(m).__name__}")

types = collections.Counter(type(m).__name__ for _, m in model.named_modules()
                            if not list(m.children()))
print(f"\n--- unikatni tipovi leaf slojeva ({len(types)}) ---")
for tn, n in types.most_common():
    print(f"  {n:4d}x  {tn}")

