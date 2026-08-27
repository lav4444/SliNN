"""slinn/plugins/detection — decode + NMS + mAP za detekciju.

JAVNI OTVOR (jedino sto jezgra smije zvati):
    pick_adapter(model)   -> per-obitelj adapter (auto po arhitekturi)
    make_gt_loader(split) -> GT loader za mAP
    eval_map(model, adapter, loader, device) -> (metrike, n)
"""

from .adapters import pick_adapter, make_gt_loader, eval_map, set_bn_eval   # noqa: F401

__all__ = ["pick_adapter", "make_gt_loader", "eval_map", "set_bn_eval"]
