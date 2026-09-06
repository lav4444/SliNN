import sys

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

from classify import classify                                # noqa: E402
from position import positional                              # noqa: E402
from task import detect_task                                 # noqa: E402
from dataset import probe_dataset, resolve_splits           # noqa: E402


def _out_kind(task):
    return "kl" if task in ("classification", "segmentation") else "mse"


def prepare(model, adapter, device, dataset_path):
    cls = classify(model, adapter, device)
    pos, meta = positional(model, adapter, device, cls=cls)
    prunable = {n for n, v in pos.items() if v["morph"]}

    probe = probe_dataset(dataset_path)
    tinfo = detect_task(model, adapter, device, probe=probe)
    full = probe["mode"] == "full"

    return {
        "data_path": dataset_path, "mode": probe["mode"], "splits": probe["splits"],
        "n_samples": probe["n_samples"], "dataset_format": probe["format"],
        "split_plan": resolve_splits(probe["splits"]),
        "task": tinfo["task"], "task_source": tinfo["source"], "metrics": tinfo["metrics"],
        "kd_core": tinfo["kd_core"], "enhancers": tinfo["enhancers"] if full else [],
        "taps": meta["taps"], "kd_mode": meta["kd_mode"], "out_kind": _out_kind(tinfo["task"]),
        "prunable": prunable,
    }
