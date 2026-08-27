"""
pipeline.py — ORKESTRACIJA (Faza 4.5): spaja sve slojeve u JEDAN tok.

  dataset_probe (format/mode/splits/oznake) + task_detector (task/metrike/enhancers)
  + classify/position (mask, tapovi, kd_mode)  ->  KD kontekst.

`prepare(model, adapter, device, dataset_path)` vrati kontekst koji engine (Faza 5) konzumira; prave ulaze
dohvaca `dataset.input_batch`, gubitak/vaznost `loss.kd_loss`/`loss.kd_importance`. Enhaneri se ukljucuju SAMO
u `mode=full` (uvjetno). `out_kind` mapira task -> tip output-KD-a (KL na [B,K] softmax; MSE inace).
"""
import sys

_MORPH = "/home/tomi/code/dipl/morphology"
if _MORPH not in sys.path:
    sys.path.insert(0, _MORPH)

from classify import classify                                # noqa: E402
from position import positional                              # noqa: E402
from task import detect_task                                 # noqa: E402
from dataset import probe_dataset, resolve_splits           # noqa: E402


def _out_kind(task):
    """Tip output-KD-a: KL@T za SOFTMAX klasne distribucije — classification [B,K] I segmentation [B,K,H,W]
    (per-piksel KL, loss.kd_terms reshapea 4D). MSE za kontinuirano/nesoftmax: regression/depth/detection-raw i
    multilabel (sigmoid, ne softmax). (5.6 nalaz: seg pod MSE-output-KD gubi mIoU jer globalni MSE favorizira
    pozadinu; per-piksel KL to rjesava.)"""
    return "kl" if task in ("classification", "segmentation") else "mse"


def prepare(model, adapter, device, dataset_path):
    """Jedan tok: struktura (classify/position) + podaci (probe_dataset) + task (detect_task) -> KD kontekst."""
    cls = classify(model, adapter, device)
    pos, meta = positional(model, adapter, device, cls=cls)
    prunable = {n for n, v in pos.items() if v["morph"]}

    probe = probe_dataset(dataset_path)
    tinfo = detect_task(model, adapter, device, probe=probe)
    full = probe["mode"] == "full"

    return {
        # podaci
        "data_path": dataset_path, "mode": probe["mode"], "splits": probe["splits"],
        "n_samples": probe["n_samples"], "dataset_format": probe["format"],
        "split_plan": resolve_splits(probe["splits"]),   # {train,val,test,method}; 'AUTO' -> stratified_split u Fazi 5
        # task
        "task": tinfo["task"], "task_source": tinfo["source"], "metrics": tinfo["metrics"],
        "kd_core": tinfo["kd_core"], "enhancers": tinfo["enhancers"] if full else [],   # uvjetno na mode=full
        # struktura + KD config (za loss/importance)
        "taps": meta["taps"], "kd_mode": meta["kd_mode"], "out_kind": _out_kind(tinfo["task"]),
        "prunable": prunable,
    }
