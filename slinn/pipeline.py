import sys


from classify import classify                                # noqa: E402
from position import positional                              # noqa: E402
from task import detect_task                                 # noqa: E402
from dataset import probe_dataset, resolve_splits           # noqa: E402
import dataset as DS                                        # noqa: E402


def _out_kind(task):
    return "kl" if task in ("classification", "segmentation") else "mse"


def apply_input_spec(adapter, dataset_path, verbose=True):
    spec = DS.input_spec(dataset_path)
    mode, old = getattr(adapter, "_mode", "image"), getattr(adapter, "imgsz", None)
    src = None
    if spec.get("length"):
        adapter.imgsz = int(spec["length"]); src = "input.json"
    elif getattr(adapter, "flexible", False):
        root = DS.detect_format(dataset_path).get("root", dataset_path)
        n = DS.native_length(root, mode, getattr(adapter, "_in_ch", 1))
        if n and n != old:
            adapter.imgsz = int(n); src = "native-iz-podataka"
    adapter.sr = int(spec["sample_rate"]) if spec.get("sample_rate") else None
    adapter.size = tuple(spec["size"]) if spec.get("size") else None
    adapter.norm = spec.get("norm") or None
    if verbose and (adapter.size or adapter.norm):
        print("[ulaz] 2D ugovor: velicina {} · normalizacija {}".format(
            adapter.size or "(izvorna)", "deklarirana" if adapter.norm else "nema"))
    if verbose and (src or adapter.sr):
        print("[ulaz] duljina {}{} · frekvencija {}".format(
            adapter.imgsz, " (bilo {}, izvor: {})".format(old, src) if src else " (probe)",
            "{} Hz (deklarirano)".format(adapter.sr) if adapter.sr else "izvorna (nije deklarirana)"))
    return adapter


def prepare(model, adapter, device, dataset_path):
    apply_input_spec(adapter, dataset_path)
    cls = classify(model, adapter, device)
    pos, meta = positional(model, adapter, device, cls=cls)
    prunable = {n for n, v in pos.items() if v["morph"]}

    import morph as _C
    tapped = _C.tap_coupled(model, adapter, device, meta["taps"], prunable)
    kd_note = None
    if tapped:
        prunable -= tapped
        print("[pipeline] {} slojeva izbaceno iz prunable — dijele tp-grupu s feature-KD tapom "
              "(rez bi pomaknuo sirinu tapa).".format(len(tapped)))

    if meta["taps"] and not prunable and (prunable | tapped):
        n_back = len(tapped)
        prunable = prunable | tapped
        meta["taps"], meta["kd_mode"] = {}, "logit"
        kd_note = ("svi rezivi slojevi bili su spregnuti s feature-KD tapovima (rezidualni tok) "
                   "-> feature clan iskljucen, KD je sada CISTI LOGIT; vraceno {} rezivih slojeva"
                   .format(n_back))
        print("[pipeline] " + kd_note.upper()[:1] + kd_note[1:])

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
        "terminal": meta.get("terminal_names", []),
        "prunable": prunable,
        "kd_note": kd_note,
    }
