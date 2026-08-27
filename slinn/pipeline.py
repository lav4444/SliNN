"""
pipeline.py — ORKESTRACIJA (Faza 4.5): spaja sve slojeve u JEDAN tok.

  dataset_probe (format/mode/splits/oznake) + task_detector (task/metrike/enhancers)
  + classify/position (mask, tapovi, kd_mode)  ->  KD kontekst.

`prepare(model, adapter, device, dataset_path)` vrati kontekst koji engine (Faza 5) konzumira; prave ulaze
dohvaca `dataset.input_batch`, gubitak/vaznost `loss.kd_loss`/`loss.kd_importance`. Enhaneri se ukljucuju SAMO
u `mode=full` (uvjetno). `out_kind` mapira task -> tip output-KD-a (KL na [B,K] softmax; MSE inace).
"""
import sys


from classify import classify                                # noqa: E402
from position import positional                              # noqa: E402
from task import detect_task                                 # noqa: E402
from dataset import probe_dataset, resolve_splits           # noqa: E402
import dataset as DS                                        # noqa: E402


def _out_kind(task):
    """Tip output-KD-a: KL@T za SOFTMAX klasne distribucije — classification [B,K] I segmentation [B,K,H,W]
    (per-piksel KL, loss.kd_terms reshapea 4D). MSE za kontinuirano/nesoftmax: regression/depth/detection-raw i
    multilabel (sigmoid, ne softmax). (5.6 nalaz: seg pod MSE-output-KD gubi mIoU jer globalni MSE favorizira
    pozadinu; per-piksel KL to rjesava.)"""
    return "kl" if task in ("classification", "segmentation") else "mse"


def apply_input_spec(adapter, dataset_path, verbose=True):
    """Namjesti ULAZNI UGOVOR adaptera: deklaracija > native-iz-podataka > sto je probe pogodio.

    Probe bira NAJVECU radnu velicinu s ljestvice. Za slike je to ispravno (fiksni FC -> najveca
    radna JEST native), ali fleksibilan 1D model (potpuno konvolucijski + adaptivni pooling) prihvati
    bilo sto, pa "najveca radna" nema veze s treningom: M5 je treniran na 8000, a dobivao je 48000.
    Duljina i frekvencija su cinjenice o RECEPTU treninga — iz tezina se ne citaju. Zato:
      1. `input.json` u datasetu/model-mapi ({"length":8000,"sample_rate":8000}) — deklarirano
      2. inace, ako je adapter fleksibilan: PRIRODNA duljina iz stvarne datoteke
      3. inace ostaje sto je probe nasao
    Frekvencija se ne da pogoditi ni iz cega — samo deklaracija. Bez nje dekoder ne resamplira."""
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
    if verbose and (src or adapter.sr):
        print("[ulaz] duljina {}{} · frekvencija {}".format(
            adapter.imgsz, " (bilo {}, izvor: {})".format(old, src) if src else " (probe)",
            "{} Hz (deklarirano)".format(adapter.sr) if adapter.sr else "izvorna (nije deklarirana)"))
    return adapter


def prepare(model, adapter, device, dataset_path):
    """Jedan tok: struktura (classify/position) + podaci (probe_dataset) + task (detect_task) -> KD kontekst."""
    apply_input_spec(adapter, dataset_path)                   # PRIJE mjerenja — sve nizvodno ovisi o imgsz
    cls = classify(model, adapter, device)
    pos, meta = positional(model, adapter, device, cls=cls)
    prunable = {n for n, v in pos.items() if v["morph"]}

    # Tap je oznacen morph=False, ali to ga stiti samo kao KORIJEN reza — sirina mu ostaje spregnuta
    # sa slojevima u istoj tp-grupi. Rez takvog sloja pomakne tap i feature-KD pukne. Izbaci ih.
    import morph as _C                                        # lokalno: pipeline se uvozi i bez tp-a
    tapped = _C.tap_coupled(model, adapter, device, meta["taps"], prunable)
    if tapped:
        prunable -= tapped
        print("[pipeline] {} slojeva izbaceno iz prunable — dijele tp-grupu s feature-KD tapom "
              "(rez bi pomaknuo sirinu tapa).".format(len(tapped)))
    if meta["taps"] and not prunable:
        print("[pipeline] UPOZORENJE: svi rezivi slojevi su spregnuti s tapovima (tipicno rezidualni "
              "tok). Feature-KD i strukturni rez se ovdje iskljucuju — razmisli o kd_mode='logit'.")

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
