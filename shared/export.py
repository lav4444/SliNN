# -*- coding: utf-8 -*-
"""export.py — torch -> ONNX, NA UREDJAJU, + vrata protiv slomljenog grafa.

ZASTO NA UREDJAJU: graf proizvodi verzija torcha koja ga izvozi. Da smo izvozili na
laptopu pa slali .onnx, razlika laptop/uredjaj bila bi skrivena varijabla u svakom
kasnijem mjerenju. Ovako je .onnx lokalni artefakt uredjaja, kao i engine na Jetsonu.

ZASTO VRATA (provjera odstupanja): bez njih, kad metrika kasnije mrdne, ne znas je li
kriv runtime ili slomljen izvoz. Vrata se prolaze PRIJE nego sto se napise ijedan
rezultat: izvoz koji ne prodje ne ostavlja .onnx, pa eval tog modela padne glasno
umjesto da tiho izmjeri nesto drugo.

Izlaz po modelu (u BASELINE_OPTIM/<model>/):
    model.onnx          graf
    model_onnx.json     opis: imena ulaza, stil izlaza, izmjereno odstupanje
                        (runners.py ga cita — .onnx je time samoopisan)

Okolina:
    OPTIM_DIR=BASELINE_OPTIM   ciljna mjerna mapa
    EXPORT_ONLY=yolo26n,...    samo navedeni modeli (prazno = svi)
    EXPORT_FAKE=1              nasumican ulaz umjesto stvarnih uzoraka; SAMO za provjeru
                               same masinerije izvan uredjaja, NIKAD za mjerenje
"""
import datetime
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIM = os.path.join(ROOT, os.environ.get("OPTIM_DIR", "BASELINE_OPTIM"))
FAKE = os.environ.get("EXPORT_FAKE", "").strip() not in ("", "0", "false", "no")
ONLY = [s.strip() for s in os.environ.get("EXPORT_ONLY", "").replace(";", ",").split(",")
        if s.strip()]
STAMP = os.environ.get("RUN_STAMP") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OPSET = 17
N_GATE = 3                     # koliko stvarnih uzoraka ide kroz vrata
TOL_REL = 1e-3                 # FP32 izvoz: sve iznad ovoga je slomljen graf, ne format


# ---------------------------------------------------------------- pomocno
def _load_mod(path, name):
    """Ucitaj data.py / evaluate.py iz zadane mape kao modul (svaki ima svoj HERE)."""
    d = os.path.dirname(path)
    sys.path.insert(0, d)
    try:
        sp = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(sp)
        sys.modules[name] = m
        sp.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(d)


def _load_pt(d, fname="model.pt"):
    """torch.load punog eager modela — s mapom modela na sys.path.

    Pickle nosi REFERENCU na klasu (`model_mlp.HousingMLP`), ne njezin kod, pa modul
    koji je definira mora biti uvoziv u trenutku ucitavanja. Put ostaje na sys.path i
    nakon povratka: klasa se referencira i kasnije, pri prvom forwardu.
    """
    if d not in sys.path:
        sys.path.insert(0, d)
    return torch.load(os.path.join(d, fname), map_location="cpu", weights_only=False).eval()


class _PickDict(nn.Module):
    """DeepLabV3 vraca {'out':..., 'aux':...}; eval koristi samo 'out'."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)["out"]


class _PickLogits(nn.Module):
    """HF model vraca objekt s .logits; ONNX treba goli tenzor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


class _PickFirst(nn.Module):
    """YOLO glava vraca (dense, feats) u eval nacinu; eval uzima dense."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        if isinstance(o, (tuple, list)):
            return o[0]
        if isinstance(o, dict):     # znak da je glava u train nacinu — ne izvozi to tiho
            raise RuntimeError(f"YOLO glava vratila dict {list(o)} — modul nije u eval nacinu")
        return o


# ---------------------------------------------------------------- graditelji
# Svaki vraca (modul_za_izvoz, uzorci, imena_ulaza, dinamicke_osi, stil_izlaza).
# `uzorci` je lista torki: prva sluzi za trasiranje, sve zajedno prolaze kroz vrata.
#
# Dinamicke osi navode se i za IZLAZ, ne samo za ulaz: sve sto se ne oznaci exporter
# zabetonira na vrijednost iz trasiranja, pa bi prva slika drugog oblika pukla u ORT-u.
# Kod midasa je izlaz [B,H,W] (bez kanala) — prostorne osi su 1 i 2, ne 2 i 3.

def _b_housing(d):
    m = _load_pt(d)
    if FAKE:
        xs = [(torch.randn(1, 8),) for _ in range(N_GATE)]
    else:
        D = _load_mod(os.path.join(d, "data.py"), "d_housing")
        X = D.data_raw("val")[0]
        xs = [(X[i:i + 1],) for i in range(N_GATE)]
    return m, xs, ["input"], {"input": {0: "b"}, "output": {0: "b"}}, "tensor"


def _b_m5(d):
    m = _load_pt(d)
    if FAKE:
        xs = [(torch.randn(1, 1, 8000),) for _ in range(N_GATE)]
    else:
        D = _load_mod(os.path.join(d, "data.py"), "d_m5")
        xs = [(xb,) for i, (xb, _) in enumerate(D.loader("val", batch=1)) if i < N_GATE]
    return m, xs, ["input"], {"input": {0: "b"}, "output": {0: "b"}}, "tensor"


def _b_distilbert(d):
    m = _load_pt(d)
    if FAKE:
        xs = [(torch.randint(0, 2000, (1, 17)), torch.ones(1, 17, dtype=torch.long))
              for _ in range(N_GATE)]
    else:
        D = _load_mod(os.path.join(d, "data.py"), "d_bert")
        xs = [(e["input_ids"], e["attention_mask"])
              for i, (e, _) in enumerate(D.loader("validation", batch=1)) if i < N_GATE]
    # collate koristi padding=True (do najduzeg u batchu) -> duljina niza NIJE fiksna
    dyn = {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
           "output": {0: "b"}}
    return _PickLogits(m), xs, ["input_ids", "attention_mask"], dyn, "logits"


def _b_voc(d):
    m = _load_pt(d)
    if FAKE:
        xs = [(torch.randn(1, 3, 520, 693),) for _ in range(N_GATE)]
    else:
        D = _load_mod(os.path.join(d, "data.py"), "d_voc")
        tf, ds = D.transform(), D.voc("val")
        xs = [(tf(ds[i][0]).unsqueeze(0),) for i in range(N_GATE)]
    # smaller-edge 520 CUVA omjer stranica -> H i W se mijenjaju od slike do slike
    return (_PickDict(m), xs, ["input"],
            {"input": {0: "b", 2: "h", 3: "w"}, "output": {0: "b", 2: "h", 3: "w"}},
            "dict_out")


def _b_midas(d):
    D = _load_mod(os.path.join(d, "data.py"), "d_midas")
    for _hd in getattr(D, "HUB_DIRS", []):              # MiDaS/geffnet klase za torch.load
        if _hd not in sys.path:
            sys.path.insert(0, _hd)
    m = _load_pt(d)
    if FAKE:
        xs = [(torch.randn(1, 3, 256, 352),) for _ in range(N_GATE)]
    else:
        tf, ds = D.midas_transform(), D.NYUVal(N_GATE)
        xs = [(tf(np.asarray(ds[i][0], dtype=np.float32) / 255.0),) for i in range(N_GATE)]
    return (m, xs, ["input"],
            {"input": {0: "b", 2: "h", 3: "w"}, "output": {0: "b", 1: "h", 2: "w"}},
            "tensor")


def _b_yolo(d):
    E = _load_mod(os.path.join(d, "evaluate.py"), "y_" + os.path.basename(d))
    y = E.YOLO(os.path.join(d, E.MODEL_NAME))
    head = E.find_detect_head(y)
    if getattr(head, "end2end", False):
        head.end2end = False        # eval treba gusti pred-NMS izlaz, ne top-K
    core = y.model.eval()
    if FAKE:
        xs = [(torch.randn(1, 3, 640, 640),) for _ in range(N_GATE)]
    else:
        imgs = E.list_images(E.split_paths("val")["img_dir"])[:N_GATE]
        xs = [(E.preprocess_image(p)[0].unsqueeze(0),) for p in imgs]
    # 640x640 letterbox je fiksan po konstrukciji -> nema dinamickih osi
    return _PickFirst(core), xs, ["input"], None, "tensor"


BUILDERS = {
    "housing_mlp": _b_housing,
    "speechcommands_m5": _b_m5,
    "sst2_distilbert": _b_distilbert,
    "voc_deeplabv3": _b_voc,
    "midas_depth": _b_midas,
    "yolo26n": _b_yolo,
    "yolo26l": _b_yolo,
}
ORDER = ["housing_mlp", "speechcommands_m5", "midas_depth", "sst2_distilbert",
         "voc_deeplabv3", "yolo26n", "yolo26l"]        # od najjeftinijeg


# ---------------------------------------------------------------- izvoz + vrata
def _export(mod, xs, names, dyn, path):
    kw = dict(input_names=names, output_names=["output"], opset_version=OPSET,
              do_constant_folding=True)
    if dyn:
        kw["dynamic_axes"] = dyn
    # `dynamo=False` trazi stari (TorchScript) izvoznik. Laptop ima torch 2.6, Pi 2.14 —
    # jedina razlika koju nije bilo gdje isprobati. Ako taj argument nestane, pada se na
    # zadani put; vrata su ionako ta koja presudjuju je li graf ispravan.
    try:
        torch.onnx.export(mod, xs[0], path, dynamo=False, **kw)
    except TypeError as e:
        if "dynamo" not in str(e):          # TypeError iz samog trasiranja — ne zataskavaj
            raise
        torch.onnx.export(mod, xs[0], path, **kw)


def _gate(mod, xs, names, path):
    """Isti ulaz kroz torch i kroz ORT. Vraca (najgori_rel, najgori_abs)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    worst_r, worst_a = 0.0, 0.0
    for xt in xs:
        with torch.no_grad():
            ref = mod(*xt)
        ref = ref.detach().cpu().numpy()
        got = sess.run(None, {n: t.cpu().numpy() for n, t in zip(names, xt)})[0]
        if got.shape != tuple(ref.shape):
            raise RuntimeError(f"oblik izlaza: torch {tuple(ref.shape)} vs onnx {got.shape}")
        a = float(np.max(np.abs(got - ref)))
        worst_a = max(worst_a, a)
        worst_r = max(worst_r, a / (float(np.max(np.abs(ref))) + 1e-12))
    return worst_r, worst_a


def one(name, log):
    d = os.path.join(OPTIM, name)
    onnx_path = os.path.join(d, "model.onnx")
    t0 = datetime.datetime.now()
    mod, xs, names, dyn, style = BUILDERS[name](d)
    # NUZNO, i to na OMOTACU, ne samo na modelu u njemu: torch.onnx.export siri nacin
    # rada vrsnog modula na sve podmodule, a svjeze konstruiran nn.Module je u train
    # nacinu. Izmjereno kad je ovo nedostajalo:
    #   distilbert  dropout aktivan -> odstupanje 3.0e-01 umjesto 5e-07
    #   voc         BatchNorm nad [1,256,1,1] -> pad u ASPPPooling
    #   yolo        glava u train nacinu vraca dict umjesto (dense, ...)
    mod = mod.eval()
    _export(mod, xs, names, dyn, onnx_path)
    mb = os.path.getsize(onnx_path) / 1024 ** 2
    rel, abs_ = _gate(mod, xs, names, onnx_path)
    secs = (datetime.datetime.now() - t0).total_seconds()

    if rel > TOL_REL:
        os.remove(onnx_path)               # bez artefakta -> eval padne glasno
        raise RuntimeError(f"VRATA: rel {rel:.2e} > {TOL_REL:.0e} (abs {abs_:.2e}) — graf nije isti")

    with open(os.path.join(d, "model_onnx.json"), "w", encoding="utf-8") as f:
        json.dump({"input_names": names, "out_style": style, "opset": OPSET,
                   "dynamic": bool(dyn), "gate_rel": rel, "gate_abs": abs_,
                   "torch": torch.__version__, "izvezeno": STAMP, "fake_ulaz": FAKE},
                  f, indent=2)
    log(f"  {name:20} OK   {mb:7.1f} MB   rel {rel:.2e}   abs {abs_:.2e}   {secs:6.1f}s")


def main():
    todo = [m for m in ORDER if (not ONLY or m in ONLY)]
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"### izvoz -> {OPTIM}")
    log(f"### opset {OPSET}   prag rel {TOL_REL:.0e}   uzoraka kroz vrata {N_GATE}"
        + ("   [FAKE ULAZ — ne za mjerenje]" if FAKE else ""))
    ok, bad = [], []
    for name in todo:
        if not os.path.isdir(os.path.join(OPTIM, name)):
            log(f"  {name:20} PRESKACEM — nema mape (pokreni make_optim.py)")
            continue
        try:
            one(name, log)
            ok.append(name)
        except Exception as e:
            bad.append(name)
            log(f"  {name:20} PAO   {type(e).__name__}: {str(e)[:120]}")
            lines.append(traceback.format_exc())
    log(f"### {len(ok)}/{len(ok) + len(bad)} izvezeno"
        + (f"   palo:{' '.join(bad)}" if bad else ""))

    p = os.path.join(OPTIM, f"export_log_{STAMP}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] -> {p}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
