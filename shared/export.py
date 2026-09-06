# -*- coding: utf-8 -*-
"""export.py — torch -> ONNX, NA UREDJAJU, + vrata protiv slomljenog grafa.

ZASTO NA UREDJAJU: graf proizvodi verzija torcha koja ga izvozi. Da smo izvozili na
laptopu pa slali .onnx, razlika laptop/uredjaj bila bi skrivena varijabla u svakom
kasnijem mjerenju. Ovako je .onnx lokalni artefakt uredjaja, kao i engine na Jetsonu.

ZASTO VRATA (provjera odstupanja): bez njih, kad metrika kasnije mrdne, ne znas je li
kriv runtime ili slomljen izvoz. Vrata se prolaze PRIJE nego sto se napise ijedan
rezultat: izvoz koji ne prodje ne ostavlja .onnx, pa eval tog modela padne glasno
umjesto da tiho izmjeri nesto drugo.

DVIJE MJERNE MAPE, JEDAN IZVOZNIK. `OPTIM_DIR=BASELINE_OPTIM` izvozi original u FP32.
`OPTIM_DIR=SLINN_OPTIM` izvozi SliNN checkpointe; celija se ondje zove `<model>__<checkpoint>`
pa se graditelj bira po IMENU MODELA ispred `__`. Sto se izvozi ne odlucuje mapa nego SAM
CHECKPOINT: nosi li fake-quant module, izlazi QDQ (INT8) graf; ako ne, izlazi obican FP32.
Tako se recept ne dvoji i ne moze se dogoditi da mapa kaze INT8 a graf bude FP32.

Izlaz po celiji:
    model.onnx          graf
    model_onnx.json     opis: imena ulaza, stil izlaza, izmjereno odstupanje, broj Q/DQ
                        (runners.py ga cita — .onnx je time samoopisan)

Okolina:
    OPTIM_DIR=BASELINE_OPTIM   ciljna mjerna mapa (ili SLINN_OPTIM)
    EXPORT_ONLY=yolo26n,...    samo navedene celije (prazno = sve; podniska je dovoljna)
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

# INT8 (QDQ) ima DRUGA vrata, i to iz mjerenja, ne iz opreza. Torch simulira kvantizaciju u
# fp32 aritmetici, a ORT/TRT QDQ uzorak FUZIONIRAJU u prave int8 kernele — zbroj se skuplja
# drukcije, i razlika se kroz mrezu pojaca. Na yolu je ranije izmjereno 6.97% (todo_finish.txt,
# odjeljak C), i to na grafu koji NIJE bio slomljen. Prag od 1e-3 bi takav izvoz odbio bez
# razloga; prag koji propusta 6.97% ne bi vise bio nikakva provjera. Zato tri pojasa:
TOL_QDQ_TIHO = 5e-2            # ispod ovoga: nista za reci
TOL_QDQ_PAD = 2.5e-1           # iznad ovoga: graf je slomljen, a ne kvantiziran
# Izmedju je UPOZORENJE: izvoz prolazi, ali broj ide u log i u model_onnx.json. O tome je li
# gubitak prihvatljiv ne presudjuju vrata nego metrika koju eval poslije izmjeri — vrata samo
# hvataju slom. Uz to stoji STRUKTURNA provjera koju odstupanje ne moze zamijeniti: fake-quant
# koji se NIJE materijalizirao u Q/DQ cvorove daje graf koji radi, mjeri se, i tiho je FP32.


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


SLINN_PT = "slinn_model.pt"       # ime pod kojim make_slinn.py ostavi checkpoint


def _weights(d, fname="model.pt"):
    """Ime datoteke s tezinama u celiji: SliNN checkpoint ako postoji, inace original."""
    return SLINN_PT if os.path.isfile(os.path.join(d, SLINN_PT)) else fname


def _cell_quant(d):
    """Ucitaj `quant.py` IZ OVE CELIJE pod imenom `quant`, izbacivsi prethodno ucitani.

    ZASTO: `torch.load` kvantiziranog checkpointa radi `import quant` da nadje QConv/QLinear/
    ActFQ. Prvi uvoz ostaje u `sys.modules`, pa bi svaka sljedeca celija dobila TUDJE razrede.
    Dok su sve celije iz istog SliNN runa to prolazi neprimijeceno; cim se u istoj mjernoj mapi
    nadju dva runa, drugi se ili srusi ili — gore — ucita s tudjim razredima i mjeri se nesto
    drugo. Uhvaceno na zrcalu:
        AttributeError: Can't get attribute 'QConv' on <module 'quant' from '...housing.../quant.py'>
    """
    p = os.path.join(d, "quant.py")
    if os.path.isfile(p):
        sys.modules.pop("quant", None)
        _load_mod(p, "quant")


def _is_qdq(mod):
    """Nosi li modul fake-quant? Provjerava se po IMENU RAZREDA, ne isinstanceom: `quant.py`
    se ucitava iz svake celije zasebno, pa su to tehnicki razliciti razredi istog imena."""
    want = {"ActFQ", "QConv", "QConv2d", "QLinear"}
    return sum(1 for m in mod.modules() if type(m).__name__ in want)


def _load_pt(d, fname="model.pt"):
    """torch.load punog eager modela — s mapom modela na sys.path.

    Pickle nosi REFERENCU na klasu (`model_mlp.HousingMLP`), ne njezin kod, pa modul
    koji je definira mora biti uvoziv u trenutku ucitavanja. Put ostaje na sys.path i
    nakon povratka: klasa se referencira i kasnije, pri prvom forwardu.
    """
    if d not in sys.path:
        sys.path.insert(0, d)
    _cell_quant(d)
    return torch.load(os.path.join(d, _weights(d, fname)), map_location="cpu",
                      weights_only=False).eval()


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
    if os.path.isfile(os.path.join(d, SLINN_PT)):
        # SliNN checkpoint JE vec `y.model` (DetectionModel), spremljen punim modulom. Ne ide
        # kroz `YOLO(...)`: taj put ocekuje ultralytics ckpt-rjecnik i zna posegnuti za
        # fuzioniranjem, sto na prorezanom modelu s nasim omotacima nema smisla.
        core = _load_pt(d).eval()
        head = list(core.model)[-1]
    else:
        y = E.YOLO(os.path.join(d, E.MODEL_NAME))
        head = E.find_detect_head(y)
        core = y.model.eval()
    if getattr(head, "end2end", False):
        head.end2end = False        # eval treba gusti pred-NMS izlaz, ne top-K
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


def _base(cell):
    """Ime modela iz imena celije: `voc_deeplabv3__ckpt_2_qat` -> `voc_deeplabv3`."""
    return cell.split("__", 1)[0]


def _cells():
    """Celije u mjernoj mapi, poredane kao ORDER (pa po imenu). BASELINE_OPTIM ima po jednu
    celiju po modelu, SLINN_OPTIM po jednu po checkpointu — popis se zato CITA S DISKA, a ne
    drzi u konstanti koju bi trebalo dopunjavati svaki put kad SliNN proizvede novu verziju."""
    if not os.path.isdir(OPTIM):
        return []
    out = [c for c in os.listdir(OPTIM)
           if os.path.isdir(os.path.join(OPTIM, c)) and _base(c) in BUILDERS]
    rank = {m: i for i, m in enumerate(ORDER)}
    return sorted(out, key=lambda c: (rank.get(_base(c), 99), c))


# ---------------------------------------------------------------- izvoz + vrata
def _export(mod, xs, names, dyn, path, qdq=False):
    # `do_constant_folding` MORA biti iskljucen za QDQ: presavijanje bi DequantizeLinear nad
    # tezinama izracunalo unaprijed i ostavilo obican float initializer. Graf bi se izvezao,
    # radio bi, i bio bi FP32 — bez ijedne poruke o gresci.
    kw = dict(input_names=names, output_names=["output"], opset_version=OPSET,
              do_constant_folding=not qdq)
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


def _struct(path):
    """Sto je stvarno u grafu: Q/DQ cvorovi, zaostali BatchNorm, tudje domene."""
    import onnx
    g = onnx.load(path, load_external_data=False).graph
    q = sum(1 for x in g.node if x.op_type in ("QuantizeLinear", "DequantizeLinear"))
    bn = sum(1 for x in g.node if x.op_type == "BatchNormalization")
    doms = sorted({x.domain for x in g.node if x.domain})
    return {"qdq": q, "bn": bn, "nodes": len(g.node), "domains": doms}


def one(name, log):
    d = os.path.join(OPTIM, name)
    onnx_path = os.path.join(d, "model.onnx")
    t0 = datetime.datetime.now()
    mod, xs, names, dyn, style = BUILDERS[_base(name)](d)
    # NUZNO, i to na OMOTACU, ne samo na modelu u njemu: torch.onnx.export siri nacin
    # rada vrsnog modula na sve podmodule, a svjeze konstruiran nn.Module je u train
    # nacinu. Izmjereno kad je ovo nedostajalo:
    #   distilbert  dropout aktivan -> odstupanje 3.0e-01 umjesto 5e-07
    #   voc         BatchNorm nad [1,256,1,1] -> pad u ASPPPooling
    #   yolo        glava u train nacinu vraca dict umjesto (dense, ...)
    mod = mod.eval()
    n_fq = _is_qdq(mod)                    # graf odlucuje checkpoint, ne mjerna mapa
    _export(mod, xs, names, dyn, onnx_path, qdq=bool(n_fq))
    mb = os.path.getsize(onnx_path) / 1024 ** 2
    st = _struct(onnx_path)
    rel, abs_ = _gate(mod, xs, names, onnx_path)
    secs = (datetime.datetime.now() - t0).total_seconds()
    tol = TOL_QDQ_PAD if n_fq else TOL_REL

    def odbij(zasto):
        os.remove(onnx_path)               # bez artefakta -> eval padne glasno
        raise RuntimeError(zasto)

    if rel > tol:
        odbij(f"VRATA: rel {rel:.2e} > {tol:.0e} (abs {abs_:.2e}) — graf nije isti")
    if n_fq and not st["qdq"]:
        # Najopasniji ishod u cijelom lancu: sve radi, nista ne prijavi gresku, a mjeri se
        # FP32 graf pod imenom INT8 — i ubrzanje i gubitak tocnosti bi bili izmisljeni.
        odbij(f"VRATA: {n_fq} fake-quant modula u .pt, a 0 Q/DQ cvorova u grafu — "
              f"kvantizacija se nije materijalizirala")
    if st["domains"]:
        odbij(f"VRATA: tudje domene u grafu {st['domains']} — runtime ih nece znati izvesti")

    meta = {"input_names": names, "out_style": style, "opset": OPSET, "dynamic": bool(dyn),
            "gate_rel": rel, "gate_abs": abs_, "kvantiziran": bool(n_fq), "fakequant": n_fq,
            "qdq_cvorova": st["qdq"], "batchnorm_cvorova": st["bn"], "cvorova": st["nodes"],
            "torch": torch.__version__, "izvezeno": STAMP, "fake_ulaz": FAKE}
    with open(os.path.join(d, "model_onnx.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    kind = f"INT8 {st['qdq']:>4} Q/DQ" if n_fq else "FP32          "
    log(f"  {name:34} OK  {kind}  {mb:7.1f} MB   rel {rel:.2e}   abs {abs_:.2e}   {secs:6.1f}s")
    if n_fq and rel > TOL_QDQ_TIHO:
        log(f"  {'':34} ^ odstupanje iznad {TOL_QDQ_TIHO:.0e}: ocekivano za int8 kernele, "
            f"ali presudjuje metrika koju eval izmjeri, ne ova vrata")
    if n_fq and st["bn"]:
        log(f"  {'':34} ^ {st['bn']} BatchNormalization cvorova u QDQ grafu — skale su racunate "
            f"na nefoldanim tezinama (slinn/quant.py: fold_bn)")


def main():
    todo = [c for c in _cells() if (not ONLY or any(o in c for o in ONLY))]
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"### izvoz -> {OPTIM}   ({len(todo)} celija)")
    log(f"### opset {OPSET}   prag FP32 {TOL_REL:.0e}   prag INT8 {TOL_QDQ_PAD:.0e}   "
        f"uzoraka kroz vrata {N_GATE}"
        + ("   [FAKE ULAZ — ne za mjerenje]" if FAKE else ""))
    if not todo:
        log("### nema nijedne celije — pokreni shared/make_optim.py ili shared/make_slinn.py")
    # Model bez ijedne celije ne smije samo nestati iz ispisa: popis se sada cita s diska, pa
    # bi neuspjela gradnja izgledala jednako kao model koji nikad nije ni trazen.
    prazni = [m for m in ORDER if not any(_base(c) == m for c in _cells())]
    if prazni:
        log("### bez celije: " + " ".join(prazni)
            + "   (nije sagradjeno — make_optim.py / make_slinn.py)")
    ok, bad = [], []
    for name in todo:
        try:
            one(name, log)
            ok.append(name)
        except Exception as e:
            bad.append(name)
            log(f"  {name:34} PAO  {type(e).__name__}: {str(e)[:110]}")
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
