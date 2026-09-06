# -*- coding: utf-8 -*-
"""make_slinn.py — sagradi SLINN_OPTIM iz SliNN checkpointa, s najmanjim mogucim diffom.

Sestra od `make_optim.py` i drzi se ISTIH pravila:
  * BASELINE_RAW se SAMO CITA — odatle dolazi mjerni aparat (eval skripta, data.py).
  * SLINN_IN se SAMO CITA — odatle dolaze tezine (checkpointi koje je proizveo SliNN).
  * Datasetovi se NE kopiraju; zive jednom, u shared/datasets/.
  * Imena skripti ostaju ista (eval_baseline.py / evaluate.py).

STO JE OVDJE DRUKCIJE NEGO KOD BASELINE_OPTIM: jedan model daje VISE celija. SliNN run
ostavlja trajektoriju checkpointa (ckpt_1, ckpt_2, ... i best_quality_model), a svaki od
njih je zasebna verzija koju treba izmjeriti. Zato je celija `<model>__<checkpoint>`, a ne
`<model>`, i CSV stupac `model` nosi ime CELIJE — inace bi svi retci istog modela u
results.csv izgledali jednako i ne bi se dalo reci koji je koji.

ULAZ (rsync s laptopa; ime podmape JE ime modela iz BASELINE_RAW):
    SLINN_IN/<model>/ckpt_1.pt  ckpt_1_qat.pt  ...  run_meta.json  quant.py

IZLAZ:
    SLINN_OPTIM/<model>__ckpt_1_qat/
        <sve iz BASELINE_RAW/<model>/, osim tudjih rezultata>
        slinn_model.pt     checkpoint — IZVOR IZVOZA (mjeri se ONNX/engine iz njega)
        quant.py           klase koje pickle referencira (QConv2d/QLinear/ActFQ)
        slinn.json         iz cega je celija nastala

`_qat` u imenu znaci kvantiziran (fake-quant u grafu -> QDQ ONNX -> INT8). Bez sufiksa je
isti checkpoint prije kvantizacije; on se gradi samo ako se izrijekom trazi, jer poanta na
uredjaju je INT8.

DVIJE MJERNE MAPE IZ ISTIH TEZINA. `SLINN_DIR=SLINN_OPTIM` gradi celiju koja mjeri izvezeni
model (ORT/TRT); `SLINN_DIR=SLINN_RAW` gradi celiju koja mjeri ISTI checkpoint u EAGERU, dakle
pod istim runtimeom kao BASELINE_RAW. Bez te druge se ne zna koliko je od razlike model a
koliko runtime. Nacin se bira po imenu mape — `RAW` u imenu znaci eager.

Okolina:
    SLINN_IN=<put>       ulazna mapa       (zadano <korijen>/slinn_in)
    SLINN_DIR=...        ciljna mjerna mapa: SLINN_OPTIM (zadano) ili SLINN_RAW
    SLINN_WHICH=qat      qat | fp32 | both  (zadano qat)
    SLINN_ONLY=voc,...   samo navedeni modeli
    FORCE=1              pregradi postojece celije

Pokretanje (na uredjaju):
    python shared/make_slinn.py
    OPTIM_DIR=SLINN_OPTIM python shared/export.py
"""
import io
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "BASELINE_RAW")
IN = os.environ.get("SLINN_IN") or os.path.join(ROOT, "slinn_in")
DST = os.path.join(ROOT, os.environ.get("SLINN_DIR", "SLINN_OPTIM"))
WHICH = os.environ.get("SLINN_WHICH", "qat").strip().lower()
ONLY = [s.strip() for s in os.environ.get("SLINN_ONLY", "").replace(";", ",").split(",")
        if s.strip()]
FORCE = os.environ.get("FORCE", "").strip() not in ("", "0", "false", "no")
RAW_MODE = "RAW" in os.path.basename(DST).upper()     # eager celija umjesto izvezene

# Tudja mjerenja i tudji artefakti ne idu u novu celiju.
SKIP = ("eval_result", "results_", "export_log", "build_log", "model_onnx.json",
        "model.onnx", ".engine", "__pycache__", ".log")

WEIGHT = "slinn_model.pt"          # ime pod kojim checkpoint zivi u celiji

EMIT_ANCHOR = "import emit as EMIT                                             # noqa: E402"

EMIT_BLOCK = EMIT_ANCHOR + """
import runners as RUN                                           # noqa: E402

# Dvije stvari se postavljaju na JEDNOM mjestu, pa svi EMIT.write pozivi nize ostaju
# nedirnuti:
#   runtime  — sto je `load_optim` stvarno ucitao (onnxruntime na Pi-ju, tensorrt na Jetsonu)
#   model    — ime CELIJE, ne modela. U SLINN_OPTIM-u zivi vise checkpointa istog modela;
#              bez ovoga bi im svi retci u results.csv bili nerazlucivi.
_EMIT_MOD = EMIT


class EMIT(object):                                             # noqa: F811
    @staticmethod
    def write(script_dir, model, *a, **k):
        k.setdefault("runtime", RUN.runtime_name())
        cell = os.path.basename(os.path.abspath(str(script_dir)))
        return _EMIT_MOD.write(script_dir, cell, *a, **k)
"""

EMIT_BLOCK_RAW = EMIT_ANCHOR + """

# Ime CELIJE umjesto imena modela: u jednoj mjernoj mapi zivi vise checkpointa istog modela,
# pa bi im inace svi retci u results.csv bili nerazlucivi. `runtime` se NE dira — ostaje
# `eager`, jer se ovdje mjeri bas to.
_EMIT_MOD = EMIT


class EMIT(object):                                             # noqa: F811
    @staticmethod
    def write(script_dir, model, *a, **k):
        cell = os.path.basename(os.path.abspath(str(script_dir)))
        return _EMIT_MOD.write(script_dir, cell, *a, **k)
"""

# RAW: mijenja se SAMO koja se datoteka ucitava. `torch.load` ostaje, jer je eager runtime
# upravo ono sto se u ovoj celiji mjeri.
PT_OLD = 'MODEL_PT = os.path.join(HERE, "model.pt")'
PT_NEW = ('# SLINN_RAW: prorezan i QAT-om dotjeran checkpoint, ucitan EAGERO — isti runtime\n'
          '# kao BASELINE_RAW, pa je razlika pripisiva modelu a ne runtimeu.\n'
          'sys.path.insert(0, HERE)                    # pickle trazi `quant` uz .pt\n'
          'MODEL_PT = os.path.join(HERE, "slinn_model.pt")')

YOLO_HELPER_RAW = """
# --- SLINN_RAW: eager forward SliNN checkpointa ------------------------------
# `YOLO(MODEL_NAME)` nize ucitava ORIGINALNI .pt, ali se on NIKAD NE FORWARDA: sluzi samo
# pomocnim funkcijama (find_detect_head, imena razreda, NMS). Svaki forward ide kroz `_core()`,
# tj. kroz SliNN checkpoint, u eageru.
_CORE = {}


def _core():
    if "m" not in _CORE:
        d = str(SCRIPT_DIR)
        if d not in sys.path:
            sys.path.insert(0, d)                    # pickle trazi `quant` uz .pt
        m = torch.load(os.path.join(d, "slinn_model.pt"), map_location=DEVICE,
                       weights_only=False).eval()
        h = list(m.model)[-1]
        if getattr(h, "end2end", False):
            h.end2end = False                        # eval treba gusti pred-NMS izlaz
        _CORE["m"] = m.to(DEVICE)
    return _CORE["m"]
# -----------------------------------------------------------------------------
"""

YOLO_PATCHES_RAW = [
    ("        out = model.model(tensor)",
     "        out = _core()(tensor)                # SLINN_RAW: eager SliNN checkpoint"),
    ("    torch_model = model.model",
     "    torch_model = _core()                    # SLINN_RAW: eager SliNN checkpoint"),
    ('PRED_ROOT = DATASET_ROOT / "yolo26n"', 'PRED_ROOT = DATASET_ROOT / SCRIPT_DIR.name'),
    ('PRED_ROOT = DATASET_ROOT / "yolo26l"', 'PRED_ROOT = DATASET_ROOT / SCRIPT_DIR.name'),
]


# ---------------------------------------------------------------- zakrpe
LOAD_OLD = '    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()'
LOAD_NEW = ('    # SLINN: isti mjerni aparat, drugi model i drugi runtime. `slinn_model.pt` je\n'
            '    # samo izvor izvoza; mjeri se ono sto je `export.py` od njega napravio.\n'
            '    model = RUN.load_optim(HERE, dev)')

CPU_OLD = '    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()'
CPU_NEW = '    m_cpu = model                      # ORT je na Pi-ju ionako CPU — ista sesija'

YOLO_HELPER = """
# --- SLINN: ONNX Runtime / TensorRT umjesto eager forwarda -------------------
# `YOLO(MODEL_NAME)` nize i dalje ucitava ORIGINALNI .pt, ali se on NIKAD NE FORWARDA:
# sluzi samo pomocnim funkcijama (find_detect_head, imena razreda, NMS). Svaki forward
# ide kroz `_ort()`, tj. kroz izvezeni SliNN checkpoint. Sesija se gradi lijeno i jednom,
# pa evaluacija i benchmark mjere isti objekt.
_ORT_CACHE = {}


def _ort():
    if "m" not in _ORT_CACHE:
        _ORT_CACHE["m"] = RUN.load_optim(str(SCRIPT_DIR), DEVICE)
    return _ORT_CACHE["m"]
# -----------------------------------------------------------------------------
"""

YOLO_PATCHES = [
    ("        out = model.model(tensor)",
     "        out = _ort()(tensor)                 # SLINN: izvezeni checkpoint"),
    ("    torch_model = model.model",
     "    torch_model = _ort()                     # SLINN: izvezeni checkpoint"),
    # Predikcije zive u shared/datasets/ -> moraju nositi ime CELIJE, inace bi dva
    # checkpointa istog modela pisala jedan preko drugoga (i preko BASELINE_RAW-a).
    ('PRED_ROOT = DATASET_ROOT / "yolo26n"', 'PRED_ROOT = DATASET_ROOT / SCRIPT_DIR.name'),
    ('PRED_ROOT = DATASET_ROOT / "yolo26l"', 'PRED_ROOT = DATASET_ROOT / SCRIPT_DIR.name'),
]


def _sub(s, old, new, where, optional=False):
    """Zamjena koja inzistira na jedinstvenom uzorku — tiha djelomicna zakrpa je gora od
    pada, jer bi ostavila skriptu koja se pokrece ali mjeri krivi objekt."""
    n = s.count(old)
    if n == 0 and optional:
        return s, False
    if n != 1:
        raise SystemExit("{}: uzorak nije jedinstven ({}x): {}".format(where, n, old.strip()[:70]))
    return s.replace(old, new), True


def patch_raw(path, model):
    """RAW celija: eval skripta ostaje eager, mijenja se samo koje tezine ucitava."""
    s = io.open(path, encoding="utf-8").read()
    if "_EMIT_MOD" in s:
        return "vec zakrpano"
    done = []
    if model.startswith("yolo"):
        s, _ = _sub(s, EMIT_ANCHOR, EMIT_BLOCK_RAW + YOLO_HELPER_RAW, model)
        for old, new in YOLO_PATCHES_RAW:
            opt = old.startswith("PRED_ROOT")
            s, hit = _sub(s, old, new, model, optional=opt)
            if hit:
                done.append(old.strip().split("=")[0].strip()[:24])
    else:
        s, _ = _sub(s, EMIT_ANCHOR, EMIT_BLOCK_RAW, model)
        s, _ = _sub(s, PT_OLD, PT_NEW, model)
        done.append("MODEL_PT")
    io.open(path, "w", encoding="utf-8").write(s)
    return "eager, zakrpano: " + ", ".join(done)


def patch(path, model):
    s = io.open(path, encoding="utf-8").read()
    if "runners" in s:
        return "vec zakrpano"
    done = []
    if model.startswith("yolo"):
        s, _ = _sub(s, EMIT_ANCHOR, EMIT_BLOCK + YOLO_HELPER, model)
        for old, new in YOLO_PATCHES:
            opt = old.startswith("PRED_ROOT")        # svaki yolo ima samo svoj redak
            s, hit = _sub(s, old, new, model, optional=opt)
            if hit:
                done.append(old.strip().split("=")[0].strip()[:24])
    else:
        s, _ = _sub(s, EMIT_ANCHOR, EMIT_BLOCK, model)
        s, _ = _sub(s, LOAD_OLD, LOAD_NEW, model)
        s, hit = _sub(s, CPU_OLD, CPU_NEW, model, optional=True)
        done.append("load" + ("+cpu_bench" if hit else ""))
    io.open(path, "w", encoding="utf-8").write(s)
    return "zakrpano: " + ", ".join(done)


# ---------------------------------------------------------------- ulaz
def checkpoints(src):
    """(.pt datoteke koje treba izgraditi) iz ulazne mape, po pravilu SLINN_WHICH."""
    pts = sorted(f for f in os.listdir(src) if f.endswith(".pt"))
    qat = [f for f in pts if f.endswith("_qat.pt")]
    fp32 = [f for f in pts if not f.endswith("_qat.pt")]
    if WHICH == "qat":
        return qat
    if WHICH == "fp32":
        return fp32
    if WHICH == "both":
        return qat + fp32
    raise SystemExit("SLINN_WHICH mora biti qat | fp32 | both, a ne {!r}".format(WHICH))


def origin(src, pt):
    """Sto se o ovom checkpointu zna iz `run_meta.json` — zapisuje se u celiju kao trag."""
    p = os.path.join(src, "run_meta.json")
    if not os.path.isfile(p):
        return {"run_meta": "nema"}
    try:
        m = json.load(io.open(p, encoding="utf-8"))
    except ValueError as e:
        return {"run_meta": "necitljiv: {}".format(e)}
    out = {k: m.get(k) for k in ("model_path", "dataset_path", "phase", "metric_name",
                                 "metric_baseline", "metric_tol")}
    for q in m.get("qat") or []:
        if q.get("file") == pt:
            out["qat"] = {k: v for k, v in q.items() if k != "file"}
    return out


def build(model, pt, src):
    stem = os.path.splitext(pt)[0]
    cell = "{}__{}".format(model, stem)
    d_dir = os.path.join(DST, cell)
    if os.path.isdir(d_dir) and not FORCE:
        return cell, "vec postoji — preskacem (FORCE=1 za ponovnu izgradnju)"
    if os.path.isdir(d_dir):
        shutil.rmtree(d_dir)
    os.makedirs(d_dir)

    s_dir = os.path.join(RAW, model)
    mb = 0.0
    for f in sorted(os.listdir(s_dir)):
        if any(k in f for k in SKIP) or os.path.isdir(os.path.join(s_dir, f)):
            continue
        # Originalne tezine ne trebaju: eval ih vise ne ucitava. Iznimka je yolo, gdje
        # `YOLO(MODEL_NAME)` u evaluate.py mora imati sto otvoriti — ali se ne forwarda.
        if f.endswith(".pt") and not model.startswith("yolo"):
            continue
        sp = os.path.join(s_dir, f)
        shutil.copy2(sp, os.path.join(d_dir, f))
        mb += os.path.getsize(sp) / 1024 ** 2

    shutil.copy2(os.path.join(src, pt), os.path.join(d_dir, WEIGHT))
    w_mb = os.path.getsize(os.path.join(d_dir, WEIGHT)) / 1024 ** 2

    # `quant.py` MORA putovati: pickle nosi REFERENCU na klase QConv2d/QLinear/ActFQ, ne
    # njihov kod. Bez modula uz .pt, `torch.load` padne na ModuleNotFoundError.
    q = os.path.join(src, "quant.py")
    if os.path.isfile(q):
        shutil.copy2(q, os.path.join(d_dir, "quant.py"))
    elif pt.endswith("_qat.pt"):
        raise SystemExit("{}: nema quant.py u {} — kvantizirani pickle se bez njega ne "
                         "moze ucitati".format(cell, src))

    json.dump({"celija": cell, "model": model, "checkpoint": pt,
               "kvantiziran": pt.endswith("_qat.pt"), "izvor": src,
               "porijeklo": origin(src, pt)},
              io.open(os.path.join(d_dir, "slinn.json"), "w", encoding="utf-8"), indent=2)

    script = "evaluate.py" if model.startswith("yolo") else "eval_baseline.py"
    fn = patch_raw if RAW_MODE else patch
    note = fn(os.path.join(d_dir, script), model)
    return cell, "{:6.1f} MB aparat + {:7.1f} MB tezine   {}".format(mb, w_mb, note)


def main():
    if not os.path.isdir(RAW):
        raise SystemExit("nema mjernog aparata: {}".format(RAW))
    if not os.path.isdir(IN):
        raise SystemExit("nema ulaza: {}\n  ocekujem SLINN_IN/<model>/ckpt_*.pt "
                         "(rsync sa slinn/runs/<model>_<pecat>/)".format(IN))
    os.makedirs(DST, exist_ok=True)
    print("### {}  +  {}  ->  {}".format(RAW, IN, DST))
    print("### checkpointi: {}   runtime: {}   datasetovi se NE kopiraju (shared/datasets/)"
          .format(WHICH, "eager (RAW)" if RAW_MODE else "izvezeni (ORT/TRT)"))

    models = sorted(m for m in os.listdir(IN) if os.path.isdir(os.path.join(IN, m)))
    if ONLY:
        models = [m for m in models if m in ONLY]
    if not models:
        raise SystemExit("nema nijedne podmape u {}".format(IN))

    n = 0
    for model in models:
        src = os.path.join(IN, model)
        print("\n  {}".format(model))
        if not os.path.isdir(os.path.join(RAW, model)):
            print("    NEMA U BASELINE_RAW — ime podmape mora biti ime modela. Postoji: {}"
                  .format(", ".join(sorted(os.listdir(RAW)))))
            continue
        pts = checkpoints(src)
        if not pts:
            print("    nema nijednog .pt za SLINN_WHICH={}".format(WHICH))
            continue
        for pt in pts:
            cell, note = build(model, pt, src)
            print("    {:34} {}".format(cell, note))
            n += 1

    print("\n### {} celija u {}".format(n, DST))
    if RAW_MODE:
        print("### sljedece:  EVAL_DIR={} bash shared/run_slinn.sh".format(os.path.basename(DST)))
    else:
        print("### sljedece:  OPTIM_DIR={} python shared/export.py".format(os.path.basename(DST)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
