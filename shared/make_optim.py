# -*- coding: utf-8 -*-
"""make_optim.py — sagradi BASELINE_OPTIM iz BASELINE_RAW, s najmanjim mogucim diffom.

PRAVILA KOJIH SE DRZI:
  * BASELINE_RAW se SAMO CITA. Nista se u njemu ne mijenja niti dodaje.
  * BASELINE_OPTIM je samodostatan: ima vlastitu kopiju tezina, vlastiti data.py i
    vlastitu eval skriptu. Ne poseze u druge mjerne mape ni u vrijeme mjerenja.
  * Datasetovi se NE kopiraju — oni su ULAZ i zive jednom, u shared/datasets/.
    data.py racuna korijen kao <korijen>/<MJERENJE>/<model>/ -> ... -> shared/datasets,
    pa kopija pogadja isti skup bez ijedne izmjene.
  * Imena skripti ostaju ista (eval_baseline.py / evaluate.py) da run_evals.sh radi
    nepromijenjen, samo s EVAL_DIR=BASELINE_OPTIM.

STO SE MIJENJA U KOPIJI EVAL SKRIPTE (i nista vise):
  1. ucitavanje modela: torch.load(model.pt)  ->  runners.load_optim(...)
  2. CSV stupac `runtime`: eager -> onnxruntime, na jednom mjestu (omotac oko EMIT)
  3. samo yolo: PRED_ROOT dobiva ime mjerne mape, da meta.json BASELINE_RAW-a
     ostane netaknut (inace bi ga OPTIM run prepisao — PRED_ROOT je u shared/datasets/)

Sve ostalo — ucitavanje podataka, predobrada, letterbox, prag, NMS, metrika, format
.txt izvjestaja — ostaje bajt u bajt isto. Inace bi se mjerila razlika izmedju dva
mjerna aparata, a ne izmedju eagera i ONNX Runtimea.

Pokretanje (na uredjaju):
    python shared/make_optim.py
Ponovno pokretanje je sigurno: postojece mape se preskacu osim ako FORCE=1.
"""
import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "BASELINE_RAW")
DST = os.path.join(ROOT, os.environ.get("OPTIM_DIR", "BASELINE_OPTIM"))
FORCE = os.environ.get("FORCE", "").strip() not in ("", "0", "false", "no")

MODELS = ["housing_mlp", "speechcommands_m5", "midas_depth", "sst2_distilbert",
          "voc_deeplabv3", "yolo26n", "yolo26l"]

# Rezultati RAW-a ne idu u OPTIM: to su tudja mjerenja.
SKIP = ("eval_result", "results_", "export_log", "model.onnx", "model_onnx.json",
        "__pycache__", ".log")

EMIT_ANCHOR = "import emit as EMIT                                             # noqa: E402"

EMIT_BLOCK = EMIT_ANCHOR + """
import runners as RUN                                           # noqa: E402

# `runtime` u CSV-u se postavlja na JEDNOM mjestu; svi EMIT.write pozivi nize ostaju
# nedirnuti, pa se kopija i izvornik razlikuju samo u ucitavanju modela.
_EMIT_MOD = EMIT


class EMIT(object):                                             # noqa: F811
    @staticmethod
    def write(*a, **k):
        k.setdefault("runtime", RUN.runtime_name())
        return _EMIT_MOD.write(*a, **k)
"""

# ---------------------------------------------------------------- zakrpe
LOAD_OLD = ('    model = torch.load(MODEL_PT, map_location=dev, weights_only=False).eval()')
LOAD_NEW = ('    # OPTIM: isti mjerni aparat, drugi runtime. `model.pt` ostaje u mapi samo\n'
            '    # kao izvor izvoza; mjeri se ONNX Runtime nad model.onnx.\n'
            '    model = RUN.load_optim(HERE, dev)')

CPU_OLD = '    m_cpu = torch.load(MODEL_PT, map_location=cpu, weights_only=False).eval()'
CPU_NEW = '    m_cpu = model                      # ORT je na Pi-ju ionako CPU — ista sesija'

YOLO_HELPER = """
# --- OPTIM: ONNX Runtime umjesto eager forwarda ------------------------------
# Ucitava se lijeno i jednom: sesija se gradi tek kad zatreba, a benchmark i
# evaluacija dijele istu, pa mjere isti objekt.
_ORT_CACHE = {}


def _ort():
    if "m" not in _ORT_CACHE:
        _ORT_CACHE["m"] = RUN.load_optim(str(SCRIPT_DIR), DEVICE)
    return _ORT_CACHE["m"]
# -----------------------------------------------------------------------------
"""

YOLO_PATCHES = [
    # forward u evaluaciji
    ("        out = model.model(tensor)",
     "        out = _ort()(tensor)                 # OPTIM: ONNX Runtime"),
    # forward u benchmarku
    ("    torch_model = model.model",
     "    torch_model = _ort()                     # OPTIM: ONNX Runtime"),
    # meta.json zivi u shared/datasets/ -> mora nositi ime mjerne mape, inace
    # bi OPTIM run prepisao meta.json koji je ostavio BASELINE_RAW
    ('PRED_ROOT = DATASET_ROOT / "yolo26n"',
     'PRED_ROOT = DATASET_ROOT / f"yolo26n_{SCRIPT_DIR.parent.name.lower()}"'),
    ('PRED_ROOT = DATASET_ROOT / "yolo26l"',
     'PRED_ROOT = DATASET_ROOT / f"yolo26l_{SCRIPT_DIR.parent.name.lower()}"'),
]


def _sub(s, old, new, where, optional=False):
    """Zamjena koja inzistira na jedinstvenom uzorku — tiha djelomicna zakrpa je gora
    od pada, jer bi ostavila skriptu koja se pokrece ali mjeri krivi objekt."""
    n = s.count(old)
    if n == 0 and optional:
        return s, False
    if n != 1:
        raise SystemExit(f"{where}: uzorak nije jedinstven ({n}x): {old.strip()[:70]}")
    return s.replace(old, new), True


def patch(path, model):
    s = io.open(path, encoding="utf-8").read()
    if "runners" in s:
        return "vec zakrpano"
    done = []
    if model.startswith("yolo"):
        # `_ort()` je lijen, pa smije stajati prije SCRIPT_DIR/DEVICE — tijelo se
        # razrjesava tek pri pozivu.
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


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"nema izvora: {SRC}")
    os.makedirs(DST, exist_ok=True)
    print(f"### {SRC}  ->  {DST}")
    print("### datasetovi se NE kopiraju (shared/datasets/)")
    total = 0
    for m in MODELS:
        s_dir, d_dir = os.path.join(SRC, m), os.path.join(DST, m)
        if not os.path.isdir(s_dir):
            print(f"  {m:20} NEMA U RAW-u — preskacem")
            continue
        if os.path.isdir(d_dir) and not FORCE:
            print(f"  {m:20} vec postoji — preskacem (FORCE=1 za ponovnu izgradnju)")
            continue
        if os.path.isdir(d_dir):
            shutil.rmtree(d_dir)
        os.makedirs(d_dir)
        mb = 0
        for f in sorted(os.listdir(s_dir)):
            if any(k in f for k in SKIP):
                continue
            sp = os.path.join(s_dir, f)
            if os.path.isdir(sp):                 # data/ i slicno ne idu
                continue
            shutil.copy2(sp, os.path.join(d_dir, f))
            mb += os.path.getsize(sp) / 1024 ** 2
        script = "evaluate.py" if m.startswith("yolo") else "eval_baseline.py"
        note = patch(os.path.join(d_dir, script), m)
        total += mb
        print(f"  {m:20} {mb:7.1f} MB   {note}")
    print(f"### ukupno {total:.1f} MB")
    print("### sljedece:  python shared/export.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
