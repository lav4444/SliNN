"""slinn/gui/backend.py — 6.1 engine-vodjeni backend (dijeljeno prep_worker + worker).

ULAZ = config.json {model_path, dataset_path, code_dirs?}. SVE ostalo (adapter/task/mode/metrika/enhaneri/
tapovi/split) je AUTO iz `pipeline.prepare` — nema per-model konfiguracije.

(6.4) morphology se VISE NE UVOZI: jezgra je `slinn/` (introspect/morph/kdterms/settings), a detekcijski
decode + mAP su izolirani u `slinn/plugins/detection/`. Ovaj modul je JEDINO mjesto gdje jezgra dodiruje
plug — kroz `build_metric_fn`. Makni taj folder i sve i dalje radi (gate padne na teacher-agreement).
"""
import copy
import os
import sys

_SLINN = "/home/tomi/code/dipl/slinn"
for _p in (_SLINN,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                 # noqa: E402
import introspect as A                                       # noqa: E402  (6.4: genericka jezgra)
from plugins import detection as DET                         # noqa: E402  (6.4: izolirani decode plug;
#                                          obrisi taj folder i jezgra radi dalje -> teacher-agreement gate)
import pipeline as PP                                        # noqa: E402
import metric as M                                           # noqa: E402
import engine as E                                           # noqa: E402
from classify import probe_adapter                           # noqa: E402

JOB = os.path.join(_SLINN, "tmp", "gui_job")                 # job-dir (config.json / *_status.json / trajectory.jsonl)


class _Tee:
    """Pise u terminal (kao i inace) I u log datoteku, da je GUI moze tail-ati.
    Bez GUI-a se log samo dodatno pise — pokretanje iz terminala ostaje nepromijenjeno."""

    def __init__(self, stream, logf):
        self.stream, self.logf = stream, logf

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        self.logf.write(s)
        self.logf.flush()

    def flush(self):
        self.stream.flush()
        self.logf.flush()


def new_run_dir(model_path):
    """Napravi `runs/<ime_modela>_<timestamp>/` za JEDNU kompresiju. Sve sto run proizvede (log,
    trajektorija, checkpointi, manifest) ide ovdje i nista se ne dijeli s drugim runovima —
    dva runa nad razlicitim modelima vise ne mogu pregaziti jedan drugome izlaze.
    Originalni model se NE kopira; `run_meta.json` cuva putanju do njega."""
    import datetime
    import settings as _S
    # Zoo konvencija je `<ime_modela>/model.pt`, pa je ime DATOTEKE cesto beskorisno ("model").
    # U tom slucaju uzmi ime MAPE — inace bi svi runovi razlicitih modela dijelili prefiks `model_`.
    GENERIC = {"model", "best", "last", "final", "checkpoint", "ckpt", "weights", "pytorch_model"}
    pth = os.path.abspath(str(model_path))
    stem = os.path.splitext(os.path.basename(pth))[0]
    if stem.lower() in GENERIC:
        stem = os.path.basename(os.path.dirname(pth)) or stem
    d = os.path.join(_S.RUNS_DIR, "{}_{}".format(stem, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(d, exist_ok=True)
    return d


def tee_log(name, out_dir=None):
    """Preusmjeri stdout+stderr u terminal I u <out_dir|JOB>/<name>. Vrati putanju loga."""
    d = out_dir or JOB
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    logf = open(path, "w")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    return path


def load_ctx(cfg, dev):
    """Eager model + probe-adapter + AUTO ctx (task/mode/metrike/enhaneri/tapovi/split) iz putanje modela+dataseta."""
    model = A.load_any(cfg["model_path"], dev, code_dirs=cfg.get("code_dirs"))
    adapter = probe_adapter(model, dev, verbose=False)
    ctx = PP.prepare(model, adapter, dev, cfg["dataset_path"])
    return model, adapter, ctx


def frozen_teacher(model, dev):
    """Smrznuta kopija ORIGINALA = KD učitelj (nikad se ne trenira)."""
    t = copy.deepcopy(model).to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _fixed_subset(pr, frac, seed=0):
    """FIKSAN podskup liste parova za monitor metriku (PLAN_KOMPRESIJA 1.7). Seedani NASUMICNI izbor,
    ne `pr[::k]`: `pairs_classification` gradi listu round-robin po razredima, pa bi korak od 2 kod
    parnog broja razreda pokupio SAMO parne razrede. Racuna se jednom -> isti uzorci svaku epohu."""
    if not pr or not frac or not (0.0 < frac < 1.0) or len(pr) < 2:
        return None
    import random
    k = max(1, int(round(frac * len(pr))))
    return [pr[i] for i in sorted(random.Random(seed).sample(range(len(pr)), k))]


def build_metric_fn(ctx, model, adapter, dataset_path, dev, n_gate=None, monitor_frac=None,
                    teacher=None):
    # n_gate=None -> mjeri na CIJELOM val splitu (kao morphology). Broj postoji samo za smoke-testove.
    """PER-TASK prava metrika za quality-gate. Vrati (metric_fn, metric_name, monitor_fn).

    `monitor_frac` in (0,1) -> uz punu metriku vrati i BRZU na FIKSNOM podskupu iste velicine udjela
    (PLAN_KOMPRESIJA 1.7): Faza 1 njome odlucuje prune/recovery i patience, a punu vrti samo kad model
    kandidira za LAST_GOOD. `monitor_fn` je None kad se ne trazi ili kad podskup nema smisla.
    VAZNO: monitor i puna metrika NISU ista brojka -> prag se mora racunati iz ODGOVARAJUCE baseline
    vrijednosti (floor_mon iz monitor-baselinea, floor_full iz punog), inace se usporedjuju jabuke i kruske.

    Bez oznaka (`mode != full`) ILI task bez generičkog evaluatora -> SLAGANJE S UCITELJEM, i to pod
    ISTIM PRAVILIMA kao svaka druga metrika: mjeri se na VAL splitu, puna + fiksna polovica za monitor
    ([[kd-only-no-gt]]). Nema zasebnog gatea ni zasebne tolerancije — samo druga metrika u istom okviru.
    Za to treba `teacher`; bez njega se vraca (None, 'teacher_agreement', None) pa engine gradi sam.
    Detekcija = morphology mAP plug."""
    task, mode = ctx["task"], ctx.get("mode")

    def _agree():
        """Label-free metrika, ISTI okvir kao ostale: VAL split, puna + fiksna polovica za monitor."""
        if teacher is None:
            return None, "teacher_agreement", None
        import engine as _E
        f, m = _E.agreement_metrics(teacher, adapter, dev, ctx, dataset_path, frac=monitor_frac)
        return f, "teacher_agreement", m

    if mode != "full":                                       # nema čitljivih oznaka
        return _agree()
    if task == "detection":                                  # decode plug = jedini per-family dio
        sample = adapter.forward_example(dev) if hasattr(adapter, "forward_example") else None
        ad_m = DET.pick_adapter(model, sample_input=sample)   # obitelj -> format izlaza -> None
        if ad_m is None:                                      # nepoznat decode: NE padaj, degradiraj
            return _agree()                                   # (pick_adapter je vec glasno objasnio zasto)
        vloader = DET.make_gt_loader("val", bs=4)
        mon = None
        if monitor_frac and 0.0 < monitor_frac < 1.0:        # podskup se fiksira U LOADERU (seedano)
            mloader = DET.make_gt_loader("val", bs=4, frac=monitor_frac)
            mon = lambda mdl: float(DET.eval_map(mdl, ad_m, mloader, dev, max_images=None)[0]["map"])
        return (lambda mdl: float(DET.eval_map(mdl, ad_m, vloader, dev, max_images=n_gate)[0]["map"]),
                "mAP", mon)
    if task == "segmentation":
        pr = M.pairs_segmentation(dataset_path, split="val", n=n_gate or 10 ** 6, size=256)
        sub = _fixed_subset(pr, monitor_frac)
        return (lambda mdl: M.evaluate(mdl, adapter, "segmentation", pr, dev)["mIoU"], "mIoU",
                None if sub is None else (lambda mdl: M.evaluate(mdl, adapter, "segmentation", sub, dev)["mIoU"]))
    if task == "regression":
        pr = M.pairs_regression(dataset_path, adapter, split="val", n=None)
        sub = _fixed_subset(pr, monitor_frac)
        return (lambda mdl: M.evaluate(mdl, adapter, "regression", pr, dev)["r2"], "r2",
                None if sub is None else (lambda mdl: M.evaluate(mdl, adapter, "regression", sub, dev)["r2"]))
    if task in ("classification", "multilabel"):
        nc = None                                            # sirina izlaza = broj razreda (za provjeru)
        try:
            import loss as _L
            with torch.no_grad():
                nc = int(_L._main_out(adapter.forward(model, adapter.forward_example(dev))).shape[-1])
        except BaseException:
            pass
        pr = M.pairs_classification(dataset_path, adapter, split="val", n=n_gate or 10 ** 6,
                                    n_classes=nc, model=model)   # model nosi HF tokenizer-ime
        if pr:                                               # citljiv par (folder_per_class) -> prava metrika
            sub = _fixed_subset(pr, monitor_frac)
            return (lambda mdl: M.evaluate(mdl, adapter, "classification", pr, dev)["f1_macro"], "f1_macro",
                    None if sub is None else
                    (lambda mdl: M.evaluate(mdl, adapter, "classification", sub, dev)["f1_macro"]))
        print("[metrika] task={} ali nema citljivog (ulaz, razred) para za format ovog dataseta "
              "(npr. hf_datasets treba tokenizer) -> mjeri se SLAGANJE S UCITELJEM.".format(task))
    return _agree()


def baseline_report(ctx, model, adapter, dev, metric_fn, metric_name):
    """Baseline za semafor + gate: GFLOPs/params + prava metrika originala (ili None ako label-free gate)."""
    return {"gflops": float(E.gflops(model, adapter, dev)), "params": int(A.count_params(model)),
            "metric_name": metric_name, "metric": (float(metric_fn(model)) if metric_fn is not None else None),
            "task": ctx["task"], "mode": ctx.get("mode"), "enhancers": bool(ctx.get("enhancers"))}
