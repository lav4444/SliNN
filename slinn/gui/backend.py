import copy
import os
import sys

_SLINN = __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)))
for _p in (_SLINN,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                 # noqa: E402
import introspect as A                                       # noqa: E402  (6.4: genericka jezgra)
from plugins import detection as DET                         # noqa: E402  (6.4: izolirani decode plug;
import pipeline as PP                                        # noqa: E402
import metric as M                                           # noqa: E402
import engine as E                                           # noqa: E402
from classify import probe_adapter                           # noqa: E402
import settings as _S_MON                                    # noqa: E402  (pod monitora)

JOB = os.path.join(_SLINN, "tmp", "gui_job")


class _Tee:

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
    import datetime
    import settings as _S
    GENERIC = {"model", "best", "last", "final", "checkpoint", "ckpt", "weights", "pytorch_model"}
    pth = os.path.abspath(str(model_path))
    stem = os.path.splitext(os.path.basename(pth))[0]
    if stem.lower() in GENERIC:
        stem = os.path.basename(os.path.dirname(pth)) or stem
    d = os.path.join(_S.RUNS_DIR, "{}_{}".format(stem, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(d, exist_ok=True)
    return d


def tee_log(name, out_dir=None):
    d = out_dir or JOB
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    logf = open(path, "w")
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    return path


def load_ctx(cfg, dev):
    model = A.load_any(cfg["model_path"], dev, code_dirs=cfg.get("code_dirs"))
    adapter = probe_adapter(model, dev, verbose=False)
    try:
        from plugins.detection import dsconfig as _DSC
        _DSC.configure(cfg["dataset_path"], model=model)
    except RuntimeError as e:
        print(str(e))
    ctx = PP.prepare(model, adapter, dev, cfg["dataset_path"])
    return model, adapter, ctx


def probe_train_batch(model, adapter, dev, ctx, dataset_path):
    import settings as _S
    if dev.type == "cuda":
        return int(E.autobatch(model, adapter, dev, ctx, dataset_path)), "proba VRAM-a"
    return int(_S.TRAIN_BATCH), "CPU, bez probe"


def resolve_train_batch(cfg, model, adapter, dev, ctx):
    given = (cfg.get("batch") or {}).get("train")
    if given:
        return int(given), "iz configa"
    return probe_train_batch(model, adapter, dev, ctx, cfg["dataset_path"])


def frozen_teacher(model, dev):
    t = copy.deepcopy(model).to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _fixed_subset(pr, frac, seed=0):
    if not pr or not frac or not (0.0 < frac < 1.0) or len(pr) < 2:
        return None
    import random
    k = min(len(pr), max(int(getattr(_S_MON, 'METRIC_MONITOR_MIN', 1)),
                         int(round(frac * len(pr)))))
    return [pr[i] for i in sorted(random.Random(seed).sample(range(len(pr)), k))]


def build_metric_fn(ctx, model, adapter, dataset_path, dev, n_gate=None, monitor_frac=None,
                    teacher=None):
    task, mode = ctx["task"], ctx.get("mode")

    def _agree():
        if teacher is None:
            return None, "teacher_agreement", None
        import engine as _E
        f, m = _E.agreement_metrics(teacher, adapter, dev, ctx, dataset_path, frac=monitor_frac)
        return f, "teacher_agreement", m

    if mode != "full":
        return _agree()
    if task == "detection":
        sample = adapter.forward_example(dev) if hasattr(adapter, "forward_example") else None
        ad_m = DET.pick_adapter(model, sample_input=sample)
        if ad_m is None:
            return _agree()
        vloader = DET.make_gt_loader("val", bs=4)
        mon = None
        if monitor_frac and 0.0 < monitor_frac < 1.0:
            mloader = DET.make_gt_loader("val", bs=4, frac=monitor_frac)
            mon = lambda mdl: float(DET.eval_map(mdl, ad_m, mloader, dev, max_images=None)[0]["map"])
        return (lambda mdl: float(DET.eval_map(mdl, ad_m, vloader, dev, max_images=n_gate)[0]["map"]),
                "mAP", mon)
    def _need(pr, why):
        if pr:
            return False
        print("[metrika] task={} ali {} -> mjeri se SLAGANJE S UCITELJEM.".format(task, why))
        return True

    if task == "segmentation":
        pr = M.pairs_segmentation(dataset_path, split="val", n=n_gate or 10 ** 6, size=256)
        if _need(pr, "nema citljivog (slika, maska) para za format ovog dataseta"):
            return _agree()
        sub = _fixed_subset(pr, monitor_frac)
        return (lambda mdl: M.evaluate(mdl, adapter, "segmentation", pr, dev)["mIoU"], "mIoU",
                None if sub is None else (lambda mdl: M.evaluate(mdl, adapter, "segmentation", sub, dev)["mIoU"]))
    if task == "regression":
        prd = M.pairs_depth(dataset_path, adapter, split="val", n=n_gate)
        if prd:
            subd = _fixed_subset(prd, monitor_frac)
            print("[metrika] guste oznake ({}) -> depth metrike; gate = delta1 (r2 na relativnoj "
                  "dubini nije upotrebljiv).".format(tuple(prd[0][1].shape)))
            return (lambda mdl: M.evaluate(mdl, adapter, "depth", prd, dev)["delta1"], "delta1",
                    None if subd is None else
                    (lambda mdl: M.evaluate(mdl, adapter, "depth", subd, dev)["delta1"]))
        pr = M.pairs_regression(dataset_path, adapter, split="val", n=None)
        if _need(pr, "nema citljivog (ulaz, cilj) para za format ovog dataseta"):
            return _agree()
        sub = _fixed_subset(pr, monitor_frac)
        return (lambda mdl: M.evaluate(mdl, adapter, "regression", pr, dev)["r2"], "r2",
                None if sub is None else (lambda mdl: M.evaluate(mdl, adapter, "regression", sub, dev)["r2"]))
    if task in ("classification", "multilabel"):
        nc = None
        try:
            import loss as _L
            with torch.no_grad():
                nc = int(_L._main_out(adapter.forward(model, adapter.forward_example(dev))).shape[-1])
        except BaseException:
            pass
        pr = M.pairs_classification(dataset_path, adapter, split="val", n=n_gate or 10 ** 6,
                                    n_classes=nc, model=model)
        if pr:
            sub = _fixed_subset(pr, monitor_frac)
            return (lambda mdl: M.evaluate(mdl, adapter, "classification", pr, dev)["f1_macro"], "f1_macro",
                    None if sub is None else
                    (lambda mdl: M.evaluate(mdl, adapter, "classification", sub, dev)["f1_macro"]))
        print("[metrika] task={} ali nema citljivog (ulaz, razred) para za format ovog dataseta "
              "(npr. hf_datasets treba tokenizer) -> mjeri se SLAGANJE S UCITELJEM.".format(task))
    return _agree()


def baseline_report(ctx, model, adapter, dev, metric_fn, metric_name):
    return {"gflops": float(E.gflops(model, adapter, dev)), "params": int(A.count_params(model)),
            "metric_name": metric_name, "metric": (float(metric_fn(model)) if metric_fn is not None else None),
            "task": ctx["task"], "mode": ctx.get("mode"), "enhancers": bool(ctx.get("enhancers"))}
