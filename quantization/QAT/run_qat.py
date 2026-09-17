
import copy
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_QROOT = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_QROOT)
_SLINN = os.path.join(_ROOT, "slinn")
for _p in (_HERE, _QROOT, _SLINN,
           os.path.join(_QROOT, "PTQ", "schoolcnn"),
           os.path.join(_QROOT, "PTQ", "deeplabv3"),
           os.path.join(_QROOT, "PTQ", "distilbert")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qat_common as QC
import quant_common as Q

MODELS = os.environ.get("QAT_MODELS", "schoolcnn,yolo26n,yolo26l,deeplabv3,distilbert").split(",")
CPU_THREADS = 8
LAT_WARMUP, LAT_ITERS = 10, 50


class HalfWrap(nn.Module):

    def __init__(self, m):
        super().__init__()
        self.m = m
        if hasattr(m, "model"):
            self.model = m.model

    def forward(self, *a, **kw):
        a = tuple(t.half() if torch.is_tensor(t) and t.is_floating_point() else t for t in a)
        kw = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in kw.items()}
        return self.m(*a, **kw)


class Shim(nn.Module):

    def __init__(self, runner, like=None, nc=None, names=None, logits=False):
        super().__init__()
        self._r = runner
        self._names = names
        self._logits = logits
        if like is not None:
            head = nn.Module(); head.nc = nc; head.end2end = False
            self.model = nn.ModuleList([head])

    def forward(self, *a, **kw):
        if kw:
            feed = {k: v for k, v in kw.items()}
            ref = next(iter(feed.values()))
        elif len(a) == 1 and torch.is_tensor(a[0]):
            feed, ref = a[0], a[0]
        elif len(a) == 1:
            feed = a[0]; ref = next(iter(feed.values()))
        else:
            feed = {n: t for n, t in zip(self._names, a)} if self._names else a
            ref = a[0]
        out = self._r.infer(feed)
        out = out.to(ref.device)
        if self._logits:
            from types import SimpleNamespace
            return SimpleNamespace(logits=out)
        return out


_SHARED_NAMES = ("data", "eval_baseline", "common", "model_cnn",
                 "ptq_schoolcnn", "ptq_deeplabv3", "ptq_distilbert")


def _use(subdir):
    for n in _SHARED_NAMES:
        sys.modules.pop(n, None)
    d = os.path.join(_QROOT, "PTQ", subdir)
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)


def _p_schoolcnn():
    _use("schoolcnn")
    import ptq_schoolcnn as PT
    C2 = PT.C2
    nc = PT.NUM_CLASSES
    val = C2.make_loader("val", 32, shuffle=False, num_workers=2)
    tr = C2.make_loader("train", 32, shuffle=True, num_workers=2)
    cal = C2.make_loader("train", 32, shuffle=False, num_workers=2, max_images=512)
    ex = next(iter(val))[0][:1].clone()
    teacher = {}

    def loss(m, b, t):
        x = b[0] if isinstance(b, (list, tuple)) else b
        with torch.no_grad():
            tt = torch.sigmoid(t(x))
        return F.binary_cross_entropy_with_logits(m(x), tt)

    def ev(mdl, dev):
        return {k: v for k, v in PT.eval_panel(PT.make_infer(mdl, dev), val, nc).items()
                if k != "per_class_ap"}

    return dict(name="schoolcnn", load=PT.load_fp32, example=ex, skip=(), fp32_layers=(),
                calib=cal, train=tr, calib_fwd=lambda m, b: m(b[0]), loss=loss, evaluate=ev,
                steps=800, lr=1e-4, keys=("map_macro", "f1_macro", "acc_macro"),
                ptq_json=PT.OUT_JSON, ptq_rows=("FP32", "INT8-PT-static"), ptq_key="map_macro",
                teacher_store=teacher)


def _p_yolo(tag):
    from plugins.detection import adapters as A

    def load():
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER
        import logging
        LOGGER.setLevel(logging.ERROR)
        m = YOLO(os.path.join(_ROOT, "baseline_models", tag, f"{tag}.pt")).model.float().eval()
        list(m.model)[-1].end2end = False
        return m

    probe = load()
    last = len(probe.model) - 1
    nc = int(list(probe.model)[-1].nc)
    del probe
    val = A.make_gt_loader("val", bs=4, workers=0)
    tr = A.make_gt_loader("train", bs=4, workers=0)
    ex = torch.randn(1, 3, 640, 640)

    def _lb(imgs, dev):
        return torch.stack([A.YoloAdapter._letterbox(im)[0] for im in imgs]).to(dev)

    def _dense(o):
        if isinstance(o, dict):
            o = o.get("one2many", next(iter(o.values())))
        return o[0] if isinstance(o, (tuple, list)) else o

    def loss(m, b, t):
        imgs = b[0] if isinstance(b, (list, tuple)) else b
        dev = next(m.parameters()).device
        lb = _lb(imgs, dev) if isinstance(imgs, list) else imgs.to(dev)
        m.eval()
        with torch.no_grad():
            tt = _dense(t(lb)).detach()
        return F.mse_loss(_dense(m(lb)), tt)

    def ev(mdl, dev):
        mm, _ = A.eval_map(mdl, A.YoloAdapter, val, dev, max_images=400)
        f = lambda k: float(mm.get(k, float("nan")))
        return {"map": f("map"), "map50": f("map_50"), "map75": f("map_75"), "mar100": f("mar_100")}

    class _Cal:
        def __iter__(self):
            for b in tr:
                yield _lb(b[0], "cpu")

    return dict(name=tag, load=load, example=ex, skip=(f"model.{last}.",),
                fp32_layers=(f"model.{last}",),
                calib=_Cal(), train=tr, calib_fwd=None,
                calib_max=16, loss=loss, evaluate=ev, steps=300, lr=1e-5,
                keys=("map", "map50", "map75", "mar100"),
                ptq_json=os.path.join(_QROOT, "PTQ", tag, f"{tag}_ptq_report.json"),
                ptq_rows=("FP32", "INT8-TRT"), ptq_key="map",
                shim_kw=dict(like=True, nc=nc))


def _p_deeplabv3():
    _use("deeplabv3")
    import ptq_deeplabv3 as PT
    PT.EVAL_LIMIT = 300
    names = PT.D.classes()
    ds_tr = PT.D.voc("train")
    tf = PT.D.transform()
    ex = torch.rand(1, 3, PT.LAT_SIZE, PT.LAT_SIZE)

    from torchvision.transforms import functional as TF
    _mean = list(getattr(tf, "mean", [0.485, 0.456, 0.406]))
    _std = list(getattr(tf, "std", [0.229, 0.224, 0.225]))

    class _DS(torch.utils.data.Dataset):

        def __len__(self):
            return min(400, len(ds_tr))

        def __getitem__(self, i):
            t = TF.to_tensor(TF.resize(ds_tr[i][0], [PT.LAT_SIZE, PT.LAT_SIZE]))
            return TF.normalize(t, _mean, _std)

    ld = torch.utils.data.DataLoader(_DS(), batch_size=2, shuffle=True, num_workers=0,
                                     collate_fn=lambda b: torch.stack([x for x in b]))

    def loss(m, b, t):
        with torch.no_grad():
            tt = PT.make_infer(t, b.device)(b).detach()
        return F.mse_loss(PT.make_infer(m, b.device)(b), tt)

    def ev(mdl, dev):
        p = PT.panel_from(PT.make_infer(mdl, dev), names)
        return {"miou": p["miou"], "pixel_acc": p["pixel_acc"]}

    def ev_sq(mdl, dev):
        import numpy as np
        ds = PT.D.voc(PT.EVAL_SPLIT)
        n = min(PT.EVAL_LIMIT or len(ds), len(ds))
        C = PT.D.NUM_CLASSES
        conf = torch.zeros(C, C, dtype=torch.long)
        with torch.no_grad():
            for i in range(n):
                img, mask = ds[i]
                t = TF.to_tensor(TF.resize(img, [PT.LAT_SIZE, PT.LAT_SIZE]))
                x = TF.normalize(t, _mean, _std).unsqueeze(0)
                o = PT.make_infer(mdl, dev)(x).float().cpu()
                m_ = torch.as_tensor(np.array(mask), dtype=torch.long)
                o = F.interpolate(o, size=m_.shape, mode="bilinear", align_corners=False)
                pr = o.argmax(1)[0]
                v = (m_ != PT.D.IGNORE) & (m_ < C)
                conf += torch.bincount(C * m_[v] + pr[v], minlength=C * C).reshape(C, C)
        miou, pacc, _ = PT.EB.metrics(conf)
        return {"miou": float(miou), "pixel_acc": float(pacc)}

    return dict(name="deeplabv3", load=PT.load_fp32, example=ex, skip=(), fp32_layers=(),
                calib=ld, train=ld, calib_fwd=None, calib_max=16, loss=loss, evaluate=ev,
                evaluate_backend=ev_sq,
                steps=200, lr=1e-5, keys=("miou", "pixel_acc"),
                ptq_json=PT.OUT_JSON, ptq_rows=("FP32", "INT8-PT-static"), ptq_key="miou",
                pick=lambda o: o["out"] if isinstance(o, dict) else o)


def _p_distilbert():
    _use("distilbert")
    import ptq_distilbert as PT
    names = PT.D.classes()
    enc = PT.fixed_encoding()
    ex = (enc["input_ids"], enc["attention_mask"])

    def pad(loader):
        for e, y in loader:
            i, a = e["input_ids"], e["attention_mask"]
            k = PT.D.MAX_LEN - i.shape[1]
            if k > 0:
                i = F.pad(i, (0, k)); a = F.pad(a, (0, k))
            yield {"input_ids": i[:, :PT.D.MAX_LEN], "attention_mask": a[:, :PT.D.MAX_LEN]}, y

    class _It:
        def __init__(self, split, bs, lim=None):
            self.a = (split, bs, lim)

        def __iter__(self):
            return pad(PT.D.loader(self.a[0], self.a[1], limit=self.a[2]))

    tr, cal = _It("train", 16, 4000), _It("train", 16, 512)

    def loss(m, b, t):
        e = b[0] if isinstance(b, (list, tuple)) else b
        with torch.no_grad():
            tt = t(**e).logits.detach()
        return F.mse_loss(m(**e).logits, tt)

    def ev(mdl, dev):
        p = PT.eval_panel(PT.make_infer(mdl, dev), _It("validation", 64), names)
        return {k: p[k] for k in ("acc", "f1_macro", "auroc")}

    return dict(name="distilbert", load=PT.load_fp32, example=ex, skip=(), fp32_layers=(),
                calib=cal, train=tr, calib_fwd=lambda m, b: m(**b[0]), calib_max=32,
                loss=loss, evaluate=ev, steps=300, lr=1e-5,
                keys=("acc", "f1_macro", "auroc"),
                ptq_json=PT.OUT_JSON, ptq_rows=("FP32", "INT8-PT-dynamic"), ptq_key="acc",
                input_names=("input_ids", "attention_mask"), kwargs_forward=True,
                shim_kw=dict(names=("input_ids", "attention_mask"), logits=True))


PROFILES = {"schoolcnn": _p_schoolcnn, "yolo26n": lambda: _p_yolo("yolo26n"),
            "yolo26l": lambda: _p_yolo("yolo26l"), "deeplabv3": _p_deeplabv3,
            "distilbert": _p_distilbert}


def run_model(key):
    t0 = time.time()
    pf = PROFILES[key]()
    name = pf["name"]
    out_dir = os.path.join(_HERE, name)
    onnx_dir = os.path.join(out_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    has_cuda = torch.cuda.is_available()
    cpu, gpu = torch.device("cpu"), torch.device("cuda" if has_cuda else "cpu")
    keys = pf["keys"]
    cols = ["format", "backend", "cpu_ms", "gpu_ms", "size_mb"] + list(keys)
    rows, lat_full = [], {}

    print(f"\n{'=' * 96}\nQAT — {name}\n{'=' * 96}", flush=True)

    def add(fmt, backend, panel, size, c, g, extra=None):
        r = {"format": fmt, "backend": backend,
             "size_mb": round(size, 4) if isinstance(size, (int, float)) else size,
             "cpu_ms": round(c["median_ms"], 4) if isinstance(c, dict) else c,
             "gpu_ms": round(g["median_ms"], 4) if isinstance(g, dict) else g}
        for k in keys:
            v = (panel or {}).get(k)
            r[k] = round(v, 5) if isinstance(v, float) and v == v else Q.na("—")
        if extra:
            r.update(extra)
        rows.append(r); lat_full[fmt] = {"cpu": c, "gpu": g}
        print(f"  [{fmt:16}] " + "  ".join(f"{k}={r[k]}" for k in keys) +
              f" | {r['size_mb']} MB | CPU {r['cpu_ms']} | GPU {r['gpu_ms']}", flush=True)

    def bench(fn, dev, slow=False):
        try:
            return Q.benchmark(fn, dev, 2 if slow else LAT_WARMUP, 5 if slow else LAT_ITERS)
        except Exception as e:
            return Q.na(f"{type(e).__name__}")

    ex_dev = lambda d: (tuple(t.to(d) for t in pf["example"]) if isinstance(pf["example"], tuple)
                        else pf["example"].to(d))
    kw_fwd = pf.get("kwargs_forward", False)

    def call(m, x):
        if kw_fwd:
            return m(input_ids=x[0], attention_mask=x[1])
        return m(*x) if isinstance(x, tuple) else m(x)

    orig = pf["load"]().to(gpu).eval()
    p_o = pf["evaluate"](orig, gpu)
    g_o = bench(lambda: call(orig, ex_dev(gpu)), gpu) if has_cuda else Q.na("nema CUDA")
    oc = pf["load"]().to(cpu).eval()
    c_o = bench(lambda: call(oc, ex_dev(cpu)), cpu, slow=True)
    add("[ORIG] FP32", "PyTorch", p_o, Q.model_size_mb(oc), c_o, g_o)
    del oc
    torch.cuda.empty_cache()

    student = pf["load"]()
    n_w = QC.wrap_model(student, skip=pf["skip"])
    print(f"  [wrap] {n_w} modula | fake-quant tocaka: {QC.n_fakequant(student)}", flush=True)
    student = student.to(gpu)

    cal_it, cmax = pf["calib"], pf.get("calib_max", 16)
    cal_list = []
    for i, b in enumerate(cal_it):
        if i >= cmax:
            break
        cal_list.append(b)
    QC.calibrate(student, cal_list, gpu, forward_fn=pf.get("calib_fwd"))
    print(f"  [calib] {len(cal_list)} batcheva", flush=True)
    del cal_list

    loss, steps = QC.qat_finetune(student, pf["train"], lambda m, b: pf["loss"](m, b, orig),
                                  steps=pf["steps"], lr=pf["lr"], device=gpu,
                                  on_step=lambda k, n, l: print(f"      {k}/{n} kd={l:.5f}", flush=True))
    print(f"  [qat] {steps} koraka, KD {loss:.5f}", flush=True)
    del orig
    torch.cuda.empty_cache()

    student = student.cpu().eval()
    paths = QC.export_all(student, pf["example"], onnx_dir, name, pick=pf.get("pick"),
                          input_names=pf.get("input_names", ("input",)))
    ver = {m_: QC.verify(p) for m_, p in paths.items() if p}
    for m_, v in ver.items():
        print(f"  [onnx {m_:4}] {v['mb']:.2f} MB | QDQ {v['qdq']}/{v['nodes']} | "
              f"domene {v['domains'] or 'std'}", flush=True)

    plain = QC.unwrap_model(student).eval()
    p32 = pf["evaluate"](plain.to(gpu), gpu)
    g32 = bench(lambda: call(plain, ex_dev(gpu)), gpu) if has_cuda else Q.na("nema CUDA")
    pc = QC.unwrap_model(student).to(cpu).eval()
    c32 = bench(lambda: call(pc, ex_dev(cpu)), cpu, slow=True)
    add("QAT-FP32", "PyTorch", p32, Q.model_size_mb(pc), c32, g32)
    del plain, pc
    torch.cuda.empty_cache()

    if has_cuda:
        try:
            h = HalfWrap(QC.unwrap_model(student).to(gpu).half().eval())
            p16 = pf["evaluate"](h, gpu)
            xh = ex_dev(gpu)
            xh = (tuple(t.half() if t.is_floating_point() else t for t in xh)
                  if isinstance(xh, tuple) else xh.half())
            g16 = bench(lambda: call(h, xh), gpu)
            add("QAT-FP16", "PyTorch", p16, Q.model_size_mb(QC.unwrap_model(student).half()),
                Q.na("CPU fp16 bez HW puta"), g16)
            del h
            torch.cuda.empty_cache()
        except Exception as e:
            add("QAT-FP16", "PyTorch", None, Q.na("—"), Q.na("—"), Q.na(f"{type(e).__name__}"))

    ev_be = pf.get("evaluate_backend") or pf["evaluate"]
    if pf.get("evaluate_backend"):
        pl = QC.unwrap_model(student).to(gpu).eval()
        add("QAT-FP32-BE", "PyTorch (backend put)", ev_be(pl, gpu), Q.na("kontrola"),
            Q.na("—"), Q.na("—"))
        del pl
        torch.cuda.empty_cache()

    for tag, src in (("QAT-FP32-OV", "fp32"), ("QAT-INT8-OV", "qdq")):
        if not paths.get(src):
            continue
        try:
            ov = QC.OVRunner(paths[src], threads=CPU_THREADS)
            sh = Shim(ov, **pf.get("shim_kw", {}))
            add(tag, "OpenVINO CPU", ev_be(sh, cpu), Q.na("ONNX = opis grafa"),
                bench(ov.bench_call(pf["example"] if not isinstance(pf["example"], tuple)
                                    else {"input_ids": pf["example"][0],
                                          "attention_mask": pf["example"][1]}), "cpu"),
                Q.na("OpenVINO CPU"))
        except Exception as e:
            add(tag, "OpenVINO CPU", None, Q.na("—"), Q.na(f"{type(e).__name__}: {str(e)[:44]}"), Q.na("—"))

    if has_cuda:
        for tag, src, kw in (("QAT-FP16-TRT", "fp32", dict(fp16=True)),
                             ("QAT-INT8-TRT", "qdq", dict(int8=True))):
            if not paths.get(src):
                continue
            try:
                eng = os.path.join(onnx_dir, f"{name}_{tag.lower()}.engine")
                QC.trt_build(paths[src], eng, fp32_layers=pf.get("fp32_layers", ()), **kw)
                r = QC.TRTRunner(eng)
                hist = r.precision_histogram()
                sh = Shim(r, **pf.get("shim_kw", {}))
                pt_ = ev_be(sh, gpu)
                r.preload(pf["example"] if not isinstance(pf["example"], tuple)
                          else {"input_ids": pf["example"][0], "attention_mask": pf["example"][1]})
                add(tag, "TensorRT", pt_, Q.file_size_mb(eng), Q.na("TRT je GPU-only"),
                    bench(r.enqueue, "cuda"), extra={"precision_layers": hist})
                print(f"        preciznost: {hist}", flush=True)
                del r
                torch.cuda.empty_cache()
            except Exception as e:
                add(tag, "TensorRT", None, Q.na("—"), Q.na("TRT GPU-only"),
                    Q.na(f"{type(e).__name__}: {str(e)[:50]}"))

    ptq = {}
    if pf.get("ptq_json") and os.path.exists(pf["ptq_json"]):
        for r in json.load(open(pf["ptq_json"]))["rows"]:
            if r.get("format") in pf["ptq_rows"]:
                ptq[r["format"]] = r.get(pf["ptq_key"])

    meta = {"model": name, "method": "QAT (rucni fake-quant -> QDQ ONNX)",
            "qat": {"steps": steps, "lr": pf["lr"], "wrapped": n_w, "skip": list(pf["skip"]),
                    "fp32_layers": list(pf.get("fp32_layers", ()))},
            "onnx_verify": ver, "ptq_reference": ptq, "lat_full": lat_full,
            "minutes": round((time.time() - t0) / 60.0, 2)}
    Q.write_report(rows, cols, os.path.join(out_dir, f"{name}_qat_report.csv"),
                   os.path.join(out_dir, f"{name}_qat_report.json"), meta)

    k0 = keys[0]
    gv = lambda f: next((r[k0] for r in rows if r["format"] == f and isinstance(r.get(k0), (int, float))), None)
    o, q32v = gv("[ORIG] FP32"), gv("QAT-FP32")
    qi = gv("QAT-INT8-TRT") or gv("QAT-INT8-OV")
    print(f"\n  --- {name}: razmak protiv razmaka ({k0}) ---")
    if o and q32v:
        print(f"    fine-tune sam:  {o:.5f} -> {q32v:.5f} ({q32v - o:+.5f})")
    if q32v and qi:
        print(f"    QAT kvantizac.: {q32v:.5f} -> {qi:.5f} ({qi - q32v:+.5f})")
    if ptq:
        vals = list(ptq.values())
        if len(vals) == 2 and all(isinstance(v, (int, float)) for v in vals):
            print(f"    PTQ (referenca): {vals[0]:.5f} -> {vals[1]:.5f} ({vals[1] - vals[0]:+.5f})")
    print(f"  --- {meta['minutes']} min ---", flush=True)
    return {"name": name, "rows": rows, "meta": meta}


def main():
    res = []
    for k in MODELS:
        k = k.strip()
        if k not in PROFILES:
            print(f"[preskacem] nepoznat profil: {k}")
            continue
        try:
            res.append(run_model(k))
        except BaseException:
            print(f"\n!!! {k} PAO:\n{traceback.format_exc()[-1200:]}", flush=True)
        torch.cuda.empty_cache()

    print("\n" + "=" * 96)
    print("ZBIRNO")
    print("=" * 96)
    for r in res:
        k0 = list(r["meta"].get("qat", {}) and r["rows"][0].keys())[5]
        print(f"\n{r['name']}  ({r['meta']['minutes']} min)")
        for row in r["rows"]:
            print(f"   {row['format']:16} {k0}={row.get(k0)}  GPU {row.get('gpu_ms')}  CPU {row.get('cpu_ms')}")
    json.dump({r["name"]: r["meta"] for r in res},
              open(os.path.join(_HERE, "qat_summary.json"), "w"), indent=2, default=str)
    print(f"\n[save] {os.path.join(_HERE, 'qat_summary.json')}")


if __name__ == "__main__":
    main()
