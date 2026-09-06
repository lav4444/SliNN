
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

QMIN, QMAX = -128, 127


def _wscale(w):
    return (w.detach().abs().flatten(1).amax(1).clamp(min=1e-8) / QMAX)


def _amax(x, pct):
    a = x.detach().abs().flatten()
    if pct is None:
        return a.amax()
    if a.numel() > 1_000_000:
        idx = torch.randint(0, a.numel(), (1_000_000,), device=a.device)
        a = a[idx]
    return torch.quantile(a.float(), pct / 100.0)


def _amax_ch(x, pct):
    if x.dim() < 2:
        return _amax(x, pct).reshape(1)
    ch = 1 if x.dim() != 3 else 2
    a = x.detach().abs().transpose(0, ch).reshape(x.shape[ch], -1)
    if pct is None:
        return a.amax(1)
    if a.shape[1] > 100_000:
        idx = torch.randint(0, a.shape[1], (100_000,), device=a.device)
        a = a[:, idx]
    return torch.quantile(a.float(), pct / 100.0, dim=1)


class _LSQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, s, qn, qp, g):
        ctx.save_for_backward(x, s)
        ctx.other = (qn, qp, g)
        return (x / s).clamp(qn, qp).round() * s

    @staticmethod
    def backward(ctx, go):
        x, s = ctx.saved_tensors
        qn, qp, g = ctx.other
        q = x / s
        lo, hi = (q < qn), (q > qp)
        mid = ~(lo | hi)
        gx = go * mid
        gs = ((lo.float() * qn + hi.float() * qp + mid.float() * (q.round() - q)) * go * g).sum()
        return gx, gs.reshape(s.shape), None, None, None


class ActFQ(nn.Module):

    def __init__(self, momentum=0.9, pct=99.9, per_channel=False, learnable=False):
        super().__init__()
        self.momentum, self.pct = momentum, pct
        self.per_channel, self.learnable = per_channel, learnable
        self.observing = False
        self.frozen = False
        self.register_buffer("amax", torch.zeros(1))
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zp", torch.zeros((), dtype=torch.int32))
        self._lsq = None

    def _observe(self, x):
        a = (_amax_ch(x, self.pct) if self.per_channel else _amax(x, self.pct).reshape(1))
        if self.amax.numel() != a.numel() or float(self.amax.abs().sum()) == 0.0:
            self.amax = a.detach().clone()
        else:
            self.amax.mul_(self.momentum).add_((1 - self.momentum) * a.detach())

    def start_lsq(self):
        if self.learnable and self._lsq is None:
            self._lsq = nn.Parameter((self.amax / QMAX).clamp(min=1e-8).clone())

    def _shape(self, s, x):
        if not self.per_channel or s.numel() == 1:
            return s.reshape(())
        v = [1] * x.dim()
        v[1 if x.dim() != 3 else 2] = -1
        return s.reshape(v)

    def forward(self, x):
        if self.observing:
            self._observe(x)
        if self._lsq is not None and not self.frozen:
            g = 1.0 / max((x.numel() * QMAX) ** 0.5, 1.0)
            sc = self._lsq.abs().clamp(min=1e-8)
            return _LSQ.apply(x, self._shape(sc, x), float(QMIN), float(QMAX), g)
        s = self.scale if self.frozen else (self.amax / QMAX).clamp(min=1e-8)
        if self.per_channel and s.numel() > 1:
            zp = torch.zeros_like(s, dtype=torch.int32)
            ax = 1 if x.dim() != 3 else 2
            return torch.fake_quantize_per_channel_affine(x, s.float(), zp, ax, QMIN, QMAX)
        return torch.fake_quantize_per_tensor_affine(x, s.reshape(()).float(), self.zp, QMIN, QMAX)

    def freeze(self):
        src = (self._lsq.detach().abs() if self._lsq is not None
               else (self.amax / QMAX)).clamp(min=1e-8)
        self.scale = src.clone()
        self.frozen = True


def _with_weight(mod, x, wq):
    par = mod._parameters.pop("weight")
    mod.weight = wq
    try:
        return mod(x)
    finally:
        del mod.weight
        mod._parameters["weight"] = par


class QConv(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fq_in = ActFQ()
        self.register_buffer("w_scale", torch.ones(c.weight.shape[0]))
        self.register_buffer("w_zp", torch.zeros(c.weight.shape[0], dtype=torch.int32))
        self.frozen = False

    def forward(self, x):
        w = self.c.weight
        s = self.w_scale if self.frozen else _wscale(w)
        wq = torch.fake_quantize_per_channel_affine(w, s, self.w_zp, 0, QMIN, QMAX)
        return _with_weight(self.c, self.fq_in(x), wq)

    def freeze(self):
        self.w_scale.copy_(_wscale(self.c.weight))
        self.frozen = True
        self.fq_in.freeze()


QConv2d = QConv


class QLinear(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l
        self.fq_in = ActFQ()
        self.register_buffer("w_scale", torch.ones(l.weight.shape[0]))
        self.register_buffer("w_zp", torch.zeros(l.weight.shape[0], dtype=torch.int32))
        self.frozen = False

    def forward(self, x):
        w = self.l.weight
        s = self.w_scale if self.frozen else _wscale(w)
        wq = torch.fake_quantize_per_channel_affine(w, s, self.w_zp, 0, QMIN, QMAX)
        return _with_weight(self.l, self.fq_in(x), wq)

    def freeze(self):
        self.w_scale.copy_(_wscale(self.l.weight))
        self.frozen = True
        self.fq_in.freeze()


def fold_bn(model):
    from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
    n = 0
    for parent in model.modules():
        ch = list(parent.named_children())
        for (n1, c1), (n2, c2) in zip(ch, ch[1:]):
            if not isinstance(c2, nn.modules.batchnorm._BatchNorm):
                continue
            if isinstance(c1, nn.modules.conv._ConvNd) and c2.num_features == c1.out_channels:
                fused = fuse_conv_bn_eval(c1.eval(), c2.eval())
            elif isinstance(c1, nn.Linear) and c2.num_features == c1.out_features:
                fused = fuse_linear_bn_eval(c1.eval(), c2.eval())
            else:
                continue
            setattr(parent, n1, fused)
            setattr(parent, n2, nn.Identity())
            n += 1
    return n


def wrap_model(model, skip=(), _prefix=""):
    n = 0
    for name, child in list(model.named_children()):
        full = f"{_prefix}.{name}" if _prefix else name
        if any(full.startswith(p) for p in skip):
            continue
        if isinstance(child, nn.modules.conv._ConvNd):
            setattr(model, name, QConv(child)); n += 1
        elif isinstance(child, nn.Linear):
            setattr(model, name, QLinear(child)); n += 1
        else:
            n += wrap_model(child, skip, full)
    if not _prefix:
        for p in model.parameters():
            p.requires_grad_(True)
    return n


def unwrap_model(model):
    m = copy.deepcopy(model)

    def rec(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, QConv):
                setattr(mod, name, child.c)
            elif isinstance(child, QLinear):
                setattr(mod, name, child.l)
            else:
                rec(child)
    rec(m)
    return m


def set_observing(model, flag):
    for m in model.modules():
        if isinstance(m, ActFQ):
            m.observing = bool(flag)


def start_lsq(model):
    k = 0
    for m in model.modules():
        if isinstance(m, ActFQ) and m.learnable:
            m.start_lsq(); k += 1
    return k


def sensitivity(load_fn, eval_fn, calib, device, skip=(), topk=None):
    base = eval_fn(load_fn().to(device))
    names = [n for n, m, _, _ in _leaf_names(load_fn())]
    out = []
    for nm in names:
        if any(nm.startswith(x) for x in skip):
            continue
        m = load_fn()
        fold_bn(m)
        wrap_model(m, skip=tuple(x for x in names if x != nm))
        m = m.to(device)
        calibrate(m, calib, device)
        out.append((nm, base - eval_fn(m)))
        del m
        torch.cuda.empty_cache()
    out.sort(key=lambda t: -t[1])
    return out[:topk] if topk else out


def _leaf_names(model, _p=""):
    for name, ch in model.named_children():
        full = f"{_p}.{name}" if _p else name
        if isinstance(ch, (nn.Conv2d, nn.Linear)):
            yield full, ch, None, None
        else:
            for t in _leaf_names(ch, full):
                yield t


def n_fakequant(model):
    return sum(1 for m in model.modules() if isinstance(m, ActFQ))


def freeze_scales(model):
    n = 0
    for m in model.modules():
        if isinstance(m, (QConv, QLinear)):
            m.freeze(); n += 1
        elif isinstance(m, ActFQ) and not m.frozen:
            m.freeze(); n += 1
    return n


def _to_dev(b, device):
    if torch.is_tensor(b):
        return b.to(device)
    if isinstance(b, dict):
        return {k: _to_dev(v, device) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(_to_dev(v, device) for v in b)
    return b


@torch.no_grad()
def calibrate(model, batches, device, forward_fn=None):
    model.eval()
    set_observing(model, True)
    n = 0
    for b in batches:
        b = _to_dev(b, device)
        (forward_fn(model, b) if forward_fn else model(b))
        n += 1
    set_observing(model, False)
    return n


def qat_finetune(model, batches, loss_fn, steps=300, lr=1e-5, device="cuda",
                 freeze_bn=True, on_step=None):
    model.to(device).train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nema parametara s gradijentom (requires_grad) — provjeri wrap_model")
    opt = torch.optim.Adam(params, lr=lr)
    tot, k, i = 0.0, 0, 0
    while k < steps:
        for b in batches:
            if k >= steps:
                break
            b = _to_dev(b, device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, b)
            loss.backward()
            opt.step()
            tot += float(loss); k += 1
            if on_step and k % max(steps // 10, 1) == 0:
                on_step(k, steps, tot / max(k, 1))
        i += 1
        if i > 10000:
            break
    model.eval()
    return tot / max(k, 1), k


class _MainOut(nn.Module):

    def __init__(self, m, pick=None):
        super().__init__()
        self.m = m
        self.pick = pick

    def forward(self, *a):
        o = self.m(*a)
        if self.pick:
            return self.pick(o)
        if isinstance(o, dict):
            o = o.get("out", next(iter(o.values())))
        while isinstance(o, (list, tuple)) and o:
            o = o[0]
        return o


def export_onnx(model, example, path, mode="qdq", pick=None, input_names=("input",),
                output_names=("out",), opset=17):
    if mode == "qdq":
        freeze_scales(model)
        m, ex, dev = model, example, "cpu"
    else:
        m = unwrap_model(model)
        dev = "cuda" if (mode == "fp16" and torch.cuda.is_available()) else "cpu"
        ex = example
    m = _MainOut(m, pick).eval().to(dev)
    ex = _to_dev(ex if isinstance(ex, (tuple, list)) else (ex,), dev)
    if mode == "fp16":
        m = m.half()
        ex = tuple(t.half() if torch.is_tensor(t) and t.is_floating_point() else t for t in ex)
    torch.onnx.export(m, tuple(ex), path, opset_version=opset,
                      input_names=list(input_names), output_names=list(output_names),
                      dynamic_axes=None)
    del m
    torch.cuda.empty_cache()
    return path


def export_all(model, example, out_dir, name, pick=None, **kw):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for mode in ("qdq", "fp32", "fp16"):
        p = os.path.join(out_dir, f"{name}_qat_{mode}.onnx")
        try:
            out[mode] = export_onnx(model, example, p, mode=mode, pick=pick, **kw)
        except Exception as e:
            out[mode] = None
            print(f"  [export {mode}] PAO: {type(e).__name__}: {str(e)[:90]}")
    return out


def verify(path):
    import onnx
    m = onnx.load(path)
    hist = {}
    for n in m.graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    q = hist.get("QuantizeLinear", 0) + hist.get("DequantizeLinear", 0)
    doms = sorted({n.domain for n in m.graph.node if n.domain})
    return {"qdq": q, "nodes": len(m.graph.node), "domains": doms,
            "mb": round(os.path.getsize(path) / 1024 ** 2, 3),
            "ok": q > 0 and not doms}


def trt_build(onnx_path, engine_path, fp16=False, int8=False, workspace_gb=4, fp32_layers=()):
    import tensorrt as trt

    def _attempt(force):
        lg = trt.Logger(trt.Logger.ERROR)
        b = trt.Builder(lg)
        net = b.create_network(0)
        parser = trt.OnnxParser(net, lg)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("ONNX parse: " + str(parser.get_error(0))[:140])
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if fp16 or int8:
            cfg.set_flag(trt.BuilderFlag.FP16)
        if int8:
            cfg.set_flag(trt.BuilderFlag.INT8)
        if force:
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            n = 0
            for i in range(net.num_layers):
                l = net.get_layer(i)
                if any(sub in l.name for sub in force):
                    try:
                        l.precision = trt.float32
                        n += 1
                    except Exception:
                        pass
            print(f"   [trt] {n}/{net.num_layers} slojeva prisiljeno u fp32")
        return b.build_serialized_network(net, cfg)

    ser = _attempt(tuple(fp32_layers))
    if ser is None and fp32_layers:
        print("   [trt] s ogranicenjima pao -> gradim IZNOVA bez njih")
        ser = _attempt(())
    if ser is None:
        raise RuntimeError("build_serialized_network vratio None")
    with open(engine_path, "wb") as f:
        f.write(ser)
    return engine_path


def _trt_dt(dt):
    import tensorrt as trt
    return {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8, trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64}.get(dt, torch.float32)


class TRTRunner:

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.torch_stream = torch.cuda.Stream()
        self.stream = self.torch_stream.cuda_stream
        self.inp, self.out_name, self.out_t = {}, None, None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            t = torch.empty(tuple(self.engine.get_tensor_shape(n)),
                            dtype=_trt_dt(self.engine.get_tensor_dtype(n)), device="cuda")
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp[n] = t
            else:
                self.out_name, self.out_t = n, t
            self.ctx.set_tensor_address(n, int(t.data_ptr()))

    def enqueue(self):
        self.ctx.execute_async_v3(self.stream)

    def infer(self, feed):
        if torch.is_tensor(feed):
            feed = {next(iter(self.inp)): feed}
        B = next(iter(feed.values())).shape[0]
        outs = []
        for i in range(B):
            for k, v in feed.items():
                self.inp[k].copy_(v[i:i + 1].to(self.inp[k].dtype))
            self.enqueue()
            self.torch_stream.synchronize()
            outs.append(self.out_t.detach().float().cpu().clone())
        return torch.cat(outs, 0)

    def preload(self, feed):
        if torch.is_tensor(feed):
            feed = {next(iter(self.inp)): feed}
        for k, v in feed.items():
            self.inp[k].copy_(v[:1].to(self.inp[k].dtype))
        torch.cuda.synchronize()

    def precision_histogram(self):
        try:
            insp = self.engine.create_engine_inspector()
            info = json.loads(insp.get_engine_information(self.trt.LayerInformationFormat.JSON))
            h = {}
            for l in info.get("Layers", []):
                if not isinstance(l, dict):
                    continue
                for o in l.get("Outputs", []) or []:
                    dt = str((o or {}).get("Format/Datatype", "?")).split("(")[0].strip()
                    h[dt] = h.get(dt, 0) + 1
            return h or {"?": 0}
        except Exception as e:
            return {"inspector": type(e).__name__}


class OVRunner:

    def __init__(self, onnx_path, device="CPU", threads=8):
        import openvino as ov
        core = ov.Core()
        core.set_property(device, {"INFERENCE_NUM_THREADS": int(threads)})
        self.model = core.read_model(onnx_path)
        self.cm = core.compile_model(self.model, device)
        self.req = self.cm.create_infer_request()
        self.in_names = [i.get_any_name() for i in self.model.inputs]

    def infer(self, feed):
        if torch.is_tensor(feed):
            feed = {self.in_names[0]: feed}
        B = next(iter(feed.values())).shape[0]
        outs = []
        for i in range(B):
            r = self.req.infer({k: v[i:i + 1].cpu().numpy() for k, v in feed.items()})
            outs.append(torch.from_numpy(next(iter(r.values()))).float())
        return torch.cat(outs, 0)

    def bench_call(self, feed):
        if torch.is_tensor(feed):
            feed = {self.in_names[0]: feed}
        nd = {k: v[:1].cpu().numpy() for k, v in feed.items()}
        return lambda: self.req.infer(nd)


def save_eager(model, path):
    torch.save(model, path)
    return path


def _slinn_loss(teacher, adapter, ctx, cache, n_batches, loss_fn=None):
    import loss as L
    st = {"i": 0}

    def fn(model, batch):
        i = st["i"] % max(n_batches, 1)
        st["i"] += 1
        if loss_fn is not None:
            out = loss_fn(model, batch)
        else:
            sig = cache.get(i, batch[0].device) if cache is not None else None
            out = L.kd_loss(model, teacher, adapter, batch, ctx["taps"], ctx["kd_mode"],
                            ctx["out_kind"], teacher_sig=sig)
        return out[0] if isinstance(out, tuple) else out
    return fn


def quantize_checkpoint(model, teacher, adapter, ctx, batches, cache=None, loss_fn=None,
                        device="cuda", monitor_fn=None, calib=None, on_step=None):
    import settings as CFG
    calib = int(CFG.QAT_CALIB_BATCHES if calib is None else calib)
    skip = tuple(ctx.get("terminal") or ())

    n_bn = fold_bn(model)

    n_w = wrap_model(model, skip=skip)
    n_fq = n_fakequant(model)
    if not n_w:
        return model, {"wrapped": 0, "why": "nema konvolucija ni Linear slojeva izvan izlaznih glava"}
    model.to(device)

    calibrate(model, list(batches[:max(1, calib)]), device,
              forward_fn=lambda mdl, b: adapter.forward(mdl, b))
    fn = _slinn_loss(teacher, adapter, ctx, cache, len(batches), loss_fn)
    start_lsq(model)
    model, info = _finetune(model, batches, fn, monitor_fn, device, on_step=on_step)
    freeze_scales(model)
    info.update({"wrapped": n_w, "fakequant": n_fq, "folded_bn": n_bn, "skip": list(skip)})
    return model, info


def _finetune(model, batches, loss_fn, monitor_fn, device, on_step=None):
    import copy as _copy
    import settings as CFG
    import engine as E
    model.to(device).train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nema parametara s gradijentom — provjeri wrap_model")
    opt = E._new_prodigy(model)

    best, best_state, no_imp, k, tot = None, None, 0, 0, 0.0
    hist = []
    while k < CFG.QAT_MAX_STEPS and no_imp < CFG.QAT_PATIENCE:
        for b in batches:
            if k >= CFG.QAT_MAX_STEPS:
                break
            b = _to_dev(b, device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, b)
            loss.backward()
            opt.step()
            tot += float(loss); k += 1
            if k % CFG.QAT_EVAL_EVERY:
                continue
            if monitor_fn is None:
                if on_step:
                    on_step(k, CFG.QAT_MAX_STEPS, tot / k)
                continue
            model.eval()
            m_now = float(monitor_fn(model))
            model.train()
            for mm in model.modules():
                if isinstance(mm, nn.modules.batchnorm._BatchNorm):
                    mm.eval()
            hist.append((k, m_now))
            if best is None or m_now > best:
                best, no_imp = m_now, 0
                best_state = _copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
            if on_step:
                on_step(k, CFG.QAT_MAX_STEPS, tot / k)
            if no_imp >= CFG.QAT_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"steps": k, "kd": tot / max(k, 1), "monitor_best": best,
                   "monitor_hist": hist,
                   "why": ("patience" if no_imp >= CFG.QAT_PATIENCE else "max_steps")}
