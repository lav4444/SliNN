import collections.abc
import copy
import hashlib
import json
import math
import os
import sys

_AA = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
if _AA not in sys.path:
    sys.path.insert(0, _AA)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import introspect as A                                       # noqa: E402  (6.4: bivsi morphology.analysis, genericki dio)
import morph as C                                            # noqa: E402  (6.4: bivsi morphology.compress mehanike)
import settings as CFG                                       # noqa: E402
import dataset as DS                                         # noqa: E402
import loss as L                                             # noqa: E402

TMP_ROOT = CFG.TMP_ROOT


def _ag_layer_table(model, adapter, device):
    if not hasattr(adapter, "forward_example"):
        return _ORIG_LT(model, adapter, device)
    leaves = A.weighted_leaves(model)
    by_id = {id(m): (name, tn, w) for name, m, tn, w in leaves}
    rec, handles = [], []

    def mk(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and o:
                o = o[0]
            name, tn, w = by_id[id(mod)]
            ishape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            oshape = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() >= 3:
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:
                    out_f, in_f = w.shape
                    flops = 2 * out_f * in_f * (o.numel() // out_f if out_f else 0)
            rec.append({"name": name, "type": tn, "role": "neuron" if w.dim() == 2 else "filter",
                        "units": int(w.shape[0]), "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                        "gflops": flops / 1e9, "in": ishape, "out": oshape})
        return hook

    for name, m, tn, w in leaves:
        handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        adapter.forward(model, adapter.forward_example(device))
    for h in handles:
        h.remove()
    return rec


def _ag_forward_ok(model, adapter, device):
    if not hasattr(adapter, "forward_example"):
        return _ORIG_FO(model, adapter, device)
    was = model.training
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward(model, adapter.forward_example(device))
        return True
    except BaseException:
        return False
    finally:
        model.train(was)


def _ag_try_grow_layer(model, adapter, device, name, k, init_filters=None):
    import torch_pruning as tp
    if not hasattr(adapter, "forward_example"):
        return _ORIG_GROW(model, adapter, device, name, k, init_filters)
    ref_imgs = adapter.forward_example(device)
    try:
        with torch.no_grad():
            ref_out = adapter.teacher_outputs(model, ref_imgs)
    except BaseException:
        return None
    trial = copy.deepcopy(model)
    leaves = {nm: (mm, w.dim()) for nm, mm, _, w in A.weighted_leaves(trial)}
    if name not in leaves:
        return None
    Lm, dim = leaves[name]
    if getattr(Lm, "groups", 1) > 1:
        return None
    old_L = Lm.weight.shape[0]
    try:
        for p in trial.parameters():
            p.requires_grad_(True)
        fn = (tp.function.prune_conv_out_channels if dim >= 3 else tp.function.prune_linear_out_channels)
        DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
        group = DG.get_pruning_group(Lm, fn, idxs=[0])
        out_mods, bns, fbns, dws, cons = [], [], [], [], {}
        for dep, idxs in group:
            tgt = getattr(getattr(dep, "target", None), "module", None)
            if tgt is None:
                continue
            hn = getattr(dep.handler, "__name__", type(dep.handler).__name__).lower()
            if isinstance(tgt, nn.modules.batchnorm._BatchNorm):
                bns.append(tgt)
            elif type(tgt).__name__ == "FrozenBatchNorm2d" or (hasattr(tgt, "running_mean") and hasattr(tgt, "weight")
                                                               and not isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear))):
                fbns.append(tgt)
            elif C._is_depthwise(tgt):
                dws.append(tgt)
            elif isinstance(tgt, nn.Conv2d) and tgt.groups > 1:
                return None
            elif "in_channel" in hn or "in_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    cons.setdefault(tgt, []).append(int(min(idxs)))
            elif "out_channel" in hn or "out_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    out_mods.append(tgt)
        if Lm not in out_mods:
            out_mods.append(Lm)
        seen = set()
        for mod in out_mods:
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if mod is Lm and init_filters is not None:
                init = init_filters
            else:
                wabs = mod.weight.detach().abs().flatten(1).mean(1)
                order = torch.argsort(wabs, descending=True)
                idx = [int(order[i % len(order)]) for i in range(k)]
                cl = mod.weight.detach()[idx].clone()
                init = cl + torch.randn_like(cl) * 0.02 * cl.abs().mean().clamp(min=1e-6)
            C._widen_out(mod, k, init)
        for dw in {id(d): d for d in dws}.values():
            C._widen_depthwise(dw, k)
        for bn in {id(b): b for b in bns}.values():
            C._widen_bn(bn, k)
        for fb in {id(f): f for f in fbns}.values():
            C._widen_frozen_bn(fb, k)
        for cmod, offs in cons.items():
            C._insert_in_zeros(cmod, k, [o + old_L for o in offs])
        if not _ag_forward_ok(trial, adapter, device):
            return None
        with torch.no_grad():
            after = adapter.teacher_outputs(trial, ref_imgs)
        if C._max_abs_diff(ref_out, after) >= 1e-3:
            return None
        return trial
    except BaseException:
        return None


_SHIMS_INSTALLED = False
_ORIG_LT = _ORIG_FO = _ORIG_GROW = None


def install_sizing_shims():
    global _SHIMS_INSTALLED, _ORIG_LT, _ORIG_FO, _ORIG_GROW
    if _SHIMS_INSTALLED:
        return
    _ORIG_LT, _ORIG_FO, _ORIG_GROW = A.layer_table, C._forward_ok, C._try_grow_layer
    A.layer_table = _ag_layer_table
    C._forward_ok = _ag_forward_ok
    C._try_grow_layer = _ag_try_grow_layer
    _SHIMS_INSTALLED = True


install_sizing_shims()


def apply_torch_backends():
    if getattr(CFG, "MATMUL_TF32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


apply_torch_backends()


def gflops(model, adapter, device):
    return A.gflops_total(model, adapter, device)


def _widths(model):
    return {nm: int(w.shape[0]) for nm, _, _, w in A.weighted_leaves(model)}


def autobatch(model, adapter, device, ctx, path, free_frac=0.9, cap=64, cands=(1, 2, 4, 8, 16, 32, 64)):
    cands = [b for b in cands if b <= cap]
    if device.type != "cuda":
        return cands[0]
    import gc

    import prodigyopt
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    split = ctx["split_plan"]["train"] if ctx.get("split_plan") else None
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rg = [p.requires_grad for p in model.parameters()]
    chosen = cands[0]
    try:
        for bs in cands:
            torch.cuda.empty_cache(); gc.collect()
            free, _ = torch.cuda.mem_get_info()
            usable = free + torch.cuda.memory_allocated()
            try:
                imgs, _ = DS.input_batch(path, adapter, device, split=split, n=bs)
                with torch.no_grad():
                    tsig = L.teacher_signals(model, adapter, imgs, taps)
                for p in model.parameters():
                    p.requires_grad_(True)
                opt = prodigyopt.Prodigy([p for p in model.parameters() if p.requires_grad], lr=1.0)
                model.eval()
                torch.cuda.reset_peak_memory_stats(device)
                loss, _ = L.kd_loss(model, model, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=tsig)
                loss.backward(); opt.step()
                peak = torch.cuda.max_memory_allocated(device)
                del loss, tsig, imgs, opt
                for p in model.parameters():
                    p.grad = None
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache(); gc.collect(); break
            if peak <= free_frac * usable:
                chosen = bs
            else:
                break
    finally:
        model.load_state_dict(sd); model.to(device)
        for p, r in zip(model.parameters(), rg):
            p.requires_grad_(r)
        torch.cuda.empty_cache(); gc.collect()
    return chosen


def _candidate_files(root, mode, split, cap, split_key=None, carve=False):
    lim = 10 ** 9 if carve else cap
    if mode == "image":
        files = DS._media_files(root, DS._IMG, split, lim, skip_masks=True)
    elif mode == "seq":
        files = DS._media_files(root, DS._AUD, split, lim)
    else:
        return []
    if carve and split_key:
        files = DS.auto_carve(files, split_key)
    if CFG.DEV_DATA_SUBSET:
        cap = min(cap, int(CFG.DEV_DATA_SUBSET))
    return files[:cap]


def to_device(batch, device):
    out = []
    for im in batch:
        t = im.to(device, non_blocking=True)
        out.append(t.float().div_(255.0) if t.dtype == torch.uint8 else t)
    return out


def _u8_safe(x):
    if not torch.is_tensor(x) or not x.is_floating_point():
        return False
    if float(x.min()) < 0.0 or float(x.max()) > 1.0:
        return False
    return bool(torch.equal((x * 255.0).round().div(255.0), x))


def pack_batches(batches, device):
    if not batches or not batches[0]:
        return batches, None
    u8 = bool(getattr(CFG, "BATCH_UINT8", False)) and all(_u8_safe(x) for x in batches[0][:4])
    budget = int(getattr(CFG, "BATCH_GPU_CACHE_MB", 0)) * 1024 ** 2
    out, used, n_gpu = [], 0, 0
    for b in batches:
        nb = [(x * 255.0).round().to(torch.uint8) if (u8 and x.is_floating_point()) else x
              for x in b]
        size = sum(x.numel() * x.element_size() for x in nb)
        if budget and used + size <= budget and device.type == "cuda":
            try:
                nb = [x.to(device) for x in nb]
                used += size
                n_gpu += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                budget = 0
        out.append(nb)
    msg = "batchevi: {}{} rezidentno na GPU-u ({:.0f} MB)".format(
        "uint8, " if u8 else "float32, ", "{}/{}".format(n_gpu, len(batches)), used / 1024 ** 2)
    return out, msg


FALLBACK_BATCHES = 8


class LazyBatches(collections.abc.Sequence):

    def __init__(self, groups, dec, in_ch, size, sr=None):
        self.groups, self._dec, self._in_ch, self._size = groups, dec, in_ch, size
        self._sr = sr
        self._cache = {}
        self._u8 = None
        self._budget = int(getattr(CFG, "BATCH_CACHE_MB", 0)) * 1024 ** 2
        self._used = 0

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        hit = self._cache.get(i)
        if hit is not None:
            return [x.float().div_(255.0) if x.dtype == torch.uint8 else x for x in hit]
        out = [self._dec(f, self._in_ch, self._size, self._sr) if self._sr is not None
               else self._dec(f, self._in_ch, self._size) for f in self.groups[i]]
        self._maybe_cache(i, out)
        return out

    def _maybe_cache(self, i, out):
        if not self._budget or not out:
            return
        if self._u8 is None:
            self._u8 = all(_u8_safe(x) for x in out[:4])
        keep = [(x * 255.0).round().to(torch.uint8) if (self._u8 and x.is_floating_point()) else x
                for x in out]
        size = sum(x.numel() * x.element_size() for x in keep)
        if self._used + size > self._budget:
            return
        self._cache[i] = keep
        self._used += size

    def cache_note(self):
        if not self._budget:
            return None
        return "kes batcheva: {}/{} dekodirano, {} ({:.0f} MB / {:.0f} MB)".format(
            len(self._cache), len(self.groups),
            "uint8" if self._u8 else "float32",
            self._used / 1024 ** 2, self._budget / 1024 ** 2)

    @property
    def bsize(self):
        return max((len(g) for g in self.groups), default=0)

    @property
    def paths(self):
        return [f for g in self.groups for f in g]


def _batch_size_of(batches):
    bs = getattr(batches, "bsize", None)
    return bs if bs is not None else max((len(b) for b in batches), default=0)


def count_train_samples(path, adapter, split_plan, split_key="train", model=None):
    mode = getattr(adapter, "_mode", "image")
    root = DS.detect_format(path).get("root", path)
    tr = (split_plan or {}).get(split_key) if isinstance(split_plan, dict) else split_plan
    split = None if (tr in (None, "AUTO")) else tr
    carve = (tr == "AUTO")
    try:
        if mode in ("image", "seq"):
            return len(_candidate_files(root, mode, split, 10 ** 9, split_key, carve))
        if mode == "vector":
            X = DS._tabular_matrix(root, adapter._in_ch, split=split_key)
            n = len(X) if X is not None else 0
            return min(n, int(CFG.DEV_DATA_SUBSET)) if CFG.DEV_DATA_SUBSET else n
        if mode == "token":
            a = DS.hf_arrow_for(root, split or split_key)
            if a is None or DS.hf_tokenizer(root, model) is None:
                return 0
            import datasets
            n = datasets.Dataset.from_file(a).num_rows
            return min(n, int(CFG.DEV_DATA_SUBSET)) if CFG.DEV_DATA_SUBSET else n
    except BaseException:
        pass
    return 0


def materialize_train_batches(path, adapter, device, split_plan, batch_size=None, n_batches=None, seed=0,
                              split_key="train", model=None):
    import random
    batch_size = int(CFG.TRAIN_BATCH if batch_size is None else batch_size)
    mode = getattr(adapter, "_mode", "image")
    root = DS.detect_format(path).get("root", path)
    tr = (split_plan or {}).get(split_key) if isinstance(split_plan, dict) else split_plan
    split = None if (tr in (None, "AUTO")) else tr
    carve = (tr == "AUTO")

    if n_batches is None:
        n_avail = count_train_samples(path, adapter, split_plan, split_key, model=model)
        n_batches = max(1, -(-n_avail // batch_size)) if n_avail else FALLBACK_BATCHES
    need = batch_size * n_batches

    samples, source = [], "fallback-random"
    try:
        if mode in ("image", "seq"):
            files = _candidate_files(root, mode, split, need * 3, split_key, carve)
            if files:
                rng = random.Random(seed); rng.shuffle(files)
                files = (files * (need // len(files) + 1))[:need]
                dec = DS._decode_image if mode == "image" else DS._decode_audio
                groups = [files[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
                groups = [g for g in groups if g]
                return (LazyBatches(groups, dec, adapter._in_ch, adapter.imgsz,
                                    getattr(adapter, "sr", None)),
                        "image-files" if mode == "image" else "audio-files")
        elif mode == "token":
            pairs = DS.hf_pairs(root, adapter, model, split or split_key, need)
            if pairs:
                samples = [t for t, _ in pairs]
                if len(samples) < need:
                    samples = (samples * (need // len(samples) + 1))[:need]
                source = "hf-arrow"
        elif mode == "vector":
            X = DS._tabular_matrix(root, adapter._in_ch, split=split_key)
            if X is not None and len(X):
                import numpy as np
                idx = np.random.default_rng(seed).integers(0, len(X), need)
                samples = [torch.from_numpy(X[i]) for i in idx]; source = "tabular"
    except BaseException:
        samples = []
    if not samples:
        samples = [adapter._one(torch.device("cpu")) for _ in range(need)]
        source = "fallback-random"
    batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
    batches, _msg = pack_batches([b for b in batches if b], device)
    if _msg:
        print("[podaci] " + _msg)
    return batches, source


def agreement_metrics(teacher, adapter, device, ctx, path, frac=None, seed=0):
    import metric as _M
    frac = CFG.METRIC_MONITOR_FRAC if frac is None else frac
    kind = ctx["out_kind"]
    vb, _src = materialize_train_batches(path, adapter, device, ctx["split_plan"],
                                         batch_size=CFG.METRIC_VAL_BATCH, seed=seed, split_key="val")
    full = [x for b in vb for x in b]
    mon = None
    if full and 0.0 < frac < 1.0 and len(full) > 1:
        import random
        k = min(len(full), max(int(getattr(CFG, 'METRIC_MONITOR_MIN', 1)),
                               int(round(frac * len(full)))))
        idx = sorted(random.Random(seed).sample(range(len(full)), k))
        sub = [full[i] for i in idx]
        mon = lambda mdl: _M.teacher_agreement(mdl, teacher, adapter, sub, device, kind=kind)["agreement"]
    return (lambda mdl: _M.teacher_agreement(mdl, teacher, adapter, full, device, kind=kind)["agreement"],
            mon)


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def model_name_of(model_path):
    p = os.path.normpath(str(model_path))
    mapa = os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(os.path.basename(p))[0]
    return mapa if mapa and stem in ("model", mapa) else (f"{mapa}_{stem}" if mapa else stem)


def _to_half_cpu(sig):
    h = {"feat": {k: v.detach().half().cpu() for k, v in sig["feat"].items()},
         "out": sig["out"].detach().half().cpu()}
    if sig.get("out_size"):
        h["out_size"] = tuple(sig["out_size"])
    return h


def _to_dev_float(sig, device):
    out = sig["out"].float().to(device)
    sz = sig.get("out_size")
    if sz and tuple(out.shape[-2:]) != tuple(sz):
        out = nn.functional.interpolate(out, size=tuple(sz), mode="bilinear", align_corners=False)
    return {"feat": {k: v.float().to(device) for k, v in sig["feat"].items()}, "out": out}


def _fingerprint(batches):
    h = hashlib.sha1()
    paths = getattr(batches, "paths", None)
    if paths is not None:
        for f in paths:
            h.update(str(f).encode())
        return h.hexdigest()[:16]
    for b in batches:
        if not b:
            continue
        t = b[0].detach().float().flatten()
        h.update(str(tuple(b[0].shape)).encode())
        step = max(1, t.numel() // 64)
        h.update(str(round(float(t[::step].sum()), 3)).encode())
    return h.hexdigest()[:16]


def _ram_free_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


class TeacherSigCache:

    def __init__(self, cache_dir, n, in_ram, budget_mb=None):
        self.dir = cache_dir; self.n = n; self.in_ram = in_ram
        self.ram = [None] * n if in_ram else None
        mb = getattr(CFG, "TEACHER_CACHE_MB", 0) if budget_mb is None else budget_mb
        self.strop_mb = float(mb)
        self.slobodno_mb = _ram_free_mb()
        if budget_mb is None and self.slobodno_mb:
            preostalo = self.slobodno_mb - getattr(CFG, "BATCH_CACHE_MB", 0)
            mb = min(mb, max(0.0, preostalo) * getattr(CFG, "CACHE_RAM_FRAC", 0.6))
        self.budget = int(mb * 1024 ** 2) if in_ram else 0
        self.used = 0

    def _path(self, i):
        return os.path.join(self.dir, f"sig_{i}.pt")

    def has_all(self):
        return all(os.path.exists(self._path(i)) for i in range(self.n))

    def _keep(self, i, h):
        if self.ram is None:
            return False
        b = _nbytes(h)
        if self.used + b > self.budget:
            return False
        self.ram[i] = h
        self.used += b
        return True

    def put(self, i, sig):
        h = _to_half_cpu(sig)
        torch.save(h, self._path(i))
        self._keep(i, h)

    def warm_ram(self):
        if self.ram is None:
            return
        for i in range(self.n):
            if self.ram[i] is None and not self._keep(i, torch.load(self._path(i),
                                                                    map_location="cpu")):
                break

    def get(self, i, device):
        h = self.ram[i] if (self.ram is not None and self.ram[i] is not None) else \
            torch.load(self._path(i), map_location="cpu")
        return _to_dev_float(h, device)

    def cache_note(self):
        if not self.budget:
            return None
        k = sum(1 for x in (self.ram or []) if x is not None)
        return ("kes ucitelja: {}/{} batcheva u RAM-u ({:.0f} MB / {:.0f} MB budzeta; "
                "strop {:.0f} MB, slobodno bilo {:.0f} MB), ostatak s diska").format(
            k, self.n, self.used / 1024 ** 2, self.budget / 1024 ** 2,
            self.strop_mb, self.slobodno_mb)


def _nbytes(o):
    if torch.is_tensor(o):
        return o.numel() * 2
    if isinstance(o, dict):
        return sum(_nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_nbytes(v) for v in o)
    return 0


def plan_teacher_cache(teacher, adapter, batches, taps, model_name, split="train", disk_frac=0.8,
                       out_tap=None):
    dev = next(teacher.parameters()).device
    n = len(batches)
    cache_dir = os.path.join(TMP_ROOT, _safe(model_name), _safe(split))
    per = 0
    if n:
        sig = L.teacher_signals(teacher, adapter, to_device(batches[0], dev), taps, out_tap=out_tap)
        per = _nbytes(sig)
    total = per * n
    try:
        st = os.statvfs(TMP_ROOT if os.path.isdir(TMP_ROOT) else os.path.dirname(TMP_ROOT) or "/")
        free = st.f_bavail * st.f_frsize
    except BaseException:
        free = 0
    fits = bool(free) and total <= free * disk_frac
    return {"n_batches": n, "batch_size": _batch_size_of(batches),
            "bytes_per_batch": per, "total_gb": total / C.GB, "free_gb": free / C.GB,
            "fits_disk": fits, "cache_dir": cache_dir,
            "n_fit": int(free * disk_frac / per) if per else 0}


def precompute_teacher(teacher, adapter, batches, taps, model_name, split="train", in_ram=True, verbose=False):
    dev = next(teacher.parameters()).device
    n = len(batches)
    bs = _batch_size_of(batches)
    cache_dir = os.path.join(TMP_ROOT, _safe(model_name), _safe(split))
    meta_path = os.path.join(cache_dir, "meta.json")

    out_tap = None
    if n and getattr(CFG, "TEACHER_CACHE_LOWRES_OUT", True):
        try:
            out_tap = L.find_out_tap(teacher, adapter, to_device(batches[0], dev))
        except BaseException:
            out_tap = None

    meta = {"model": model_name, "n_batches": n, "batch_size": bs, "out_tap": out_tap,
            "taps": sorted(taps), "fingerprint": _fingerprint(batches)}
    cache = TeacherSigCache(cache_dir, n, in_ram)
    if os.path.exists(meta_path) and _load_json(meta_path) == meta and cache.has_all():
        cache.warm_ram()
        note = cache.cache_note()
        if note:
            print("  " + note + "  [reuse]", flush=True)
        if verbose:
            print(f"  reuse teacher-cache (meta valjan): {cache_dir}")
        return cache
    os.makedirs(cache_dir, exist_ok=True)
    stari = [f for f in os.listdir(cache_dir) if f.startswith("sig_") and f.endswith(".pt")]
    povrat = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in stari)

    frac = getattr(CFG, "TEACHER_CACHE_DISK_FRAC", 0.8)
    plan = plan_teacher_cache(teacher, adapter, batches, taps, model_name, split=split, disk_frac=frac,
                              out_tap=out_tap)
    po_batchu = plan["bytes_per_batch"]
    treba, slobodno = po_batchu * n, plan["free_gb"] * C.GB + povrat
    if plan["free_gb"] and treba > slobodno * frac:
        raise RuntimeError(
            "teacher-cache ne stane na disk: treba {:.1f} GB ({} batcheva x {:.0f} MB), slobodno "
            "{:.1f} GB (od toga {:.1f} GB iz starog kesa), granica {:.0%}. Stane ~{} batcheva. {}"
            .format(treba / C.GB, n, po_batchu / 1024 ** 2, slobodno / C.GB, povrat / C.GB,
                    frac, int(slobodno * frac / po_batchu) if po_batchu else 0, cache_dir))
    print("  teacher-cache: {} batcheva x {:.1f} MB = {:.1f} GB (slobodno {:.1f} GB){}"
          .format(n, po_batchu / 1024 ** 2, treba / C.GB, slobodno / C.GB,
                  "  [izlaz pred-upsample: " + out_tap + "]" if out_tap else ""), flush=True)

    for f in stari:
        os.remove(os.path.join(cache_dir, f))
    for i, b in enumerate(batches):
        cache.put(i, L.teacher_signals(teacher, adapter, to_device(b, dev), taps, out_tap=out_tap))
    json.dump(meta, open(meta_path, "w"))
    note = cache.cache_note()
    if note:
        print("  " + note, flush=True)
    if verbose:
        print(f"  precompute teacher-cache: {n} batcheva -> {cache_dir}")
    return cache


def _load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except BaseException:
        return None


def prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, steps, clip=5.0, loss_fn=None,
                     offset=0):
    import prodigyopt
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                             lr=1.0, d_coef=0.9, safeguard_warmup=True)
    last, nb = None, len(batches)
    for s in range(steps):
        i = (offset + s) % nb
        imgs = to_device(batches[i], device)
        sig = cache.get(i, device) if cache is not None else None
        opt.zero_grad(set_to_none=True)
        loss, _ = (loss_fn(student, imgs) if loss_fn is not None
                   else L.kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=sig))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], clip)
        opt.step()
        last = float(loss)
    return last


def _new_prodigy(model):
    import prodigyopt
    return prodigyopt.Prodigy([p for p in model.parameters() if p.requires_grad],
                              lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)


def lr_eff(opt):
    pg = opt.param_groups[0]
    return float(pg.get("d", 1.0)) * float(pg["lr"])


def kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep, warmup,
             clip=5.0, loss_fn=None, on_batch=None):
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    warmup = max(int(warmup), 1)
    tot, n = 0.0, 0
    for i in range(len(batches)):
        for g in opt.param_groups:
            g["lr"] = min(1.0, (gstep + 1) / warmup)
        imgs = to_device(batches[i], device)
        sig = cache.get(i, device) if cache is not None else None
        opt.zero_grad(set_to_none=True)
        loss, _ = (loss_fn(student, imgs) if loss_fn is not None
                   else L.kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=sig))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], clip)
        opt.step()
        tot += float(loss); n += 1; gstep += 1
        if on_batch is not None:
            on_batch(i + 1, len(batches), tot / n)
    return (tot / max(n, 1)), gstep


class MorphState:

    def __init__(self, prunable, g0, step_frac=None, reinvest_frac=None, cooldown=None):
        self.prunable = set(prunable)
        self.g0 = float(g0)
        self.step_target = (step_frac or CFG.F1_PRUNE_STEP_FRAC) * self.g0
        self.reinvest_frac = CFG.PHASE2_REINVEST_FRAC if reinvest_frac is None else reinvest_frac
        self.cooldown = CFG.PHASE2_CHURN_COOLDOWN if cooldown is None else cooldown
        self.banned = set()
        self.grown_at, self.pruned_at = {}, {}
        self.morph_idx = 0
        self.total_pruned = self.total_grown = 0.0

    def align_best(self, model):
        return C.best_align_score(model, {nm: True for nm in self.prunable}, self.banned)


def morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=None, grow=True,
               step_target=None):
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    target = st.step_target if step_target is None else float(step_target)
    st.morph_idx += 1
    mi = st.morph_idx
    grow_protected = {l for l, k in st.grown_at.items() if mi - k <= st.cooldown}
    prune_protected = {l for l, k in st.pruned_at.items() if mi - k <= st.cooldown}
    elig_all = st.prunable - st.banned
    imp, gavg = L.kd_importance(student, teacher, adapter, imp_dev, taps, kd_mode, out_kind,
                                prunable=elig_all, loss_fn=loss_fn)
    cost, flops_per, units = C.prune_costs(student, adapter, device, elig_all)
    info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in elig_all}
    flops_per0, units0 = dict(flops_per), dict(units)

    g_before = g_cur = gflops(student, adapter, device)
    w_step0 = _widths(student)
    cd_override, n_rem, pruned_names, touched = False, 0, set(), set()
    est_freed = r1_freed = 0.0
    remaining, rounds, stall = target, 0, 0
    for _rnd in range(CFG.PHASE2_PRUNE_ROUNDS):
        if remaining <= CFG.PHASE2_PRUNE_SLACK * target or stall >= 2:
            break
        excl = (st.banned | touched) if cd_override else (st.banned | touched | grow_protected)
        elig_r = elig_all - excl
        if not elig_r:
            if not cd_override and grow_protected:
                cd_override = True; continue
            break
        if rounds:
            cost, flops_per, units = C.prune_costs(student, adapter, device, elig_r)
            info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in elig_r}
        plan, _r = C._select_prune_plan({nm: True for nm in elig_r}, imp, cost, flops_per, units, info,
                                        remaining, CFG.PHASE2_PRUNE_LAYER_CAP, CFG.PHASE2_MIN_ALIVE_FRAC,
                                        CFG.PHASE2_MIN_ALIVE, exclude=excl)
        if not plan:
            if not cd_override and grow_protected:
                cd_override = True; continue
            break
        if not rounds:
            est_freed = float(_r)
        student, k_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, device, plan)
        st.banned |= bad
        pruned_names |= (set(plan) - bad)
        n_rem += k_rem
        rounds += 1
        w_now = _widths(student)
        touched |= {nm for nm, w0 in w_step0.items() if w_now.get(nm, w0) != w0}
        g_new = gflops(student, adapter, device)
        got = g_cur - g_new
        g_cur = g_new
        if rounds == 1:
            r1_freed = max(got, 0.0)
        remaining -= max(got, 0.0)
        stall = stall + 1 if got <= 1e-9 else 0
    freed = g_before - g_cur
    st.total_pruned += max(freed, 0.0)
    for nm in pruned_names:
        st.pruned_at[nm] = mi

    pool = st.reinvest_frac * st.total_pruned - st.total_grown
    grown_info = []
    if grow and pool > 0:
        info_g = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in st.prunable}
        struct_g = {nm: (nm in st.prunable and nm not in pruned_names and nm not in prune_protected
                         and nm not in st.banned) for nm in st.prunable}
        g2, ginfos, spent = C._grow_decide(student, adapter, device, struct_g, gavg, flops_per0, units0,
                                           info_g, pool)
        if g2 is not None:
            student = g2; st.total_grown += max(spent, 0.0); grown_info = ginfos
            for gi in ginfos:
                st.grown_at[gi["layer"]] = mi
    return student, {"n_rem": n_rem, "grown": grown_info, "cd_override": cd_override,
                     "step_target": target, "est_freed": est_freed, "r1_freed": r1_freed,
                     "act_freed": max(freed, 0.0), "prune_rounds": rounds,
                     "grow_protected": len(grow_protected)}


def morph_loop(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, step_frac=None, reinvest_frac=None, max_steps=None, ft_steps=6,
               batch_size=None, n_batches=None, imp_batches=None, seed=0, cache=None, batches=None,
               cooldown=None, on_step=None,
               metric_fn=None, metric_tol=None, metric_baseline=None, loss_fn=None):
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    prunable = set(ctx["prunable"])
    step_frac = step_frac or CFG.F1_PRUNE_STEP_FRAC
    reinvest_frac = CFG.PHASE2_REINVEST_FRAC if reinvest_frac is None else reinvest_frac
    max_steps = CFG.F1_MAX_STEPS if max_steps is None else max_steps
    if metric_tol is None:
        metric_tol = CFG.FT_RECOVERY_FRAC
    cooldown = CFG.PHASE2_CHURN_COOLDOWN if cooldown is None else cooldown
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, n_batches, seed)
    if cache is None and loss_fn is None:
        cache = precompute_teacher(teacher, adapter, batches, taps, model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches or CFG.IMP_BATCHES]]

    g0 = gflops(student, adapter, device)
    st = MorphState(prunable, g0, step_frac, reinvest_frac, cooldown)
    ft_cursor = 0
    best_model, best_gflops, best_step = None, float("inf"), 0
    if metric_fn is not None and metric_baseline is None:
        metric_baseline = metric_fn(student)
    metric_floor = (metric_tol * metric_baseline) if metric_fn is not None else None
    below = 0
    _p0 = A.count_params(student)
    traj = [{"step": 0, "gflops": g0, "params": _p0, "kd": None, "removed_ch": 0, "grown": [],
             "metric": metric_baseline, "size_mb": _p0 * 4 / (1024 ** 2),
             "gflops_freed": 0.0, "gflops_reinvested": 0.0,
             "align_score": C.model_align_score(student), "align_best": st.align_best(student)}]
    if on_step:
        on_step(traj[0])
    for step in range(1, max_steps + 1):
        if (g0 - traj[-1]["gflops"]) >= target_frac * g0:
            break
        student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=loss_fn)
        n_rem, grown_info = mi["n_rem"], mi["grown"]

        kd = prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, ft_steps,
                              loss_fn=loss_fn, offset=ft_cursor)
        ft_cursor += ft_steps
        cur_metric = metric_fn(student) if metric_fn is not None else None
        n_par = A.count_params(student)
        rec = {"step": step, "gflops": gflops(student, adapter, device), "params": n_par,
               "kd": kd, "metric": cur_metric, "removed_ch": n_rem,
               "grown": [(gi["layer"], gi["k"]) for gi in grown_info],
               "cd_override": mi["cd_override"], "banned": len(st.banned),
               "size_mb": n_par * 4 / (1024 ** 2),
               "step_target": mi["step_target"],
               "est_freed": mi["est_freed"],
               "r1_freed": mi["r1_freed"],
               "act_freed": mi["act_freed"],
               "prune_rounds": mi["prune_rounds"],
               "gflops_freed": st.total_pruned,
               "gflops_reinvested": st.total_grown,
               "align_score": C.model_align_score(student),
               "align_best": st.align_best(student)}
        traj.append(rec)
        if on_step:
            on_step(rec)
        if metric_fn is not None:
            if cur_metric >= metric_floor:
                below = 0
                if rec["gflops"] < best_gflops:
                    best_model = copy.deepcopy(student); best_gflops = rec["gflops"]; best_step = step
            else:
                below += 1
                if below >= 3:
                    break
        if torch.cuda.is_available():
            import gc as _gc
            torch.cuda.empty_cache(); _gc.collect()
        if n_rem == 0 and not grown_info and not mi["grow_protected"]:
            break
    return {"g0": g0, "trajectory": traj, "final_gflops": traj[-1]["gflops"], "banned": sorted(st.banned),
            "grown_at": st.grown_at, "pruned_at": st.pruned_at, "student": student, "cache": cache,
            "best_model": best_model, "best_gflops": best_gflops, "best_step": best_step,
            "metric_baseline": metric_baseline}


def phase1_loop(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                monitor_fn=None, metric_baseline=None, monitor_baseline=None, metric_tol=None,
                step_frac=None, reinvest_frac=None, cooldown=None, max_steps=None,
                batch_size=None, imp_batches=None, seed=0, cache=None, batches=None,
                loss_fn=None, on_step=None, on_batch=None):
    taps = ctx["taps"]
    prunable = set(ctx["prunable"])
    max_steps = CFG.F1_MAX_STEPS if max_steps is None else max_steps
    metric_tol = CFG.FT_RECOVERY_FRAC if metric_tol is None else metric_tol
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if cache is None and loss_fn is None:
        cache = precompute_teacher(teacher, adapter, batches, taps, model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches or CFG.IMP_BATCHES]]

    g0 = gflops(student, adapter, device)
    st = MorphState(prunable, g0, step_frac, reinvest_frac, cooldown)
    opt, gstep, warmup = _new_prodigy(student), 0, len(batches)

    if metric_baseline is None:
        metric_baseline = float(metric_fn(student))
    if monitor_fn is not None and monitor_baseline is None:
        monitor_baseline = float(monitor_fn(student))
    floor_full = metric_tol * metric_baseline
    gate_fn = monitor_fn if monitor_fn is not None else metric_fn
    gate_floor = (metric_tol * monitor_baseline) if monitor_fn is not None else floor_full

    last_good = copy.deepcopy(student)
    lg_gflops, lg_metric, lg_step = g0, metric_baseline, 0
    ft_used = no_imp = 0
    curr_best = None
    above = True
    prev = monitor_baseline if monitor_fn is not None else metric_baseline
    _p0 = A.count_params(student)
    traj = [{"step": 0, "phase": "baseline", "gflops": g0, "params": _p0, "kd": None,
             "metric": prev, "metric_full": metric_baseline, "monitor": prev,
             "removed_ch": 0, "grown": [],
             "size_mb": _p0 * 4 / (1024 ** 2), "gflops_freed": 0.0, "gflops_reinvested": 0.0,
             "align_score": C.model_align_score(student), "align_best": st.align_best(student),
             "is_best": True}]
    if on_step:
        on_step(traj[0])
    reason = "max_steps ({})".format(max_steps)

    for step in range(1, max_steps + 1):
        mode = "morph" if above else "ft"
        mi = None
        if mode == "morph":
            student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=loss_fn)
            opt = _new_prodigy(student)
            if mi["n_rem"] == 0 and not mi["grown"] and not mi["grow_protected"]:
                reason = "nema vise rezivih kandidata (off-limits/banned/floor/cap)"
                break
        kd, gstep = kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep,
                             warmup, loss_fn=loss_fn, on_batch=on_batch)
        m = float(gate_fn(student))
        g_now = gflops(student, adapter, device)

        above = m >= gate_floor
        full = None
        if above and g_now < lg_gflops - 1e-9:
            full = m if monitor_fn is None else float(metric_fn(student))
            if full >= floor_full:
                last_good = copy.deepcopy(student)
                lg_gflops, lg_metric, lg_step = g_now, full, step
            else:
                above = False

        stop = None
        if above:
            ft_used = no_imp = 0
            curr_best = None
        elif mode == "morph":
            curr_best, ft_used, no_imp = m, 0, 0
        else:
            ft_used += 1
            if curr_best is None or m > curr_best + 1e-4:
                curr_best, no_imp = m, 0
            else:
                no_imp += 1
            if no_imp >= CFG.F1_FT_PATIENCE:
                stop = "oporavak stagnira ({} FT epohe bez novog najboljeg)".format(CFG.F1_FT_PATIENCE)
            elif ft_used >= CFG.F1_FT_MAX_EPOCHS:
                stop = "oporavak nije uspio u {} FT epoha".format(CFG.F1_FT_MAX_EPOCHS)

        n_par = A.count_params(student)
        rec = {"step": step, "phase": mode, "gflops": g_now, "params": n_par, "kd": kd,
               "metric": m, "monitor": m, "metric_full": full,
               "removed_ch": (mi["n_rem"] if mi else 0),
               "grown": [(gi["layer"], gi["k"]) for gi in (mi["grown"] if mi else [])],
               "banned": len(st.banned), "cd_override": bool(mi and mi["cd_override"]),
               "size_mb": n_par * 4 / (1024 ** 2),
               "step_target": (mi["step_target"] if mi else None),
               "est_freed": (mi["est_freed"] if mi else None),
               "r1_freed": (mi["r1_freed"] if mi else None),
               "act_freed": (mi["act_freed"] if mi else None),
               "prune_rounds": (mi["prune_rounds"] if mi else 0),
               "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
               "align_score": C.model_align_score(student), "align_best": st.align_best(student),
               "lr": lr_eff(opt), "ft_used": ft_used, "no_imp": no_imp, "curr_best": curr_best,
               "is_best": lg_step == step}
        traj.append(rec)
        if on_step:
            on_step(rec)

        if stop:
            reason = stop
            break
        prev = m

    return {"model": last_good, "gflops": lg_gflops, "metric": lg_metric, "step": lg_step,
            "g0": g0, "metric_baseline": metric_baseline, "monitor_baseline": monitor_baseline,
            "floor_full": floor_full, "floor_monitor": gate_floor, "trajectory": traj,
            "reason": reason, "banned": sorted(st.banned), "student": student, "cache": cache,
            "batches": batches, "state": st}


def run_phase1(student, teacher, adapter, device, ctx, path, model_name, metric_fn=None,
               monitor_fn=None, batch_size=None, imp_batches=None, seed=0, batches=None, cache=None, **kw):
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)

    if not kw.get("loss_fn"):
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    if cache is None and not kw.get("loss_fn"):
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")

    if metric_fn is None:
        metric_fn, monitor_fn = agreement_metrics(teacher, adapter, device, ctx, path, seed=seed)
    return phase1_loop(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                       monitor_fn=monitor_fn, batches=batches, cache=cache,
                       imp_batches=imp_batches, seed=seed, **kw)


def ft_until(student, teacher, adapter, device, ctx, batches, cache, opt, gstep, warmup, gate_fn,
             max_epochs, patience, loss_fn=None, on_epoch=None, on_batch=None):
    best, no_imp, m, ep = -float("inf"), 0, None, 0
    for ep in range(1, int(max_epochs) + 1):
        kd, gstep = kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep,
                             warmup, loss_fn=loss_fn, on_batch=on_batch)
        m = float(gate_fn(student))
        if m > best + 1e-4:
            best, no_imp = m, 0
        else:
            no_imp += 1
        if on_epoch is not None:
            on_epoch(ep, kd, m, no_imp)
        if no_imp >= int(patience):
            break
    return gstep, m, ep


def phase2_ladder(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                  monitor_fn=None, out_dir=None, n_ckpt=None, g_min=None, banned=(),
                  step_frac=None, reinvest_frac=None, cooldown=None, batch_size=None, imp_batches=None,
                  seed=0, cache=None, batches=None, loss_fn=None,
                  on_step=None, on_batch=None, on_probe=None, max_steps_per_rung=None):
    n_ckpt = int(CFG.F2_CHECKPOINTS if n_ckpt is None else n_ckpt)
    max_steps_per_rung = int(CFG.F2_MAX_STEPS_PER_RUNG if max_steps_per_rung is None
                             else max_steps_per_rung)
    out_dir = out_dir or os.path.join(CFG.TMP_ROOT, "gui_job")
    os.makedirs(out_dir, exist_ok=True)
    prunable = set(ctx["prunable"])
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if cache is None and loss_fn is None:
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches or CFG.IMP_BATCHES]]
    gate_fn = monitor_fn if monitor_fn is not None else metric_fn

    g_start = gflops(student, adapter, device)
    if g_min is None:
        g_min, banned = C.probe_min_gflops(student, adapter, device, prunable, banned=banned,
                                           gflops_fn=lambda m: gflops(m, adapter, device),
                                           on_progress=on_probe)
    g_min = min(float(g_min), g_start)
    delta = (g_start - g_min) / max(n_ckpt, 1)
    targets = [g_start - delta * (i + 1) for i in range(n_ckpt - 1)] + [g_min]

    st = MorphState(prunable, g_start, step_frac or CFG.F2_PRUNE_STEP_FRAC,
                    reinvest_frac, cooldown)
    st.banned |= set(banned)
    opt, gstep, warmup = _new_prodigy(student), 0, len(batches)
    traj, ckpts = [], []
    step = 0

    def _emit(rec):
        traj.append(rec)
        if on_step:
            on_step(rec)

    _emit({"step": 0, "phase": "baseline", "rung": 0, "gflops": g_start,
           "params": A.count_params(student), "kd": None, "metric": None,
           "g_min": g_min, "target": None, "removed_ch": 0, "grown": [],
           "size_mb": A.count_params(student) * 4 / (1024 ** 2),
           "gflops_freed": 0.0, "gflops_reinvested": 0.0,
           "align_score": C.model_align_score(student), "align_best": st.align_best(student)})

    exhausted = None
    stopped = None
    rung_tol = 1.0 + 2 * CFG.PHASE2_PRUNE_SLACK

    def _hit(g, t):
        return g <= t * rung_tol + 1e-9

    for i, t in enumerate(targets):
        last_rung = (i == len(targets) - 1)
        for _ in range(max_steps_per_rung):
            g_now = gflops(student, adapter, device)
            if _hit(g_now, t):
                break
            step += 1
            student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev,
                                     loss_fn=loss_fn, grow=not last_rung,
                                     step_target=min(st.step_target, g_now - t))
            if mi["n_rem"] == 0 and not mi["grown"] and not mi["grow_protected"]:
                exhausted = "nema vise rezivih kandidata na {:.4f} GFLOPs".format(g_now)
                break

            low_yield = None
            if CFG.F2_MIN_YIELD is not None and not mi["cd_override"]:
                narrowed = (g_now - t) < st.step_target
                want, got = mi["step_target"], mi["act_freed"]
                if not narrowed and want > 0 and got < CFG.F2_MIN_YIELD * want:
                    low_yield = ("rez oslobodio {:.0%} trazenog (prag {:.0%}) na {:.4f} GFLOPs"
                                 .format(got / want, CFG.F2_MIN_YIELD, g_now))

            opt = _new_prodigy(student)
            g_step = gflops(student, adapter, device)
            _ep = {"n": 0}

            def _on_ep(ep, kd, m, no_imp):
                _ep["n"] = ep
                n_par = A.count_params(student)
                _emit({"step": step, "phase": "morph" if ep == 1 else "ft", "rung": i + 1,
                       "gflops": g_step, "params": n_par, "kd": kd,
                       "metric": m, "g_min": g_min, "target": t, "ft_epoch": ep, "no_imp": no_imp,
                       "removed_ch": mi["n_rem"] if ep == 1 else 0,
                       "grown": [(gi["layer"], gi["k"]) for gi in mi["grown"]] if ep == 1 else [],
                       "banned": len(st.banned), "size_mb": n_par * 4 / (1024 ** 2),
                       "step_target": mi["step_target"], "est_freed": mi["est_freed"],
                       "r1_freed": mi["r1_freed"], "act_freed": mi["act_freed"],
                       "prune_rounds": mi["prune_rounds"],
                       "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
                       "align_score": C.model_align_score(student), "align_best": st.align_best(student),
                       "lr": lr_eff(opt)})

            ft_max = CFG.F2_YIELD_FT_EPOCHS if low_yield else CFG.F2_FT_MAX_EPOCHS
            ft_pat = CFG.F2_YIELD_FT_EPOCHS if low_yield else CFG.F2_FT_PATIENCE
            gstep, _m, _n = ft_until(student, teacher, adapter, device, ctx, batches, cache, opt,
                                     gstep, warmup, gate_fn, ft_max, ft_pat,
                                     loss_fn=loss_fn, on_epoch=_on_ep, on_batch=on_batch)
            if low_yield:
                stopped = low_yield
                break
        g_ck = gflops(student, adapter, device)
        m_full = float(metric_fn(student)) if metric_fn is not None else None
        f = os.path.join(out_dir, "ckpt_{}.pt".format(i + 1))
        torch.save(student, f)
        ckpts.append({"i": i + 1, "path": f, "target": t, "gflops": g_ck,
                      "params": int(A.count_params(student)), "metric": m_full,
                      "reached": _hit(g_ck, t), "grow": not last_rung})
        if on_step:
            on_step({"step": step, "phase": "checkpoint", "rung": i + 1, "gflops": g_ck,
                     "params": int(A.count_params(student)), "metric": m_full, "target": t,
                     "g_min": g_min, "path": f, "reached": _hit(g_ck, t),
                     "size_mb": A.count_params(student) * 4 / (1024 ** 2),
                     "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
                     "align_score": C.model_align_score(student), "align_best": st.align_best(student),
                     "stopped": stopped})
        if exhausted or stopped:
            break

    man = os.path.join(out_dir, "ladder.json")
    json.dump({"g_start": g_start, "g_min": g_min, "delta": delta, "targets": targets,
               "rung_tol": rung_tol - 1.0, "min_yield": CFG.F2_MIN_YIELD,
               "checkpoints": ckpts, "exhausted": exhausted, "stopped": stopped},
              open(man, "w"), indent=1)
    return {"g_start": g_start, "g_min": g_min, "delta": delta, "targets": targets,
            "checkpoints": ckpts, "manifest": man, "trajectory": traj, "exhausted": exhausted,
            "stopped": stopped, "student": student, "banned": sorted(st.banned), "state": st}


def run_phase2(student, teacher, adapter, device, ctx, path, model_name, metric_fn=None,
               monitor_fn=None, batch_size=None, imp_batches=None, seed=0, batches=None, cache=None, **kw):
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if not kw.get("loss_fn"):
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    if cache is None and not kw.get("loss_fn"):
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")
    if metric_fn is None:
        metric_fn, monitor_fn = agreement_metrics(teacher, adapter, device, ctx, path, seed=seed)
    return phase2_ladder(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                         monitor_fn=monitor_fn, batches=batches, cache=cache,
                         imp_batches=imp_batches, seed=seed, **kw)


def dead_removal(student, adapter, device, ctx, batches, census_max=None):
    struct = {nm: True for nm in ctx["prunable"]}
    loader = [(b, None) for b in batches]
    return C.remove_dead_neardead(student, adapter, device, loader, struct, census_max=census_max)


def full_cycle(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, ft_steps=6, dead_ft_steps=8,
               batch_size=8, n_batches=None, imp_batches=3, seed=0, max_steps=None, on_step=None,
               dead=False, **kw):
    batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, n_batches, seed)

    if not kw.get("loss_fn"):
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    cache = None if kw.get("loss_fn") else precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")

    n_dead = n_lay = 0
    if dead:
        student, n_dead, n_lay, _bad = dead_removal(student, adapter, device, ctx, batches)
        if dead_ft_steps:
            prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, dead_ft_steps, loss_fn=kw.get("loss_fn"))

    if not kw.get("metric_fn"):
        kw["metric_fn"] = agreement_metrics(teacher, adapter, device, ctx, path)[0]

    res = morph_loop(student, teacher, adapter, device, ctx, path, model_name,
                     target_frac=target_frac, ft_steps=ft_steps, batches=batches, cache=cache,
                     imp_batches=imp_batches, max_steps=max_steps, on_step=on_step, **kw)
    res.update({"n_dead": n_dead, "n_dead_layers": n_lay})
    if res["best_model"] is None:
        res["best_model"] = res["student"]; res["best_gflops"] = res["final_gflops"]
    return res
