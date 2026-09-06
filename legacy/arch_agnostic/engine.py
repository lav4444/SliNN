import copy
import hashlib
import json
import math
import os
import sys

_MORPH = "/home/tomi/code/dipl/morphology"
_AA = "/home/tomi/code/dipl/arch_agnostic"
for d in (_MORPH, _AA):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import analysis as A                                         # noqa: E402
import compress as C                                         # noqa: E402
import config as CFG                                         # noqa: E402
import dataset as DS                                         # noqa: E402
import loss as L                                             # noqa: E402
from classify import weighted_leaves as _WL_AG

TMP_ROOT = os.path.join(_AA, "tmp")


def _ag_layer_table(model, adapter, device):
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


def install_sizing_shims():
    global _SHIMS_INSTALLED
    if _SHIMS_INSTALLED:
        return
    A.layer_table = _ag_layer_table
    A.weighted_leaves = _WL_AG
    C._forward_ok = _ag_forward_ok
    C._try_grow_layer = _ag_try_grow_layer
    _SHIMS_INSTALLED = True


install_sizing_shims()


def gflops(model, adapter, device):
    return A.gflops_total(model, adapter, device)


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


def _candidate_files(root, mode, split, cap):
    if mode == "image":
        return DS._media_files(root, DS._IMG, split, cap, skip_masks=True)
    if mode == "seq":
        return DS._media_files(root, DS._AUD, split, cap)
    return []


def to_device(batch, device):
    return [im.to(device) for im in batch]


def materialize_train_batches(path, adapter, device, split_plan, batch_size=8, n_batches=8, seed=0):
    import random
    mode = getattr(adapter, "_mode", "image")
    root = DS.detect_format(path).get("root", path)
    tr = (split_plan or {}).get("train") if isinstance(split_plan, dict) else split_plan
    split = None if (tr in (None, "AUTO")) else tr
    need = batch_size * n_batches
    samples, source = [], "fallback-random"
    try:
        if mode in ("image", "seq"):
            files = _candidate_files(root, mode, split, need * 3)
            if files:
                rng = random.Random(seed); rng.shuffle(files)
                files = (files * (need // len(files) + 1))[:need]
                dec = DS._decode_image if mode == "image" else DS._decode_audio
                samples = [dec(f, adapter._in_ch, adapter.imgsz) for f in files]
                source = "image-files" if mode == "image" else "audio-files"
        elif mode == "vector":
            X = DS._tabular_matrix(root, adapter._in_ch)
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
    return [b for b in batches if b], source


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def _to_half_cpu(sig):
    return {"feat": {k: v.detach().half().cpu() for k, v in sig["feat"].items()},
            "out": sig["out"].detach().half().cpu()}


def _to_dev_float(sig, device):
    return {"feat": {k: v.float().to(device) for k, v in sig["feat"].items()},
            "out": sig["out"].float().to(device)}


def _fingerprint(batches):
    h = hashlib.sha1()
    for b in batches:
        if not b:
            continue
        t = b[0].detach().float().flatten()
        h.update(str(tuple(b[0].shape)).encode())
        step = max(1, t.numel() // 64)
        h.update(str(round(float(t[::step].sum()), 3)).encode())
    return h.hexdigest()[:16]


class TeacherSigCache:
    def __init__(self, cache_dir, n, in_ram):
        self.dir = cache_dir; self.n = n; self.in_ram = in_ram
        self.ram = [None] * n if in_ram else None

    def _path(self, i):
        return os.path.join(self.dir, f"sig_{i}.pt")

    def has_all(self):
        return all(os.path.exists(self._path(i)) for i in range(self.n))

    def put(self, i, sig):
        h = _to_half_cpu(sig)
        torch.save(h, self._path(i))
        if self.in_ram:
            self.ram[i] = h

    def warm_ram(self):
        if self.in_ram:
            for i in range(self.n):
                if self.ram[i] is None:
                    self.ram[i] = torch.load(self._path(i), map_location="cpu")

    def get(self, i, device):
        h = self.ram[i] if (self.in_ram and self.ram[i] is not None) else torch.load(self._path(i), map_location="cpu")
        return _to_dev_float(h, device)


def precompute_teacher(teacher, adapter, batches, taps, model_name, split="train", in_ram=True, verbose=False):
    dev = next(teacher.parameters()).device
    n = len(batches)
    bs = max((len(b) for b in batches), default=0)
    cache_dir = os.path.join(TMP_ROOT, _safe(model_name), _safe(split))
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = {"model": model_name, "n_batches": n, "batch_size": bs,
            "taps": sorted(taps), "fingerprint": _fingerprint(batches)}
    cache = TeacherSigCache(cache_dir, n, in_ram)
    if os.path.exists(meta_path) and _load_json(meta_path) == meta and cache.has_all():
        cache.warm_ram()
        if verbose:
            print(f"  reuse teacher-cache (meta valjan): {cache_dir}")
        return cache
    os.makedirs(cache_dir, exist_ok=True)
    for f in os.listdir(cache_dir):
        if f.startswith("sig_") and f.endswith(".pt"):
            os.remove(os.path.join(cache_dir, f))
    for i, b in enumerate(batches):
        cache.put(i, L.teacher_signals(teacher, adapter, to_device(b, dev), taps))
    json.dump(meta, open(meta_path, "w"))
    if verbose:
        print(f"  precompute teacher-cache: {n} batcheva -> {cache_dir}")
    return cache


def _load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except BaseException:
        return None


def prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, steps, clip=5.0, loss_fn=None):
    import prodigyopt
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                             lr=1.0, d_coef=0.9, safeguard_warmup=True)
    last, nb = None, len(batches)
    for s in range(steps):
        i = s % nb
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


def morph_loop(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, step_frac=None, reinvest_frac=None, max_steps=20, ft_steps=6,
               batch_size=8, n_batches=8, imp_batches=3, seed=0, cache=None, batches=None,
               cooldown=None, on_step=None,
               metric_fn=None, metric_tol=0.90, metric_baseline=None, loss_fn=None):
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    prunable = set(ctx["prunable"])
    step_frac = step_frac or CFG.PHASE2_PRUNE_STEP_FRAC
    reinvest_frac = CFG.PHASE2_REINVEST_FRAC if reinvest_frac is None else reinvest_frac
    cooldown = CFG.PHASE2_CHURN_COOLDOWN if cooldown is None else cooldown
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, n_batches, seed)
    if cache is None and loss_fn is None:
        cache = precompute_teacher(teacher, adapter, batches, taps, model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches]]

    g0 = gflops(student, adapter, device)
    step_target = step_frac * g0
    banned = set()
    grown_at, pruned_at = {}, {}
    morph_idx = 0
    total_pruned = total_grown = 0.0
    best_model, best_gflops, best_step = None, float("inf"), 0
    if metric_fn is not None and metric_baseline is None:
        metric_baseline = metric_fn(student)
    metric_floor = (metric_tol * metric_baseline) if metric_fn is not None else None
    below = 0
    traj = [{"step": 0, "gflops": g0, "params": A.count_params(student), "kd": None,
             "removed_ch": 0, "grown": []}]
    for step in range(1, max_steps + 1):
        if (g0 - traj[-1]["gflops"]) >= target_frac * g0:
            break
        morph_idx += 1
        grow_protected = {l for l, k in grown_at.items() if morph_idx - k <= cooldown}
        prune_protected = {l for l, k in pruned_at.items() if morph_idx - k <= cooldown}
        elig_all = prunable - banned
        imp, gavg = L.kd_importance(student, teacher, adapter, imp_dev, taps, kd_mode, out_kind,
                                    prunable=elig_all, loss_fn=loss_fn)
        cost, flops_per, units = C.prune_costs(student, adapter, device, elig_all)
        info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in elig_all}

        g_before = gflops(student, adapter, device)
        prune_elig = elig_all - grow_protected
        cd_override, n_rem, pruned_names = False, 0, set()
        for _try in range(5):
            struct = {nm: True for nm in prune_elig}
            excl = banned if cd_override else (banned | grow_protected)
            plan, _r = C._select_prune_plan(struct, imp, cost, flops_per, units, info, step_target,
                                            CFG.PHASE2_PRUNE_LAYER_CAP, CFG.PHASE2_MIN_ALIVE_FRAC,
                                            CFG.PHASE2_MIN_ALIVE, exclude=excl)
            if not plan and grow_protected and not cd_override:
                cd_override = True; prune_elig = elig_all; continue
            if not plan:
                break
            student, n_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, device, plan)
            if bad:
                banned |= bad
            if n_rem > 0:
                pruned_names = set(plan.keys()) - bad
                break
        freed = g_before - gflops(student, adapter, device)
        total_pruned += max(freed, 0.0)
        for nm in pruned_names:
            pruned_at[nm] = morph_idx

        pool = reinvest_frac * total_pruned - total_grown
        grown_info = []
        if pool > 0:
            info_g = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in prunable}
            struct_g = {nm: (nm in prunable and nm not in pruned_names and nm not in prune_protected
                             and nm not in banned) for nm in prunable}
            g2, ginfos, spent = C._grow_decide(student, adapter, device, struct_g, gavg, flops_per, units, info_g, pool)
            if g2 is not None:
                student = g2; total_grown += max(spent, 0.0); grown_info = ginfos
                for gi in ginfos:
                    grown_at[gi["layer"]] = morph_idx

        kd = prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, ft_steps, loss_fn=loss_fn)
        cur_metric = metric_fn(student) if metric_fn is not None else None
        rec = {"step": step, "gflops": gflops(student, adapter, device), "params": A.count_params(student),
               "kd": kd, "metric": cur_metric, "removed_ch": n_rem,
               "grown": [(gi["layer"], gi["k"]) for gi in grown_info], "cd_override": cd_override, "banned": len(banned)}
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
        if n_rem == 0 and not grown_info and not grow_protected:
            break
    return {"g0": g0, "trajectory": traj, "final_gflops": traj[-1]["gflops"], "banned": sorted(banned),
            "grown_at": grown_at, "pruned_at": pruned_at, "student": student, "cache": cache,
            "best_model": best_model, "best_gflops": best_gflops, "best_step": best_step,
            "metric_baseline": metric_baseline}


def dead_removal(student, adapter, device, ctx, batches, census_max=None):
    struct = {nm: True for nm in ctx["prunable"]}
    loader = [(b, None) for b in batches]
    return C.remove_dead_neardead(student, adapter, device, loader, struct, census_max=census_max)


def full_cycle(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, ft_steps=6, dead_ft_steps=8,
               batch_size=8, n_batches=8, imp_batches=3, seed=0, max_steps=20, on_step=None,
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
        import metric as _M
        gate_inputs = [s for b in batches for s in b][:64]
        kind = ctx["out_kind"]
        kw["metric_fn"] = lambda mdl: _M.teacher_agreement(mdl, teacher, adapter, gate_inputs, device, kind=kind)["agreement"]
        kw.setdefault("metric_tol", 0.97)

    res = morph_loop(student, teacher, adapter, device, ctx, path, model_name,
                     target_frac=target_frac, ft_steps=ft_steps, batches=batches, cache=cache,
                     imp_batches=imp_batches, max_steps=max_steps, on_step=on_step, **kw)
    res.update({"n_dead": n_dead, "n_dead_layers": n_lay})
    if res["best_model"] is None:
        res["best_model"] = res["student"]; res["best_gflops"] = res["final_gflops"]
    return res
