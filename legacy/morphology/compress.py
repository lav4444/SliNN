
import copy
import json
import math
import os
import shutil
import time

import torch
import torch.nn as nn

import analysis as A
import config
from config import (TMP_ROOT, MODELS_DIR, VAL_CAP, EVAL_MAX, EVAL_BATCH, TRAIN_BATCH,
                    CENSUS_MAX, FT_PATIENCE, FT_MAX_EPOCHS, FT_METRICS, FT_RECOVERY_FRAC, DEV_DATA_SUBSET,
                    PHASE2_PRUNE_STEP_FRAC, PHASE2_PRUNE_LAYER_CAP, PHASE2_MIN_ALIVE_FRAC,
                    PHASE2_COST_FLOPS_W, PHASE2_PRUNE_PATIENCE, PHASE2_MAX_STEPS,
                    PHASE2_REINVEST_FRAC, PHASE2_GROW_DOM,
                    PHASE2_CHURN_COOLDOWN, PHASE2_GROW_MAX_LAYERS)

GB = 1024 ** 3


def _free_ram_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None


def _free_disk_bytes(path):
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free


def _half_bytes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel() * (2 if obj.is_floating_point() else obj.element_size())
    if isinstance(obj, dict):
        return sum(_half_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_half_bytes(v) for v in obj)
    return 0


def _to_half_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return (obj.half() if obj.is_floating_point() else obj).cpu()
    if isinstance(obj, dict):
        return {k: _to_half_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_half_cpu(v) for v in obj)
    return obj


def _to_dev_float(obj, device):
    if isinstance(obj, torch.Tensor):
        return (obj.float() if obj.is_floating_point() else obj).to(device)
    if isinstance(obj, dict):
        return {k: _to_dev_float(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dev_float(v, device) for v in obj)
    return obj


class TeacherCache:
    def __init__(self, cache_dir, n_batches, in_ram):
        self.dir = cache_dir; self.n = n_batches; self.in_ram = in_ram
        self.ram = [None] * n_batches if in_ram else None

    def _path(self, i):
        return os.path.join(self.dir, f"batch_{i}.pt")

    def has_all(self):
        return all(os.path.exists(self._path(i)) for i in range(self.n))

    def put(self, i, td):
        h = _to_half_cpu(td)
        torch.save(h, self._path(i))
        if self.in_ram:
            self.ram[i] = h

    def warm_ram(self):
        if self.in_ram:
            for i in range(self.n):
                if self.ram[i] is None:
                    self.ram[i] = torch.load(self._path(i), map_location="cpu")

    def get(self, i, device):
        td = self.ram[i] if (self.in_ram and self.ram[i] is not None) else torch.load(self._path(i), map_location="cpu")
        return _to_dev_float(td, device)


def gpu_status():
    if not torch.cuda.is_available():
        return {"available": False, "ok": False, "msg": "nema CUDA GPU-a (radit ce na CPU, sporo)"}
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return {"available": True, "name": torch.cuda.get_device_name(0), "total_gb": total / GB,
            "used_gb": used / GB, "free_gb": free / GB, "used_pct": used / total * 100.0,
            "ok": used < total * 0.5}


def teacher_mem_plan(teacher, adapter, loader, model_name, ram_frac=0.5, max_batches=None):
    sub = f"_dev{DEV_DATA_SUBSET}" if DEV_DATA_SUBSET else ""
    cache_dir = os.path.join(TMP_ROOT, model_name, f"train{sub}")
    meta_path = os.path.join(cache_dir, "meta.json")
    dev = next(teacher.parameters()).device
    n_batches = max_batches or len(loader)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    sample = None
    for imgs, _ in loader:
        sample = adapter.teacher_outputs(teacher, [im.to(dev) for im in imgs]); break
    vram = None
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        vram = {"total": total, "free": free, "batch_peak": torch.cuda.max_memory_allocated(dev)}
    comp = {k: _half_bytes(v) for k, v in sample.items() if k != "sizes"}
    per_batch = sum(comp.values()); total_b = per_batch * n_batches
    free_disk = _free_disk_bytes(cache_dir); free_ram = _free_ram_bytes()
    meta = {"model": model_name, "n_batches": n_batches, "batch_size": getattr(loader, "batch_size", None),
            "schema": sorted(comp.keys())}
    cached = (os.path.exists(meta_path) and json.load(open(meta_path)) == meta
              and TeacherCache(cache_dir, n_batches, False).has_all())
    return {"components": comp, "per_batch": per_batch, "total": total_b, "n_batches": n_batches,
            "free_disk": free_disk, "free_ram": free_ram, "fits_disk": total_b < free_disk * 0.9,
            "in_ram": bool(free_ram) and total_b < free_ram * ram_frac, "ram_frac": ram_frac,
            "cache_dir": cache_dir, "cached": bool(cached), "vram": vram}


def precompute_teacher(teacher, adapter, loader, model_name, ram_frac=0.5, max_batches=None, plan=None, verbose=False):
    plan = plan or teacher_mem_plan(teacher, adapter, loader, model_name, ram_frac, max_batches)
    cache_dir = plan["cache_dir"]; n_batches = plan["n_batches"]
    meta_path = os.path.join(cache_dir, "meta.json")
    dev = next(teacher.parameters()).device

    if verbose:
        print("\n=== TEACHER PRECOMPUTE — memorija (fp16) ===")
        print(f"  {'komponenta':<12}{'po batchu':>12}{'ukupno (x' + str(n_batches) + ')':>16}")
        for k, b in plan["components"].items():
            print(f"  {k:<12}{b/GB:>9.4f} GB{b*n_batches/GB:>13.4f} GB")
        print(f"  {'UKUPNO':<12}{plan['per_batch']/GB:>9.4f} GB{plan['total']/GB:>13.4f} GB")
        ram_str = f"{plan['free_ram']/GB:.1f} GB" if plan["free_ram"] else "?"
        print(f"  slobodno: disk {plan['free_disk']/GB:.1f} GB | RAM {ram_str}")
        print(f"  odluka: SVE -> disk ({cache_dir}); radni cache: "
              f"{'RAM-rezident' if plan['in_ram'] else 'disk-stream'}; stane na disk: {'DA' if plan['fits_disk'] else 'NE'}")
    if not plan["fits_disk"]:
        raise SystemExit("Nema dovoljno diska za teacher cache.")

    cache = TeacherCache(cache_dir, n_batches, plan["in_ram"])
    meta = {"model": model_name, "n_batches": n_batches, "batch_size": getattr(loader, "batch_size", None),
            "schema": sorted(plan["components"].keys())}
    if os.path.exists(meta_path) and json.load(open(meta_path)) == meta and cache.has_all():
        if verbose:
            print("  reuse: postojeci cache valjan -> preskacem precompute")
        cache.warm_ram()
        return cache

    os.makedirs(cache_dir, exist_ok=True)
    for f in os.listdir(cache_dir):
        if f.startswith("batch_") and f.endswith(".pt"):
            os.remove(os.path.join(cache_dir, f))
    t0 = time.time()
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        cache.put(i, adapter.teacher_outputs(teacher, [im.to(dev) for im in imgs]))
    json.dump(meta, open(meta_path, "w"))
    if verbose:
        print(f"  precompute: {n_batches} batcheva u {time.time()-t0:.0f}s -> {cache_dir}")
    return cache


def autobatch(model, adapter, device, free_frac=0.9, cap=64, cands=(1, 2, 4, 8, 16, 32, 64)):
    cands = [b for b in cands if b <= cap]
    if device.type != "cuda":
        return cands[0]
    import gc
    import prodigyopt
    ds = A._DetDataset("train"); nimg = len(ds)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rg = [p.requires_grad for p in model.parameters()]
    chosen = cands[0]
    try:
        for bs in cands:
            if bs > nimg:
                break
            torch.cuda.empty_cache(); gc.collect()
            free, _ = torch.cuda.mem_get_info()
            usable = free + torch.cuda.memory_allocated()
            try:
                imgs = [ds[i][0].to(device) for i in range(bs)]
                with torch.no_grad():
                    tgt = _to_dev_float(adapter.teacher_outputs(model, imgs), device)
                for p in model.parameters():
                    p.requires_grad_(True)
                opt = prodigyopt.Prodigy([p for p in model.parameters() if p.requires_grad], lr=1.0)
                model.train()
                torch.cuda.reset_peak_memory_stats(device)
                loss, _ = adapter.kd_loss(model, tgt, imgs)
                loss.backward(); opt.step()
                peak = torch.cuda.max_memory_allocated(device)
                del loss, tgt, imgs, opt
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


def _forward_ok(model, adapter, device):
    was = model.training
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward(model, [torch.rand(3, 320, 320, device=device)])
        return True
    except BaseException:
        return False
    finally:
        model.train(was)


def remove_dead_neardead(model, adapter, device, loader, struct, census_max=None, eps=1e-6, weak=0.01):
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)
    act, _ = A.activation_stats(model, adapter, device, loader, census_max, eps, weak)
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(model)}
    plan = []
    for leaf, st in act.items():
        if (struct is not None and not struct.get(leaf)) or leaf not in info:
            continue
        m = info[leaf][0]
        if getattr(m, "groups", 1) > 1:
            continue
        idx = sorted(set(st["dead_idx"]) | set(st["near_idx"]))
        if idx and len(idx) >= st["C"]:
            idx = idx[:st["C"] - 1]
        if idx:
            plan.append((leaf, idx))
    cur = model
    n_rem = n_lay = 0
    n_bad = 0
    bad = set()
    for leaf, idx in plan:
        trial = copy.deepcopy(cur)
        try:
            info_now = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(trial)}
            if leaf not in info_now:
                continue
            m, dim = info_now[leaf]
            fn = pconv if dim >= 3 else plin
            DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
            g = DG.get_pruning_group(m, fn, idxs=idx)
            if not DG.check_pruning_group(g):
                bad.add(leaf); continue
            g.prune()
            if not _forward_ok(trial, adapter, device):
                n_bad += 1; bad.add(leaf)
                continue
            cur = trial; n_rem += len(idx); n_lay += 1
        except BaseException:
            n_bad += 1; bad.add(leaf)
    if n_bad:
        print(f"[dead/near-dead] preskoceno {n_bad} slojeva (rez bi slomio model)")
    return cur, n_rem, n_lay, bad


def _random_val_loader(n=VAL_CAP, bs=4):
    import random
    ds = A._DetDataset("val")
    idx = list(range(len(ds)))
    if len(idx) > n:
        idx = random.sample(idx, n)
    sub = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2,
                                       collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]))


def _metric_label(metric):
    return "val_map50:95" if metric == "map" else f"val_{metric}"


def ft_recover(student, adapter, cache, train_loader, device, max_epochs=FT_MAX_EPOCHS, patience=FT_PATIENCE,
               val_cap=VAL_CAP, w_feat=1.0, w_rpn=1.0, grad_clip=5.0, on_epoch=None,
               metrics=None, recover_targets=None):
    import copy
    import prodigyopt
    metrics = list(metrics) if metrics else list(FT_METRICS)
    recover_targets = recover_targets or {}
    primary = "map" if "map" in metrics else metrics[0]
    for p in student.parameters():
        p.requires_grad_(True)
    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad], lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
    nb = min(cache.n, len(train_loader)); warmup = max(1, nb); gstep = 0
    best = {mm: -1.0 for mm in metrics}; no_imp = {mm: 0 for mm in metrics}
    best_comb = -1.0; best_state = None
    for ep in range(1, max_epochs + 1):
        student.train()
        racc = {}; cnt = 0; t0 = time.time()
        for i, (imgs, _) in enumerate(train_loader):
            if i >= nb:
                break
            imgs = [im.to(device) for im in imgs]
            td = cache.get(i, device)
            for g in opt.param_groups:
                g["lr"] = 1.0 * min(1.0, (gstep + 1) / warmup)
            opt.zero_grad(set_to_none=True)
            loss, info = adapter.kd_loss(student, td, imgs, w_feat=w_feat, w_rpn=w_rpn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], grad_clip)
            opt.step()
            for k, v in info.items():
                racc[k] = racc.get(k, 0.0) + v
            cnt += 1; gstep += 1
        m, _ = A.evaluate(student, adapter, _random_val_loader(val_cap), device)
        cur = {mm: m.get(mm, 0.0) for mm in metrics}
        vmap = m.get("map", 0.0); vmar = m.get("mar_100", 0.0)
        for mm in metrics:
            if cur[mm] > best[mm] + 1e-4:
                best[mm] = cur[mm]; no_imp[mm] = 0
            else:
                no_imp[mm] += 1
        ratios = [cur[mm] / recover_targets[mm] for mm in metrics if recover_targets.get(mm, 0) > 0]
        comb = sum(ratios) / len(ratios) if ratios else sum(cur.values()) / max(len(cur), 1)
        star = comb > best_comb + 1e-4
        if star:
            best_comb = comb; best_state = copy.deepcopy(student.state_dict())
        pg = opt.param_groups[0]
        lr_eff = pg.get("d", 1.0) * pg["lr"]
        opt_mark = lambda mm: "*" if mm in metrics else ""
        mstr = f"val_map50:95={vmap:.4f}{opt_mark('map')}  val_mar_100={vmar:.4f}{opt_mark('mar_100')}"
        kstr = " ".join(f"{k}={racc[k]/max(cnt,1):.3f}" for k in racc)
        print(f"  [FT ep{ep:2d}] {kstr} | {mstr} | lr(d·)={lr_eff:.2e} | {time.time()-t0:.0f}s{'  *best' if star else ''}")
        if on_epoch:
            ev = {"epoch": ep, "val_map": cur[primary], "vmap": vmap, "vmar": vmar, "lr": lr_eff}
            ev.update({k: racc[k] / max(cnt, 1) for k in racc})
            on_epoch(ev)
        if all(recover_targets.get(mm, 0) > 0 and cur[mm] >= recover_targets[mm] for mm in metrics):
            print("  >>> early-stop (recovery: " +
                  ", ".join(f"{_metric_label(mm)}={cur[mm]:.4f}>={recover_targets[mm]:.4f}" for mm in metrics) + ")")
            break
        if all(no_imp[mm] >= patience for mm in metrics):
            print(f"  >>> early-stop (patience {patience}: sve optimizirane metrike stagniraju)")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    return student, best_comb


def _model_name(spec):
    return A.model_name(spec)


def run_dead_ft(spec, device, on_event=None, eval_max=EVAL_MAX, val_cap=VAL_CAP, eval_batch=EVAL_BATCH,
                train_batch=TRAIN_BATCH, census_max=CENSUS_MAX, models_dir=MODELS_DIR, do_analysis=True,
                precompute_max=None, ft_max_epochs=FT_MAX_EPOCHS, metrics=None, final_report=True):
    name = _model_name(spec)
    metrics = list(metrics) if metrics else list(FT_METRICS)
    model = A.load_any(spec, device); adapter = A.pick_adapter(model)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _step = [0]
    def emit(phase, gflops, params, val, freed, reinvested=0.0, vmap=None, vmar=None):
        if on_event:
            on_event({"step": _step[0], "phase": phase, "gflops": gflops, "params": params,
                      "size_mb": params * 4 / (1024 ** 2), "val_metric": val,
                      "val_map": vmap, "val_mar": vmar,
                      "gflops_freed": freed, "gflops_reinvested": reinvested,
                      "align_score": model_align_score(model),
                      "align_best": best_align_score(model, struct, dead_frozen, config.ALIGN_M), "quant_score": None})
        _step[0] += 1

    if A.dev_subset_note():
        print("\n" + A.dev_subset_note())
    print(f"\n########## BASELINE ({name}, kind={adapter.kind}) ##########")
    rep0 = A.baseline_perf(spec, device, eval_max=eval_max, model=model, adapter=adapter)
    A.print_perf(rep0, tag="baseline (iz analize)")
    base_g = rep0["gflops"]
    primary = "map" if "map" in metrics else metrics[0]
    base = {mm: rep0["maps"].get("val", {}).get(mm, 0.0) for mm in metrics}
    targets = {mm: FT_RECOVERY_FRAC * base[mm] for mm in metrics}
    v0 = rep0["maps"].get("val", {})
    struct, _ = A.structural_flags(model, adapter, device)
    dead_frozen = set()
    emit("baseline", base_g, rep0["params"], base[primary], 0.0, vmap=v0.get("map", 0.0), vmar=v0.get("mar_100", 0.0))

    if do_analysis:
        print(f"\n########## ANALIZA ({name}) ##########")
        A.run_analysis(spec, device)

    model, n_rem, n_lay, _dead_bad = remove_dead_neardead(model, adapter, device, A.make_gt_loader("train", bs=4), struct, census_max)
    dead_frozen |= _dead_bad
    print(f"\n[dead/near-dead] maknuto {n_rem} kanala u {n_lay} slojeva")
    if n_rem == 0:
        print("[Faza 1] nema mrtvih/near-dead kanala -> preskačem rez i FT recovery (model nepromijenjen).")
        emit("final", base_g, rep0["params"], base[primary], 0.0, vmap=v0.get("map", 0.0), vmar=v0.get("mar_100", 0.0))
        return model, rep0

    train_loader = A.make_gt_loader("train", bs=train_batch)
    cache = precompute_teacher(teacher, adapter, train_loader, name, max_batches=precompute_max)
    g_dead = A.gflops_total(model, adapter, device); freed = base_g - g_dead
    md = A.evaluate(model, adapter, _random_val_loader(val_cap), device)[0]
    emit("dead", g_dead, A.count_params(model), md.get(primary, 0.0), freed,
         vmap=md.get("map", 0.0), vmar=md.get("mar_100", 0.0))

    opt_str = ", ".join(f"{_metric_label(mm)} cilj>={targets[mm]:.4f} (={FT_RECOVERY_FRAC:.0%}×{base[mm]:.4f})" for mm in metrics)
    print(f"\n########## FT RECOVERY (pure-KD) ## optimizira: {opt_str} ##########")
    def ft_cb(info):
        emit("ft", g_dead, A.count_params(model), info["val_map"], freed,
             vmap=info.get("vmap"), vmar=info.get("vmar"))
    ft_recover(model, adapter, cache, train_loader, device, val_cap=val_cap,
               max_epochs=ft_max_epochs, on_epoch=ft_cb, metrics=metrics, recover_targets=targets)

    rep1 = None
    if final_report:
        print(f"\n########## NAKON DEAD/NEAR-DEAD + FT ({name}) ##########")
        eval_loaders = {s: A.make_gt_loader(s, bs=eval_batch) for s in A.list_splits()}
        rep1 = A.perf_report(model, adapter, eval_loaders, device, eval_max)
        A.print_perf(rep1, tag="best-quality-compressed")
        v1 = rep1["maps"].get("val", {})
        emit("final", rep1["gflops"], rep1["params"], v1.get(primary, 0.0), base_g - rep1["gflops"],
             vmap=v1.get("map", 0.0), vmar=v1.get("mar_100", 0.0))
    else:
        print(f"\n[Faza 1 gotova] dead/near-dead + FT zavrseni; PERF se mjeri na KRAJU (nakon Faze 2) na best_quality modelu.")

    models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    out = os.path.join(models_dir, f"{name}_best_quality_compressed.pt")
    torch.save(model, out); print(f"\n[save] {out}")
    return model, rep1


def _kd_grad_importance(student, adapter, cache, train_loader, device, n_batches=4, w_feat=1.0, w_rpn=1.0):
    leaves = A.weighted_leaves(student)
    acc = {nm: torch.zeros(w.shape[0]) for nm, _, _, w in leaves}
    gacc = {nm: torch.zeros(w.shape[0], w.reshape(w.shape[0], -1).shape[1]) for nm, _, _, w in leaves}
    for p in student.parameters():
        p.requires_grad_(True)
    student.train()
    bn_snap = [(m, m.running_mean.clone(), m.running_var.clone(),
                None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
               for m in student.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None]
    nb = 0
    for i, (imgs, _) in enumerate(train_loader):
        if i >= min(n_batches, cache.n):
            break
        imgs = [im.to(device) for im in imgs]
        td = cache.get(i, device)
        for p in student.parameters():
            p.grad = None
        loss, _ = adapter.kd_loss(student, td, imgs, w_feat=w_feat, w_rpn=w_rpn)
        loss.backward()
        for nm, m, _, w in leaves:
            g = m.weight.grad
            if g is not None:
                gd = g.detach().cpu()
                acc[nm] += gd.abs().flatten(1).mean(1)
                gacc[nm] += gd.reshape(gd.shape[0], -1)
        nb += 1
    for p in student.parameters():
        p.grad = None
    for m, rm, rv, nbt in bn_snap:
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        if nbt is not None:
            m.num_batches_tracked.copy_(nbt)
    nb = max(nb, 1)
    return {nm: acc[nm] / nb for nm in acc}, {nm: gacc[nm] / nb for nm in gacc}


def coupled_unit_cost(model, adapter, device, prunable):
    import torch_pruning as tp
    rec = A.layer_table(model, adapter, device)
    gpf, ppf, units, in_ch = {}, {}, {}, {}
    for r in rec:
        gpf[r["name"]] = r["gflops"]; ppf[r["name"]] = r["params"]
        units[r["name"]] = max(int(r["units"]), 1)
        ish = r.get("in")
        in_ch[r["name"]] = max(int(ish[1]), 1) if ish and len(ish) >= 2 else units[r["name"]]
    id2name = {id(m): nm for nm, m, _, _ in A.weighted_leaves(model)}
    name2mod = {nm: m for nm, m, _, _ in A.weighted_leaves(model)}
    for p in model.parameters():
        p.requires_grad_(True)
    try:
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=adapter.tp_example(device))
    except Exception:
        DG = None
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    flops_per, params_per = {}, {}
    for nm in prunable:
        m = name2mod.get(nm)
        if m is None or nm not in gpf:
            continue
        own_f, own_p = gpf[nm] / units[nm], ppf[nm] / units[nm]
        if DG is None:
            flops_per[nm], params_per[nm] = own_f, own_p; continue
        fn = pconv if m.weight.dim() >= 3 else plin
        try:
            g = DG.get_pruning_group(m, fn, idxs=[0])
        except BaseException:
            flops_per[nm], params_per[nm] = own_f, own_p; continue
        fl = pr = 0.0
        for item in g:
            dep = getattr(item, "dep", None) or (item[0] if isinstance(item, (tuple, list)) else None)
            mm = getattr(getattr(dep, "target", None), "module", None)
            mnm = id2name.get(id(mm))
            if mnm is None or mnm not in gpf:
                continue
            hn = ""
            h = getattr(dep, "handler", None)
            if h is not None:
                hn = getattr(h, "__name__", "") or getattr(getattr(h, "__func__", None), "__name__", "") or str(h)
            if "in_channel" in hn:
                denom = in_ch[mnm]
            elif "out_channel" in hn:
                denom = units[mnm]
            else:
                denom = units[mnm] if mm is m else in_ch[mnm]
            fl += gpf[mnm] / max(denom, 1); pr += ppf[mnm] / max(denom, 1)
        flops_per[nm] = fl if fl > 0 else own_f
        params_per[nm] = pr if pr > 0 else own_p
    return flops_per, params_per, units


def prune_costs(model, adapter, device, prunable, cost_flops_w=PHASE2_COST_FLOPS_W):
    flops_per, params_per, units = coupled_unit_cost(model, adapter, device, prunable)
    tot_f = sum(flops_per.values()) + 1e-12
    tot_p = sum(params_per.values()) + 1e-12
    cost = {nm: cost_flops_w * (flops_per[nm] / tot_f) + (1.0 - cost_flops_w) * (params_per[nm] / tot_p)
            for nm in flops_per}
    return cost, flops_per, units


def align_factors(width, m=None, beta=None, p=None):
    m = config.ALIGN_M if m is None else m
    beta = config.ALIGN_BETA if beta is None else beta
    p = config.ALIGN_POW if p is None else p
    if beta <= 0 or m <= 1 or width <= 0:
        return 1.0, 1.0
    r = width % m
    g = (-width) % m
    ps = (1.0 - r / m) ** p if r else 0.0
    gs = (1.0 - g / m) ** p if g else 0.0
    return 1.0 - beta * ps, 1.0 + beta * gs


def model_align_score(model, m=None):
    m = config.ALIGN_M if m is None else m
    if m <= 1:
        return 1.0
    us = []
    for _, _, _, w in A.weighted_leaves(model):
        c = int(w.shape[0])
        if c > 0:
            us.append(c / (m * math.ceil(c / m)))
    return sum(us) / len(us) if us else 1.0


def best_align_score(model, struct, frozen, m=None):
    m = config.ALIGN_M if m is None else m
    if m <= 1:
        return 1.0
    us = []
    for nm, _, _, w in A.weighted_leaves(model):
        c = int(w.shape[0])
        if c <= 0:
            continue
        touchable = (struct is None or struct.get(nm)) and nm not in frozen
        us.append(1.0 if touchable else c / (m * math.ceil(c / m)))
    return sum(us) / len(us) if us else 1.0


def prune_candidates(imp, cost, info, struct):
    cand = []
    for nm, v in imp.items():
        if struct is not None and not struct.get(nm):
            continue
        if nm not in cost or nm not in info:
            continue
        if getattr(info[nm][0], "groups", 1) > 1:
            continue
        pf, _ = align_factors(info[nm][0].weight.shape[0])
        base = pf / (cost[nm] + 1e-12)
        v = v.float()
        for i in range(v.numel()):
            cand.append((float(v[i]) * base, nm, i))
    cand.sort(key=lambda x: x[0])
    return cand


def grow_potential(gavg):
    out = {}
    for nm, G in gavg.items():
        M = G.reshape(G.shape[0], -1).float()
        try:
            out[nm] = torch.linalg.svdvals(M)
        except Exception:
            out[nm] = torch.zeros(min(M.shape))
    return out


def grow_candidates(sigma, flops_per, struct, units):
    cand = []
    for nm, s in sigma.items():
        if struct is not None and not struct.get(nm):
            continue
        if nm not in flops_per or flops_per[nm] <= 0 or s is None or len(s) == 0:
            continue
        smax = float(s[0])
        if smax > 0:
            _, gf = align_factors(units.get(nm, 0))
            cand.append(( (smax / flops_per[nm]) * gf, nm, smax))
    cand.sort(key=lambda x: -x[0])
    return cand


def _align_prune_score(imp_val, cost_nm, alive):
    pf, _ = align_factors(alive)
    return imp_val * pf / (cost_nm + 1e-12)


def _select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                       layer_cap, min_alive_frac, min_alive, exclude=()):
    import heapq
    order, ptr, alive, floor, quota, taken = {}, {}, {}, {}, {}, {}
    heap = []
    for nm, v in imp.items():
        if struct is not None and not struct.get(nm):
            continue
        if nm not in cost or nm not in info or nm in exclude:
            continue
        if getattr(info[nm][0], "groups", 1) > 1:
            continue
        vf = v.float()
        C = units.get(nm, vf.numel())
        order[nm] = torch.argsort(vf).tolist()
        ptr[nm] = 0; alive[nm] = C; taken[nm] = 0
        floor[nm] = max(min_alive, math.ceil(min_alive_frac * C))
        quota[nm] = max(0, math.floor(layer_cap * C))
        if alive[nm] > floor[nm] and quota[nm] > 0:
            heapq.heappush(heap, (_align_prune_score(float(vf[order[nm][0]]), cost[nm], alive[nm]), nm))
    removed = 0.0; sel = {}
    while heap and removed < target_gflops:
        _s, nm = heapq.heappop(heap)
        ch = order[nm][ptr[nm]]
        sel.setdefault(nm, []).append(ch)
        alive[nm] -= 1; taken[nm] += 1; ptr[nm] += 1; removed += flops_per[nm]
        if alive[nm] > floor[nm] and taken[nm] < quota[nm] and ptr[nm] < len(order[nm]):
            heapq.heappush(heap, (_align_prune_score(float(imp[nm][order[nm][ptr[nm]]]), cost[nm], alive[nm]), nm))
    return {nm: sorted(idx) for nm, idx in sel.items() if idx}, removed


def _apply_prune_plan(model, adapter, device, plan):
    import torch_pruning as tp
    for p in model.parameters():
        p.requires_grad_(True)
    pconv = getattr(tp, "prune_conv_out_channels", None) or tp.function.prune_conv_out_channels
    plin = getattr(tp, "prune_linear_out_channels", None) or tp.function.prune_linear_out_channels
    cur = model; n_rem = n_lay = n_bad = 0; bad = set()
    for leaf, idx in plan.items():
        trial = copy.deepcopy(cur)
        try:
            info_now = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(trial)}
            if leaf not in info_now:
                continue
            m, dim = info_now[leaf]
            C = m.weight.shape[0]
            idx2 = [j for j in idx if j < C]
            if not idx2 or len(idx2) >= C:
                continue
            fn = pconv if dim >= 3 else plin
            DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
            g = DG.get_pruning_group(m, fn, idxs=idx2)
            if not DG.check_pruning_group(g):
                bad.add(leaf); continue
            g.prune()
            if not _forward_ok(trial, adapter, device):
                n_bad += 1; bad.add(leaf); continue
            cur = trial; n_rem += len(idx2); n_lay += 1
        except BaseException:
            n_bad += 1; bad.add(leaf)
    return cur, n_rem, n_lay, n_bad, bad


def _kd_epoch(student, adapter, cache, train_loader, device, opt, gstep, warmup, w_feat=1.0, w_rpn=1.0, grad_clip=5.0):
    student.train()
    racc = {}; cnt = 0; nb = min(cache.n, len(train_loader))
    for i, (imgs, _) in enumerate(train_loader):
        if i >= nb:
            break
        imgs = [im.to(device) for im in imgs]
        td = cache.get(i, device)
        for g in opt.param_groups:
            g["lr"] = 1.0 * min(1.0, (gstep + 1) / max(warmup, 1))
        opt.zero_grad(set_to_none=True)
        loss, info = adapter.kd_loss(student, td, imgs, w_feat=w_feat, w_rpn=w_rpn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], grad_clip)
        opt.step()
        for k, v in info.items():
            racc[k] = racc.get(k, 0.0) + v
        cnt += 1; gstep += 1
    return {k: racc[k] / max(cnt, 1) for k in racc}, gstep


def run_morph(spec, device, on_event=None, eval_max=EVAL_MAX, val_cap=VAL_CAP, train_batch=TRAIN_BATCH,
              eval_batch=EVAL_BATCH, models_dir=MODELS_DIR, metrics=None, start_model=None,
              max_steps=PHASE2_MAX_STEPS, precompute_max=None):
    import prodigyopt
    name = _model_name(spec)
    metrics = list(metrics) if metrics else list(FT_METRICS)
    primary = "map" if "map" in metrics else metrics[0]

    teacher = A.load_any(spec, device); adapter = A.pick_adapter(teacher)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    if A.dev_subset_note():
        print("\n" + A.dev_subset_note())
    print(f"\n########## FAZA 2 — KONTINUIRANI PRUNE ({name}, kind={adapter.kind}) ##########")
    rep0 = A.baseline_perf(spec, device, eval_max=eval_max, model=teacher, adapter=adapter)
    base = {mm: rep0["maps"].get("val", {}).get(mm, 0.0) for mm in metrics}
    tol_line = {mm: FT_RECOVERY_FRAC * base[mm] for mm in metrics}
    base_gflops = rep0["gflops"]
    target_gflops = PHASE2_PRUNE_STEP_FRAC * base_gflops
    print("  tol (vs original): " + ", ".join(f"{_metric_label(mm)}>={tol_line[mm]:.4f} (={FT_RECOVERY_FRAC:.0%}×{base[mm]:.4f})" for mm in metrics))

    models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
    if start_model is not None:
        student = start_model
    else:
        bq = os.path.join(models_dir, f"{name}_best_quality_compressed.pt")
        if os.path.exists(bq):
            student = torch.load(bq, map_location=device, weights_only=False); print(f"  start = {bq}")
        else:
            student = A.load_any(spec, device); print("  start = ORIGINAL (nema Faza-1 best_quality na disku)")
    student.to(device)

    train_loader = A.make_gt_loader("train", bs=train_batch)
    cache = precompute_teacher(teacher, adapter, train_loader, name, max_batches=precompute_max)
    warmup = max(1, min(cache.n, len(train_loader)))

    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad], lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
    gstep = 0
    _step = [0]

    def emit(phase, gf, params, m, freed=0.0, reinvested=0.0):
        if on_event:
            on_event({"step": _step[0], "phase": phase, "gflops": gf, "params": params,
                      "size_mb": params * 4 / (1024 ** 2), "val_metric": m.get(primary, 0.0),
                      "val_map": m.get("map", 0.0), "val_mar": m.get("mar_100", 0.0),
                      "gflops_freed": freed, "gflops_reinvested": reinvested,
                      "align_score": model_align_score(student),
                      "align_best": best_align_score(student, struct0, unsafe, config.ALIGN_M), "quant_score": None})
        _step[0] += 1

    def measure():
        mm, _ = A.evaluate(student, adapter, _random_val_loader(val_cap), device)
        return {k: mm.get(k, 0.0) for k in ("map", "mar_100")}

    g0 = A.gflops_total(student, adapter, device)
    out_path = os.path.join(models_dir, f"{name}_best_quality_compressed.pt")
    prune_patience = 0
    unsafe = set()
    struct0, _ = A.structural_flags(student, adapter, device)
    _w = lambda mdl: {nm: int(w.shape[0]) for nm, _, _, w in A.weighted_leaves(mdl)}
    start_w = _w(student)
    pruned_tot, grown_tot = {}, {}
    best_snap = {"p": {}, "g": {}}
    morph_idx = 0
    grown_at = {}
    pruned_at = {}
    total_pruned = total_grown = 0.0
    best = {"q": -1.0, "g": float("inf"), "in_tol": False, "state": None, "m": None}

    def _qscore(mm):
        rs = [mm.get(x, 0.0) / tol_line[x] for x in metrics if tol_line[x] > 0]
        return sum(rs) / len(rs) if rs else 0.0

    def _in_tol(mm):
        return all(mm.get(x, 0.0) >= tol_line[x] for x in metrics)

    def _lr_eff():
        pg = opt.param_groups[0]; return pg.get("d", 1.0) * pg["lr"]

    def _consider_best(mm, gf):
        q, it = _qscore(mm), _in_tol(mm)
        if best["state"] is None:
            better = True
        elif it != best["in_tol"]:
            better = it
        elif it:
            better = gf < best["g"] - 1e-6 or (abs(gf - best["g"]) <= 1e-6 and q > best["q"] + 1e-9)
        else:
            better = q > best["q"] + 1e-9 or (abs(q - best["q"]) < 1e-9 and gf < best["g"] - 1e-6)
        if better:
            best.update(q=q, g=gf, in_tol=it, state=copy.deepcopy(student), m=dict(mm))
            best_snap["p"], best_snap["g"] = dict(pruned_tot), dict(grown_tot)
            torch.save(best["state"], out_path)
        return better

    m = measure()
    _consider_best(m, g0)
    emit("morph", g0, A.count_params(student), m, 0.0, 0.0)

    for step in range(1, max_steps + 1):
        gf = A.gflops_total(student, adapter, device)
        in_tol = _in_tol(m)
        rez_line = grow_line = None

        if not in_tol:
            prune_patience += 1
            if prune_patience >= PHASE2_PRUNE_PATIENCE:
                print(f"  >>> KRAJ: metrika {PHASE2_PRUNE_PATIENCE} epoha ispod tolerancije nakon reza, nema oporavka.")
                break
            racc, gstep = _kd_epoch(student, adapter, cache, train_loader, device, opt, gstep, warmup)
            phase_label = f"FT-RECOVERY {prune_patience}/{PHASE2_PRUNE_PATIENCE}"
        else:
            prune_patience = 0
            morph_idx += 1
            w_pre = _w(student)
            grow_protected = {l for l, k in grown_at.items() if morph_idx - k <= PHASE2_CHURN_COOLDOWN}
            prune_protected = {l for l, k in pruned_at.items() if morph_idx - k <= PHASE2_CHURN_COOLDOWN}
            struct, prunable, imp, gavg, cost, flops_per, units, info = _grow_ctx(student, adapter, cache, train_loader, device)
            n_rem = n_lay = n_bad = 0; plan = None; cd_override = False
            while True:
                plan, est = _select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                                               PHASE2_PRUNE_LAYER_CAP, PHASE2_MIN_ALIVE_FRAC, config.PHASE2_MIN_ALIVE,
                                               exclude=unsafe | grow_protected)
                if not plan and grow_protected:
                    plan, est = _select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                                                   PHASE2_PRUNE_LAYER_CAP, PHASE2_MIN_ALIVE_FRAC, config.PHASE2_MIN_ALIVE,
                                                   exclude=unsafe)
                    cd_override = bool(plan)
                if not plan:
                    break
                student, n_rem, n_lay, n_bad, bad = _apply_prune_plan(student, adapter, device, plan)
                unsafe |= bad
                if n_rem > 0 or not bad:
                    break
            if n_rem == 0:
                print("  >>> KRAJ: nema vise rezivih kandidata (off-limits/unsafe/floor/cap odbili sve)."); break
            opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad], lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
            gf2 = A.gflops_total(student, adapter, device); total_pruned += max(0.0, gf - gf2); pruned_names = set(plan)
            for nm in pruned_names:
                pruned_at[nm] = morph_idx
            w_mid = _w(student)
            pool = max(0.0, PHASE2_REINVEST_FRAC * total_pruned - total_grown)
            rez_line = (f"[REZ ep{step:2d}] -{n_rem} kanala/{n_lay} slojeva{(', %d odbijeno' % n_bad) if n_bad else ''}"
                        f"{' [cooldown override: rez nuzan]' if cd_override else ''} | "
                        f"GFLOPs {gf:.3f}->{gf2:.3f} ({gf2/base_gflops*100:.1f}% orig.) | reinvest pool {pool:.3f}")
            if pool > 0:
                info2 = {nm: (mod, w.dim()) for nm, mod, _, w in A.weighted_leaves(student)}
                grow_flops = {kk: vv for kk, vv in flops_per.items()
                              if kk not in pruned_names and kk not in prune_protected
                              and kk not in grow_protected and kk in info2}
                grown, ginfos, spent = _grow_decide(student, adapter, device, struct, gavg, grow_flops, units, info2, pool)
                if grown is not None:
                    student = grown; total_grown += spent
                    for gi in ginfos:
                        grown_at[gi["layer"]] = morph_idx
                    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad], lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
                    detail = ", ".join(f"+{gi['k']} {gi['layer']} (imao {gi['had']}, {gi['dom']:.1f}×)" for gi in ginfos)
                    grow_line = (f"[GROW ep{step:2d}] {len(ginfos)} sloj(a): {detail} | "
                                 f"+{spent:.3f} GFLOPs {gf2:.3f}->{A.gflops_total(student, adapter, device):.3f}")
            w_post = _w(student)
            for nm, pw in w_pre.items():
                dp = pw - w_mid.get(nm, pw)
                if dp > 0:
                    pruned_tot[nm] = pruned_tot.get(nm, 0) + dp
            for nm, mw in w_mid.items():
                dg = w_post.get(nm, mw) - mw
                if dg > 0:
                    grown_tot[nm] = grown_tot.get(nm, 0) + dg
            racc, gstep = _kd_epoch(student, adapter, cache, train_loader, device, opt, gstep, warmup)
            phase_label = "MORPH"

        m = measure()
        gf = A.gflops_total(student, adapter, device); params = A.count_params(student)
        isbest = _consider_best(m, gf)
        emit("morph", gf, params, m, base_gflops - gf, total_grown)
        opt_mark = lambda mm: "*" if mm in metrics else ""
        mstr = (f"val_map50:95={m.get('map',0.0):.4f}{opt_mark('map')}  "
                f"val_mar_100={m.get('mar_100',0.0):.4f}{opt_mark('mar_100')}")
        kstr = " ".join(f"{kk}={racc[kk]:.3f}" for kk in racc)
        print(f"[ep{step:2d}] [{phase_label}] {kstr} | {mstr} | lr(d·)={_lr_eff():.2e} | GFLOPs={gf:.3f}{'  *best' if isbest else ''}")
        if rez_line:
            print(f"    {rez_line}")
        if grow_line:
            print(f"    {grow_line}")
        print()
    else:
        print(f"  >>> KRAJ: dosegnut max_steps={max_steps}.")

    best_state = best["state"]
    eval_loaders = {s: A.make_gt_loader(s, bs=eval_batch) for s in A.list_splits()}
    status = "U TOLERANCIJI" if best["in_tol"] else "IZVAN tolerancije (najbolja postignuta kvaliteta)"
    print(f"\n########## FAZA 2 GOTOVA ({name}) — best_quality_compressed: {status} ##########")
    rep1 = A.perf_report(best_state, adapter, eval_loaders, device, eval_max)
    A.print_perf(rep1, tag="best-quality-compressed (Faza 1+2)")
    print(f"  GFLOPs: pocetak Faze 2 {g0:.3f} ({g0/base_gflops*100:.1f}%) -> best {rep1['gflops']:.3f} ({rep1['gflops']/base_gflops*100:.1f}% originala)")
    end_w = _w(best_state)
    changed = sorted({*best_snap["p"], *best_snap["g"]}, key=lambda n: -(best_snap["p"].get(n, 0) + best_snap["g"].get(n, 0)))
    if changed:
        print(f"\n  === Arhitekturne izmjene po sloju (do isporucenog best modela; GROSS — svaka promjena se broji, ne neto) ===")
        print(f"  {'sloj':<42}{'start':>6}{'pruned':>9}{'grown':>8}{'end':>7}")
        for nm in changed:
            p, g = best_snap["p"].get(nm, 0), best_snap["g"].get(nm, 0)
            print(f"  {nm:<42}{start_w.get(nm, 0):>6}{('-%d' % p) if p else '·':>9}{('+%d' % g) if g else '·':>8}{end_w.get(nm, 0):>7}")
    v1 = rep1["maps"].get("val", {})
    emit("final", rep1["gflops"], rep1["params"], {"map": v1.get("map", 0.0), "mar_100": v1.get("mar_100", 0.0)},
         base_gflops - rep1["gflops"], total_grown)
    torch.save(best_state, out_path); print(f"\n[save] {out_path}")
    return best_state, rep1


def _max_abs_diff(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            return 0.0
        if not a.is_floating_point():
            return 0.0 if torch.equal(a, b.to(a.dtype)) else float("inf")
        return float((a.float() - b.float()).abs().max())
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return float("inf")
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return float("inf")
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def _widen_out(mod, k, init_w):
    w = mod.weight.data
    mod.weight = nn.Parameter(torch.cat([w, init_w.to(w.device, w.dtype)], dim=0))
    if mod.bias is not None:
        mod.bias = nn.Parameter(torch.cat([mod.bias.data, torch.zeros(k, device=w.device, dtype=w.dtype)]))
    if hasattr(mod, "out_channels"):
        mod.out_channels += k
    if hasattr(mod, "out_features"):
        mod.out_features += k


def _widen_bn(bn, k):
    dev, dt = bn.weight.device, bn.weight.dtype
    bn.weight = nn.Parameter(torch.cat([bn.weight.data, torch.ones(k, device=dev, dtype=dt)]))
    bn.bias = nn.Parameter(torch.cat([bn.bias.data, torch.zeros(k, device=dev, dtype=dt)]))
    if bn.running_mean is not None:
        bn.running_mean = torch.cat([bn.running_mean, torch.zeros(k, device=dev, dtype=bn.running_mean.dtype)])
        bn.running_var = torch.cat([bn.running_var, torch.ones(k, device=dev, dtype=bn.running_var.dtype)])
    bn.num_features += k


def _widen_frozen_bn(fbn, k):
    for nm_, val in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
        buf = getattr(fbn, nm_, None)
        if isinstance(buf, torch.Tensor):
            ext = torch.full((k,), val, device=buf.device, dtype=buf.dtype)
            fbn.register_buffer(nm_, torch.cat([buf, ext]))


def _widen_depthwise(dw, k):
    w = dw.weight.data
    dw.weight = nn.Parameter(torch.cat([w, w.new_zeros(k, *w.shape[1:])], dim=0))
    if dw.bias is not None:
        dw.bias = nn.Parameter(torch.cat([dw.bias.data, torch.zeros(k, device=w.device, dtype=w.dtype)]))
    dw.groups += k; dw.in_channels += k; dw.out_channels += k


def _is_depthwise(m):
    return isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels == m.out_channels


def _insert_in_zeros(mod, k, positions):
    for pos in sorted(positions, reverse=True):
        w = mod.weight.data
        zshape = list(w.shape); zshape[1] = k
        z = torch.zeros(*zshape, device=w.device, dtype=w.dtype)
        mod.weight = nn.Parameter(torch.cat([w[:, :pos], z, w[:, pos:]], dim=1))
    tot = k * len(positions)
    if hasattr(mod, "in_channels"):
        mod.in_channels += tot
    if hasattr(mod, "in_features"):
        mod.in_features += tot


def _try_grow_layer(model, adapter, device, name, k, init_filters=None):
    import torch_pruning as tp
    sz = getattr(adapter, "imgsz", 320)
    ref_imgs = [torch.rand(3, sz, sz, device=device)]
    try:
        with torch.no_grad():
            ref_out = adapter.teacher_outputs(model, ref_imgs)
    except BaseException:
        return None
    trial = copy.deepcopy(model)
    leaves = {nm: (mm, w.dim()) for nm, mm, _, w in A.weighted_leaves(trial)}
    if name not in leaves:
        return None
    L, dim = leaves[name]
    if getattr(L, "groups", 1) > 1:
        return None
    old_L = L.weight.shape[0]
    try:
        for p in trial.parameters():
            p.requires_grad_(True)
        fn = (tp.function.prune_conv_out_channels if dim >= 3 else tp.function.prune_linear_out_channels)
        DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
        group = DG.get_pruning_group(L, fn, idxs=[0])
        out_mods, bns, fbns, dws, cons = [], [], [], [], {}
        for dep, idxs in group:
            tgt = getattr(getattr(dep, "target", None), "module", None)
            if tgt is None:
                continue
            hn = getattr(dep.handler, "__name__", type(dep.handler).__name__).lower()
            if isinstance(tgt, nn.modules.batchnorm._BatchNorm):
                bns.append(tgt)
            elif type(tgt).__name__ == "FrozenBatchNorm2d" or (hasattr(tgt, "running_mean") and hasattr(tgt, "weight")
                                                               and not isinstance(tgt, (nn.Conv2d, nn.Linear))):
                fbns.append(tgt)
            elif _is_depthwise(tgt):
                dws.append(tgt)
            elif isinstance(tgt, nn.Conv2d) and tgt.groups > 1:
                return None
            elif "in_channel" in hn or "in_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Linear)):
                    cons.setdefault(tgt, []).append(int(min(idxs)))
            elif "out_channel" in hn or "out_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Linear)):
                    out_mods.append(tgt)
        if L not in out_mods:
            out_mods.append(L)
        seen = set()
        for mod in out_mods:
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if mod is L and init_filters is not None:
                init = init_filters
            else:
                wabs = mod.weight.detach().abs().flatten(1).mean(1)
                order = torch.argsort(wabs, descending=True)
                idx = [int(order[i % len(order)]) for i in range(k)]
                cl = mod.weight.detach()[idx].clone()
                init = cl + torch.randn_like(cl) * 0.02 * cl.abs().mean().clamp(min=1e-6)
            _widen_out(mod, k, init)
        for dw in {id(d): d for d in dws}.values():
            _widen_depthwise(dw, k)
        for bn in {id(b): b for b in bns}.values():
            _widen_bn(bn, k)
        for fb in {id(f): f for f in fbns}.values():
            _widen_frozen_bn(fb, k)
        for cmod, offs in cons.items():
            _insert_in_zeros(cmod, k, [o + old_L for o in offs])
        if not _forward_ok(trial, adapter, device):
            return None
        with torch.no_grad():
            after = adapter.teacher_outputs(trial, ref_imgs)
        if _max_abs_diff(ref_out, after) >= 1e-3:
            return None
        return trial
    except BaseException:
        return None


def _grow_ctx(student, adapter, cache, train_loader, device):
    struct, _ = A.structural_flags(student, adapter, device)
    prunable = [nm for nm, _, _, _ in A.weighted_leaves(student) if struct is None or struct.get(nm)]
    imp, gavg = _kd_grad_importance(student, adapter, cache, train_loader, device)
    cost, flops_per, units = prune_costs(student, adapter, device, prunable, PHASE2_COST_FLOPS_W)
    info = {nm: (mod, w.dim()) for nm, mod, _, w in A.weighted_leaves(student)}
    return struct, prunable, imp, gavg, cost, flops_per, units, info


def _select_grow_plan(sigma, flops_per, units, struct, pool_gflops):
    import heapq, statistics
    cands = [nm for nm in sigma if (struct is None or struct.get(nm)) and flops_per.get(nm, 0.0) > 0
             and len(sigma[nm]) > 0 and float(sigma[nm][0]) > 0]
    if not cands:
        return {}, 0.0, 1e-12
    def _gscore(nm, w, p):
        _, gf = align_factors(w)
        return float(sigma[nm][p]) / flops_per[nm] * gf
    med = statistics.median(sorted(_gscore(nm, units.get(nm, 1), 0) for nm in cands)) + 1e-12
    thr = PHASE2_GROW_DOM * med
    ptr, width, cap, heap = {}, {}, {}, []
    for nm in cands:
        ptr[nm] = 0; width[nm] = units.get(nm, 1)
        cap[nm] = config.ALIGN_M
        s = _gscore(nm, width[nm], 0)
        if cap[nm] >= 1 and s >= thr:
            heapq.heappush(heap, (-s, nm))
    remaining = pool_gflops; plan = {}; grown = set()
    while heap:
        neg, nm = heapq.heappop(heap)
        if -neg < thr:
            break
        if nm not in grown and len(grown) >= PHASE2_GROW_MAX_LAYERS:
            continue
        if flops_per[nm] > remaining:
            break
        plan[nm] = plan.get(nm, 0) + 1; grown.add(nm)
        ptr[nm] += 1; width[nm] += 1; remaining -= flops_per[nm]
        if plan[nm] < cap[nm] and ptr[nm] < len(sigma[nm]):
            s2 = _gscore(nm, width[nm], ptr[nm])
            if s2 >= thr:
                heapq.heappush(heap, (-s2, nm))
    return plan, pool_gflops - remaining, med


def _grow_decide(student, adapter, device, struct, gavg, flops_per, units, info, pool_gflops):
    if pool_gflops <= 0:
        return None, [], 0.0
    def _fresh(nm, info_now):
        if nm not in info_now:
            return False
        W = info_now[nm][0].weight
        return (gavg[nm].shape[0] == W.shape[0]
                and gavg[nm].reshape(gavg[nm].shape[0], -1).shape[1] == W.reshape(W.shape[0], -1).shape[1])
    growable = {nm: gavg[nm] for nm in gavg
                if (struct is None or struct.get(nm)) and flops_per.get(nm, 0.0) > 0 and _fresh(nm, info)}
    sigma = grow_potential(growable)
    plan, _est, med = _select_grow_plan(sigma, flops_per, units, struct, pool_gflops)
    if not plan:
        return None, [], 0.0
    infos = []; total_spent = 0.0
    for nm, k in plan.items():
        info_now = {nm2: (mod, w.dim()) for nm2, mod, _, w in A.weighted_leaves(student)}
        if not _fresh(nm, info_now):
            continue
        C = info_now[nm][0].weight.shape[0]
        G = gavg[nm].reshape(gavg[nm].shape[0], -1).float()
        try:
            _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        except Exception:
            continue
        kk = min(k, Vh.shape[0])
        if kk < 1:
            continue
        W = info_now[nm][0].weight.detach()
        fnorm = float(W.reshape(W.shape[0], -1).norm(dim=1).mean().clamp(min=1e-6))
        init = (Vh[:kk] * fnorm).reshape((kk,) + tuple(W.shape[1:])).to(W.device, W.dtype)
        g_before = A.gflops_total(student, adapter, device)
        grown = _try_grow_layer(student, adapter, device, nm, kk, init)
        if grown is None:
            continue
        spent = A.gflops_total(grown, adapter, device) - g_before
        student = grown; total_spent += spent
        _, gf0 = align_factors(units.get(nm, C))
        infos.append({"layer": nm, "k": kk, "had": C, "dom": (float(sigma[nm][0]) / flops_per[nm] * gf0) / med})
    if not infos:
        return None, [], 0.0
    return student, infos, total_spent
