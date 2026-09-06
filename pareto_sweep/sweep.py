
import csv
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MORPH = os.path.join(os.path.dirname(_HERE), "morphology")
sys.path.insert(0, _MORPH)

import config
import analysis as A
import compress as C

MODEL_SPEC = config.YOLO_PATH

PRUNE_STEP_FRAC   = 0.03
REINVEST_FRAC     = 0.10
QUALITY_METRIC    = "map"
MIN_GFLOPS_FRAC   = 0.10
MIN_PROGRESS_FRAC = 0.0015
FT_EPOCHS_PER_MORPH = 1

PRUNE_LAYER_CAP   = config.PHASE2_PRUNE_LAYER_CAP
MIN_ALIVE_FRAC    = config.PHASE2_MIN_ALIVE_FRAC
CHURN_COOLDOWN    = config.PHASE2_CHURN_COOLDOWN
MAX_STEPS         = 1000

DEV_DATA_SUBSET   = None
VAL_CAP           = config.VAL_CAP
TRAIN_BATCH       = config.TRAIN_BATCH
EVAL_BATCH        = config.EVAL_BATCH
REUSE_MORPH_CACHE = True

OUT_CSV  = os.path.join(_HERE, "pareto_run.csv")
OUT_META = os.path.join(_HERE, "pareto_run_meta.json")
OUT_MODEL = os.path.join(_HERE, "pareto_final.pt")

CSV_COLS = ["step", "phase", "gflops", "gflops_pct", "params", "size_mb",
            "val_map", "val_mar", "quality", "gflops_freed_cum", "gflops_reinvested_cum",
            "ft_epochs", "n_pruned", "n_grown", "kd_loss", "wall_s"]


def _apply_overrides():
    for mod in (config, A, C):
        if hasattr(mod, "DEV_DATA_SUBSET"):
            mod.DEV_DATA_SUBSET = DEV_DATA_SUBSET
        if not REUSE_MORPH_CACHE and hasattr(mod, "TMP_ROOT"):
            mod.TMP_ROOT = os.path.join(_HERE, "tmp")


def run_sweep(spec=MODEL_SPEC):
    import copy
    import prodigyopt

    _apply_overrides()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    name = C._model_name(spec)
    qmetric = QUALITY_METRIC

    teacher = A.load_any(spec, device); adapter = A.pick_adapter(teacher)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    note = A.dev_subset_note()
    if note:
        print("\n" + note)
    print(f"\n########## PARETO-SWEEP ({name}, kind={adapter.kind}) — cilj: cijela Pareto krivulja ##########")
    rep0 = A.baseline_perf(spec, device, eval_max=None, model=teacher, adapter=adapter)
    base = rep0["maps"].get("val", {})
    base_q = base.get(qmetric, 0.0)
    base_gflops = rep0["gflops"]
    floor_gflops = MIN_GFLOPS_FRAC * base_gflops
    target_gflops = PRUNE_STEP_FRAC * base_gflops
    min_progress = MIN_PROGRESS_FRAC * base_gflops
    print(f"  baseline: GFLOPs={base_gflops:.3f}  val_map50:95={base.get('map',0.0):.4f}  "
          f"val_mar_100={base.get('mar_100',0.0):.4f}  | metrika kvalitete='{qmetric}'")
    print(f"  rez/korak={target_gflops:.3f} GFLOPs ({PRUNE_STEP_FRAC:.1%} orig.) | "
          f"reinvest<= {REINVEST_FRAC:.0%} | POD={floor_gflops:.3f} GFLOPs ({MIN_GFLOPS_FRAC:.0%} orig.) | "
          f"FT/korak={FT_EPOCHS_PER_MORPH}" +
          (f" | early-stop ako rez < {min_progress:.4f} GFLOPs ({MIN_PROGRESS_FRAC:.2%} orig.)" if MIN_PROGRESS_FRAC > 0 else ""))

    student = A.load_any(spec, device); student.to(device)
    for p in student.parameters():
        p.requires_grad_(True)

    train_loader = A.make_gt_loader("train", bs=TRAIN_BATCH)
    cache = C.precompute_teacher(teacher, adapter, train_loader, name)
    warmup = max(1, min(cache.n, len(train_loader)))

    def new_opt():
        return prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                                  lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
    opt = new_opt()
    gstep = 0

    def measure():
        mm, _ = A.evaluate(student, adapter, C._random_val_loader(VAL_CAP), device)
        return {"map": mm.get("map", 0.0), "mar_100": mm.get("mar_100", 0.0)}

    _w = lambda mdl: {nm: int(w.shape[0]) for nm, _, _, w in A.weighted_leaves(mdl)}

    unsafe = set()
    morph_idx = 0
    grown_at, pruned_at = {}, {}
    total_pruned = total_grown = 0.0

    csv_f = open(OUT_CSV, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_COLS); writer.writeheader()

    def record(step, phase, gf, params, m, n_pruned, n_grown, kd_loss, ft_eps):
        row = {"step": step, "phase": phase, "gflops": round(gf, 4),
               "gflops_pct": round(100.0 * gf / base_gflops, 3), "params": params,
               "size_mb": round(params * 4 / (1024 ** 2), 3),
               "val_map": round(m.get("map", 0.0), 5), "val_mar": round(m.get("mar_100", 0.0), 5),
               "quality": round(m.get(qmetric, 0.0), 5), "gflops_freed_cum": round(base_gflops - gf, 4),
               "gflops_reinvested_cum": round(total_grown, 4), "ft_epochs": ft_eps,
               "n_pruned": n_pruned, "n_grown": n_grown, "kd_loss": round(kd_loss, 5),
               "wall_s": round(time.time() - t_start, 1)}
        writer.writerow(row); csv_f.flush()
        return row

    m = measure()
    g = A.gflops_total(student, adapter, device)
    print(f"[ep 0] [ORIGINAL] val_map50:95={m['map']:.4f}  val_mar_100={m['mar_100']:.4f} | GFLOPs={g:.3f}")
    record(0, "original", g, A.count_params(student), m, 0, 0, 0.0, 0)

    end_reason = f"dosegnut MAX_STEPS={MAX_STEPS}"
    for step in range(1, MAX_STEPS + 1):
        gf = A.gflops_total(student, adapter, device)
        if gf <= floor_gflops:
            end_reason = f"GFLOPs {gf:.3f} <= POD {floor_gflops:.3f} ({MIN_GFLOPS_FRAC:.0%} orig.)"
            break
        gf_start = gf

        morph_idx += 1
        w_pre = _w(student)
        grow_protected = {l for l, k in grown_at.items() if morph_idx - k <= CHURN_COOLDOWN}
        prune_protected = {l for l, k in pruned_at.items() if morph_idx - k <= CHURN_COOLDOWN}
        struct, prunable, imp, gavg, cost, flops_per, units, info = \
            C._grow_ctx(student, adapter, cache, train_loader, device)

        n_rem = n_lay = n_bad = 0; plan = None; cd_override = False
        while True:
            plan, est = C._select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                                             PRUNE_LAYER_CAP, MIN_ALIVE_FRAC, config.PHASE2_MIN_ALIVE,
                                             exclude=unsafe | grow_protected)
            if not plan and grow_protected:
                plan, est = C._select_prune_plan(struct, imp, cost, flops_per, units, info, target_gflops,
                                                 PRUNE_LAYER_CAP, MIN_ALIVE_FRAC, config.PHASE2_MIN_ALIVE,
                                                 exclude=unsafe)
                cd_override = bool(plan)
            if not plan:
                break
            student, n_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, device, plan)
            unsafe |= bad
            if n_rem > 0 or not bad:
                break
        if n_rem == 0:
            end_reason = "nema više rezivih kandidata (off-limits/unsafe/floor/cap odbili sve)"
            break
        opt = new_opt()
        gf2 = A.gflops_total(student, adapter, device); total_pruned += max(0.0, gf - gf2)
        pruned_names = set(plan)
        for nm in pruned_names:
            pruned_at[nm] = morph_idx
        w_mid = _w(student)

        n_grown = 0
        pool = max(0.0, REINVEST_FRAC * total_pruned - total_grown)
        if pool > 0:
            info2 = {nm: (mod, w.dim()) for nm, mod, _, w in A.weighted_leaves(student)}
            grow_flops = {kk: vv for kk, vv in flops_per.items()
                          if kk not in pruned_names and kk not in prune_protected
                          and kk not in grow_protected and kk in info2}
            grown, ginfos, spent = C._grow_decide(student, adapter, device, struct, gavg,
                                                  grow_flops, units, info2, pool)
            if grown is not None:
                student = grown; total_grown += spent; n_grown = sum(gi["k"] for gi in ginfos)
                for gi in ginfos:
                    grown_at[gi["layer"]] = morph_idx
                opt = new_opt()

        racc = {}
        for _ in range(FT_EPOCHS_PER_MORPH):
            racc, gstep = C._kd_epoch(student, adapter, cache, train_loader, device, opt, gstep, warmup)
        kd_loss = sum(racc.values())

        m = measure()
        gf = A.gflops_total(student, adapter, device); params = A.count_params(student)
        record(step, "morph+ft", gf, params, m, n_rem, n_grown, kd_loss, FT_EPOCHS_PER_MORPH)
        kstr = " ".join(f"{k}={racc[k]:.3f}" for k in racc)
        co = " [cd-override]" if cd_override else ""
        gstr = f" +{n_grown}g" if n_grown else ""
        print(f"[ep{step:3d}] [MORPH+FT] -{n_rem}k/{n_lay}sl{gstr}{co} | {kstr} | "
              f"val_map50:95={m['map']:.4f} val_mar_100={m['mar_100']:.4f} | "
              f"GFLOPs={gf:.3f} ({gf/base_gflops*100:.1f}% orig.)")
        freed_step = gf_start - gf
        if MIN_PROGRESS_FRAC > 0 and freed_step < min_progress:
            end_reason = (f"early-stop: korak oslobodio {freed_step:.4f} GFLOPs < prag "
                          f"{min_progress:.4f} ({MIN_PROGRESS_FRAC:.2%} orig.) — diminishing returns")
            print(f"  >>> KRAJ: {end_reason}")
            break
    else:
        end_reason = f"dosegnut MAX_STEPS={MAX_STEPS}"

    csv_f.close()
    torch.save(student, OUT_MODEL)
    total_time = time.time() - t_start
    final_gf = A.gflops_total(student, adapter, device)
    meta = {
        "model": name, "model_spec": str(spec), "kind": adapter.kind, "device": str(device),
        "dataset_root": str(config.DATASET_ROOT), "dev_data_subset": DEV_DATA_SUBSET,
        "quality_metric": qmetric, "baseline_gflops": base_gflops, "baseline_quality": base_q,
        "baseline_map": base.get("map", 0.0), "baseline_mar_100": base.get("mar_100", 0.0),
        "baseline_params": rep0["params"],
        "params": {"prune_step_frac": PRUNE_STEP_FRAC, "reinvest_frac": REINVEST_FRAC,
                   "min_gflops_frac": MIN_GFLOPS_FRAC, "min_progress_frac": MIN_PROGRESS_FRAC,
                   "ft_epochs_per_morph": FT_EPOCHS_PER_MORPH,
                   "prune_layer_cap": PRUNE_LAYER_CAP, "min_alive_frac": MIN_ALIVE_FRAC,
                   "churn_cooldown": CHURN_COOLDOWN, "align_m": config.ALIGN_M,
                   "val_cap": VAL_CAP, "train_batch": TRAIN_BATCH},
        "n_steps": morph_idx, "final_gflops": final_gf,
        "final_gflops_pct": 100.0 * final_gf / base_gflops, "end_reason": end_reason,
        "total_time_s": round(total_time, 1), "csv": OUT_CSV, "final_model": OUT_MODEL,
    }
    json.dump(meta, open(OUT_META, "w"), indent=2)

    print(f"\n########## PARETO-SWEEP GOTOV ({name}) ##########")
    print(f"  razlog kraja: {end_reason}")
    print(f"  koraka: {morph_idx} | GFLOPs {base_gflops:.3f} -> {final_gf:.3f} ({final_gf/base_gflops*100:.1f}% orig.)")
    print(f"  CSV:   {OUT_CSV}")
    print(f"  meta:  {OUT_META}")
    print(f"  model: {OUT_MODEL}")
    print(f"  UKUPNO VRIJEME: {total_time:.1f} s  ({total_time/60:.1f} min)")
    return meta


if __name__ == "__main__":
    run_sweep()
