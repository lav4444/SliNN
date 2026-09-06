
import math
import os

import torch

import introspect as A
import morph as C
import settings as CFG
import engine as E

_HERE = os.path.dirname(os.path.abspath(__file__))


def _tile_use(width, m):
    return width / (m * math.ceil(width / m)) if (m > 1 and width > 0) else 1.0


def summary(model, adapter, device, ctx=None):
    leaves = A.weighted_leaves(model)
    types = {}
    for _, m, tn, _ in leaves:
        types[tn] = types.get(tn, 0) + 1
    all_types = {}
    for _, m in model.named_modules():
        if not list(m.children()):
            tn = type(m).__name__
            all_types[tn] = all_types.get(tn, 0) + 1

    out = {"params": A.count_params(model), "gflops": float(E.gflops(model, adapter, device)),
           "n_weighted": len(leaves), "n_leaves": sum(all_types.values()),
           "weighted_types": types, "leaf_types": all_types,
           "size_mb": A.count_params(model) * 4 / (1024 ** 2),
           "align_m": CFG.ALIGN_M, "align_score": C.model_align_score(model)}
    if ctx:
        out.update({
            "task": ctx.get("task"), "task_source": ctx.get("task_source"), "mode": ctx.get("mode"),
            "metrics": ctx.get("metrics"), "kd_core": ctx.get("kd_core"),
            "enhancers": ctx.get("enhancers"), "kd_mode": ctx.get("kd_mode"),
            "taps": ctx.get("taps"), "n_prunable": len(ctx.get("prunable") or ()),
            "dataset_format": ctx.get("dataset_format"), "n_samples": ctx.get("n_samples"),
            "splits": ctx.get("splits"), "split_plan": ctx.get("split_plan"),
            "kd_note": ctx.get("kd_note"),
        })
    return out


def layer_rows(model, adapter, device, ctx=None, with_cost=True):
    rec = A.layer_table(model, adapter, device)
    prunable = set(ctx.get("prunable") or ()) if ctx else set()
    taps = set(ctx.get("taps") or ()) if ctx else set()
    m = CFG.ALIGN_M

    flops_per, units = {}, {}
    if with_cost and prunable:
        try:
            _cost, flops_per, units = C.prune_costs(model, adapter, device, prunable)
        except BaseException as e:
            print("[overview] prune_costs preskocen: {}: {}".format(type(e).__name__, str(e)[:80]))

    rows = []
    for r in rec:
        nm = r["name"]
        w = int(r["units"])
        rows.append({
            "sloj": nm, "tip": r["type"], "uloga": r["role"], "sirina": w,
            "params": int(r["params"]), "GFLOPs": round(float(r["gflops"]), 5),
            "prunable": nm in prunable, "tap": nm in taps,
            "poravnanje": round(_tile_use(w, m), 3),
            "do_xM": (-w) % m,
            "cijena/kanal GFLOPs": round(float(flops_per.get(nm, 0.0)), 6) if flops_per else None,
        })
    return rows


def top_prune(rows, k=10):
    cand = [r for r in rows if r["prunable"] and r.get("cijena/kanal GFLOPs")]
    return sorted(cand, key=lambda r: -r["cijena/kanal GFLOPs"])[:k]


def worst_aligned(rows, k=10):
    return sorted([r for r in rows if r["prunable"]], key=lambda r: r["poravnanje"])[:k]


def report(model, adapter, device, ctx=None, deep=False, teacher=None, batches=None, progress=None):
    def tick(f, msg):
        if progress:
            progress(f, msg)

    tick(0.1, "sazetak modela")
    rep = {"summary": summary(model, adapter, device, ctx)}
    tick(0.4, "per-layer mjerenje (forward + tp graf)")
    rows = layer_rows(model, adapter, device, ctx)
    rep["layers"] = rows
    rep["top_prune"] = top_prune(rows)
    rep["worst_aligned"] = worst_aligned(rows)

    if deep and teacher is not None and batches:
        tick(0.7, "KD-vaznost (gradijentni prolaz)")
        try:
            import loss as L
            imp, _gavg = L.kd_importance(model, teacher, adapter, batches, ctx["taps"],
                                         ctx["kd_mode"], ctx["out_kind"], prunable=ctx["prunable"])
            fp = {r["sloj"]: (r.get("cijena/kanal GFLOPs") or 0.0) for r in rows}
            rr = [{"sloj": nm, "vaznost": float(v.abs().mean()) if torch.is_tensor(v) else float(v),
                   "cijena": fp.get(nm, 0.0)} for nm, v in imp.items()]
            for x in rr:
                x["risk/reward"] = (x["vaznost"] / x["cijena"]) if x["cijena"] else float("inf")
            rep["importance"] = sorted(rr, key=lambda x: x["risk/reward"])[:15]
        except BaseException as e:
            rep["importance_error"] = "{}: {}".format(type(e).__name__, str(e)[:160])
    tick(1.0, "gotovo")
    return rep


def capabilities():
    import json

    def _load(fn):
        p = os.path.join(_HERE, fn)
        return json.load(open(p)) if os.path.exists(p) else {}

    tasks = _load("SUPPORTED_TASKS.json").get("tasks", {})
    fmts = _load("SUPPORTED_DATASET_FORMATS.json").get("formats", {})
    reg = _load("LAYER_REGISTER.json").get("types", {})

    task_rows = []
    for name, d in tasks.items():
        if name.startswith("_") or not isinstance(d, dict):
            continue
        task_rows.append({"task": name, "metrics": d.get("metrics", []),
                          "kd_core": d.get("kd_core", []), "enhancers": d.get("enhancers", []),
                          "decode": bool(d.get("decode")), "note": d.get("_note", "")})

    layer_rows_ = []
    for tname, e in reg.items():
        if tname.startswith("_") or not isinstance(e, dict):
            continue
        layer_rows_.append({"tip": tname, "status": e.get("status", "?"),
                            "prunable": e.get("prunable"), "growable": e.get("growable"),
                            "trainable": e.get("trainable"), "razlog": e.get("reason", "")})

    from plugins.detection import outfmt as _of
    return {"tasks": sorted(task_rows, key=lambda r: r["task"]),
            "formats": {k: v for k, v in fmts.items() if not k.startswith("_")},
            "layers": sorted(layer_rows_, key=lambda r: r["tip"]),
            "out_formats": _of.FORMATS, "out_adapters": _of.FORMAT_ADAPTER,
            "align_m": CFG.ALIGN_M, "dev_subset": CFG.DEV_DATA_SUBSET}
