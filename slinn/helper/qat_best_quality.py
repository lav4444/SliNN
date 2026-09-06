# -*- coding: utf-8 -*-
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.dirname(HERE)
R = os.path.dirname(SL)
for p in (SL, os.path.join(SL, "gui")):
    if p not in sys.path:
        sys.path.insert(0, p)

RUNS = [
    ("midas_depth_20260906_102110",
     R + "/baseline_models/midas_depth/model.pt",
     R + "/baseline_models/midas_depth/data"),
    ("voc_deeplabv3_20260906_070720",
     R + "/baseline_models/voc_deeplabv3/model.pt",
     R + "/baseline_models/voc_deeplabv3/data"),
]
CKPT = "best_quality_model.pt"
FORCE = False


def hhmm(s):
    return "{:d}h{:02d}m".format(int(s // 3600), int((s % 3600) // 60))


def run_one(run_name, model_path, dataset_path):
    import torch
    import worker as W
    import quant as Q

    RUN = os.path.join(SL, "runs", run_name)
    src = os.path.join(RUN, CKPT)
    dst = os.path.join(RUN, os.path.splitext(CKPT)[0] + "_qat.pt")
    if not os.path.exists(src):
        print("  PRESKOCEN — nema {}".format(src))
        return None
    if os.path.exists(dst) and not FORCE:
        print("  PRESKOCEN — {} vec postoji (FORCE=True da pregradis)".format(os.path.basename(dst)))
        return None

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"model_path": model_path, "dataset_path": dataset_path, "code_dirs": None, "phase": "1"}
    sh = W._setup(cfg, dev)
    mfn, mname, monfn = sh["mfn"], sh["mname"], sh["monfn"]

    print("\n### KVANTIZACIJA · {} · {}".format(run_name, os.path.splitext(CKPT)[0]), flush=True)
    model = torch.load(src, map_location=dev, weights_only=False)
    prije = float(mfn(model)) if mfn is not None else None
    model, info = Q.quantize_checkpoint(
        model, sh["teacher"], sh["adapter"], sh["ctx"], sh["batches"],
        cache=sh["cache"], loss_fn=sh["loss_fn"], device=dev, monitor_fn=monfn,
        on_step=lambda k, n, l: print("    {}/{} · KD {:.4f}".format(k, n, l), flush=True))
    if not info.get("wrapped"):
        print("  preskocen: {}".format(info.get("why")))
        return None
    poslije = float(mfn(model)) if mfn is not None else None
    info.update({"metric_name": mname, "metric_prije": prije, "metric_poslije": poslije,
                 "delta": (None if prije is None else poslije - prije)})
    Q.save_eager(model, dst)
    print("  {} {:.4f} -> {:.4f}  ({:+.4f}) · {} koraka · kraj: {}".format(
        mname, prije, poslije, poslije - prije, info["steps"], info["why"]))
    print("  spremljeno: {}".format(dst))

    mp = os.path.join(RUN, "run_meta.json")
    try:
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        q = [x for x in (meta.get("qat") or []) if x.get("file") != os.path.basename(dst)]
        q.append({"file": os.path.basename(dst),
                  **{k: v for k, v in info.items() if k != "monitor_hist"}})
        meta["qat"] = q
        prod = meta.get("produced") or []
        if os.path.basename(dst) not in prod:
            prod.append(os.path.basename(dst))
        meta["produced"] = prod
        json.dump(meta, open(mp, "w"), indent=1)
        print("  run_meta.json azuriran")
    except BaseException as e:
        print("  run_meta.json NIJE azuriran: {}".format(e))
    return {"run": run_name, "metric": mname, "prije": prije, "poslije": poslije}


if __name__ == "__main__":
    print("#" * 74)
    print("### QAT samo nad {} · {} run(ova)".format(CKPT, len(RUNS)))
    print("#" * 74)
    t_all, done = time.time(), []
    for run_name, mp_, dp_ in RUNS:
        print("\n" + "=" * 74)
        print("==== {}   start {}".format(run_name, datetime.datetime.now().strftime("%H:%M:%S")))
        t0 = time.time()
        try:
            r = run_one(run_name, mp_, dp_)
            if r:
                r["trajanje"] = hhmm(time.time() - t0)
                done.append(r)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            print("==== {} PAO: {}".format(run_name, e))
    print("\n" + "#" * 74)
    print("### GOTOVO za {} · uspjelo {}/{}".format(hhmm(time.time() - t_all), len(done), len(RUNS)))
    for r in done:
        print("  {:34} {:8} {:.4f} -> {:.4f}  ({:+.4f})  {}".format(
            r["run"], r["metric"], r["prije"], r["poslije"], r["poslije"] - r["prije"], r["trajanje"]))
