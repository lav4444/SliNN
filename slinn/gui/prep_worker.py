import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backend as B                                          # noqa: E402
import torch                                                 # noqa: E402
import morph as C                                            # noqa: E402  (6.4: slinn gpu_status)
import engine as E                                           # noqa: E402

STATUS = os.path.join(B.JOB, "prep_status.json")


def main():
    os.makedirs(B.JOB, exist_ok=True)
    B.tee_log("prep.log")
    try:
        os.remove(STATUS)
    except OSError:
        pass
    steps = {"gpu": {"light": "gray"}, "task": {"light": "gray"}, "batch": {"light": "gray"},
             "plan": {"light": "gray"}, "perf": {"light": "gray"}}

    _seen = set()

    def write(state):
        json.dump({"state": state, "pid": os.getpid(), "steps": steps}, open(STATUS, "w"))
        for k, v in steps.items():
            sig = (k, v.get("light"), v.get("msg"))
            if v.get("light") != "gray" and sig not in _seen:
                _seen.add(sig)
                print("[{}] {:<6} {}".format(
                    {"green": "OK  ", "yellow": "UPOZ", "red": "GRES"}.get(v.get("light"), "?   "),
                    k, v.get("msg") or ""))
                if v.get("data") and k == "perf":
                    print("        " + str(v["data"]))

    write("preparing")
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(B.JOB, "config.json")))

        g = C.gpu_status()
        if not g.get("available"):
            steps["gpu"] = {"light": "yellow", "msg": g.get("msg", "nema GPU-a (CPU)")}
        elif g.get("ok"):
            steps["gpu"] = {"light": "green", "msg": f"{g['name']} · zauzeto {g['used_pct']:.0f}%"}
        else:
            steps["gpu"] = {"light": "yellow", "msg": f"{g['name']} ZAUZET {g['used_pct']:.0f}% (>50%)"}
        write("preparing")

        model, adapter, ctx = B.load_ctx(cfg, dev)
        steps["task"] = {"light": "green",
                         "msg": f"task={ctx['task']} · mode={ctx.get('mode')} · enhaneri={'da' if ctx.get('enhancers') else 'ne'}",
                         "data": {"task": ctx["task"], "mode": ctx.get("mode"), "enhancers": bool(ctx.get("enhancers")),
                                  "metrics": ctx.get("metrics")}}
        write("preparing")

        steps["batch"] = {"light": "yellow", "msg": "tražim najveći batch (≤90% slob. VRAM-a)…"}
        write("preparing")
        tb, _how = B.probe_train_batch(model, adapter, dev, ctx, cfg["dataset_path"])
        steps["batch"] = {"light": "green" if dev.type == "cuda" else "yellow",
                          "msg": f"TRAIN={tb} ({_how})",
                          "data": {"train": tb}}
        write("preparing")

        steps["plan"] = {"light": "yellow", "msg": "mjerim koliko teacher cache zauzme…"}
        write("preparing")
        n_avail = E.count_train_samples(cfg["dataset_path"], adapter, ctx["split_plan"], model=model)
        batches, src = E.materialize_train_batches(cfg["dataset_path"], adapter, dev,
                                                   ctx["split_plan"], batch_size=tb, model=model)
        plan = E.plan_teacher_cache(B.frozen_teacher(model, dev), adapter, batches,
                                    ctx["taps"], E.model_name_of(cfg["model_path"]))
        n_gate_all = 0
        try:
            n_gate_all = E.count_train_samples(cfg["dataset_path"], adapter, {"train": "val"})
        except BaseException:
            pass
        plan.update({"n_samples": sum(len(b) for b in batches), "source": src,
                     "n_avail": n_avail, "n_gate": n_gate_all})
        steps["plan"] = {"light": "green" if plan["fits_disk"] else "red",
                         "msg": ("{} uzoraka ({} batcheva) · cache {:.2f} GB / slobodno {:.0f} GB"
                                 .format(plan["n_samples"], plan["n_batches"],
                                         plan["total_gb"], plan["free_gb"])
                                 + ("" if plan["fits_disk"] else "  — NE STANE NA DISK")),
                         "data": plan}
        write("preparing")

        mfn, mname, _ = B.build_metric_fn(ctx, model, adapter, cfg["dataset_path"], dev)
        steps["perf"] = {"light": "yellow", "msg": f"mjerim baseline ({mname})…"}
        write("preparing")
        rep = B.baseline_report(ctx, model, adapter, dev, mfn, mname)
        steps["perf"] = {"light": "green", "data": rep}
        write("ready")
    except KeyboardInterrupt:
        write("stopped")
    except BaseException:
        steps.setdefault("err", {})["msg"] = traceback.format_exc()[-1500:]
        write("error")


if __name__ == "__main__":
    main()
