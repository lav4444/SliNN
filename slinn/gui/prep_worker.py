"""slinn/gui/prep_worker.py — 6.1 engine-vodjena PRIPREMA (semafor). BEZ argv; čita tmp/gui_job/config.json
{model_path, dataset_path}. Koraci -> prep_status.json (GUI crta semafor):
  1. gpu   — GPU dostupan/slobodan
  2. task  — AUTO task/mode/enhaneri (pipeline.prepare)
  3. batch — auto TRAIN batch (engine.autobatch, cache-FT-mod)
  4. perf  — per-task baseline (GFLOPs/params + prava metrika, ili teacher-agreement ako nema oznaka)
Teacher-cache se NE gradi ovdje — engine.full_cycle ga sam materijalizira/preskoči (enhaneri) tijekom kompresije.
"""
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
    B.tee_log("prep.log")                                    # ispis ide u terminal I u log (GUI ga tail-a)
    try:
        os.remove(STATUS)
    except OSError:
        pass
    steps = {"gpu": {"light": "gray"}, "task": {"light": "gray"}, "batch": {"light": "gray"},
             "plan": {"light": "gray"}, "perf": {"light": "gray"}}

    _seen = set()

    def write(state):
        json.dump({"state": state, "pid": os.getpid(), "steps": steps}, open(STATUS, "w"))
        for k, v in steps.items():                           # ispisi svaki korak JEDNOM kad dobije boju
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

        # 1) GPU --------------------------------------------------------------
        g = C.gpu_status()
        if not g.get("available"):
            steps["gpu"] = {"light": "yellow", "msg": g.get("msg", "nema GPU-a (CPU)")}
        elif g.get("ok"):
            steps["gpu"] = {"light": "green", "msg": f"{g['name']} · zauzeto {g['used_pct']:.0f}%"}
        else:
            steps["gpu"] = {"light": "yellow", "msg": f"{g['name']} ZAUZET {g['used_pct']:.0f}% (>50%)"}
        write("preparing")

        # model + AUTO ctx (task/mode/enhaneri) -------------------------------
        model, adapter, ctx = B.load_ctx(cfg, dev)
        steps["task"] = {"light": "green",
                         "msg": f"task={ctx['task']} · mode={ctx.get('mode')} · enhaneri={'da' if ctx.get('enhancers') else 'ne'}",
                         "data": {"task": ctx["task"], "mode": ctx.get("mode"), "enhancers": bool(ctx.get("enhancers")),
                                  "metrics": ctx.get("metrics")}}
        write("preparing")

        # 2) AUTO BATCH -------------------------------------------------------
        steps["batch"] = {"light": "yellow", "msg": "tražim najveći batch (≤90% slob. VRAM-a)…"}
        write("preparing")
        tb = E.autobatch(model, adapter, dev, ctx, cfg["dataset_path"]) if dev.type == "cuda" else 4
        steps["batch"] = {"light": "green" if dev.type == "cuda" else "yellow",
                          "msg": (f"TRAIN={tb}" if dev.type == "cuda" else f"CPU — bez probe, TRAIN={tb}"),
                          "data": {"train": tb}}
        write("preparing")

        # 3) PLAN TEACHER CACHEA (koliko podataka + stane li) ------------------
        # NISTA se ne pita: KD uzima CIJELI train split. Ovdje se samo IZMJERI koliko to zauzme
        # (port morphology `teacher_mem_plan` — kartica koju je stari GUI imao).
        steps["plan"] = {"light": "yellow", "msg": "mjerim koliko teacher cache zauzme…"}
        write("preparing")
        n_avail = E.count_train_samples(cfg["dataset_path"], adapter, ctx["split_plan"])
        batches, src = E.materialize_train_batches(cfg["dataset_path"], adapter, dev,
                                                   ctx["split_plan"], batch_size=tb)
        plan = E.plan_teacher_cache(B.frozen_teacher(model, dev), adapter, batches,
                                    ctx["taps"], os.path.basename(cfg["model_path"]))
        n_gate_all = 0
        try:                                                 # koliko uzoraka ima val (samo za prikaz)
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

        # 4) BASELINE PERF (per-task) -----------------------------------------
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
