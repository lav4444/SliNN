"""
worker.py — pozadinski radnik za GUI Compress. Cita tmp/gui_job/config.json (BEZ argv),
pokrece compress.run_dead_ft, a trajektorija-tocke pise u tmp/gui_job/trajectory.jsonl
te stanje u status.json. GUI samo tail-a te datoteke.
"""

import json
import os
import sys
import traceback

import torch

import compress as C
import config

JOB = os.path.join(os.path.dirname(__file__), "tmp", "gui_job")


class _Tee:
    """Pise u terminal (kao i prije) I u worker.log da ga GUI moze tail-ati. Bez GUI-a log se samo dodatno pise."""
    def __init__(self, stream, logf):
        self.stream, self.logf = stream, logf

    def write(self, s):
        self.stream.write(s); self.stream.flush()
        self.logf.write(s); self.logf.flush()

    def flush(self):
        self.stream.flush(); self.logf.flush()


def main():
    os.makedirs(JOB, exist_ok=True)
    logf = open(os.path.join(JOB, "worker.log"), "w")        # tee terminal -> log (za GUI tail)
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    cfg = json.load(open(os.path.join(JOB, "config.json")))
    traj = os.path.join(JOB, "trajectory.jsonl")
    status = os.path.join(JOB, "status.json")
    open(traj, "w").close()                              # reset log

    def set_status(d):
        d["pid"] = os.getpid()
        json.dump(d, open(status, "w"))

    def on_event(p):
        with open(traj, "a") as f:
            f.write(json.dumps(p) + "\n")
        set_status({"state": "running", "phase": p["phase"], "step": p["step"]})

    set_status({"state": "running", "phase": "start", "step": -1})
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = cfg.get("metrics", list(config.FT_METRICS))                         # GUI bira koje metrike optimizirati
        align_m = int(cfg.get("align_m", config.ALIGN_M))                             # GUI kvantizacija: INT8->32, FP16->8
        config.ALIGN_M = align_m                                                      # compress.align_factors cita config DINAMICKI
        config.PHASE2_MIN_ALIVE = align_m // 2                                        # min kanala/sloj = M/2 (vezan uz ALIGN_M)
        print(f"[align] kvantizacija M={align_m} -> PHASE2_MIN_ALIVE={align_m // 2}")
        bat = cfg.get("batch") or {}                                                  # auto-batch iz prep probe (prazno -> config default)
        tb = int(bat.get("train", config.TRAIN_BATCH)); eb = int(bat.get("eval", config.EVAL_BATCH))
        config.GRAD_BATCH = int(bat.get("grad", config.GRAD_BATCH))                   # GRAD koristi Overview; TRAIN cache MORA biti isti kao u prepu
        print(f"[autobatch] TRAIN={tb} EVAL={eb} GRAD={config.GRAD_BATCH}")
        # FAZA 1: dead/near-dead rez + KD FT recovery -> best_quality (start tocka Faze 2). final_report=False:
        # skupi PERF (puni mAP+brzina) se NE racuna ovdje, nego SAMO na KRAJU (run_morph, na best_quality modelu).
        model, _ = C.run_dead_ft(config.MODEL_SPEC, dev, on_event=on_event, do_analysis=False,   # model FIKSIRAN u config.py
                                 metrics=metrics, final_report=False, train_batch=tb, eval_batch=eb)
        # FAZA 2: kontinuirani prune (+ uvjetni grow) -> najmanji model unutar tolerancije (ista trajektorija)
        set_status({"state": "running", "phase": "faza2", "step": -1})
        C.run_morph(config.MODEL_SPEC, dev, on_event=on_event, metrics=metrics, start_model=model,
                    train_batch=tb, eval_batch=eb)
        set_status({"state": "done", "phase": "final"})
    except KeyboardInterrupt:
        set_status({"state": "stopped"})                 # Ctrl+C / SIGINT -> uredno "stopped"
    except BaseException:
        set_status({"state": "error", "msg": traceback.format_exc()[-2500:]})


if __name__ == "__main__":
    main()
