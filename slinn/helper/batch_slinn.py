# -*- coding: utf-8 -*-
import datetime
import json
import os
import subprocess
import sys
import time

R = "/home/tomi/code/dipl"
SL = os.path.join(R, "slinn")
JOB = os.path.join(SL, "tmp", "gui_job")
PY = sys.executable
PHASE = "1"
CAP_H = 10
ONLY = [x.strip() for x in os.environ.get("BATCH_ONLY", "").split(",") if x.strip()]
LOGS = os.path.join(SL, "tmp", "batch")

MODELS = [
    ("housing_mlp",       R + "/baseline_models/housing_mlp/model.pt",
                          R + "/baseline_models/housing_mlp/data"),
    ("speechcommands_m5", R + "/baseline_models/speechcommands_m5/model.pt",
                          R + "/baseline_models/speechcommands_m5/data"),
    ("midas_depth",       R + "/baseline_models/midas_depth/model.pt",
                          R + "/baseline_models/midas_depth/data"),
    ("sst2_distilbert",   R + "/baseline_models/sst2_distilbert/model.pt",
                          R + "/baseline_models/sst2_distilbert/data"),
    ("voc_deeplabv3",     R + "/baseline_models/voc_deeplabv3/model.pt",
                          R + "/baseline_models/voc_deeplabv3/data"),
    ("yolo26n",           R + "/baseline_models/yolo26n/yolo26n.pt",
                          R + "/datasets/mini_set/sub10k_open_images_v7"),
    ("yolo26l",           R + "/baseline_models/yolo26l/yolo26l.pt",
                          R + "/datasets/mini_set/sub10k_open_images_v7"),
]


def hhmm(sec):
    return "{}h{:02d}m".format(int(sec // 3600), int(sec % 3600) // 60)


def main():
    sys.path.insert(0, SL)
    import settings as CFG
    if CFG.DEV_DATA_SUBSET is not None:
        raise SystemExit("DEV_DATA_SUBSET={} — batch trazi None, inace su svi brojevi niske "
                         "vjernosti.".format(CFG.DEV_DATA_SUBSET))
    os.makedirs(JOB, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = os.path.join(LOGS, "batch_{}.txt".format(stamp))
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
        with open(summary, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    log("#" * 74)
    todo = [m for m in MODELS if not ONLY or m[0] in ONLY]
    log("### SliNN batch · faza {} + QAT · {} modela · bez kapiranja podataka".format(
        PHASE, len(todo)))
    log("### pecat {}   granica po modelu {} h".format(stamp, CAP_H))
    log("#" * 74)

    t_all = time.time()
    ok, bad = [], []
    for name, mp, dp in MODELS:
        if ONLY and name not in ONLY:
            continue
        for p in (mp, dp):
            if not os.path.exists(p):
                log("  {:20} PRESKOCEN — nema {}".format(name, p))
                bad.append(name)
                break
        else:
            cfg = {"model_path": mp, "dataset_path": dp, "code_dirs": None, "phase": PHASE}
            json.dump(cfg, open(os.path.join(JOB, "config.json"), "w"), indent=1)
            wlog = os.path.join(LOGS, "{}_{}.log".format(name, stamp))
            log("")
            log("=" * 74)
            log("==== {}   start {}".format(name, datetime.datetime.now().strftime("%H:%M:%S")))
            log("     log: {}".format(wlog))
            t0 = time.time()
            with open(wlog, "w", encoding="utf-8") as f:
                try:
                    rc = subprocess.call([PY, "-u", os.path.join(SL, "gui", "worker.py")],
                                         stdout=f, stderr=subprocess.STDOUT, cwd=SL,
                                         timeout=CAP_H * 3600)
                except subprocess.TimeoutExpired:
                    rc = -9
                    f.write("\n[batch] PREKINUT nakon {} h\n".format(CAP_H))
            dt = time.time() - t0
            tail = ""
            try:
                runs = sorted(d for d in os.listdir(os.path.join(SL, "runs"))
                              if d.startswith(name + "_"))
                if runs:
                    m = json.load(open(os.path.join(SL, "runs", runs[-1], "run_meta.json")))
                    q = m.get("qat") or []
                    tail = "  ·  {} · {} ckpt kvantiziran".format(runs[-1], len(q))
            except BaseException:
                pass
            if rc == 0:
                ok.append(name)
                log("==== {:20} GOTOVO  {}{}".format(name, hhmm(dt), tail))
            else:
                bad.append(name)
                log("==== {:20} PAO (rc={})  {}   zadnjih 12 redaka:".format(name, rc, hhmm(dt)))
                try:
                    with open(wlog, encoding="utf-8", errors="replace") as f:
                        for x in f.read().splitlines()[-12:]:
                            log("       " + x)
                except OSError:
                    pass

    log("")
    log("#" * 74)
    log("### BATCH GOTOV  {}   proslo {}/{}".format(hhmm(time.time() - t_all),
                                                    len(ok), len(MODELS)))
    if bad:
        log("### palo: {}".format(" ".join(bad)))
    log("### sazetak -> {}".format(summary))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
