"""6.1 headless validacija: config.json {model_path, dataset_path} -> prep_worker -> worker -> artefakti.
Workere pokreće kao SUBPROCESE (kako GUI stvarno radi via subprocess.Popen) — bez kolizije importa
(morphology/ i slinn/gui/ oba imaju worker.py/prep_worker.py). -> REPORTS/run61.txt"""
import json
import os
import subprocess
import sys

_SLINN = "/home/tomi/code/dipl/slinn"
GUI = os.path.join(_SLINN, "gui")
JOB = os.path.join(_SLINN, "tmp", "gui_job")
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(_SLINN, "REPORTS", "run61.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(JOB, exist_ok=True)

CFG = {"model_path": f"{BM}/housing_mlp/model.pt", "dataset_path": f"{BM}/housing_mlp/data",
       "code_dirs": [f"{BM}/housing_mlp"], "target_frac": 0.15, "ft_steps": 4, "max_steps": 6, "metric_tol": 0.90}
json.dump(CFG, open(os.path.join(JOB, "config.json"), "w"))

L = []


def _r(x):
    return round(x, 4) if isinstance(x, float) else x


def emit(s=""):
    print(s, flush=True); L.append(str(s))


def run_worker(fname):
    p = subprocess.run([sys.executable, "-u", os.path.join(GUI, fname)], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


emit("===== 6.1 HEADLESS BACKEND (housing_mlp, subprocess launch) =====")
emit("config.json: " + json.dumps(CFG))
emit("")

emit("--- PREP (subprocess prep_worker.py) ---")
rc, so, se = run_worker("prep_worker.py")
emit("rc=%d" % rc)
if se.strip():
    emit("stderr(tail): " + se.strip()[-400:])
ps = json.load(open(os.path.join(JOB, "prep_status.json")))
emit("prep state = %s" % ps["state"])
for k, v in ps["steps"].items():
    emit("  %-6s [%s] %s" % (k, v.get("light"), v.get("msg", v.get("data", ""))))

emit("")
emit("--- COMPRESS (subprocess worker.py) ---")
rc, so, se = run_worker("worker.py")
emit("rc=%d" % rc)
if se.strip():
    emit("stderr(tail): " + se.strip()[-600:])
st = json.load(open(os.path.join(JOB, "status.json")))
emit("worker state = %s  phase=%s" % (st.get("state"), st.get("phase")))
if st.get("state") == "error":
    emit("ERROR: " + str(st.get("msg", ""))[-800:])
else:
    emit("  metric=%s  g0=%s  best_gflops=%s  best_step=%s  baseline=%s" %
         (st.get("metric_name"), _r(st.get("g0")), _r(st.get("best_gflops")),
          st.get("best_step"), _r(st.get("metric_baseline"))))
    emit("  trajektorija:")
    for line in open(os.path.join(JOB, "trajectory.jsonl")):
        r = json.loads(line)
        mt = ("%.4f" % r["metric"]) if r.get("metric") is not None else "-"
        emit("   step %d  GFLOPs=%.6f  params=%d  rez=%d  metric=%s" %
             (r["step"], r["gflops"], r["params"], r["removed_ch"], mt))
    emit("  compressed.pt spremljen: %s" % os.path.exists(os.path.join(JOB, "compressed.pt")))

open(OUT, "w").write("\n".join(L) + "\n")
emit("\n-> " + OUT)
