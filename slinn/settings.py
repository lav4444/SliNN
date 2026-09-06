
ALIGN_M = 32
ALIGN_BETA = 1.0
ALIGN_POW = 1

F1_PRUNE_STEP_FRAC = 0.015
PHASE2_PRUNE_LAYER_CAP = 0.15
PHASE2_MIN_ALIVE_FRAC = 0.00001
PHASE2_MIN_ALIVE = ALIGN_M // 2
PHASE2_COST_FLOPS_W = 0.60
PRUNE_IMPORTANCE = "grad"
PRUNE_IMP_NORM = "none"
F1_MAX_STEPS = 200
PHASE2_PRUNE_ROUNDS = 4
PHASE2_PRUNE_SLACK = 0.02

PHASE2_REINVEST_FRAC = 0.15
PHASE2_GROW_DOM = 4.0
PHASE2_GROW_MAX_LAYERS = 3
PHASE2_CHURN_COOLDOWN = 2

F1_FT_MAX_EPOCHS = 5
F1_FT_PATIENCE = 2

F2_FT_MAX_EPOCHS = 3
F2_FT_PATIENCE = 1
F2_CHECKPOINTS = 2
F2_MIN_YIELD = 0.80
F2_YIELD_FT_EPOCHS = 3
F2_MAX_STEPS_PER_RUNG = 200

QAT_ENABLE = True
QAT_MAX_STEPS = 2000
QAT_EVAL_EVERY = 100
QAT_PATIENCE = 2
QAT_CALIB_BATCHES = 8

BATCH_UINT8 = True
BATCH_CACHE_MB = 3072
BATCH_GPU_CACHE_MB = 1024
TEACHER_CACHE_MB = 2048
CACHE_RAM_FRAC = 0.6
TEACHER_CACHE_DISK_FRAC = 0.8
TEACHER_CACHE_LOWRES_OUT = True
MATMUL_TF32 = True
F2_PRUNE_STEP_FRAC = 0.05

FT_RECOVERY_FRAC = 0.95
METRIC_MONITOR_FRAC = 0.25
METRIC_MONITOR_MIN = 64

AGREEMENT_SUGGEST = 0.95

TRAIN_BATCH = 8
METRIC_VAL_BATCH = 16
IMP_BATCHES = 3

DEV_DATA_SUBSET = None

from pathlib import Path  # noqa: E402

_HERE = Path(__file__).parent
TMP_ROOT = str(_HERE / "tmp")
MODELS_DIR = str(_HERE / "models")
RUNS_DIR = str(_HERE / "runs")


def effective(phases="12"):
    g = globals()
    blocks = [("zajednicko (obje faze)",
               ["F1_PRUNE_STEP_FRAC" if "1" in phases else None,
                "F2_PRUNE_STEP_FRAC" if "2" in phases else None,
                "PHASE2_PRUNE_LAYER_CAP", "PHASE2_MIN_ALIVE", "PHASE2_PRUNE_ROUNDS",
                "PHASE2_PRUNE_SLACK", "PRUNE_IMPORTANCE", "PRUNE_IMP_NORM",
                "PHASE2_REINVEST_FRAC", "PHASE2_GROW_DOM", "PHASE2_GROW_MAX_LAYERS",
                "PHASE2_CHURN_COOLDOWN", "ALIGN_M", "METRIC_MONITOR_FRAC",
                "TRAIN_BATCH", "IMP_BATCHES", "DEV_DATA_SUBSET"])]
    if "1" in phases:
        blocks.append(("faza 1", ["F1_MAX_STEPS", "F1_FT_MAX_EPOCHS", "F1_FT_PATIENCE",
                                  "FT_RECOVERY_FRAC"]))
    if "2" in phases:
        blocks.append(("faza 2", ["F2_CHECKPOINTS", "F2_FT_MAX_EPOCHS", "F2_FT_PATIENCE",
                                  "F2_MIN_YIELD", "F2_YIELD_FT_EPOCHS",
                                  "F2_MAX_STEPS_PER_RUNG"]))
    out = []
    for naslov, imena in blocks:
        out.append("  {}:".format(naslov))
        row = []
        for n in [x for x in imena if x]:
            row.append("{}={}".format(n, g[n]))
            if len(row) == 3:
                out.append("    " + "   ".join(row))
                row = []
        if row:
            out.append("    " + "   ".join(row))
    if DEV_DATA_SUBSET is not None:
        out.append("  UPOZORENJE: DEV_DATA_SUBSET={} — svaki split je kapiran, NISKA vjernost."
                   .format(DEV_DATA_SUBSET))
    return out
