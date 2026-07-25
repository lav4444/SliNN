"""
config.py — JEDINSTVENA konfiguracija morphology pipelinea.
Promijeni ovdje -> vidi se SVUGDJE (analysis.py, compress.py, main.py, gui.py, worker.py sve uvoze odavde).
GUI ne nudi tuning korisniku — samo cita ove vrijednosti (cilj: generalizacija, ne mnostvo izbora).
"""

from pathlib import Path

# =========================== DATASET =========================== #
DATASET_ROOT = Path("/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7")  # root sub10k Open Images (6 kl)
CLASS_NAMES = ["Person", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]            # redoslijed = label idx 0..5
IMG_EXTS = (".jpg", ".jpeg", ".png")                                                # podrzane ekstenzije slika

# ⚠️ PRIVREMENO (dev/prototyping): kapira SVAKI split na prvih N slika -> cijeli pipeline (eval/census/grad/
# precompute/FT/val-monitor) radi kao da ima samo N po setu. DRASTICNO brze, ali NISKA vjernost. None = iskljuceno.
# MAKNI (postavi None) prije pravih runova! (GUI/terminal glasno upozoravaju dok je aktivno.)
DEV_DATA_SUBSET = 200 #None

# =========================== MODELI =========================== #
NUM_CLASSES = len(CLASS_NAMES) + 1   # +1 background (torchvision konvencija za fasterrcnn glavu)
COCO_IDS = {"Person": 1, "Car": 3, "Truck": 8, "Bus": 6, "Motorcycle": 4, "Bicycle": 2}      # nasih 6 -> torchvision COCO-91
COCO_YOLO_IDS = {"Person": 0, "Car": 2, "Truck": 7, "Bus": 5, "Motorcycle": 3, "Bicycle": 1}  # nasih 6 -> ultralytics COCO-80
YOLO_PATH = "/home/tomi/code/dipl/baseline_models/yolo26n/yolo26n.pt"               # putanja yolo eager .pt
YOLO26L_PATH = "/home/tomi/code/dipl/baseline_models/yolo26l/yolo26l.pt"

# Koji model se default analizira/komprimira (GUI moze odabrati drugi za pojedini run).
MODEL_SPEC = YOLO_PATH #YOLO_PATH            # "fasterrcnn" (string) ILI varijabla YOLO_PATH (bez navodnika!) ILI putanja .pt

USE_PROFILE_ADAPTERS = True   # True = novi per-komponenta profili (profiles.py); False = stari monolitni adapteri (sigurnost).

# =========================== ANALIZA (Overview / dijagnostika) =========================== #
GRAD_MAX_IMAGES = 1000   # broj slika za deep-analizu (grad-prolaz + census aktivnosti). Vise = stabilnije rangiranje, sporije.
GRAD_BATCH = 4           # batch za gradijentni prolaz u analizi. Veci = brze ali vise VRAM-a.
SHOW_LAYER_TABLE = True  # ispisati punu per-layer tablicu (terminal + Overview detalji).

# =========================== EVALUACIJA / MJERENJE =========================== #
EVAL_MAX = None    # cap slika za perf_report na TRAIN/TEST (None = svi). Manji = brzi report, manje tocan baseline/final.
VAL_CAP = 1000     # val-MONITOR tijekom procesa (FT/prune/grow): random <=N val slika SVAKU epohu (metrika za early-stop).
                   # perf_report (baseline/final) IGNORIRA ovo i mjeri na PUNOM val skupu (tocan broj za usporedbu).
EVAL_BATCH = 8     # batch za eval (mAP/brzina, bez grada). Veci = brze; sigurnije na 8GB nego trening batch.

# =========================== TRENING / KOMPRESIJA =========================== #
TRAIN_BATCH = 8    # batch za precompute teachera + FT. BAKED u teacher cache (promjena INVALIDIRA cache!). Veci = vise VRAM-a.
CENSUS_MAX = 1000  # slike za dead/near-dead census prije reza. Vise = stabilniji skup mrtvih jedinica, sporije.
FT_PATIENCE = 5    # FT recovery early-stop: koliko epoha bez popravka val-metrike prije prekida. Veci = duze trazi.
FT_MAX_EPOCHS = 20  # gornja granica FT epoha (sigurnosna granica ako patience ne okine).
FT_METRICS = ["map"]  # koje metrike FT optimizira (early-stop). Subset {"map"=mAP@[.50:.95], "mar_100"}. Default ["map"].
                      # Vise odjednom = stroza kontrola kvalitete (SVE moraju doseci cilj), ali sporije i teze za naci.
FT_RECOVERY_FRAC = 0.75 #0.98  # prijevremeni stop kad se SVAKA optimizirana metrika oporavi na >= ovaj udio ORIGINALA (-2%).

# =========================== FAZA 2 (uvjet zaustavljanja; GUI ga nudi, ali jos se NE koristi) =========================== #
PHASE2_STOP_METRIC = "gflops"  # JEDINI uvjet zaustavljanja Faze 2: "gflops" (default) | "params" | "map" | "mar_100".
PHASE2_STOP_FRAC = 0.25        # ciljni udio POCETNE vrijednosti (default 25% pocetnih GFLOPs). GUI slider raspon 0.05–0.95.

# =========================== FAZA 2 — KONTINUIRANI PRUNE (+ uvjetni GROW) =========================== #
# Quality-gated petlja vs ORIGINAL baseline (tol = FT_RECOVERY_FRAC × original). Cilj: NAJMANJI model unutar tolerancije.
# Prune kad je metrika iznad praga (po risk/reward score/cost), inace FT (+grow) dok se ne vrati; stop kad rez ne oporavi.
# HW-ALIGNMENT (GUI bira kvantizaciju -> M): soft DIREKCIONI nudge u prune/grow rangiranju + driver za MIN_ALIVE.
# Poravnava broj kanala na visekratnik M radi int8/fp16 brzine (NE forsira korake od M; samo motivira smjer).
ALIGN_M = 32        # ciljani visekratnik = kvantizacija: INT8/CHW32 -> 32, FP16 -> 8 (GUI bira po runu; 32 nadskup i za fp16/fp32).
ALIGN_BETA = 1.0 #1.0    # jacina alignment nudge-a (0 = ISKLJUCENO). prune faktor in [1-beta,1], grow faktor in [1,1+beta].
ALIGN_POW = 1       # ostrina krivulje (1=linearno/siroko; 2=ostrije, samo tik uz granice).
PHASE2_PRUNE_STEP_FRAC = 0.015   # GFLOPs maknuti PO prune koraku = 1.5% ORIGINALNIH GFLOPs (apsolutno, konstantno).

PHASE2_PRUNE_LAYER_CAP = 0.15    # max udio kanala JEDNOG sloja maknut u JEDNOM koraku (sprjecava kolaps sloja).
PHASE2_MIN_ALIVE_FRAC = 0.00001 #0.05     # floor: u svakom sloju ostaje >= 5% kanala.
PHASE2_MIN_ALIVE = ALIGN_M // 2  # floor: min kanala/sloj = pola pocice (M/2): INT8->16, FP16->4. VEZAN uz ALIGN_M (kvantizaciju).
PHASE2_COST_FLOPS_W = 0.60       # cost = w·(FLOPs udio) + (1-w)·(params udio); udjeli normalizirani -> skale nestaju. Veci w = vise tezi rezu FLOP-teskih.
PHASE2_PRUNE_PATIENCE = 5        # nakon reza: koliko epoha (FT/grow) bez povratka >= tol prije KRAJA.
PHASE2_MAX_STEPS = 200           # sigurnosna gornja granica broja morph koraka.
# GROW (function-preserving rast iz reinvest-poola; SAMO u Morph koraku, uz prune).
PHASE2_REINVEST_FRAC = 0.30      # grow smije potrositi <= 30% OSLOBODENIH GFLOPs (reinvest pool).
# GROW per-event cap = ALIGN_M kanala (FIKSNO = velicina pocice; vezano uz kvantizaciju, cita se dinamicki iz config.ALIGN_M
# u _select_grow_plan). Bira se tako da gap do sljedeceg ×M (g = (-w)%M < M) UVIJEK stane u jedan event -> align snap se moze dovrsiti.
PHASE2_GROW_DOM = 4.0            # grow samo ako top benefit/FLOP >= 2x medijan kandidata (jedinstveni dominance prag).
PHASE2_GROW_MAX_LAYERS = 3       # max RAZLICITIH slojeva narastenih u JEDNOM grow koraku (per-kanal heap; ostali cekaju iduci korak).
PHASE2_CHURN_COOLDOWN = 2        # anti grow<->prune churn: narasli sloj NE smije biti rezan (ni svjez rezani narastao) iducih
                                 # N MORPH-dogadaja (recovery epohe se NE broje). Daje function-preserving kanalima vremena da
                                 # skupe importance (inace im je grad~0 -> prune ih odmah kosi -> grow ih opet vrati = period-N vrtnja).

# =========================== PUTANJE =========================== #
_HERE = Path(__file__).parent
TMP_ROOT = str(_HERE / "tmp")        # teacher cache (tmp/<model>/) + GUI job (tmp/gui_job/)
MODELS_DIR = str(_HERE / "models")   # spremljeni kompresirani modeli (.pt)
