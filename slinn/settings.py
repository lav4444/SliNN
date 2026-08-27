"""slinn/config.py — kompresijski hiperparametri jezgre.

TASK-AGNOSTICNO: ovdje smiju samo postavke koje vrijede za BILO KOJI model i task.
Nista o datasetu, razredima, putanjama modela ni obiteljima (yolo/frcnn) — to je ili
korisnicki ulaz (job config.json: model_path/dataset_path) ili ide u
slinn/plugins/detection/. Preslikano iz morphology/config.py, bez promjene vrijednosti.
"""

# =========================== HW-PORAVNANJE (kvantizacija) =========================== #
# Soft DIREKCIONI nudge u prune/grow rangiranju + driver za MIN_ALIVE. Poravnava broj kanala
# na visekratnik M radi int8/fp16 brzine (NE forsira korake od M; samo motivira smjer).
ALIGN_M = 32        # ciljani visekratnik: INT8/CHW32 -> 32, FP16 -> 8 (GUI bira po runu).
ALIGN_BETA = 1.0    # jacina nudge-a (0 = ISKLJUCENO). prune faktor in [1-beta,1], grow in [1,1+beta].
ALIGN_POW = 1       # ostrina krivulje (1=linearno/siroko; 2=ostrije, samo tik uz granice).

# =========================== KONTINUIRANI PRUNE =========================== #
PHASE2_PRUNE_STEP_FRAC = 0.015   # GFLOPs maknuti PO koraku = 1.5% ORIGINALNIH GFLOPs (apsolutno, konstantno).
PHASE2_PRUNE_LAYER_CAP = 0.15    # max udio kanala JEDNOG sloja maknut u JEDNOM koraku (sprjecava kolaps sloja).
PHASE2_MIN_ALIVE_FRAC = 0.00001  # floor: udio kanala koji u svakom sloju ostaje.
PHASE2_MIN_ALIVE = ALIGN_M // 2  # floor: min kanala/sloj = pola plocice (M/2). VEZAN uz ALIGN_M.
PHASE2_COST_FLOPS_W = 0.60       # cost = w·(FLOPs udio) + (1-w)·(params udio); udjeli normalizirani.
# --- KRITERIJ VAZNOSTI ZA PRUNE (rang kanala) ---------------------------------------------------
# "grad"   = mean|d(KD)/dw| po izlaznoj jedinici        (naslijedjeno iz morphology `_kd_grad_importance`)
# "taylor" = mean|d(KD)/dw · w| po izlaznoj jedinici    (g·w; pobjednik u critereum_experiment3)
# GROW signal (`gavg`, signed grad za GradMax SVD) je ISTI u oba slucaja - mijenja se samo PRUNE rang.
PRUNE_IMPORTANCE = "grad"
# Per-sloj normalizacija imp PRIJE globalnog rangiranja (usporedivost medju slojevima):
# "none" = sirovo (dosad; skalu korigira samo dijeljenje spregnutim troskom u _align_prune_score)
# "mean" = v / mean(v)   "max" = v / max(v)   (ekvivalenti tp `_normalize` iz importance_normalisation)
PRUNE_IMP_NORM = "none"
PHASE2_PRUNE_PATIENCE = 5        # nakon reza: koliko epoha bez povratka >= tol prije KRAJA.
PHASE2_MAX_STEPS = 200           # sigurnosna gornja granica broja morph koraka.
PHASE2_PRUNE_ROUNDS = 4          # ZATVORENA PETLJA (6.11): max krugova plan->rez->IZMJERI->doplaniraj
                                 # unutar JEDNOG koraka. 1 = staro ponasanje (procjena se uzima zdravo
                                 # za gotovo -> rez padne i na 54% cilja). Svaki krug kosta jedan
                                 # prune_costs (tp graf) + gflops; importance se NE racuna ponovno.
PHASE2_PRUNE_SLACK = 0.02        # cilj se smatra pogodenim kad ostatak padne ispod 2% step budzeta.

# =========================== GROW (function-preserving, iz reinvest-poola) =========================== #
PHASE2_REINVEST_FRAC = 0.15      # grow smije potrositi <= 15% OSLOBODENIH GFLOPs.
                                 # (Bilo 0.30. Mjereno je trosio tek 1.4-2.3% oslobodjenog, pa 30%
                                 #  ionako nikad nije bilo iskoristeno; 15% je posteniji strop i manje
                                 #  radi protiv cilja Faze 1 = NAJMANJI model iznad praga.)
PHASE2_GROW_DOM = 4.0            # grow samo ako top benefit/FLOP >= ovaj faktor medijana kandidata.
PHASE2_GROW_MAX_LAYERS = 3       # max RAZLICITIH slojeva narastenih u JEDNOM grow koraku.
PHASE2_CHURN_COOLDOWN = 2        # anti grow<->prune churn: narasli sloj NE smije biti rezan (ni svjez rezani
                                 # narastao) iducih N MORPH-dogadaja (recovery epohe se NE broje). Inace
                                 # function-preserving kanali (grad~0) budu odmah pokoseni -> period-N vrtnja.

# =========================== FAZA 1: find best quality =========================== #
# Dvomodalni automat (PLAN_KOMPRESIJA 1.2): iznad praga -> MORPH (rez+rast+epoha), ispod praga ->
# FT-RECOVERY (samo epohe). Oba broaca se RESETIRAJU na svaki povratak iznad praga, pa granice
# vrijede PO RECOVERY EPIZODI, ne kroz cijelu fazu.
F1_FT_MAX_EPOCHS = 7     # max FT epoha u JEDNOJ recovery epizodi prije KRAJA faze.
                         # (10 -> 5 -> 7. Sa `patience 3` strop je postao ogranicenje, a najduza
                         #  USPJESNA epizoda na yolo26n trajala je tocno 7 epoha — sa 5 bi bila
                         #  odsjecena dvije epohe prije potvrde, i to bas ona koja je dala najbolji
                         #  model. 7 pokriva sve sto je u tom runu stvarno uspjelo.)
F1_FT_PATIENCE = 3       # FT epoha bez NOVOG `curr_best` prije KRAJA faze.
                         # `curr_best` se ZASIJE vrijednoscu odmah NAKON REZA i pomice se samo kad je
                         # epoha stvarno bolja od dosad najbolje u toj epizodi. NE usporedjuje se s
                         # prethodnom epohom — inace pilasti oporavak (dolje-gore-dolje) beskonacno
                         # resetira brojac i faza se oslanja samo na strop epoha.

# =========================== FAZA 2: compression ladder =========================== #
# Od izlaza Faze 1 do IZMJERENOG strukturnog minimuma, uz F2_CHECKPOINTS verzija na linearno
# rasporedjenim razinama GFLOPs-a. Kvaliteta ovdje NIJE gate — samo se mjeri i zapisuje.
F2_FT_MAX_EPOCHS = 10    # max FT epoha izmedju dva reza.
F2_FT_PATIENCE = 1       # epoha bez popravka metrike -> dosta FT-a, rezi dalje.
F2_CHECKPOINTS = 3       # verzija na ljestvici; ZADNJA je g_min. (delta = (g_start - g_min)/N)
F2_PRUNE_STEP_FRAC = 0.05  # rez po koraku u FAZI 2 = 5% ORIGINALNIH GFLOPs (Faza 1 ostaje 1.5%).
                         # Vece je opravdano jer kvaliteta ovdje NIJE gate — razlog za sitne korake
                         # (ostati iznad praga) ne postoji. Na yolo26n: raspon ~3.2 GFLOPs znaci
                         # ~11 rezova umjesto ~36, tj. sati umjesto pola dana. I dalje 3-4 reza po
                         # precki, pa se vaznost preracuna vise puta unutar svakog segmenta.

# =========================== FT / OPORAVAK =========================== #
FT_PATIENCE = 5          # FT recovery early-stop: epoha bez popravka metrike prije prekida.
FT_MAX_EPOCHS = 20       # gornja granica FT epoha.
FT_RECOVERY_FRAC = 0.95  # TOLERANCIJA: metrika smije pasti najvise na ovaj udio ORIGINALA.
                         # Ovo je JEDINI izvor: `engine.morph_loop(metric_tol=None)` ga uzima, a GUI
                         # klizac time samo pocinje. (Do 6.9 su postojale TRI vrijednosti: ova 0.75,
                         # hardkodirano 0.90 u morph_loop i 0.97 u full_cycle.)
# =========================== MJERENJE METRIKE (PLAN_KOMPRESIJA 1.7) =========================== #
METRIC_MONITOR_FRAC = 0.5  # udio val skupa za BRZI monitor (odluka prune/recovery + patience).
                           # Podskup je FIKSAN: izracuna se jednom (seedano) i koristi svaku epohu.
                           # Mora biti fiksan jer patience usporedjuje UZASTOPNE epohe — slucajan
                           # uzorak bi u tu razliku ubacio sum uzorkovanja (morphology je tako radio
                           # i odlucivao early-stop na sumu). None/1.0 = uvijek puna metrika.

AGREEMENT_SUGGEST = 0.95 # SAMO PRIJEDLOG za GUI, NE drugi gate. Postoji JEDAN prag kvalitete
                         # (`FT_RECOVERY_FRAC` / klizac), bez obzira mjeri li se task-metrika ili
                         # slaganje s uciteljem. Ali te dvije brojke NISU ista vrsta:
                         #   task-metrika: baseline je sto model zna (npr. mAP 0.4126); 0.90 = -10% kvalitete
                         #   agreement:    baseline je TOCNO 1.0 (student JE ucitelj na koraku 0);
                         #                 0.90 = svaki deseti ulaz dobije DRUGI odgovor
                         # Agreement je usto ZAMJENA, ne ono sto nas zanima — model se ne isporucuje jer se
                         # slaze s drugim modelom, nego jer radi. Kad to ne mozemo provjeriti, od zamjene
                         # trazimo vise. 0.95 = ~5% preokrenutih odgovora (najgore ~5 pb kvalitete).

# =========================== BATCH / EVAL =========================== #
TRAIN_BATCH = 8   # fallback batch (autobatch ga nadglasa kad ima GPU). BAKED u teacher cache.
EVAL_BATCH = 8    # batch za eval bez gradijenata.
VAL_CAP = None    # cap val slika za monitor tijekom procesa (ne za zavrsni izvjestaj).

# ⚠️ PRIVREMENO (dev): kapira SVAKI split na prvih N uzoraka -> DRASTICNO brze, NISKA vjernost.
# MAKNI (None) prije pravih runova.
DEV_DATA_SUBSET = 5000

# =========================== PUTANJE =========================== #
from pathlib import Path  # noqa: E402

_HERE = Path(__file__).parent
TMP_ROOT = str(_HERE / "tmp")        # teacher cache (tmp/<model>/) + GUI job (tmp/gui_job/)
MODELS_DIR = str(_HERE / "models")   # spremljeni kompresirani modeli (.pt)
RUNS_DIR = str(_HERE / "runs")       # JEDAN FOLDER PO KOMPRESIJI: runs/<model>_<timestamp>/
                                     # log + trajektorija + svi checkpointi + run_meta.json.
                                     # tmp/gui_job/ ostaje samo radni prostor GUI-ja (config,
                                     # status) — izlazi vise ne zive ondje pa se ne mogu pregaziti.
