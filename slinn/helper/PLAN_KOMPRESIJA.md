# PLAN_KOMPRESIJA.md — dvofazna kompresijska petlja (tehnicki plan)

> **Status: OBJE FAZE IMPLEMENTIRANE I PROVJERENE** (2026-08-23). §1–§6 su izvorni plan i drze;
> §7–§8c su zapisnik izvedbe, ukljucujuci mjerenja koja su promijenila tri odluke iz plana.
> Gdje se plan i kod razilaze, KOD je istina — sekcije nize su uskladjene. Zapisnik mergea i faza
> 1–6 ostaje u [PLAN.md](PLAN.md). Nastao nakon detaljne analize `legacy/morphology`.
>
> **Sto NIJE napravljeno:** pun run Faze 2 na detekciji (samo housing_mlp lancano); yolo Faza 1 treba
> ponovni run pod novim pravilom patiencea.

### ODLUCENO (2026-08-23, korisnik)

| # | pitanje | odluka |
|---|---|---|
| Q1 | reset brojača u recoveryju | **po epizodi** — svaki povratak iznad praga briše `ft_used` i `no_imp` |
| Q2 | grow u Fazi 1 | **ostaje, ali `PHASE2_REINVEST_FRAC` 0.30 → 0.15** |
| Q3 | mjerenje metrike | **monitor na FIKSNOJ polovici val skupa + puna metrika samo pri kandidaturi za `LAST_GOOD`** |
| Q4 | veličina reza | **fiksnih 1.5%** u Fazi 1 (bez adaptacije) |
| Q5 | `g_min` | **izmjeren probom** (trial-rez na `MIN_ALIVE`, izmjeri, baci) |
| Q6 | grow u Fazi 2 | **ostaje 15%**, ista postavka kao Faza 1 — uz izuzetak na zadnjoj prečki (v. §2.5) |
| Q7 | ljestvica | **Δ = (g_start − g_min)/3** → 3 nove verzije + Faza 1 = **4 ukupno** |
| Q8 | korak u Fazi 2 | ~~ostaje 1.5%~~ → **5%** (`F2_PRUNE_STEP_FRAC`), v. §8c |

**Tri odluke naknadno promijenjene MJERENJEM** (obrazlozenje u §8b/§8c):

| | plan | sada | zasto |
|---|---|---|---|
| patience Faze 1 | 2, vs prethodna epoha | **3, vs `curr_best`** | s 2 je brojao SUM (padovi 0.0002–0.0005) |
| strop FT epoha | 10 → 5 | **7** | najduza USPJESNA epizoda trajala je tocno 7 |
| korak Faze 2 | 1.5% | **5%** | kvaliteta ondje nije gate; ~11 rezova umjesto ~36 |

---

## 0. Sto se mijenja u odnosu na danasnji `morph_loop`

| | danas | novi plan |
|---|---|---|
| dead/near-dead | opcija (`dead=False`) | **potpuno uklonjeno** — ni kao opcija |
| struktura | jedna petlja, rez svaki korak | **dvije faze**, svaka sa svojim ciljem |
| oporavak | nema; 3× ispod praga → stop | **FT-RECOVERY mod**: prestani rezati, vrati se, nastavi |
| FT | 6 gradijentnih koraka | **puna epoha** preko svih batcheva |
| optimizer | novi Prodigy svakih 6 koraka | **jedan perzistentan** po fazi, warmup jednom |
| izlaz | 1 model | **Faza 1: 1 checkpoint** + **Faza 2: 3 checkpointa** |
| cilj Faze 2 | `target_frac` iz GUI-ja | **izmjeren strukturni minimum**, pa linearna ljestvica |

**Zasto se dead-removal mice:** vec je bio `dead=False` po defaultu jer je census-fragilan i nije
KD-siguran (voc mIoU 0.47→0.10 od samog dead-removala, PLAN.md 5.6). KD-grad prune ga svejedno
subsumira — truly-dead kanal ima ~0 KD-grad pa se rezuje prvi, ali output-sigurno. Kod ostaje u
`morph.py` (koristi ga jos `_apply_prune_plan` obrazac), samo ga vise nitko ne zove.

---

## 1. FAZA 1 — `find_best_quality`

### 1.1 Cilj

Naci **najmanji model koji je JOS iznad praga kvalitete**. Prag je udio metrike ucitelja i zadaje ga
korisnik klizacem (`FT_RECOVERY_FRAC = 0.95` je default). To je JEDINI prag — i kad se mjeri
task-metrika i kad se mjeri slaganje s uciteljem (v. §8c).

Izlaz: `best_quality_model.pt` — **garantirano iznad praga**. U najgorem slucaju identican pocetnom
modelu (pocetni je original, cija je metrika 100% sebe → uvijek prolazi prag < 1.0).

### 1.2 Automat

```
                    ┌─────────────────────────────────────┐
                    │  metrika >= prag ?                  │
                    └──────────┬──────────────┬───────────┘
                              DA             NE
                               │              │
                    ┌──────────▼───────┐  ┌───▼────────────────────┐
                    │  MORPH korak     │  │  FT-RECOVERY korak     │
                    │  ─────────────   │  │  ──────────────────    │
                    │  1 prune (1.5%)  │  │  BEZ reza, BEZ rasta   │
                    │  1 grow (pool)   │  │  1 puna KD epoha       │
                    │  1 puna KD epoha │  │  ft_used++             │
                    │                  │  │  no_imp++ ili reset    │
                    │  snapshot        │  │                        │
                    │  LAST_GOOD  ★    │  │                        │
                    └──────────┬───────┘  └───┬────────────────────┘
                               │              │
                               └──────┬───────┘
                                      ▼
                                izmjeri metriku
                                      │
                    ┌─────────────────┴──────────────────┐
                    │ vratio se iznad praga?             │
                    │   → ft_used = 0, no_imp = 0        │
                    │   → natrag u MORPH                 │
                    └────────────────────────────────────┘
```

**★ LAST_GOOD** = deepcopy studenta svaki put kad izmjerena metrika ≥ prag. Cuva se ONAJ s
**najmanje GFLOPs-a**. To je jedini izvor `best_quality_model.pt` — nikad se ne isporucuje model
koji nije izmjeren iznad praga.

### 1.3 Izlazni uvjeti Faze 1

| uvjet | vrijednost | znacenje |
|---|---|---|
| **oporavak ne uspijeva** | `ft_used >= F1_FT_MAX_EPOCHS = 7` | 7 FT epoha u JEDNOJ epizodi bez povratka iznad praga |
| **oporavak stagnira** | `no_imp >= F1_FT_PATIENCE = 3` | 3 FT epohe bez NOVOG `curr_best` |
| **nema kandidata** | `n_rem == 0` bez cooldowna | off-limits/banned/floor/cap odbili sve |
| **sigurnosna** | `PHASE2_MAX_STEPS = 200` | |

`curr_best` = najbolja metrika u TEKUCOJ epizodi. Zasijava se vrijednoscu ODMAH NAKON REZA i pomice
se samo kad je epoha bolja od nje. NE usporedjuje se s prethodnom epohom — pilasti oporavak
(dolje-gore-dolje oko iste razine) bi inace beskonacno resetirao brojac.

`ft_used` i `no_imp` se **resetiraju na ulasku u svaku novu recovery epizodu** (tj. cim se model
vrati iznad praga) — odluka **Q1**. Posljedica: trajanje faze nije unaprijed poznato, jedina tvrda
granica je `PHASE2_MAX_STEPS = 200`.

### 1.4 Pseudokod

```python
g0        = gflops(student)
baseline  = metric_fn(student)              # metrika ORIGINALA
floor     = tol * baseline                  # tol = klizac (default FT_RECOVERY_FRAC)
opt       = new_prodigy(student)            # JEDAN optimizer, gstep persistira
gstep     = 0
last_good = deepcopy(student); lg_gflops = g0
ft_used = no_imp = 0
prev_metric = baseline

for step in 1..MAX_STEPS:
    if prev_metric >= floor:                          # ---------- MORPH ----------
        prune_closed_loop(student, target = 0.015 * g0)     # 6.11, mjeri i doplanira
        grow_from_pool(student)
        opt = rebuild_prodigy(student)                # arh. se promijenila -> novi tenzori
        ft_used = no_imp = 0
    else:                                             # ------- FT-RECOVERY -------
        pass                                          # bez reza i rasta

    kd, gstep = kd_epoch(student, opt, gstep)         # PUNA epoha preko svih batcheva
    m = metric_fn(student)

    if m >= floor and gflops(student) < lg_gflops:
        full = metric_fn(student)                     # POTVRDA na punom skupu (monitor je optimistican)
        if full >= floor_full:
            last_good = deepcopy(student); lg_gflops = gflops(student)
        else:
            m_above = False                           # tretiraj kao pad

    if m_above:                                       # vratili smo se -> epizoda gotova
        curr_best = None; ft_used = no_imp = 0
    elif mode == "morph":                             # rez nas je oborio -> POCETAK epizode
        curr_best, ft_used, no_imp = m, 0, 0          # zasij `curr_best` vrijednoscu nakon reza
    else:                                             # cista FT epoha
        ft_used += 1
        if m > curr_best + 1e-4:
            curr_best, no_imp = m, 0                  # napredak
        else:
            no_imp += 1
        if no_imp >= F1_FT_PATIENCE or ft_used >= F1_FT_MAX_EPOCHS:
            break                                     # oporavak propao -> kraj Faze 1

save(last_good, "best_quality_model.pt")              # garantirano iznad praga
```

### 1.5 FT epoha (`kd_epoch`) — sto tocno

```
za svaki batch u `batches`:                  # SVI materijalizirani train batchevi
    lr = 1.0 * min(1.0, (gstep+1) / warmup)  # linearni warmup kroz PRVU epohu
    loss = loss_fn(student, imgs)            # enhancer-KD ili genericki kd_loss
    loss.backward()
    clip_grad_norm_(5.0)
    opt.step(); gstep += 1
```

- **BN ostaje zamrznut** (`_bn_eval`) — nasa disciplina, [[bn-eval-detection-trainmode]].
  Morphology je pustao BN da trenira; to je bio izvor lazno niskih mAP-ova. **Ne kopiramo to.**
- **Bez GT-a.** Loss je iskljucivo KD vs zamrznuti original, [[kd-only-no-gt]].
- `warmup = len(batches)` — jedna epoha, izracunato jednom na pocetku faze.

### 1.6 Prodigy — perzistencija

Danas: novi Prodigy pri svakom pozivu `prune_ft_recover`, bez warmupa, 6 koraka. Prodigy treba
desetke koraka da procijeni `d` — 6 koraka ga baci prije nego sto se stabilizira.

Novo (morphology recept):

```python
prodigyopt.Prodigy(params, lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)
```

- **rebuild SAMO na arhitekturnu promjenu** (prune ili grow zamijene tenzore pa stari state pokazuje
  na mrtve parametre)
- **`gstep` se NE resetira** → warmup se ne restarta; `d` procjena se gubi na rebuildu, ali to je
  neizbjezno i dogadja se jednom po morph koraku umjesto svakih 6 koraka.

### 1.7 Mjerenje metrike — MONITOR vs POTVRDA (odluka Q3)

Puna val metrika traje ~36 s. Mjeriti je nakon svake epohe znaci ~30 min po runu samo na mjerenje.
Rjesenje: **dvije razine**.

| razina | skup | kada | sto odlucuje |
|---|---|---|---|
| **MONITOR** | **fiksna polovica** val skupa (`METRIC_MONITOR_FRAC = 0.5`) | nakon SVAKE epohe | prune vs recovery, `ft_used`, `no_imp` |
| **POTVRDA** | **cijeli** val skup | samo kad monitor kaze "iznad praga I manji GFLOPs od dosadasnjeg `LAST_GOOD`" | upis u `LAST_GOOD` |

**Podskup mora biti FIKSAN, ne slucajan svaki put.** Patience usporedjuje epohu N s N−1; ako se skup
mijenja, razlika sadrzi sum uzorkovanja pa "nema popravka" moze biti cista slucajnost. To je tocno
ono sto je morphology radio krivo (`_random_val_loader` je vrtio `random.sample` pri svakom pozivu).
Metrika ne ulazi u loss (KD je bez oznaka, GT samo u gateu) pa nema putanje kojom bi se model
"prenaucio" na monitor — jedini argument za random otpada.

**Podskup mora biti KORACNI/stratificiran, ne `files[:n/2]`.** Vec smo se opekli: `pairs_classification`
je uzimao prvih N sortiranih datoteka i dobio sve iz jedne klase → f1 = 0.0000 (PLAN.md 6.8). Za
monitor treba `files[::2]` ili per-klasa round-robin, izvedeno deterministicki iz strukture skupa.
Postojeci `metric._cap(n)` uzima PRVIH n — **ne smije se koristiti za monitor bez izmjene**.

Ako `POTVRDA` na punom skupu ispadne ispod praga (monitor je bio optimistican), model se **ne** upisuje
u `LAST_GOOD` i petlja nastavlja kao da je pao ispod praga.

---

## 2. FAZA 2 — `compression_ladder`

### 2.1 Cilj

Krenuti od `best_quality_model.pt` i rezati **do strukturnog minimuma**, spremajuci 3 verzije na
linearno rasporedjenim razinama GFLOPs-a. Kvaliteta ovdje **nije gate** — ona se samo mjeri i
zapisuje uz svaki checkpoint. Korisnik na kraju bira verziju iz trajektorije ([[morph-pipeline-plan]]).

### 2.2 Minimum ostvarivih GFLOPs-a — `probe_min_gflops()`

Trebamo `g_min` PRIJE nego krenemo, da mozemo podijeliti raspon.

**Predlozena metoda (mjerena, ne procijenjena):**

```
trial = deepcopy(best_quality_model)
za svaki prunabilan, ne-banned, ne-depthwise-root sloj:
    ciljna sirina = max(PHASE2_MIN_ALIVE, ceil(PHASE2_MIN_ALIVE_FRAC * C))
    rezi na tu sirinu preko tp, uz trial+forward+rollback
g_min = gflops(trial); trial se BACA
```

Nuspojava koja se isplati: probe usput napuni `banned` skup (slojevi koji puknu na forward-checku),
pa Faza 2 krece s vec ociscenom listom kandidata.

**Zasto mjereno, a ne analiticki:** nauceno u 6.11 — spregnuti cost model precjenjuje. Analiticki
zbroj po sloju ne uzima u obzir da rez producenta vec suzuje potrosaca. Jedina brojka kojoj vjerujemo
je izmjerena.

**Cijena:** jedan deepcopy + tp grupa po sloju. Za yolo26n (~80 prunabilnih listova) procjena je
~1–2 min. Vidi **Q4**.

### 2.3 Ljestvica

```
Δ = (g_start − g_min) / 3

  g_start ──────── g_start−Δ ──────── g_start−2Δ ──────── g_min
  (= Faza 1)          CP1                 CP2              CP3
  vec spremljen       novi                novi             novi
```

Ukupno **4 verzije** na disku: `best_quality_model.pt` + `ckpt_1.pt`, `ckpt_2.pt`, `ckpt_3.pt`.
Svaki se sprema **full-eager** (`torch.save(model)`, ne state_dict) — [[save-models-full-eager]].
Uz svaki ide zapis u trajektoriji: GFLOPs, params, MB, metrika, align_score.

### 2.4 Petlja Faze 2

```python
g_min   = probe_min_gflops(student)
delta   = (gflops(student) - g_min) / 3
targets = [gflops(student) - delta, gflops(student) - 2*delta, g_min]
opt     = new_prodigy(student); gstep = 0

for i, t in enumerate(targets):
    last_rung = (i == len(targets) - 1)        # zadnja precka = g_min -> grow OFF (v. 2.5)
    while gflops(student) > t:
        prune_closed_loop(student, target = min(F2_PRUNE_STEP_FRAC * g_start, gflops(student) - t))
        if not last_rung:
            grow_from_pool(student)            # Q6: 15%, isto kao Faza 1
        opt = rebuild_prodigy(student)
        ft_until(opt, max_epochs = F2_FT_MAX_EPOCHS = 10, patience = F2_FT_PATIENCE = 1)
        if nema_kandidata: break
    save_eager(student, f"ckpt_{i+1}.pt")
```

`ft_until(patience=1)` = trenirati dok jedna epoha ne prodje bez popravka metrike. Tipicno 2 epohe.

### 2.5 Zasto grow pada na zadnjoj precki (izvedeno iz Q5+Q6)

`g_min` se mjeri probom koja radi **cisti rez** — grow u njoj ne sudjeluje. Ako grow ostane aktivan
dok se spustamo prema toj brojci, blizu dna se korak reza suzava na `gflops − t`, a grow svejedno
smije vratiti do 15% oslobodjenog — pa petlja oscilira oko `g_min` i nikad ga ne dosegne. Meta je
po konstrukciji nedostizna dok grow radi.

Zato: **grow aktivan na preckama 1 i 2** (tamo kvaliteta jos vrijedi i reinvest ima smisla),
**iskljucen na zadnjoj** (dno je definirano bez njega). Ovo NIJE promjena odluke Q6 — Q6 kaze da
grow ostaje u Fazi 2; ovo je samo uvjet pod kojim je zadnja precka uopće dostizna.

### 2.6 Trosak Faze 2 (Q8: korak ostaje 1.5%)

Ako je raspon `g_start → g_min` npr. 60% GFLOPs-a, pri koraku 1.5% originala to je **~40 rezova**,
svaki s FT-om (patience=1 → tipicno 2 epohe) = **~80 epoha**. Na punom skupu to su sati. Odluka je
svjesna (gusca trajektorija, jedna konstanta manje). Ako se u praksi pokaze preskupo, prvo sto treba
probati je **FT samo pred checkpoint** (kratki FT izmedju, pun FT tek prije spremanja) — to cuva
kvalitetu isporucenih verzija, a ne placa punu cijenu na medjukoracima.

---

## 3. Sto vec imamo (NE graditi ponovno)

| mehanizam | gdje | stanje |
|---|---|---|
| **churn cooldown** | `engine.py` `grown_at`/`pruned_at`, `PHASE2_CHURN_COOLDOWN = 2` | ✅ prenesen iz morphology, sa soft-overrideom kad blokira sve |
| precizan rez (zatvorena petlja) | `engine.py` 6.11 | ✅ 98–99% cilja |
| spregnuti cost model | `morph.py` `coupled_unit_cost` | ✅ + `in_ch` fix za (B,S,C) |
| align nudge (prune i grow) | `morph.py` `align_factors` | ✅ dinamican, po kanalu |
| reinvest pool | `engine.py`, `PHASE2_REINVEST_FRAC = 0.30` | ✅ |
| GradMax grow | `morph.py` `_select_grow_plan` / `_grow_decide` | ✅ function-preserving, provjereno |
| trial + forward + rollback + ban | `morph.py` `_apply_prune_plan` | ✅ |
| teacher signal cache | `engine.py` `TeacherSigCache` | ✅ RAM+disk plan |
| autobatch | `engine.py` | ✅ |
| metrika / teacher-agreement fallback | `metric.py`, `full_cycle` | ✅ |

## 4. Sto treba napisati

| # | zahvat | datoteka | stanje |
|---|---|---|---|
| 1 | `kd_epoch()` — puna epoha, warmup, perzistentan `gstep` | `engine.py` | ✅ |
| 2 | perzistentan Prodigy + rebuild-na-arh-promjenu | `engine.py` | ✅ |
| 3 | dvomodalni automat (MORPH / FT-RECOVERY) | `engine.py` `phase1_loop` | ✅ |
| 4 | `LAST_GOOD` snapshot + garancija izlaza | `engine.py` | ✅ |
| 5 | `probe_min_gflops()` | `morph.py` | ✅ |
| 6 | `phase2_ladder()` + eager checkpointi | `engine.py` | ✅ |
| 7 | GUI: dva checkboxa, jedan klizac praga, run-folderi | `gui/gui.py`, `worker.py` | ✅ |
| 8 | izbaciti `dead_removal` iz `full_cycle` | `engine.py` | ⏳ ceka da nove faze preuzmu |

Usput izdvojeno: `MorphState` + `morph_step` (prune+grow u JEDAN dom, dijele ga obje faze),
`ft_until`, `run_phase1`/`run_phase2`, `new_run_dir`.

## 5. Nove konstante (`settings.py`)

Stanje u `settings.py` NAKON svih mjerenja:

```python
# --- PRAG KVALITETE (jedan, za obje vrste metrike) ---
FT_RECOVERY_FRAC    = 0.95   # default klizaca; JEDINI prag
AGREEMENT_SUGGEST   = 0.95   # SAMO prijedlog za GUI kad nema task-metrike, NE drugi gate

# --- FAZA 1 ---
F1_FT_MAX_EPOCHS    = 7      # max FT epoha u JEDNOJ recovery epizodi
F1_FT_PATIENCE      = 3      # FT epoha bez NOVOG `curr_best`
                             # (Q1: OBA se resetiraju na svakom povratku iznad praga)

# --- MJERENJE (Q3) ---
METRIC_MONITOR_FRAC = 0.5    # udio val skupa za brzi monitor. FIKSAN (isti uzorci svaku epohu) i
                             # seedano NASUMICAN, NE `[::k]` — round-robin po razredima bi kod parnog
                             # broja razreda dao samo parne. Puna metrika samo pri kandidaturi za LAST_GOOD.

# --- FAZA 2 ---
F2_PRUNE_STEP_FRAC  = 0.05   # rez po koraku (Faza 1 ostaje 1.5%); racuna se od g_start FAZE 2
F2_FT_MAX_EPOCHS    = 10     # max FT epoha izmedju dva reza
F2_FT_PATIENCE      = 1      # epoha bez popravka -> dosta FT-a, rezi dalje
F2_CHECKPOINTS      = 3      # verzija na ljestvici; ZADNJA je g_min

# --- promijenjeno ---
PHASE2_REINVEST_FRAC = 0.15  # bilo 0.30 (Q2); mjereno je trosio tek 1.4-2.3% oslobodjenog

# --- putanje ---
RUNS_DIR = slinn/runs        # jedan folder po kompresiji (v. §8c)
```

Nepromijenjene: `PHASE2_PRUNE_STEP_FRAC` (0.015, Faza 1), `PHASE2_PRUNE_LAYER_CAP`,
`PHASE2_MIN_ALIVE(_FRAC)`, `PHASE2_COST_FLOPS_W`, `PHASE2_GROW_*`, `PHASE2_CHURN_COOLDOWN`,
`PHASE2_PRUNE_ROUNDS/SLACK`, `ALIGN_*`.

Mrtve, mogu se maknuti: `FT_PATIENCE`, `FT_MAX_EPOCHS` (zamijenjene s `F1_*`/`F2_*`),
`PHASE2_PRUNE_PATIENCE`, `MODELS_DIR` (izlazi idu u `RUNS_DIR`).

---

## 6. PITANJA

### ✅ ZATVORENO (2026-08-23)

**Q1 — reset brojaca.** → **po epizodi.** Svaki povratak iznad praga brise `ft_used` i `no_imp`.
Faza staje tek kad JEDNA recovery epizoda ne uspije (10 epoha ili 2 bez popravka). Trajanje nije
unaprijed poznato; gornja granica je `PHASE2_MAX_STEPS = 200`.

**Q2 — grow u Fazi 1.** → **ostaje, pool prepolovljen na 15%.** Mehanizam se ne dira (dokazan,
function-preserving, dominance prag 4× medijan), ali strop od 30% ionako nikad nije bio iskoristen
(mjereno 1.4–2.3%), pa 15% postenije opisuje stvarnost i manje radi protiv cilja faze.

**Q3 — cijena metrike.** → **monitor + potvrda**, v. §1.7. Monitor = **fiksna, koracna polovica**
val skupa nakon svake epohe; puna metrika samo pri kandidaturi za `LAST_GOOD`. Fiksan (ne slucajan)
jer patience usporedjuje uzastopne epohe — slucajan uzorak bi u tu usporedbu ubacio sum, sto je
tocno morphologyjeva greska.

**Q4 — velicina reza.** → **fiksnih 1.5%** originalnih GFLOPs. Bez adaptacije prema pragu ili prema
brzini oporavka; zatvorena petlja (6.11) sad pogadja 98–99% pa je korak stvarno 1.5%. Adaptacija se
razmatra tek ako se pokaze da faza vecinu vremena provede u recoveryju.

### ✅ ZATVORENO — Faza 2 (2026-08-23)

**Q5 — `g_min`.** → **izmjeren probom.** Trial-kopija, rez svih prunabilnih slojeva na `MIN_ALIVE`,
izmjeri GFLOPs, kopiju baci. Analiticka procjena odbijena jer je to tocno onaj tip racuna koji je u
6.11 precjenjivao za 46% — dobili bismo prenizak `g_min` i ljestvicu koju model nikad ne dosegne.
Nuspojava koja se isplati: proba usput napuni `banned` skup.

**Q6 — grow u Fazi 2.** → **ostaje 15%**, ista postavka kao Faza 1 (jedna konstanta manje za pratiti).
IZVEDENO OGRANICENJE: na **zadnjoj precki grow se gasi**, jer je `g_min` mjeren cistim rezom pa je uz
aktivan grow nedostizan. Obrazlozenje u §2.5.

**Q7 — ljestvica.** → **Δ = (g_start − g_min)/3**; checkpointi na `g_start−Δ`, `g_start−2Δ`, `g_min`.
Tri nove verzije + `best_quality_model.pt` iz Faze 1 = **4 ukupno**. Zadnja verzija JE minimum.

**Q8 — korak u Fazi 2.** → **ostaje 1.5%** originalnih GFLOPs, isti kao Faza 1. Trosak je poznat i
prihvacen (~80 epoha pri sirokom rasponu, v. §2.6); prva mjera ako ispadne preskupo je "FT samo pred
checkpoint", ne veci korak.

---

## 7. STANJE IMPLEMENTACIJE — FAZA 1 ✅ RADI (2026-08-23)

### Sto je napisano

| # | zahvat | gdje |
|---|---|---|
| 1 | `kd_epoch()` — PUNA epoha, linearni warmup, vanjski `gstep` | `engine.py` |
| 2 | `_new_prodigy()` + `lr_eff()` — perzistentan optimizer, rebuild samo na arh. promjenu | `engine.py` |
| — | `MorphState` + `morph_step()` — prune+grow izdvojen iz `morph_loop` u JEDAN dom | `engine.py` |
| 3 | `phase1_loop()` — dvomodalni automat MORPH / FT-RECOVERY | `engine.py` |
| 4 | `LAST_GOOD` + zajamcen izlaz | `engine.py` |
| — | `run_phase1()` — priprema (batchevi, enhancer-loss, cache, gate) bez dead-removala | `engine.py` |
| — | MONITOR metrika: `build_metric_fn(..., monitor_frac=)` -> 3-torka, `_fixed_subset`, `make_gt_loader(frac=)` | `gui/backend.py`, plug |
| — | `main_phase1()` + prekidac `cfg["phase"] == 1` | `gui/worker.py` |
| — | `F1_FT_MAX_EPOCHS/PATIENCE`, `METRIC_MONITOR_FRAC`, `REINVEST 0.30->0.15` | `settings.py` |

### Refaktor `morph_loop` -> `morph_step`: provjera

Prune/grow tijelo je izdvojeno da ga Faza 1 i stara petlja DIJELE. Usporedba tri runa istog joba
(A i B = identican kod PRIJE refaktora, C = poslije):

    korak     A         B         C
      1     5.8716    5.8716    5.8716     <- identicno u sva tri
      2     5.7839    5.7861    5.7832
      3     5.6955    5.6978    5.6934
      4     5.6067    5.6101    5.6048

Korak 1 (jedina potpuno deterministicka tocka) poklapa se ZNAMENKU PO ZNAMENKU u sva tri, ukljucujuci
84 kanala, 3 kruga, est 0.0899 -> act 0.0875, iste narasle slojeve i 2 banana sloja. Od koraka 2 sve
tri divergiraju — ali A i B divergiraju JEDNAKO kao B i C, iako je A<->B isti kod. To je postojeci
nedeterminizam (cuDNN backward), ne refaktor. Digit-for-digit paritet na cijeloj trajektoriji NIJE
moguc jer ga ni kod sa sobom nema.

### 🔴 NALAZ: BN je bio u train modu tijekom FT-a (detekcija)

Prvi smoke Faze 1 pokazao je nesto sto se sa starim 6-koracnim FT-om nije vidjelo: **KD loss pada,
a mAP se rusi.**

    train_bn=True    KD 0.90 -> 0.75 -> 0.51    mAP 0.3847 -> 0.3843 -> 0.3091
    train_bn=False   KD 0.24 -> 0.23 -> 0.32    mAP 0.4086 -> 0.3943 -> 0.3909

Uzrok: `plugins/detection/profiles.py::_yolo_student_signals` je zvao `_dense_decode(train_bn=True)`,
naslijedjeno iz morphology. Loss se time racuna pod BATCH statistikama, a mAP se mjeri pod RUNNING
statistikama — dva razlicita rezima. Uz 6 koraka FT-a razlika je bila u sumu; uz PUNE epohe model se
optimizira u rezim u kojem ga nikad ne evaluiramo.

FIX: `train_bn=False`. Sigurno je jer tp reze BN `running_mean/var` zajedno s kanalima, pa preostale
statistike ostaju ispravne. Ovo je nasa vec zapisana disciplina ([[bn-eval-detection-trainmode]]) koja
na ovom putu jednostavno nije bila primijenjena.

**Napomena za plan §1.5:** ondje pise "BN ostaje zamrznut" — to sada i JE istina, ali nije bila:
`student.eval()` u `kd_epoch` je enhancer nizvodno pregazio za BN module. Tvrdnja je sada tocna.

### Kraj-do-kraja run (yolo26n, DEV_DATA_SUBSET=300, tol 97%, max_steps=8)

    kor    mod     GFLOPs      params   monitor      puna    rez       KD
      0   base     5.9584   2,572,280    0.4057         —   0.0%        —
      1  MORPH     5.8685   2,535,130    0.4086    0.4179   1.5%   0.2437  <- NAJBOLJI
      2  MORPH     5.7794   2,502,010    0.3943    0.4077   3.0%   0.2346  <- NAJBOLJI
      3  MORPH     5.6872   2,453,466    0.3909         —   4.6%   0.3153     pad ispod praga
      4     FT     5.6872   2,453,466    0.3840         —   4.6%   0.2494     oporavak 0/10
      5     FT     5.6872   2,453,466    0.3976    0.4016   4.6%   0.2219     oporavak 1/10
      6     FT     5.6872   2,453,466    0.4023    0.4073   4.6%   0.1942     oporavak 2/10
      8     FT     5.6872   2,453,466    0.4001    0.4058   4.6%   0.1728     oporavak 4/10
    FAZA 1 GOTOVA · razlog: max_steps (8)
      isporuceno: korak 2 · 5.9584 -> 5.7794 (3.0% rez) · mAP 0.4077 (prag 0.4074) POTVRDENO

Sto ovo potvrdjuje:
* automat prebacuje MORPH <-> FT-RECOVERY i broaci se resetiraju po epizodi (Q1)
* `LAST_GOOD` se upisuje SAMO uz potvrdu na punom skupu — koraci 5, 6 i 8 su prosli monitor
  (0.3976/0.4023/0.4001 > monitor-prag 0.3935) ali pali punu potvrdu (< 0.4074), pa su ispravno
  tretirani kao pad. Pravilo iz §1.7 radi.
* oporavak stvarno oporavlja: mAP se penje 0.3909 -> 0.3840 -> 0.3976 -> 0.4023
* garancija izlaza drzi — prvi smoke (s BN bugom) nije nasao nijedan valjan rez i uredno je
  isporucio POCETNI model

### Kako pokrenuti

U `tmp/gui_job/config.json` dodaj `"phase": 1`. Bez toga se vrti stara `full_cycle` petlja — namjerno,
da radni put ne padne prije nego Faza 1 prodje pun run na cijelom skupu.

### GUI toggle ✅ (2026-08-23)

Prekidac **FAZA 1** na stranici Kompresija, default UKLJUCEN. Kad je ukljucen, sakriva kontrole koje u
toj fazi nemaju znacenje (`target_frac`, `ft_steps`, `dead`) i umjesto njih ispisuje granice oporavka i
udio monitor-skupa. Payload dobiva `"phase": 1`; worker po tome bira petlju.

Usput popravljena semantika metrike u trajektoriji: `metric` je sada UVIJEK vrijednost kojom petlja
ODLUCUJE (monitor kad postoji), a `metric_full` zasebna, rjedja potvrda na punom skupu. Prije su se
mijesale u istoj seriji, pa bi graf crtao dvije skale na jednoj liniji a linija praga bila bi kriva.
GUI je dobio i zaseban graf za punu metriku.

## 8. STANJE IMPLEMENTACIJE — FAZA 2 ✅ NAPISANA (2026-08-23)

| # | zahvat | gdje |
|---|---|---|
| 5 | `probe_min_gflops()` — IZMJERI strukturni minimum | `morph.py` |
| — | `ft_until()` — FT dok se popravlja (max/patience), bez praga | `engine.py` |
| 6 | `phase2_ladder()` — ljestvica + eager checkpointi + `ladder.json` | `engine.py` |
| — | `run_phase2()` — priprema pa ljestvica | `engine.py` |
| — | `main_phase2()` + `cfg["phase"] == 2` | `gui/worker.py` |
| — | GUI: radio **Faza 1 / Faza 2 / stara petlja** | `gui/gui.py` |
| — | `F2_FT_MAX_EPOCHS`, `F2_FT_PATIENCE`, `F2_CHECKPOINTS` | `settings.py` |

### `probe_min_gflops` — kako radi

Kopija modela, pa sloj po sloj rez na `floor = max(MIN_ALIVE, ceil(MIN_ALIVE_FRAC x C))`, pa mjerenje;
kopija se baca. Ciljna sirina se racuna na TRENUTNOM stanju kopije, ne na pocetnom — rez jednog sloja
spregnuto mijenja sirine drugih. Depthwise se preskace. REUSE `_apply_prune_plan`, dakle ista
trial+forward+rollback disciplina kao u pravoj petlji, pa proba usput vrati `banned` skup.

Smoke (housing_mlp, CPU): `g0 = 0.000217` -> `g_min = 0.000003` (**1.6% od g0**), 0 banananih, <0.1 s,
original nakon probe bitno netaknut (`assert` na GFLOPs). Ljestvica ispala 67.2% -> 34.4% -> 1.6%.

### Izmjereno na yolo26n (CPU, paralelno s punim runom Faze 1)

    g0     = 5.9584 GFLOPs · 85 prunabilnih slojeva · MIN_ALIVE 16
    g_min  = 2.3364 GFLOPs  (39.2% od g0)
    banned = 18 slojeva pronadjeno TIJEKOM probe
    trajanje = 17 s (5 CPU niti, nice 19) — procjena od 1-2 min bila je previsoka
    original netaknut nakon probe (assert na GFLOPs)

    ljestvica (delta = 1.2073):
      ckpt_1  ->  4.7511  (79.7% od g0)
      ckpt_2  ->  3.5437  (59.5% od g0)
      ckpt_3  ->  2.3364  (39.2% od g0)   <- g_min, grow OFF

**Sto ovo znaci:** strukturni strop kompresije yolo26n je **60.8%**, ne "koliko god". Dno drze tri
stvari: 8 slojeva izbacenih iz prunable (dijele tp-grupu s feature-KD tapom), **18 banananih**
(rez razbije forward — C2f split/concat), i floor od 16 kanala po sloju. Sve sto nije prunabilno
(glava, off-limits) ostaje pune sirine.

Tih 18 slojeva je konkretan trag za poboljsanje: kad bi se rjesio forward-safe rez kroz C2f
split/concat, `g_min` bi pao. NE dirati sada — samo zapisano.

**Napomena:** proba je vrtjena na ORIGINALU; `phase2_ladder` je svejedno ponovno mjeri iz svog
STVARNOG starta (izlaz Faze 1), jer se banned skup moze razlikovati. Brojka odavde je orijentacija.

Izvjestaj: `helper/REPORTS/probe_min_yolo26n.json`.

### Chaining

Faza 2 krece od `best_quality_model.pt` ako postoji u JOB folderu, inace od originala (i to kaze
naglas). Ucitelj je UVIJEK original — ne izlaz Faze 1.

### Sto Faza 2 sprema

`ckpt_1.pt` ... `ckpt_N.pt` full-eager ([[save-models-full-eager]]) + `ladder.json` s
`{g_start, g_min, delta, targets, checkpoints[{i, path, target, gflops, params, metric, reached, grow}]}`.
`reached=false` znaci da je precka promasena (kandidati iscrpljeni) — verzija se svejedno sprema, uz
oznaku, umjesto da se tiho preskoci.

### Cijena mjerenja u Fazi 2

Kvaliteta nije gate, pa monitor sluzi SAMO FT patienceu, a puna metrika se vrti **jednom po
checkpointu** (za manifest). To je 3 puna mjerenja po cijeloj fazi umjesto jednog po epohi.

### Sto jos nije napravljeno

* ~~pun run Faze 1~~ ✅ GOTOV — v. §8b
* **Faza 2 nije jos vrtjena na detekciji** — `probe_min_gflops` JEST izmjeren na yolo26n (v. gore),
  ali sama ljestvica (rez + FT + checkpointi) ceka slobodan GPU.
* `full_cycle`/`dead_removal` jos stoje (§4 stavka 8) — mice se tek kad obje faze prodju
* **FAZA 2** cijela — `probe_min_gflops()`, ljestvica, eager checkpointi (§4 stavke 5–6)
* `full_cycle`/`dead_removal` jos stoje (§4 stavka 8) — mice se tek kad Faza 1 preuzme

---

## 8b. PUN RUN FAZE 1 — yolo26n, cijeli skup (2026-08-23)

Postavke: tolerancija **97%**, batch 16, epoha = 367 batcheva, monitor = fiksna polovica val skupa.
Trajanje **~2 h**, 32 koraka.

    baseline   monitor 0.4202 (prag 0.4076) · puna 0.4126 (prag 0.4002)

    epizoda   koraci      epoha oporavka   ishod
      1        1-3             2          POTVRDEN  1.5%
      2        4-7             3          POTVRDEN  3.0%
      3        8-13            5          POTVRDEN  4.5%
      4       14-16            2          POTVRDEN  6.0%
      5       17-24            7          POTVRDEN  7.4%
      6       25-32            8          PROPAO -> kraj faze

    ISPORUCENO: korak 24 · 5.9584 -> 5.5153 GFLOPs (7.4% rez) · mAP 0.4007 (prag 0.4002)

**Sto je run dokazao:**
* dvomodalni automat radi na pravim podacima — pet uspjesnih recovery epizoda
* oporavak stvarno oporavlja: epizoda 5 je isla 0.4026 -> 0.4090 kroz 7 epoha i vratila model iznad praga
* tezina raste s dubinom (2, 3, 5, 2, 7, 8 epoha) — zadnja epizoda dosla do 0.4041 uz potreban 0.4076
* garancija drzi: isporucen je model IZMJEREN iznad praga, potvrdjen na PUNOM val skupu
* kalibracija monitora stabilna: razmak monitor−puna ostao u pojasu 0.0070–0.0083 kroz cijeli run

**Ogranicenje:** ovo NIJE kontrolirana usporedba sa starom petljom (mehanizmi se previse razlikuju),
i ne govori nista o drugim modelima ni taskovima.

### 🔬 Patience: `prev` vs `curr_best` — izmjereno, ne pretpostavljeno

Run je isao pod pravilom "bolji od PRETHODNE epohe". Korisnik je specificirao strozije: `curr_best`
zasijan vrijednoscu nakon reza, patience 2, strop 5. Oba pravila pustena preko ISTE zapisane
trajektorije (cisto knjigovodstvo, bez ponovnog racunanja):

    pravilo                              kraj       isporuka
    prev,      pat 2, strop 10  (staro)  korak 32   korak 24 · 7.4% rez
    curr_best, pat 2, strop 5   (novo)   korak 12   korak  7 · 3.0% rez
    curr_best, pat 3, strop 5            korak 22   korak 16 · 6.0% rez
    curr_best, pat 2, strop 8            korak 12   korak  7 · 3.0% rez

Strop NIJE ogranicenje (8 daje isto sto i 5). Vezuje **patience 2 na `curr_best`**. Mjesto gdje puca:

     9  FT  0.4050   novi curr_best
    10  FT  0.4067   novi curr_best
    11  FT  0.4062   −0.0005 od najbolje  -> 1
    12  FT  0.4065   −0.0002 od najbolje  -> 2  KRAJ
    13  FT  0.4107   <- nikad se ne dogodi; ovo bi potvrdilo 4.5%

Padovi od **0.0005 i 0.0002** su unutar podrhtavanja mjerenja mAP-a na monitor skupu. Patience 2 na
`curr_best` dakle broji SUM, ne zastoj — model je i dalje isao gore, samo ne monotono.

Implementirano je korisnikovo pravilo (patience 2, strop 5). Ako se pokaze prestrogo, jedina promjena
je `F1_FT_PATIENCE = 3` (daje 6.0%) ili prag napretka veci od suma (npr. +0.0005 umjesto +0.0001).

## 8c. GUI, IZLAZI I FINALNE KONSTANTE (2026-08-23)

### Izbor faza — dva nezavisna checkboxa

    ☑ FAZA 1 — najmanji model iznad praga
    ☐ FAZA 2 — ljestvica do strukturnog minimuma

Oba -> lancano (Faza 2 krece od izlaza Faze 1). Samo Faza 2 -> krece od **ORIGINALA**. Nijedna ->
gumb onemogucen. Stara `full_cycle` petlja je maknuta iz GUI-ja (kod ostaje, dosezan bez `phase`).

Zastavica je STRING: `"1"`, `"2"` ili `"12"`. Kljucno: odakle Faza 2 krece odlucuje CONFIG, ne
"postoji li datoteka na disku" — prije bi stari `best_quality_model.pt` iz ranijeg runa tiho postao ulaz.

Priprema (model, adapter, task, batchevi, teacher cache, baseline) radi se **jednom** i dijeli medju
fazama; `run_phase1`/`run_phase2` sad primaju gotove `batches`/`cache`.

### JEDAN prag kvalitete

`AGREEMENT_TOL` kao drugi, skriveni gate je UKINUT. Postoji jedan parametar (`FT_RECOVERY_FRAC = 0.95`,
klizac ga nadglasava). Kad nema task-metrike, GUI **sugerira** `AGREEMENT_SUGGEST = 0.95` i objasni
zasto to nije ista vrsta broja:

| | task-metrika | teacher agreement |
|---|---|---|
| baseline | sto model zna (mAP 0.4126) | **tocno 1.0** — student JE ucitelj na koraku 0 |
| 0.90 znaci | -10% kvalitete | svaki deseti ulaz dobije DRUGI odgovor |

Agreement je usto ZAMJENA, ne ono sto nas zanima; kad ne mozemo provjeriti radi li model, od zamjene
trazimo vise. Klizac se sam pomakne na 0.95, ali ostaje klizac — i poruka to kaze.

### JEDAN FOLDER PO KOMPRESIJI

    runs/<ime_modela>_<YYYYmmdd_HHMMSS>/
        worker.log            cijeli ispis
        trajectory.jsonl      sve tocke obje faze (svaka nosi `loop: 1|2`)
        best_quality_model.pt Faza 1, full-eager
        ckpt_1..N.pt          Faza 2, full-eager
        ladder.json           ciljevi + izmjereni GFLOPs/metrika po verziji
        run_meta.json         model_path, dataset, code_dirs, config, prag, sto je proizvedeno

`tmp/gui_job/` je od sada SAMO radni prostor GUI-ja (`config.json`, `status.json`, prep). `status.json`
ostaje ondje jer GUI anketira fiksnu putanju, i nosi `run_dir` — GUI odatle cita log i trajektoriju
(fallback na JOB za stare runove). **Originalni model se NE kopira**; `run_meta.json` cuva putanju.

Ime foldera: kad je ime datoteke genericko (`model.pt`, `best.pt`, `weights.pt`...) uzima se ime MAPE,
inace ime datoteke. Zoo konvencija je `<ime>/model.pt`, pa bi inace svi runovi bili `model_*`.

**Povod:** test na housing_mlp pregazio je `best_quality_model.pt` iz dvosatnog yolo runa, jer su svi
runovi pisali u isti `gui_job`. Log je prezivio (arhiviran u REPORTS), model nije. Per-run folder to
rjesava u korijenu; ranija zakrpa (`_archive_if_other_model`) je maknuta kao suvisna.

### Finalne konstante

| konstanta | vrijednost | zasto |
|---|---|---|
| `FT_RECOVERY_FRAC` | **0.95** | jedini prag kvalitete; klizac pocinje odavde |
| `AGREEMENT_SUGGEST` | 0.95 | samo prijedlog za GUI, ne gate |
| `F1_FT_PATIENCE` | **3** | FT epoha bez novog `curr_best`; s 2 je brojao SUM (padovi 0.0002-0.0005) |
| `F1_FT_MAX_EPOCHS` | **7** | najduza USPJESNA epizoda na yolo26n trajala je tocno 7 |
| `PHASE2_PRUNE_STEP_FRAC` | 0.015 | Faza 1 |
| `F2_PRUNE_STEP_FRAC` | **0.05** | Faza 2 reze krupnije — kvaliteta ondje nije gate |
| `PHASE2_REINVEST_FRAC` | 0.15 | grow; OFF na zadnjoj precki Faze 2 |
| `METRIC_MONITOR_FRAC` | 0.5 | fiksni podskup za brzi monitor |

Napomena uz `F2_PRUNE_STEP_FRAC`: korak je 5% GFLOPs-a **na pocetku Faze 2**, ne originala — Faza 2
svoj `MorphState` gradi iz vlastitog `g_start`.

### Provjereno (housing_mlp, lancano `phase="12"`, ~90 s)

    g_start 2.022e-04 (izlaz Faze 1; original 2.17e-04)   g_min 3.456e-06
      ckpt_1  1.355e-04   67.0% starta   r2 0.7540   DOSEGNUT   grow ON
      ckpt_2  6.899e-05   34.1% starta   r2 0.7536   DOSEGNUT   grow ON
      ckpt_3  3.456e-06    1.7% starta   r2 0.7418   DOSEGNUT   grow OFF

Lanac radi, sve tri precke dosegnute, grow ugasen na zadnjoj, svih 7 datoteka u run-folderu.
Usput popravljeno: FT epohe unutar precke dijelile su broj koraka pa je log izgledao zamrznut —
sad nose oznaku `34.2`, `34.3`...

---

## 8d. METRIKA IDE NA VAL — I VAL KONACNO POSTOJI (2026-08-23)

Dva povezana defekta, oba nadjena citanjem korisnikovog m5 loga.

### (1) Slaganje s uciteljem je bilo special-case

    prije:  gate_inputs = [s for b in batches for s in b][:64]
            ^ 64 uzoraka        ^ iz TRAIN batcheva

Dvije greske u jednom retku. Kvaliteta se mjerila na podacima na kojima se TRENIRA, i to na uzorku
tako malom da je rezolucija bila 1/64 = **1.6 postotnih bodova**. Uz prag 0.97 ne postoji dostizna
vrijednost izmedju 62/64 = 0.9688 i 63/64 = 0.9844 — gate nije mogao izraziti "jedva iznad praga".
Potvrda iz loga: SVAKA izmjerena vrijednost bila je tocan visekratnik 1/64 (0.8125=52/64,
0.8594=55/64, 0.9219=59/64, 0.9844=63/64...). Faza 1 je stala na patience mjereci kvantizacijski sum.

**FIX:** `engine.agreement_metrics(teacher, adapter, device, ctx, path)` — slaganje je od sada
OBICNA metrika kvalitete, pod ISTIM pravilima kao mAP/f1/r2/mIoU:

* mjeri se na **VAL splitu**, ne na train ulazima
* `metric_fn` = cijeli val, `monitor_fn` = fiksna seedana polovica (kao svugdje)
* prag iz odgovarajuce baseline vrijednosti; **nema zasebnog gatea ni zasebne tolerancije**

Tri special-casea u `engine`-u (`full_cycle`, `run_phase1`, `run_phase2`) zamijenjena jednim pozivom.
Slaganje sad ide kroz `build_metric_fn` — JEDNO mjesto odlucuje kako se mjeri kvaliteta.
`AGREEMENT_TOL` i `AGREEMENT_SAMPLES` obrisani. Za to je trebalo generalizirati
`materialize_train_batches(..., split_key=)`: isti citac sluzi za KD ulaze (`train`) i za mjerenje
kvalitete (`val`).

Izmjereno na m5: **64 -> 5000 uzoraka**, rezolucija 1.6 pb -> 0.02 pb.

### (2) `AUTO` split je znacio "cijelo stablo"

Politika je bila odlucena jos u [PLAN.md §4.5a](PLAN.md) i `dataset.stratified_split` napisan i
validiran (`helper/_splitcheck.py`) — ali **nikad spojen u izvedbeni put**. Plan to i priznaje,
doslovno na dva mjesta (`engine.py` zaglavlje i PLAN.md 5.1):

> `'AUTO'` -> pool (split=None) + seeded subset ... `stratified_split` ostaje za val/test METRIKU u 5.6

Faza 5.6 to nije napravila. `stratified_split` se pozivao SAMO iz validacijske skripte. Posljedica na
auto-pool datasetima: `count(train) == count(val) == count(test)`, iste datoteke.

**FIX:**

* `dataset.auto_carve(files, split_key, ratios=(0.70,0.15,0.15), seed=0)` — dijeli **POPIS DATOTEKA**,
  ne oznake. Grupni kljuc je ime NADFOLDERA: kod `folder_per_class` nadfolder JE razred pa je podjela
  STRATIFICIRANA; kad su sve datoteke u istom folderu ostaje jedna grupa -> obicna nasumicna (seedana)
  podjela. Degradacija je automatska, bez ijedne grane po tasku.
* `engine._candidate_files(..., split_key, carve)` — `carve=True` kad `split_plan[key] == 'AUTO'`.
  Cap se primjenjuje **tek nakon** podjele, pa `DEV_DATA_SUBSET` znaci N PO SPLITU, ne N na pool.
* `dataset._tabular_matrix(root, in_ch, split=None)` — bio slijep na split i vracao PRVI 2D array
  odgovarajuce sirine, a to je po redoslijedu kljuceva `X_train`. Sad prvo trazi `X_<split>`.
  (`metric.pairs_regression` je bio ispravan jer sam trazi `X_<split>` — zato je housing r2 bio posten.)
* `dataset._media_files` — GLASNO upozorenje kad trazeni split ne postoji u putanjama; dosad je tiho
  padalo na cijelo stablo.

### Provjereno

    m5 · auto-pool    train 74 086 | val 15 878 | test 15 871   (ukupno 105 835)
                      preklapanja 0 / 0 / 0  ->  DISJUNKTNO
                      omjer razreda  train: five .038  zero .038  yes .038
                                     val:   five .038  zero .038  yes .038
    housing · as-is   train 14 448 | val 3 096 | test 3 096
    yolo26n · as-is   train  5 860 | val   837 | test  1 674     DISJUNKTNO

Prije je m5 imao train = val = test = istih 5000 datoteka. Sad 70/15/15, disjunktno, omjer razreda
sacuvan u trecu decimalu.

**Ispravak ranije tvrdnje u ovom dokumentu:** housing NIJE auto-pool. `pipeline.prepare` za tabularne
podatke cita splitove iz NPZ KLJUCEVA (`X_train`/`X_val`/`X_test`), pa je oduvijek `as-is`. Raniji
zakljucak je izveden iz `detect_format` direktno, koji gleda samo foldere — krivi poziv, krivi
zakljucak. Jedini stvarno pokvaren bio je m5.

### Posljedica za ranije runove

Korisnikov m5 run (Faza 1 + 2) treba baciti: nastao je uz slaganje mjereno na 64 trening uzorka, uz
val koji je bio isti skup kao train. Yolo runovi nisu pogodjeni — taj dataset ima prave val foldere i
koristi mAP, ne slaganje.

---

## 8e. PRAVA METRIKA ZA m5 I sst2 — CETIRI KVARA (2026-08-23)

**Pitanje:** zasto m5 i sst2 padaju na `teacher_agreement` kad im je task uredno prepoznat i
`SUPPORTED_TASKS.json` kaze `metrics: ["f1_macro", "accuracy"]`?

**Odgovor:** task NIJE problem. Oba su `task=classification`, `mode=full`. Puca CITAC OZNAKA, i to iz
dva razlicita razloga:

    m5    folder_per_class · 36 foldera · model ima 12 izlaza
          pairs BEZ guarda = 36   -> citac RADI, guard ga (ispravno) odbija
    sst2  hf_datasets · mode=token · model ima 2 izlaza
          pairs BEZ guarda = 0    -> citac uopce ne moze procitati

### Sto je legacy vec znao (provjereno prije diranja koda)

* `legacy/arch_agnostic/dataset.py:517` — **citanje HF OZNAKA je rijeseno**: stupac se nalazi po TIPU
  znacajke (`ClassLabel`), s imenom (`label`/`labels`/`target`) kao rezervom; arrow datoteke se sortiraju
  tako da se preferira `valid`/`test`. Kod je identican u slinnu — zadrzan pri mergeu.
* `_survey_hf` cita splitove iz IMENA arrow datoteka: sst2 = `{train: 67349, val: 872, test: 1821}`.
* Tokenizer je **svjesno izostavljen**, dokumentirano: *"token bez tokenizera -> frozen random"*.
  Oznake citljive, ulazi nisu.
* `legacy/.../metric.py` **NEMA `pairs_classification`** — klasifikacija nikad nije imala pravu metriku;
  njihov `agree_gate.txt` navodi m5 pod "unknown-task fallback" iako task nije nepoznat.
* Bonus: legacyjeve agreement vrijednosti za m5 (0.9219, 0.9688, 0.8438) su **sve visekratnici 1/64** —
  bug od 64 uzorka naslijedjen je odande (popravljen u §8d).

### Cetiri kvara, cetiri genericka popravka

**(1) HF citac (ulaz+oznaka).** `dataset.hf_pairs` sparuje postojeci citac oznaka s NOVOM ulaznom
stranom: `hf_tokenizer` bira tokenizer ISKLJUCIVO iz lokalnog cachea (`local_files_only`), a kandidate
sortira po `model.config.name_or_path` (HF cache mapu imenuje `models--<org>--<ime>`) — inace bi se kod
vise kesiranih modela uzeo prvi na koji se naidje. Genericno za bilo koji HF tekstualni klasifikator.

**(2) `classes.json` konvencija.** Kad ime foldera NIJE oznaka razreda, dataset to smije DEKLARIRATI:

    {"yes":0, "no":1, ..., "_background_noise_":11, "*":10}

Kljuc `"*"` je catch-all; bez njega se Wardenova konvencija (10 naredbi + 'unknown' za ostalih 25
rijeci + 'silence') ne da izraziti. Kod ne zna nista o modelu — dataset nosi vlastito mapiranje.

**(3) ULAZNI UGOVOR — `input.json` + native-iz-podataka.** Probe bira NAJVECU radnu velicinu s
ljestvice ([classify.py](../classify.py) `for sz in sorted(ladder, reverse=True)`). Za slike je to
tocno (fiksni FC -> najveca radna JEST native; tako je nadjen schoolcnn 320), ali fleksibilna 1D mreza
(potpuno konvolucijska + adaptivni pooling) prihvati sve — pa je M5 dobivao **48000 umjesto 8000**.

Ljestvica dvaju mehanizama u `pipeline.apply_input_spec`:

    1. `input.json`  {"length":8000, "sample_rate":8000}   <- DEKLARIRANO
    2. native-iz-podataka (dekodiraj stvarnu datoteku)     <- AGNOSTICNO, ali samo duljina
    3. sto je probe pogodio

Trebaju OBA jer rjesavaju razlicite stvari: iz podataka se moze doci do prirodne duljine (16000 @
16 kHz), ali NE do odluke recepta da se prije treninga radi downsample na 8 kHz. Ta cinjenica postoji
samo u `data.py` (`about.txt`: *"Downsample na 8 kHz -> [1, 8000]"*). Frekvencija se ne da pogoditi ni
iz cega — samo deklaracija.

**(4) Resampling u dekoderu.** `_decode_audio(..., sr=)`: prvo uzme isjecak ODGOVARAJUCEG TRAJANJA
(`length/sr` sekundi pri izvornoj frekvenciji), pa resamplira. Redoslijed je bitan — resampliranje
60-sekundne pozadinske snimke ravno na 8000 uzoraka trazi omjer 16000->133 Hz i gradi ogroman filtar
(izmjereno: proces bude UBIJEN). Bez `sr` se ponasa kao prije (samo rez/pad).

**(5) Kvota po RAZREDU, ne po folderu.** Skriveni kvar koji je isplivao tek nakon (1)-(4). Kad
`classes.json` slije 25 foldera u `unknown`, kvota-po-folderu tom razredu da 25x vise uzoraka:

    prije:  {0-9: 11 svaki, 10: 275, 11: 6}   -> 275/391 = 70% skupa u jednom razredu
    poslije:{0-10: 33 svaki, 11: 6}

Uz to round-robin po folderima unutar razreda, da razred ne bude sav iz jedne rijeci.

### Rezultat

    speechcommands_m5   f1_macro  0.0506 -> 0.7279 -> 0.9263   (monitor 0.9281)
    sst2_distilbert     f1_macro  fallback -> 0.9000            (monitor 0.8848)

Referenca: m5 `eval_result.txt` na sluzbenom val splitu = **0.8709**; DistilBERT SST-2 ~0.91.

### Ostaje nesavrseno

`silence` ima samo **6 uzoraka** — toliko datoteka ima `_background_noise_`. `data.py` iz njih GENERIRA
tisuce nasumicnih 1-sekundnih izrezaka; genericki citac zna za datoteke, ne za izreske. Ne dira ostale
razrede, ali cini taj razred statisticki nestabilnim.

### Pouka za ocjenjivanje nalaza

Kvar (3) je VEC bio zapisan — [PLAN.md:456](PLAN.md): *"m5 @48000 (6x native nakon reversal-a):
'najveca radna' overshoota native za audio; KD self-konzistentan, ali opc. refine na
native-iz-podataka. **(minor)**"*, i ponovljen u backlogu na retku 1424. Ranije je stajalo suprotno:
*"SIZE_LADDER_1D (1024..48000; **M5 pogadja 8000**)"* — dakle REVERSAL ljestvice (uveden zbog slika)
je slomio ono sto je prije bilo tocno.

Ocjena "minor" bila je TOCNA U TRENUTKU PISANJA: bez prave metrike KD je self-konzistentan i nista ne
odaje problem — i ucitelj i student gledaju isto smece. Cim je uvedena prava metrika, isti nalaz je
vrijedio f1 0.05 naspram 0.87. **Nalaz oznacen kao minoran vrijedi preispitati kad se promijeni ono
sto ga je cinilo minornim.**

---

## 9. SLJEDECI KORAK

Obje faze su napisane, provjerene i uskladjene s planom. Preostaje:

1. **Yolo Faza 1 ponovno** — postojeci rezultat (7.4%) nastao je pod STARIM pravilom patiencea
   (prev-based, strop 10). Pod danasnjim (`curr_best`, patience 3, strop 7) trajektorija je druga.
   Model iz tog runa je usput izgubljen (v. §8c), log je u `REPORTS/phase1_full_yolo26n.log`. ~2 h.
2. **Faza 2 na detekciji** — dosad samo housing_mlp lancano. Proba je vec izmjerila `g_min = 2.3364`
   (39.2% originala), pa ljestvica ima realan raspon. Uz korak od 5% ocekivano ~11 rezova.
   Najbrze: pokrenuti `phase="12"` i dobiti oboje u jednom prolazu.
3. **Izbaciti `dead_removal` i staru `full_cycle`** (§4 stavka 8) — tek kad 1 i 2 prodju.

### Brzi razvoj bez cekanja yola

`speechcommands_m5` uz `DEV_DATA_SUBSET ~6000` je ~28x laksi po epohi od yola, uz pravi task
(klasifikacija) i pravu metriku (f1). Bez podskupa NIJE brz — model je 29x laksi, ali mu je dataset
18x veci (105 835 uzoraka naspram yolovih 5 860), pa se usteda pojede.
`housing_mlp` je za cistu logicku provjeru (~11 000x laksi po epohi), ne za zakljucke o kompresiji.

### Zapisani nalazi koji nisu dirani

* **`sst2_distilbert` ima 0 prunabilnih slojeva** — pipeline javlja da su svi rezivi slojevi spregnuti
  s tapovima (rezidualni tok) i predlaze `kd_mode='logit'`. Model se u trenutnoj konfiguraciji NE moze
  komprimirati. To nije sporost, to je zid.
* **`midas_depth` se ne ucitava** — `ModuleNotFoundError: No module named 'midas'`; treba mu `code_dirs`
  (auto-dodavanje mape .pt-a ne pomaze jer je modul drugdje).
* **18 forward-nesigurnih slojeva na yolo26n** (C2f split/concat) drzi `g_min` na 39.2%. Kad bi se
  rijesio forward-safe rez kroz njih, dno bi palo. Konkretan trag, nije diran.
