# Plan — generalizirani, model-agnostičan compression pipeline

Cilj: pipeline za kompresiju (prune + grow + KD) koji na **novom modelu radi odmah** (out of the box),
bez ručnog pisanja adaptera po modelu. Razvija se u `arch_agnostic/` kao paralelna grana nad dokazanim
mehanizmima iz `morphology/`.

---

## 0. Tvrda pravila (nepregovaračka)

- **Izolacija:** sve novo se piše u `arch_agnostic/`. `morphology/` se **NE dira** — koristi se samo
  read-only (`import analysis`, `import compress`, `import kd`). Kad se grana pokaže dobrom → merge u glavni.
- **Puni eager modul:** svi modeli spremljeni kao `torch.save(model)` (cijeli modul), nikad state_dict/
  TorchScript — inače forward-hookovi i analiza ne rade.
- **Težine/datasetovi izvan gita** (već `.gitignore`).
- **Bez magičnih brojeva** gdje god ide: pravila se izvode iz grafa/mjerenja, pragovi su eksplicitni i branjivi.

---

## 1. Gdje smo (stanje `arch_agnostic/`)

Riješena je **strukturno-pozicijska ravnina** (što/gdje morphati) mjerenjem umjesto ručnih adaptera:

- `classify.py` — `ProbeAdapter` (ulazni ugovor probingom) + klasifikacija leafova mjerenjem
  (`_is_activation` mjeri, briše globalni `ACT_TYPES`) + samograđujući `LAYER_REGISTER.json`
  (sposobnosti dobivene STVARNIM pokušajem prune/grow kroz pipeline; ručni `hazard` unosi netaknuti).
- `position.py` — pozicijsko sužavanje: terminal / attention / SE / **feature-KD tap** (block-boundary:
  prvi morphabilan sloj iznad glave koji se grana na ne-terminalnog potrošača ili mijenja rezoluciju;
  net-fallback + guard → `kd_mode` "feature+logit" | "logit").

Testirano čisto na 3 modela: yolo26n (3 neck tapa), fasterrcnn (2 FPN tapa), SchoolCNN (1 tap).

Time su pokrivene per-model točke: matching, forward, tp_example, prunable/growable, aktivacije, protect+tapovi.

---

## 2. Dvije preostale ravnine

| Ravnina | Status | Plan |
|---|---|---|
| **Strukturno/pozicijsko** | ✅ gotovo (`classify`+`position`+`register`) | održavati |
| **Gubitak / KD** (teacher/student signali, KD članovi, važnost) | ⛔ još model-vezano u morphology | **generički hook-KD + KD-grad važnost** (koraci 1–2) |
| **Evaluacija** (decode, NMS, letterbox, mAP) | ⚠️ vezano, ali odvojivo (samo za metriku) | izolirati kao opcijski "scorer" (korak 4) |
| **Engine** (dead→FT→prune/grow petlja, budžet, cooldown, align, rollback, autobatch, verzije) | ♻️ već agnostičan u morphology | **preuzeti kakav jest** (korak 3) |

---

## 3. Auto-detekcija taska (arhitektura + oznake + backup)  — ✅ IMPLEMENTIRANO (Faza 3.5, v. §4g)

Task NE treba engineu (KD-vođen); određuje samo *dodatne* KD članove i *prijavljenu* metriku. Zato smije
graciozno degradirati.

**Dva signala, križno provjeravana:**
- **A. Oznake (presudno):** skalar/one-hot → klasifikacija; multi-hot → multilabel; lista okvira
  `[cls,x,y,w,h]` → detekcija; per-piksel maska → segmentacija; kontinuirano → regresija.
- **B. Arhitektura/izlaz (potvrdno):** probe zna oblik/tip izlaza; `morphology/introspect.py` već
  strukturno klasificira obitelj/task (nije nigdje spojen — kandidat za ponovnu upotrebu).

**Backup — ljestvica povjerenja (siguran default):**
1. **Ručni override** (`TASK=...`) uvijek pobjeđuje.
2. **A ∩ B se slažu** → koristi task.
3. **Neslaganje/djelomično** → vjeruj oznakama, spusti razinu enhancera.
4. **Unknown/kvar → univerzalni KD-only mod:** feature-KD (hookovi na tapovima) + output-KD (hookovi na
   terminalnim glavama), bez decode-a, a **KD-gubitak vs učitelj = validacijski proxy** za early-stop.
   Pipeline i dalje komprimira; samo ne prijavljuje task-specifičnu metriku.

---

## 4. Koraci (redoslijed razvoja)

**Korak 0 — proširiti skup model+dataset parova** (prije developa).
Raznolikost = jedini pravi test agnostičnosti. Spremati kao pun eager modul (`torch.save(model,"model.pt")`),
`model.pt` izvan gita. Svaki par ide u **`baseline_models/<dataset>_<model>/`** (uz postojeće yolo/frcnn),
s obrascem: `model_<...>.py` (klasa u uvozivom modulu za eager reload) · `data.py` (lokalni `data/` + uniformni
`data(split)` loader koji vraća `(inputs, labels)` da task-detektor onjuši oznake) · `build.py` (trening +
`train_summary.txt`) · `eval_baseline.py` (→ `eval_result.txt`, format kao yolo) · `about.txt` (opis).
Referentni gotov primjer: `baseline_models/housing_mlp/`.

Već u ruci: yolo26n + FasterRCNN (detekcija, slika), SchoolCNN (multilabel klasifikacija, slika).

Finalizirani novi parovi:

| # | Task | Data type | Model | Dataset | Pretrained | Compute |
|---|---|---|---|---|---|---|
| A1 | klasifikacija (ključne riječi) | audio (waveform) | M5 1D-CNN | SpeechCommands (torchaudio) | trenirati | lagano | ✅ GOTOV (baseline_models/speechcommands_m5; izložio 1D Conv1d gap, v. §4b) |
| A2 | klasifikacija (žanr) | audio (spektrogram) | 2D-CNN | GTZAN | trenirati | lagano | ✗ ODUSTAO (strukturno = 2D CNN kao SchoolCNN, ne donosi novo) |
| T1 | klasifikacija (sentiment) | tekst | DistilBERT | SST-2 | ✅ HF | srednje | ✅ GOTOV (baseline_models/sst2_distilbert; val acc 0.91; izložio token/vise-ulazni gap, v. §4b) |
| S1 | segmentacija | slika (per-piksel) | DeepLabV3-MobileNetV3 | VOC | ✅ torchvision | srednje | ✅ GOTOV (baseline_models/voc_deeplabv3; mIoU 0.727; probe RADI — PRVI novi model koji pipeline čita: 0 unknown, 35 SE + 2 terminala OK, 5 tapova) |
| R1 | dubina (dense regresija) | slika→dubina | MiDaS_small | NYU v2 val (HF) | ✅ torch.hub | srednje | ✅ GOTOV (baseline_models/midas_depth; δ1 0.60 / AbsRel 0.22; probe RADI, 2 unknown; NYU .mat host mrtav → HF val 654) |
| M1 | regresija / čisti MLP | tabular | mali MLP | California housing | trenirati (trivijalno) | ~0 | ✅ GOTOV (baseline_models/housing_mlp) |

Osi stresa pokrivene: 1D/Conv1d (A1), spektrogram-2D (A2), transformer/attention+embedding-hazard (T1),
per-piksel decode (S1), dense regresija (R1), čisti-MLP/`flat`/tap-net-fallback (M1). TinyStories-1M
(generativni LM) je moguć kasniji dodatak ako zatreba generativna os (strukturno redundantan s T1).

**Korak 1 — generički hook-KD** (najveća poluga).
Tap je već *imenovani sloj* → feature-KD povuci forward-hookom na tom sloju (isti hook na student i
teacher), MSE. Output-KD = hook na terminalnim glavama (MSE/KL). BEZ `student_signals`, BEZ decode-a.
Detekcijski članovi (box_giou, dense_cls, objectness) postaju opcijski *enhanceri* → točno gradacija
`feature+logit` vs `pure logit` koju već imamo. Reuse: `morphology/kd.py` (handleri po tipu su agnostični).

**Korak 2 — generička KD-grad važnost.**
Gradijent generičkog KD-gubitka po kanalu → rang za prune + smjer za grow (GradMax). Briše zadnju ovisnost
o `gt_loss` (koji je u morphology ionako samo za Overview analizu, ne za petlju).

**Korak 3 — preuzeti morphology engine kakav jest.**
`coupled_unit_cost`, `prune_costs`, `prune_candidates`, `align_factors`, forward-safe trial/rollback,
churn cooldown, autobatch, BN-eval disciplina, `_consider_best` — svi već agnostični (trebaju samo
`tp_example`/`forward`, što probe daje). Zamijeni ulaz: adapter → (probe + mask iz classify/position).
Zadržati redoslijed: prep (gpu/autobatch/teacher cache) → Phase 1 (dead-removal + KD FT recovery) →
Phase 2 (kontinuirani prune+grow pod GFLOPs budžetom, cooldown, align, forward-safe) → verzije/trajektorija.
⚠️ REVIDIRANO (Faza 3, §4e): pretpostavka "engine je već agnostičan" NE vrijedi za 1D — `_apply_prune_plan`
tiho vraća `n_rem=0`, `_try_grow_layer` vraća `None` na Conv1d (2D pretpostavka `dim in (2,4)`, isti korijen
kao stari `weighted_leaves` blindspot). To je konkretan zadatak za Korak 3/Fazu 5 (generalizirati engine na
dim-3), izoliranom granom (morphology read-only).

**Korak 4 — izolirati JEDINI preostali per-model dio = "scorer"** (decode → mAP).
Opcijski plug; kad ga nema (novi model) → KD-gubitak kao proxy metrika. Klasifikaciji decode ne treba
(logiti su gotova metrika).

**Korak 5 — kasnije:** task/metrika registar + class-remap samo ondje gdje treba prava metrika.

---

## 4b. Poznati gapovi (iz zoo stres-testa)

- **1D / Conv1d ulaz** (otkrio `speechcommands_m5`, M5) — ✅ RIJESENO (Faza 1a, 2026-08).
  `classify._input_spec` / `ProbeAdapter` prepoznavali su samo Conv2d (dim-4) i Linear (dim-2); Conv1d (dim-3)
  se nije hvatao → ulaz se krivo tumacio kao vektor, forward padao, probe vracao `None`. Izmjene (sve u
  `arch_agnostic/`, morphology netaknut): (1) `_input_spec` hvata dim-3, vraca `ndim` (2 slika / 1 sekvenca /
  0 vektor) umjesto `vector` bool; (2) `ProbeAdapter._one` gradi `[C,L]` za 1D; (3) `SIZE_LADDER_1D`
  (1024..48000; M5 pogadja 8000); (4) `classify_leaf` measured-width cita kanal s osi 1 za conv (Conv1d izlaz
  je `(B,C,L)`), ne sa zadnje osi; (5) NOVI lokalni `weighted_leaves` (dim 2/3/4) jer `morphology.weighted_leaves`
  filtrira dim in (2,4) → 1D lanac bio nevidljiv grafu; `position.py` sad koristi taj lokalni. Rezultat: M5
  probe YES, 0 unknown, 1 net-fallback tap (kao `housing_mlp`), nula regresija na 2D. Bilanca probe 6/8 → 7/8.
- **Token / vise-ulazni forward** (otkrio `sst2_distilbert`, DistilBERT) — ✅ RIJESENO (Faza 1b, 2026-08).
  Vidi §4c za detalje; probe YES, bilanca 7/8 → 8/8 (cijeli zoo se sad cita).

## 4c. Faza 1b — plan (token / vise-ulazni, DistilBERT)

Introspekcija (`DistilBertForSequenceClassification`): forward `(input_ids, attention_mask, inputs_embeds, ...)`
keyword; `word_embeddings` vocab 30522 dim 768, `position_embeddings` 512; 6× `DistilBertSelfAttention`
(position ih vec skida); leaf tipovi Embedding×2/LayerNorm×13/Dropout×14/Linear×38/GELU×6; izlaz
`SequenceClassifierOutput` (`.logits`), ne goli tensor. `_input_spec` vraca `(768, 0)` — POGRESNO (768 je
embedding DIM, ruta na vektor mod), probe → `None`.

4 slomljene pretpostavke (probe pretpostavlja JEDAN float tenzor, in_ch = weight.shape[1]):
1. ulaz float → `input_ids` su **long indeksi** < vocab.
2. in_ch = shape[1] → Embedding weight je `[vocab, dim]`, treba **shape[0]=vocab** (ne dim 768).
3. poziv list/batch → treba **keyword** `model(input_ids=…, attention_mask=…)`.
4. izlaz tensor → **`SequenceClassifierOutput`** (`.logits`).

Pod-koraci (RIZICNI GATE prvi):
- **1b.0 SPIKE (prvo, de-risk):** provjeriti trasira li `torch_pruning.DependencyGraph().build_dependency`
  DistilBERT s token-ulazom + `output_transform=lambda o: o.logits`. Najtvrdji dio (embedding + attention
  reshape za glave). AKO NE trasira → DistilBERT ide u **logit-only KD** (probe YES, ali `consumer_map` prazan,
  bez tapova). Domet 1b odlucuje se OVDJE, mjerenjem.
- **1b.1 token mod u `ProbeAdapter`:** `_input_spec` → ako je prvi weighted leaf `nn.Embedding` → `mode="token"`,
  `vocab = num_embeddings`; `_one` = `randint(0, vocab, (L,), long)`, batch `[B,L]`; `SEQ_LADDER=(8,16,32,64)`
  (≤512 zbog position_embeddings). Embedding je vec `hazard` — probe ga samo CITA.
- **1b.2 keyword/multi-input konvencija:** uz list/batch dodati `kwargs`; imena iz `inspect.signature`; probaj
  `input_ids` sam → ako padne dodaj `attention_mask=ones`.
- **1b.3 normalizacija HF izlaza:** `.logits` / `.to_tuple()` prije `_finite`/`_detach`/`teacher_outputs` i za
  `output_transform` u tp-u.
- **1b.4 verifikacija classify:** ocekivano 0 unknown BEZ registarskih unosa (pravila pokrivaju: LayerNorm→norm,
  Dropout(eval)→prolaz, GELU→aktivacija, Linear FFN→morph, classifier→terminal, attention-Linear→skip).

Definicija gotovog: minimalno `sst2_distilbert` probe YES + unk 0 (matrica 7/8 → 8/8); rastegnuto (ovisi o 1b.0)
smisleni FFN tapovi ili dokumentiran logit-only fallback. Ne dira se: morphology, LAYER_REGISTER, `_res_of`,
postojeci modovi.

REZULTAT (2026-08): probe YES (`kwargs · 30522vocab @ 8 token`), bilanca 8/8. Izmjene u `classify.py`
(morphology netaknut): (1) `_input_spec` 3-tier conv>Embedding>Linear, vraca `mode`
("image"/"seq"/"vector"/"token") umjesto `ndim`; Embedding se hvata PRIJE dim-2 (weight mu je isto dim-2
[vocab,dim]); token in_ch = vocab = num_embeddings. (2) `ProbeAdapter` mode-vodjen: `_one` za token =
`randint(0,vocab,(L,),long)`; nova `kwargs` konvencija poziva = `model(input_ids=stack)`. (3) `SEQ_LADDER_TOK
=(8,16,32,64)`. (4) `_unwrap` helper: HF `SequenceClassifierOutput` -> `.logits`/`.to_tuple()` prije
`_finite`/`_detach`. 1b.0 SPIKE potvrdio: tp trasira DistilBERT (build_dependency + get_pruning_group na
ffn.lin1 rade i BEZ output_transforma), pa `consumer_map`/`position` rade neizmijenjeni. `input_ids` sam je
dovoljan (bez attention_mask).

Klasifikacija cista: 0 tip-gapova (LayerNorm→norm, GELU→aktivacija, Dropout→prolaz, Linear FFN→morph,
classifier→terminal — sve pravilima, bez registra). 8 unknown je RAZLOZENO i benigno: 2× Embedding = namjeran
hazard-alarm (stite se od reza), 6× `attention.dropout` = HF SDPA proslijedi dropout_p u
scaled_dot_product_attention pa se `nn.Dropout` modul nikad ne izvrsi (0 param, izvan grafa). Pozicija: guard
opali (6 FFN-boundary tapova > cap 5) → pure logit-KD (kanonski BERT-KD mod); morph 38→13 (6×lin1+6×lin2+
pre_classifier rezljivi, 24 attention-Lineara skip, classifier terminal).

OTVORENO za Fazu 2 (unknown→0): 6× SDPA-dropout "nije se izvrsio". Vidi §4d — RIJESENO.

## 4d. Faza 2 — unknown → 0 (2026-08) ✅

Dijagnostika je pokazala da SVIH 8 preostalih unknown-a (DistilBERT 8, MiDaS 2) spada u tocno dvije benigne
kategorije — nijedan nije stvarni tip-gap (Interpolate/ReLU6/ZeroPad2d se uredno klasificiraju pravilima; samo
nisu u registru = Faza 3). MiDaS-ova 2 su zapravo `Identity` quant-stubovi (`skip_add.activation_post_process`),
NE Interpolate/ReLU6/ZeroPad2d. Dvije principijelne korekcije u `classify_leaf` (bez magicnih brojeva):

- **Fix 1 — hazard je POZNAT, ne unknown.** Hazard branch → `is_unknown=False` (bilo True). Hazard je
  prepoznat i obradjen (zasticen `morph=False`), suprotno od coverage gapa; `why` i dalje kaze "hazard: …".
  Cisti 2× Embedding (DistilBERT).
- **Fix 2 — 0-param leaf izvan compute-grafa → is_unknown=False.** Grana `nparams==0 and not fired`.
  Dokaz: leaf bez tezine koji nije u USPJESNOM forward-grafu ne moze biti prune-root (nema tezine), ni
  census-tocka ni topoloski op (nije u grafu). probe je vec validirao pun forward, pa "nije se izvrsio" =
  stvarno neaktivan (SDPA-inline Dropout, quant-stub Identity), a NE slomljen probe. Cisti 6× DistilBERT
  `attention.dropout` + 2× MiDaS `Identity`.

Prave uzbune ostaju `is_unknown=True`: weighted "shape mismatch"/"nema dokaza o izlaznoj sirini" i finalni
`neprepoznato (fqn, params=…)` za leaf S parametrima koji ne padne ni u jedno pravilo.

REZULTAT: cijeli zoo **unk=0** (8/8), nula regresije (svi probe YES, tapovi i reg nepromijenjeni). Semantika
`is_unknown` sad je cista: iskljucivo stvarni coverage gap (leaf s parametrima koji pipeline ne zna svrstati).

## 4e. Faza 3 — populacija registra (2026-08) ✅

Runner `_populate.py`: za svih 8 modela probe → classify → `capabilities_by_type` (STVARNI prune/grow pokusaj
kroz `morphology.compress`, deepcopy-izoliran) → `merge_register` (aditivno, OR, hazardi zasticeni). Registar
**18 → 27 tipova**, 9 novih upisano s empirijski provjerenim zastavicama:

  Interpolate/ReLU6/ZeroPad2d/MaxPool1d/Dropout/GELUActivation → sve p/g/t=False (topolosko/aktivacija, bez
  params). BatchNorm1d/LayerNorm → t=True (imaju affine params), p/g=False (nisu prune-root). Conv1d → t=True,
  **p/g=False**. 2D netaknuti (Conv2d/Linear p/g/t=True), 2 hazarda ocuvana. OR preko zooa ispravlja lazne
  negative: housing/M5/DistilBERT lokalno daju Linear prune=False (BN1d-coupling / attention-first / terminal),
  ali register zadrzi Linear=True iz frcnn/schoolcnn.

KLJUCNI NALAZ (stress-test upravo ovo trazio): **morphology prune/grow engine tiho no-opira na 1D.**
`_apply_prune_plan(M5 body.0.0)` → `n_rem=0` (ne krusi, samo nista ne reze); `_try_grow_layer` → `None`.
Korijen = ista 2D pretpostavka (`dim in (2,4)`) kao stari `weighted_leaves` blindspot. Registar to POSTENO
biljezi (Conv1d p/g=False); kad Faza 5 generalizira engine na dim-3, re-run `_populate.py` OR-flipne na True
(self-correcting). Zadatak prebacen u Korak 3/Fazu 5. Registar semantika potvrdjena: capability je TIP-razine
i "sto pipeline STVARNO moze", ne "sto bi teoretski trebalo".

## 4g. Faza 3.5 — Task-detekcija (pred-F4, 2026-08) ✅

ODLUKA: task-detekcija PRIJE Faze 4 (ne poslije). Detektor je samostojeći (čita oznake+arhitekturu, ne treba
KD-jezgru ni engine → nema inverzije ovisnosti), pa se u F4 pišu generička KD-jezgra I task-enhanceri u jednom
prolazu. ČUVAR: generička jezgra ostaje samostalna za `task=unknown` (KD-only dno); enhanceri se kače uvjetno.

Datoteke (u `arch_agnostic/`): **`task.py`** (detektor) + **`SUPPORTED_TASKS.json`** (katalog = single source of
truth: 5 taskova + unknown, svaki s label/arch potpisima, metrikama, `decode`, `kd_core`, `enhancers`).

HIBRID, dva signala, ljestvica `override > A∩B > A > B > unknown`:
- **A `_label_signature`** — duck-typing DEKODIRANOG in-memory oblika oznaka (data.py parsira YOLO/COCO/VOC/PNG/
  BIO s diska; detektor gleda samo tenzor/listu). Ključno: **binarnost se provjerava PRIJE dtype-a** (multi-hot
  zna doći kao float 0.0/1.0, npr. schoolcnn). Neprepoznato → unknown (nikad crash/pogađanje).
- **B `_arch_signature`** — FORMAT-NEOVISNO (čita model): strukturni markeri detekcije (`Detect`/
  `AnchorGenerator`/`RoIHeads`/`RegionProposalNetwork` u imenu tipa) + oblik probe-izlaza (`[B,K,H,W]`→seg,
  `[B,1,H,W]`/`[B,H,W]` float→depth/reg). cls/multilabel/reg su `[B,K]`=dvosmisleni → prepušteno oznakama.

Depth = pod-tip regresije (dense float); seg vs depth diskriminira dtype oznake (int klase vs float). Depth-
specifične metrike (AbsRel/δ1) su scorer-razina (Korak 4), ne jezgra. Per-token `[B,L]` (NER/BIO) prepoznat ali
nepodržan → unknown (budući task, v. `future` u JSON-u).

VALIDACIJA (`_tasksweep.py` → `task_report.txt`): **8/8** točno.

SPOJENO s DatasetProbe-om (2026-08): `detect_task(model, adapter, device, probe=…)` sad prima rezultat
`dataset.probe_dataset` — koji je JEDINI izvor oznaka (`_labels` uzorak) + `task_hint`. Signal A = `_label_signature`
na tim oznakama, a ako se ne izjasni → `probe.task_hint` (format-kontekst). **Per-model glue UKLONJEN** iz
`_tasksweep` (nema više label-gettera/importlib-a); po modelu se navodi samo DATASET-PATH (pravi korisnički ulaz).
Signal B (`_arch_signature`) sad vraća i OBLIK izlaza (`detection`/`spatial`/`dense1`/`flat`).
**B OGRANIČAVA A** (novo): strukturirani task (detection/seg) je NEMOGUĆ na `flat [B,K]` izlazu → `_recast_flat`
reinterpretira (boxevi → per-uzorak prisutnost klasa: >1 klasa = multilabel). Time schoolcnn (isti sub10k
box-podaci, ali klasifikator [B,6]) → **multilabel** (izvor `A|B-shape`), a ne lažno detection. midas: labeli
nečitljivi (core_kd_only) → A iz `probe.task_hint=regression`, B=dense1 → A∩B. Rezultat po modelu: yolo/frcnn→
detection (A∩B), schoolcnn→multilabel (A|B-shape), housing→regression (A), M5/DistilBERT→classification (A),
voc→segmentation (A∩B), midas→regression (A∩B). Ništa ručno; jedan tok probe→task.

## 4h. Faza 3.6 — DatasetProbe (path → format, 2026-08) ✅

ODLUKA (korisnik): ulaz je SAMO `(eager model, path na dataset)`. Naš je posao detektirati i task i **format
zapisa oznaka na disku** — nula ručnog readera od korisnika. Ovo je TRECI sloj auto-detekcije (uz model-probe
i label-duck-typing): on-disk → `(format, task_hint, oznake)`.

Datoteke: **`dataset.py`** (DatasetProbe) + **`SUPPORTED_DATASET_FORMATS.json`** (katalog on-disk formata +
otisci + task_hint + reader-status).

`detect_format(path)` = kaskada strukturnih otisaka (najspecifičnije prvo): coco → voc → yolo → seg_masks →
folder_per_class → tabular → nlp → unknown_format. Njuši STRUKTURU foldera + **SADRŽAJ uzorka label-datoteke**
(npr. yolo redak `cls cx cy w h` = 5 tokena → detection). Wrapper-folderi (VOCdevkit/VOC2012,
SpeechCommands/xxx) se preskaču (`_descend`). Format = ŠTO JE NA DISKU; stvarni task odredi `task.py` križanjem
s modelom. VAŽAN uvid: isti `images/+labels/` (5-stupčani boxovi) yolo koristi za detection, a schoolcnn
PRENAMJENJUJE u multilabel (čita samo klasu) — zato je format ≠ task, i model ima zadnju riječ.

`label_sample(path)` → uzorak oznaka u memoriji (kanonski oblik za `_label_signature`). Implementirano:
folder_per_class (class-idx), yolo (lista `[N,5]`), voc (maske `[B,H,W]` int), tabular (npz/csv target),
hf_datasets (arrow ClassLabel). coco/seg_masks/nlp = **detect-only, reader `todo`**.

FALLBACK `agnostic_sniffer(path)` (kad kaskada padne, PRIJE unknown_format): generalizirajuce + CONTENT-AWARE,
EGZAUSTIVNO (bez limita dubine, cap 2M). Koraci: (1) ULAZI = dominantna prava-media kategorija rekurzivnim
walkom (image/audio/video/tabular; text dvosmislen). (2) CONTENT-PEEK (mjerenje, ne ime/ekstenzija):
  · slike → foto vs maska po SADRZAJU (`_img_role`: palette/malo-vrijednosti = maska) → maske poravnate s
    fotama = segmentation; ulazi = samo fote (VOC: 17125 fota, ne 22951 s maskama)
  · sidecar tekst → `_peek_text_kind` cita sadrzaj: box=klasa-int+4..7 koord (5=GT,6=+conf,7=+track)→detection,
    poligoni→seg, BIO→token, skalar→reg; **binarno (.pt/.npy cache) se preskace** (null-bajt/printable-ratio)
  · tabular kontejner → `_peek_table` cita STUPCE: in-file label (y/target/depth) + pravi input-tip
    (image-u-parquetu!) → nyu: input=image, label=depth(regression)
(3) OZNAKE strukturno: stem-sidecar (peek, preskace prazne) → folder=klasa (SAMO ako DISJUNKTNI stemovi:
parallel view-i = <=4 dira + overlap>=0.85; koincidentalni overlap kao SpeechCommands `<govornik>_nohash` NIJE
parallel) → manifest → none. KLJUC: **ulaz i oznaka ODVOJENI** — i bez oznaka (`none`) ostaju ULAZI za KD.
CISTI sniffer (bez kaskade) sad TOCNO rjesava svih 6: sub10k→detection, housing→regression, speechcommands→
classification, sst2→classification, voc→segmentation, nyu→regression. `unknown_format` = ni sniffer ne nadje
nista citljivo. Naucene lekcije robusnosti (sve generalizirajuce, sadrzaj/struktura, ne per-format): binary-skip,
box=class-int+coords, mask=palette/low-unique, parallel=malo-dira+visok-overlap, in-file labeli preko stupaca.

FINALNI IZLAZ `probe_dataset(path)` (jedinstveni ugovor za pipeline): `samples_found` / `labels_found` (ODVOJENO)
+ `n_samples` (EGZAUSTIVNO prebrojano, ne-pohlepno preko cijelog stabla) + `splits` (`{split: n}`) + `mode`:
  `full` (uzorci+oznake → task-detektor + enhanceri) · `core_kd_only` (samo uzorci → general core KD logit+feature)
  · `stop` (nista citljivo → nema smisla dalje). Brojanje: `_survey` (media po splitu-pretku), `_survey_tabular`
(npz/csv/parquet redovi), `_survey_hf` (arrow num_rows) — sve NE-pohlepno (obidje train/val/test/eval, zbroji, pa
zakljuci). Rezultat sweepa: voc full/22951, speechcommands full/105835, sub10k full/8371{train5860,test1674,val837},
housing full/20640{train,val,test}, sst2 full/70042{train,val,test}, nyu **core_kd_only**/654{val} (uzorci da,
oznake ne → samo core KD). Odluka `mode` je točka grananja: enhanceri samo u `full`.

VALIDACIJA (`_datasweep.py` → `dataset_report.txt`): 5/6 riješeno punim lancem path→format→oznake→
`_label_signature`: voc→segmentation, speechcommands→folder_per_class/classification, sub10k→yolo/detection
(splits train/val/test), housing→tabular/regression, **sst2→hf_datasets/classification** (riješeno: `hf_datasets`
recognizer čita `dataset_info.json` shemu → ClassLabel=classification, i vadi oznake iz `.arrow`). **nyu (raw
parquet) → `sniffed`/tabular** preko `agnostic_sniffer` fallbacka (našao 3 parquet kao ULAZE, oznake=none →
KD-only; parquet-dense label-reader je backlog). Dakle NIJEDAN od 8 dataseta više ne dead-enda u unknown_format —
sniffer uvijek nađe barem ulaze ako postoji čitljiv medij. Namjera potvrđena: standardni+HF layouti kaskadom,
sve ostalo s medijem preko sniffera, samo istinski neproziran folder → unknown.

OTVORENO (backlog/kasnije): (a) readeri za coco/seg_masks/nlp + parquet-with-dense-columns (nyu); (b) puni
INPUT-readeri (trenutno `label_sample` vadi samo oznake; KD treba i dekodirane ulaze) — nadogradnja istih
recognizera; (c) txt-list splitovi (VOC ImageSets, SpeechCommands) — sad se hvataju samo dir-bazirani splitovi.

NAPOMENA (organizacija): `arch_agnostic/` = pravi moduli (`classify`/`position`/`task`/`dataset`) + registri
(3 JSON-a). `arch_agnostic/helper/` = svi `_*` harnessi + ovaj PLAN.md. `arch_agnostic/REPORTS/` (PARALELNO s
helper, ne pod njim) = `*_report.txt`. Harnessi imaju arch_agnostic apsolutno na `sys.path` (importi rade iz
helper/-a); OUT = `../REPORTS/` (`dirname(dirname(__file__))`, makedirs za svjež clone).

DATASET POKRIVENOST (2026-08, svih 8 modela): yolo/fasterrcnn/schoolcnn dijele `sub10k_open_images_v7`
(`images/+labels/`, 5-stupčani YOLO labeli) → format `yolo` (task se razlikuje: detection vs multilabel, odredi
ga model). housing→tabular, m5→folder_per_class, distilbert→hf_datasets, voc→voc — **7/8 prepoznato**. Jedini
promašaj: midas/nyu (raw parquet s image+depth stupcima) → `unknown_format` (backlog: parquet-dense reader).

PROPAGACIJA `unknown_format` (glasno, nikad tiho-krivo): `detect_format` NIKAD ne baca (vrati unknown_format
dict s `why`), pa se format može INSPEKTIRATI bez pada. Tek `label_sample` na unknown_format diže jasan
`ValueError`. Kroz pipeline: (A) TASK — oznake (A-signal) otpadaju → ljestvica pada na B (arhitektura,
format-neovisna); modeli s odlučivom arhitekturom (detekcijska glava, spatial/[B,1,H,W] izlaz) svejedno dobiju
task — npr. midas dobije `regression` preko B iako mu je nyu parquet neprepoznat; samo [B,K]-dvosmisleni modeli
+ nepoznat dataset → `unknown` task → KD-only. (B) KD-KOMPRESIJA — treba prave ULAZE iz podataka; na
unknown_format pipeline STANE s glasnom porukom (daj reader / podržani format), umjesto da trenira na smeću.

## 4i. Faza 4 — plan (ravnina gubitka: generička KD-jezgra) 🔜

CILJ: KD-gubitak koji na NOVOM modelu radi ODMAH — generička jezgra (feature + output KD) + uvjetni
task-enhaneri + KD-grad važnost (briše zadnju ovisnost o gt_loss). Ovime je zadnja ravnina (LOSS) agnostična.

ULAZI (sve gotovo): `probe_dataset` (inputs/splits/mode) · `task_detector` (task/metrics/kd_core/enhancers) ·
`classify`+`position` (mask, TAP slojevi = feature-hook točke, TERMINALNE glave = output-hook točke, `kd_mode`) ·
`kd.py` handleri (reuse). TVRDA PRAVILA: morphology read-only (reuse `kd._LOSS`+`kd_total`, teacher-cache obrazac,
BN-eval disciplina); KD-ONLY (GT nije u petlji); generička jezgra SAMOSTALNA za `mode=core_kd_only`/`unknown`;
enhaneri UVJETNI (`mode=full` + decode).

Pod-koraci (rizik-poredano):
- **4.0 Generički INPUT-reader** (`dataset.py`): `input_batch(path, split, n)` dekodira medij → tenzori (slika PIL,
  audio torchaudio, tabular numpy/parquet, token). Generičiji od label-readera (medij se dekodira univerzalno).
  KVAKA: token-modeli (DistilBERT) trebaju TOKENIZER (model-specifičan) za pravi tekst → fallback = probe-random
  tokeni (degradiran KD, ali radi). Za KD trebamo prave ulaze iz `train` splita.
- **4.1 Generički hook-KD** (novi `loss.py`) — GLAVNA POLUGA: forward-hookovi na TAP slojevima (feature MSE,
  kanali poravnati jer position štiti tapove) + na TERMINALNIM glavama (output-KD). Tip output-gubitka bira TASK:
  KL@T za distribucije (classification/multilabel/segmentation), MSE za kontinuirano/sirovo (regression/detection-
  glava). Sastavi `terms` → reuse `kd.kd_total`. BEZ `student_signals`/decode. Poštuje `kd_mode` (feature+logit vs
  logit-guard). Teacher = smrznuta kopija ORIGINALA (prije kompresije).
- **4.2 Teacher cache**: predračunaj teacher tap+terminal aktivacije po batchu, cachiraj (kao morphology `tmp/`),
  reuse kroz epohe. Batch-size u meta (feature-cache mismatch, v. [[kd-featlogit-experiment]] gotcha).
- **4.3 KD-grad važnost**: gradijent generičkog KD-gubitka po kanalu → rang za PRUNE + smjer za GROW (GradMax).
  Briše zadnju ovisnost o `gt_loss`. Ovo OTKLJUČAVA engine (Faza 5).
- **4.4 Uvjetni enhaneri**: iz `task_detector.enhancers` (detection → dense_cls/box_giou/objectness). Samo
  `mode=full` + decode; po-task decode-svjestan extractor plug. Core radi i bez njih. (detection prvi.)
- **4.5 Orkestracija + validacija** (`_kdsweep.py`): jedan tok `probe → task → KD`. Za svih 8 modela potvrdi:
  KD-loss se RAČUNA (konačan) + PADA kroz par koraka (student oponaša teachera) + KD-grad daje per-kanal skorove +
  `mode` grananje radi (full→enhaneri, core_kd_only→samo jezgra). To je MJERLJIVI cilj Faze 4.

REUSE (izolacija): `kd._LOSS`+`kd_total`, teacher-cache obrazac, `set_bn_eval`. NOVO u arch_agnostic: hook-
ekstrakcija signala (zamjena za `student_signals`), generički input-reader, KD-grad važnost (probe+mask vođena).
REDOSLIJED: 4.1 odmah (glavna poluga; za smoke može probe-random ulaz prije 4.0); 4.3 otključava Fazu 5.
DEFINICIJA GOTOVOG: na svih 8 KD-loss računa+pada, KD-grad daje rang, `mode` grananje radi → LOSS ravnina
agnostična; ostaje Faza 5 (engine end-to-end, uklj. 1D generalizaciju iz §4e).

**4.1 GOTOVO (2026-08):** `arch_agnostic/loss.py` — `kd_terms`/`kd_loss` (feature-hook na tapovima + output-KD na
`_main_out` finalnom izlazu; reuse `kd.kd_total`). `_main_out` reducira izlaz na primarni tenzor (HF `.logits` |
torchvision seg dict `'out'` | ugnijezdjena lista→prvi).
NORMALIZACIJA (skala-invarijantno, riješeno odmah): `_rel(s,t)` dijeli studenta i teachera s teacher-RMS →
plain MSE preko `kd._feature_loss` postaje RELATIVNI MSE = MSE/mean(teacher²) ~O(1), po tapu (nijedan ne dominira)
i na MSE-izlazu; KL je već skala-invarijantan. Nazivnik je od SMRZNUTOG teachera (konstanta) → progress-signal
ostaje (brojnik→0). Auto-LR = **Prodigy** (reuse iz morphology KD-FT; bez grad-clipa/per-model tuninga).
Smoke `_kdsmoke.py` → `REPORTS/kd_smoke.txt`: **5/5** KD-loss RAČUNA + PADA, i sad SVE magnitude O(1):
schoolcnn 0.124→0.007, housing 0.052→0.002, m5(kl) 2.67→0.070, voc(5 tap) 0.223→0.009, **midas 0.114→0.008
(prije normalizacije bio 26168, eksplodirao)**. feat i out doprinosi usporedivih magnituda → težine `w` smislene.

**4.3 GOTOVO (2026-08):** `loss.py` `kd_importance(student, teacher, adapter, batches, taps, kd_mode, out_kind,
prunable)` — port morphology `_kd_grad_importance` na generički `kd_loss` (pa AUTOMATSKI uključi enhanere kad
4.4 doda; isti loss). Iz JEDNOG backwarda po batchu vrati dva signala: `imp[name]` = mean|d(KD)/dw| po izlaznoj
jedinici (PRUNE rang; mean-norm → usporedivo među slojevima), `gavg[name]` = signed grad matrica [O, in·k]
(GROW/GradMax SVD). BN u eval (bez korupcije running-statsa + bez dropouta → determinist. grad; jednostavnije od
morphology snapshot/restore). NAPOMENA: pri student==teacher KD=0→grad=0; u Fazi 1/2 je student već razIđen pa
smislen (smoke perturbira). Smoke `_kdimp.py` → `REPORTS/kd_imp.txt`: važnost KONAČNA + DISKRIMINIRA na svih 5
(schoolcnn 6/6, housing 3/3, m5 3/3, voc 33/34, midas 70/70; voc 1 sloj nul-grad = izvan KD-signala = prune-safe),
gavg oblik OK svugdje. → PRUNE rang + GROW smjer spremni za engine (Faza 5).

**4.0 GOTOVO (2026-08):** `dataset.py` `input_batch(path, adapter, device, split, n)` → (batch, source) — dekodira
PRAVE ulaze usklađene s adapter-ugovorom (mode/in_ch/size): slika (PIL, resize, **maske preskočene** preko
`_img_role`), audio (torchaudio, crop/pad na duljinu), tabular (npz/csv matrica [N,in_ch]). Generickije od
label-readera (medij univerzalno). Fallback = probe-random (`adapter._one`) uz `source='fallback-random'` gdje
nema readera (token bez tokenizera, image-u-parquetu) → KD i dalje radi (treba samo ulaze). Smoke `_inputsweep.py`
→ `REPORTS/input_report.txt`: 4/5 pravi podaci + forward konačan (schoolcnn←sub10k [3,320,320], housing←npz [8],
m5←SpeechCommands [1,8000], voc←VOC [3,96,96] maske preskočene), midas←nyu → fallback (image-u-parquetu, backlog).

**4.5 GOTOVO (2026-08):** `pipeline.py` `prepare(model, adapter, device, dataset_path)` — JEDAN tok:
classify/position (mask, tapovi, kd_mode) + `probe_dataset` (mode/splits/oznake) + `detect_task` (task/metrike/
enhancers) → KD kontekst {task, mode, metrics, kd_core, enhancers (uvjetno na mode=full), taps, kd_mode, out_kind,
prunable, splits}. `out_kind` = KL za classification, MSE inače. Engine (Faza 5) konzumira kontekst; ulaze daje
`input_batch`, loss/rang `kd_loss`/`kd_importance`. Smoke `_pipesweep.py` → `REPORTS/pipe_report.txt`: cijeli lanac
probe→task→input→loss→importance radi na 6 modela (schoolcnn→multilabel, housing→regression, m5/sst2→classification,
voc→segmentation, midas→regression); loss konačan+O(1), importance DA svugdje, mode grananje radi (midas
core_kd_only). → **LOSS RAVNINA AGNOSTIČNA.** Preostaje 4.2 (teacher cache=optimizacija) + 4.4 (detection enhaneri).

**4.5a SPLIT-POLITIKA + TAP-TRIM (2026-08):** `dataset.resolve_splits(splits)` — train/val/test as-is; train+val→
test=val; train+test→val=test; samo-train→auto-carve val/test; bez train/splitova→auto-pool + stratified 70/15/15.
`dataset.stratified_split(labels, ratios, seed)` — disjunktni indeksi, stratifikacija po klasi (int) / kvantilnim
binovima (float) / random (multilabel/None); GRUPNI leakage (govornik/pacijent) NIJE pokriven bez grupnog ključa
(granica). `resolve_splits` u `pipeline.prepare` kao `ctx['split_plan']` ('AUTO'→stratified_split u Fazi 5).
Validirano (`_splitcheck.py`): as-is za sub10k/housing/sst2, auto-pool za voc/speechcommands/nyu; stratifikacija
čuva omjer klasa po splitu + disjunktno. TAP-TRIM (position.pick_taps): >cap → trim ravnomjerno po dubini (v. §4f);
DistilBERT 6→5 feature-tapova. `pipeline.prepare` timing: 0.01–2.2s/model (jednokratni setup; probe_dataset ~0.5s
i za 105k fajlova). Brzo, zanemarivo naspram FT petlje.

## 4f. KONSOLIDIRANI ZAOSTACI (F1–F5, stanje nakon 5.7 + čišćenja 2026-08)

### ✅ RIJEŠENO tijekom F1–F5
- **Probe gapovi 1D+token** (F1) · **unknown→0** (F2) · **registar populiran** (F3) · **task-detekcija** (F3.5) ·
  **DatasetProbe** (F3.6) · **hook-KD + normalizacija (`_rel`)** (4.1) · **KD-grad važnost** (4.3) ·
  **input-reader** (4.0) · **orkestracija** (4.5) · **split-politika + tap-trim** (4.5a).
- **Teacher-cache** (5.1) · **prep: autobatch + adapter-size shim** (5.2) · **grow+churn-cooldown** (5.3) ·
  **dead-removal** (5.4, sad `dead=False` default — nije KD-siguran/census-fragilan) · **1D-engine + Conv1d
  capability flip** (5.5) · **prava metrika (r2/mIoU/f1) + real-metric gate** (5.6a) · **teacher-agreement gate**
  (unknown/bez-oznaka) · **baked-normalizacija u model** (housing; zoo-ugovor: sirovi ulaz→izlaz) ·
  **seg per-piksel KL** · **detekcijski enhaneri** (5.7, `enhancers.py`, task-uvjetno).
- **LAYER_REGISTER Conv1d** False→True (5.5, self-correcting). **Native-res** (probe bira NAJVEĆU radnu veličinu,
  640→niže; rješava "96 presitno" + GFLOPs sad pravi). **Feature-KD normalizacija** (4.1).

### ⏳ OTVORENO (backlog, po prioritetu)
- **Detekcija — paritet:** ✅ VEĆINOM RIJEŠENO (2026-08). (a) **BN-u-važnosti fix** GOTOVO: `kd_importance`
  snapshot/restore BN buffera (try/finally) — detekcijski `_dense_decode(train_bn=True)` više ne drifta stat.;
  yolo mAP +0.02-0.03/korak (best 92.1%). (b) **frcnn two-stage** GOTOVO: enhancer-put radi kroz naš engine
  (FRCNN_PROFILE), 99.5% mAP @3.8% reza, 0 banano. (c) teacher-cache se preskače kad je loss_fn (enhaner) aktivan
  (frcnn dict-izlaz + detekcija ima vlastite teacher-signale). OSTAJE samo: dublja-kompresija tuning (NIJE potrebno
  — pipeline koristi fiksni 1.5-3%/korak) i opc. FT-budžet.
- **Readeri (format-specifični, po dizajnu):** coco/seg_masks/nlp **label**-readeri; **parquet-dense** input+label
  (nyu/midas → sad fallback-random ulazi); **tokenizer** za tekst (DistilBERT → sad random tokeni); txt-list
  splitovi u `dataset.splits()` (VOC ImageSets — `metric.py` ih čita, ali `dataset` ne).
- **Final multi-term KD težine** (feature+output+enhaneri): sad jednake (pošteno jer `_rel`→O(1)); opc.
  uncertainty-weighting/GradNorm.
- **Group-leakage** u `stratified_split` (govornik/pacijent) — bez grupnog ključa (granica, označeno).
- **m5 @48000** (6× native nakon reversal-a): "najveća radna" overshoota native za audio; KD self-konzistentan,
  ali opc. refine na native-iz-podataka. (minor)

### 🧹 ČIŠĆENJE (2026-08, uklonjene redundancije)
- **`continuous_prune` UKLONJEN** (5.2, prune-only) — `morph_loop(reinvest_frac=0)` daje isti čisti prune.
- **KD-proxy gate (`kd_tol`/`kd_ref`/`kd_slack`) UKLONJEN** iz `morph_loop`/`full_cycle` — mrtav put otkad
  `full_cycle` UVIJEK postavi `metric_fn` (prava metrika ili teacher-agreement). Gate = jedan mehanizam (metrika).
- `full_cycle` reorganiziran: loss_fn (enhaneri) se gradi PRIJE dead-recovera; dead-FT samo kad `dead=True`.
- Smokovi `_prune52`/`_cycle54` ažurirani na novi API. Granica agnostično/decode ostaje: `loss.py` bez
  decode-ovisnosti; `enhancers.py` = decode-svjesni plug.

## 5. FAZA 5 — engine (kompresija end-to-end) 🔜 plan

CILJ: pokrenuti pun `prune+grow+KD-FT` loop kroz arch_agnostic `ctx` (pipeline.prepare), REUSE dokazanih
morphology mehanika ("kralježnica"), + riješiti 1D-engine gap (§4e). Svi ulazi spremni (4.0/4.1/4.3/4.5).

REUSE-MAPA (morphology daje, ostaje kralježnica): dead-removal · `coupled_unit_cost`/`prune_costs`/
`prune_candidates` (FLOPs+importance) · `align_factors` · `_apply_prune_plan` (forward-safe trial/rollback) ·
`_try_grow_layer` (GradMax iz `gavg`) · churn-cooldown · GFLOPs budžet · autobatch/prep · BN-eval · verzije/
trajektorija · `_consider_best`. ZAMJENA ULAZA: adapter/profile → `ctx` (probe+mask) + `loss.kd_loss`/
`kd_importance` + `dataset.input_batch` + teacher-cache.

Sub-koraci (rizik-gate):
- **5.0 SPIKE:** jedan `prune-korak + KD-FT recovery` na 2D modelu (schoolcnn/voc, gdje prune radi) kroz `ctx` →
  dokaz da integracija (morphology mehanika ← naš ctx/loss/importance) stoji. Prije pune petlje.
  **✅ GOTOVO (2026-08, `_spike50.py` → `REPORTS/spike50.txt`):** pun lanac na `voc_deeplabv3` PROLAZI —
  `pipeline.prepare`→ctx (seg, 5 tapova, 34 prunable) · `dataset.input_batch` pravi `[3,96,96]` uzorci ·
  `loss.kd_importance` rang 34 sloja · morphology `prune_costs`/`_select_prune_plan`/`_apply_prune_plan`
  (REUSE, forward-safe) reže 243 kanala/6 slojeva, 0 bad, forward OK · GFLOPs 0.6954→0.6856 (-1.40%),
  params -1.69% · KD-FT (`loss.kd_loss`+Prodigy, 8 kor.) recovery 0.851→0.291. TRI NALAZA za 5.2/6:
  (a) **input-size hardkodovi** = per-model rezidual: `analysis.layer_table` hardkodira `640×640`,
  `compress._forward_ok` `320×320` — ignoriraju `adapter.imgsz` (probom izmjeren). Spike ih generalizira
  **runtime monkeypatchom** (`A.layer_table` → shim koji koristi `adapter.forward_example`; morphology fajlovi
  NETAKNUTI). Faza 5.2/6 formalizira: JEDINI izvor velicine ulaza = adapter, nikad hardkod (uklj. `_forward_ok`).
  (b) **BN-eval disciplina u KD-FT nije samo detekcija:** `student.train()` + mali batch (4) napuse KD-loss
  (0→7) jer BN batch-stat student vs `eval()` teacher = jabuke/kruske ([[bn-eval-detection-trainmode]] siri se i
  na klasifikaciju/seg). Fix u spike-u: `student.eval()` cijelo vrijeme (BN zamrznut; eval NE gasi grad → tezine
  se uce). 5.4 mora BN-eval u FT-u za sve taskove.
  (c) `schoolcnn_pareto_final` NEUPOTREBLJIV kao spike-target (vec maks. prorezan: sve sirine 6-9 < floor
  `PHASE2_MIN_ALIVE=ALIGN_M//2=16` → 0 kandidata). Za engine-testove treba model S PROSTOROM (voc/nepokresani).
- **5.1 Teacher-cache (4.2) + split-materijalizacija:** predračunaj teacher tap+terminal aktivacije po batchu
  (`split_plan.train`; `AUTO`→`stratified_split`), reuse kroz epohe (batch-size u meta).
  **✅ GOTOVO (2026-08, `engine.py` + `_enginecache.py` → `REPORTS/engine_cache.txt`):** dvije cigle za petlju.
  (1) `loss.teacher_signals(teacher, adapter, imgs, taps)` izdvojen → `{feat:{tap}, out}` (detached); `kd_terms`/
  `kd_loss` sad primaju opc. `teacher_sig` (predračunat) i preskaču teacher-forward. (2) `engine.py`:
  `materialize_train_batches(path, adapter, dev, split_plan, batch_size, n_batches, seed)` → FIKSNI CPU-batchevi
  (reuse kroz epohe → cache-alignment); `split_plan.train` realno ime → ti fajlovi, `'AUTO'` → pool (split=None) +
  seeded subset (KD bez oznaka → TRAIN ulazi ne trebaju stratifikaciju; `stratified_split` ostaje za val/test
  metriku u 5.6); fallback-random se generira JEDNOM i drži (token/parquet). `TeacherSigCache` + `precompute_teacher`
  = disk fp16 + opc. RAM, reuse KONCEPTA morphology `TeacherCache` (NAŠI hook-signali); META (model/n_batches/
  batch_size/tapovi/**fingerprint**) invalidira cache pri promjeni ulaza/tapova (fingerprint = oblik+prorijeđena
  suma → hvata drugi seed/split iako se brojevi poklope). Cache dom `arch_agnostic/tmp/` (gitignore). VALIDIRANO
  na image/vector/seq (voc/housing/m5): materijalizacija deterministična, **cached≈inline kd_loss** (rel-diff
  1e-5..3e-4, fp16), reuse (valjan meta, ~15-30× brže od builda), invalidacija (promjena tapova prepiše meta),
  speedup 1.3-1.9× (preskočen teacher-forward; voc najveći jer je teacher najteži). morphology NETAKNUT.
- **5.2 Prune-petlja:** `prune_costs` (coupled FLOPs) + `kd_importance` rang → select pod GFLOPs budžetom,
  forward-safe, cooldown, align (reuse mehanike, feed NAŠU važnost).
  **✅ GOTOVO (2026-08, `engine.py` + `_prune52.py` → `REPORTS/prune52.txt`):** PREP + PETLJA.
  **PREP:** (a) `install_sizing_shims()` (aktivan na import enginea) — formalizira nalaz 5.0-a: `A.layer_table`
  (hardkod 640) i `compress._forward_ok` (320) zamijenjeni adapter-verzijama (`adapter.forward_example`);
  interni pozivatelji (`gflops_total`/`coupled_unit_cost`/`_apply_prune_plan`) transparentno pokupe zakrpu →
  `gflops`/prune rade na SVIM modama (housing/m5 bi inace puknuli). morphology fajlovi NETAKNUTI (Faza 6 =
  isti ispravak direktno). (b) `autobatch(model, adapter, dev, ctx, path)` — port morphology `autobatch` na ctx
  (`_DetDataset`→`input_batch`, `adapter.kd_loss`→`loss.kd_loss`); probira kandidate 1..64 mimicirajuci CACHIRANI
  FT korak (student fwd + KD vs predracunati teacher-signal + backward + Prodigy.step), mjeri vrsni VRAM, uzme
  najveci pod `free_frac`. Probe u `eval()` modu (zrcali BN-eval FT; peak ~isti jer se aktivacije za backward
  cuvaju i u eval; usput rjesava bs=1 BN-crash na global-pool granama poput voc ASPP `[1,C,1,1]`).
  **PETLJA:** `continuous_prune(...)` do uštede `target_frac×baseline GFLOPs` (ili `max_steps`); svaki korak:
  `kd_importance` (KD-grad rang) + `prune_costs` → `_select_prune_plan` (per-step GFLOPs budžet, cap/floor, align)
  → `_apply_prune_plan` (forward-safe; forward-nesigurne BANANE + re-select, [[phase2-prune-forward-unsafe-ban]])
  → `prune_ft_recover` (KD-FT, Prodigy, BN-eval, teacher-cache iz 5.1). Prunable FIKSAN (channel-rez ne mijenja
  uloge). VALIDIRANO na voc: 8 koraka, GFLOPs 0.6954→0.6106 (**12.19% ušteda**, cilj pogođen), MONOTONO pada,
  params 11.0M→9.85M, KD-loss oporavlja svaki korak (0.61→0.51), 0 banano, forward OK. Ostaje: cooldown je iz
  morphology mehanike (grow još nije uključen — 5.3), quality-gate/best-selekcija (5.4/5.6), position-recompute
  po koraku (refinement; channel-rez ne mijenja uloge pa je fiksni prunable zasad korektan).
- **5.3 Grow:** GradMax iz `gavg` + churn-cooldown.
  **✅ GOTOVO (2026-08, `engine.morph_loop` + `_grow53.py` → `REPORTS/grow53.txt`):** `morph_loop(...)` = pun
  PRUNE+GROW pod GFLOPs budžetom. Svaki morph korak: `kd_importance` → imp (prune rang) + gavg (grow smjer,
  jedan prolaz) → PRUNE (cooldown-svjestan: elig − `grow_protected`; soft-override ako cooldown blokira SVE) →
  GROW (reinvest pool = `reinvest_frac×total_pruned − total_grown`; elig − upravo_rezano − `prune_protected`)
  preko REUSE `compress._grow_decide` (GradMax: σ=svdvals(gavg) → `_select_grow_plan` → `_try_grow_layer` s SVD
  init, function-preserving commit diff<1e-3) → `prune_ft_recover` (teacher-cache). Churn-cooldown
  ([[grow-prune-churn-rootcause]]): `morph_idx` broji MORPH dogadaje (ne FT epohe), `grown_at`/`pruned_at` dicti,
  simetrična zaštita `PHASE2_CHURN_COOLDOWN=2`; `_grow_decide._fresh` je dodatni guard (rezani sloj = stari
  gavg-oblik → preskočen). VALIDIRANO na voc: (A) morph_loop 8 koraka, **13 grow-događaja**, **0 churn-prekršaja**
  (nijedan narastao sloj rezan u cooldownu), net GFLOPs −12.07%, forward OK; (B) izravni `_try_grow_layer` test:
  **9/34** prunable slojeva prima function-preserving +1 rast (max |Δizlaz| 5.2e-04 < 1e-3; ostali padnu na
  tp-fragilnosti → korektno rollback). NAPOMENA: grow ekonomija na PUNOM modelu zna vratiti None (top kandidati
  ne prolaze `_try_grow_layer`), radi tek nakon reza (petlja to prirodno rješava). morphology NETAKNUT.
- **5.4 Dead-removal + KD-FT recovery** (Phase-1 ekvivalent) na `ctx` + `loss.kd_loss` (Prodigy, BN-eval).
  **✅ GOTOVO (2026-08, `engine.full_cycle`/`dead_removal` + `_cycle54.py` → `REPORTS/cycle54.txt`):**
  `dead_removal(...)` = REUSE `compress.remove_dead_neardead` s NAŠIM loaderom (materijalizirani batchevi kao
  `(imgs,None)`); `full_cycle(...)` = pun quality-gated ciklus: dead-removal → KD-FT (referentni KD) → `morph_loop`
  (prune+grow) s **KD-PROXY quality-gateom** + best-model selekcijom. GATE = KD-loss vs SMRZNUTI teacher (PLAN §3
  backup: KD-loss = validacijski proxy; **GT NE ulazi u loss** [[kd-only-no-gt]]; task-specifična metrika se sloji
  u 5.6). `kd_ref` = oporavljeni KD nakon dead-removala; `kd_tol=(1+kd_slack)×kd_ref`; best = model s NAJMANJE
  GFLOPs čiji je oporavljeni KD još ≤ kd_tol. VALIDIRANO na voc: dead-removal maknuo **509 kanala/17 slojeva**
  (GFLOPs 0.6954→0.6787 samo od dead-a), kd_ref=0.60→kd_tol=0.91, 10 morph koraka (KD PADA 0.61→0.44 kroz
  akumulirani FT, svi ≤ tol), **best=korak 10 (15.22% ušteda), forward OK**. morphology NETAKNUT.
  **⚠️ REVIDIRANO u 5.6:** (a) `dead=False` je sad DEFAULT — activation-frequency dead/near-dead removal je
  CENSUS-FRAGILAN i NIJE KD-siguran: na densnom izlazu (seg) reže rijetko-opaljujuće kanale kritične za rijetke
  foreground klase (voc mIoU 0.47→0.10 od dead-a samog na malom census-u; čak truly-dead weak=0 → 0.28). KD-grad
  prune petlja (output-sigurna) ga subsumira i daje ~74% mIoU retention. (b) KD-proxy gate je nadograđen na
  REAL-METRIC gate u 5.6 (v. dolje) jer KD-loss monotono raste s kompresijom → apsolutni prag teško kalibrirati.
- **5.5 1D-engine generalizacija (§4e):** `_apply_prune_plan`/`_try_grow_layer` na `Conv1d` (dim-3);
  re-run `_populate` → Conv1d prune/grow True (self-correcting registar).
  **✅ GOTOVO (2026-08, `engine.install_sizing_shims` prošireno + `_prune1d55.py` → `REPORTS/prune1d55.txt`):**
  korijen 1D-gapa je bio SAMO `A.weighted_leaves` filter (dim 2,4) → Conv1d nevidljiv SVIM morphology internalsima
  (`_apply_prune_plan` `if leaf not in info_now: continue` → tihi n_rem=0; `coupled_unit_cost`/`layer_table`
  isto; gflops≈0). ISPRAVAK (2 nova patcha u `install_sizing_shims`, morphology NETAKNUT): (1) `A.weighted_leaves`
  → lokalni `classify.weighted_leaves` (dim 2,3,4) → Conv1d/1D lanac vidljiv svim pozivateljima; (2)
  `compress._try_grow_layer` → `engine._ag_try_grow_layer` (JEDINA izmjena naspram morphology: referentni ulaz
  `adapter.forward_example` umjesto hardkod `rand(3,sz,sz)`; widen-helperi `_widen_out/_widen_bn/_insert_in_zeros`
  su već dim-generički; dodan `Conv1d` u isinstance-liste). tp podržava Conv1d prune/grow nativno. VALIDIRANO na
  m5: gflops sad 0.0316 (>0, prije ~0), morph_loop rezao **60 Conv1d kanala** (prije tihi 0), 15% ušteda, forward
  OK, KD-FT oporavlja; izravni grow-1D **3/3 Conv1d sloja** function-preserving (|Δ|=0.0). **Re-run `_populate.py`
  (uz `import engine`) → Conv1d capability FLIPNUO prune/grow False→True** (self-correcting registar, kako §4e
  predvidio); 27 tipova, 2D nepromijenjen, hazardi netaknuti. 2D regresija-provjera: voc ciklus identičan (15.20%).
  Registar backup: `LAYER_REGISTER.json.pre55.bak`.
- **5.6 END-TO-END trajektorija na ≥2 modela:** 2D **detekcija** (yolo/frcnn — PARITET sa starim, da ne
  regresiramo aktivni produkt) + **ne-detekcija** (voc/housing — dokaz generalnosti). Verzije + metrika iz
  SUPPORTED_TASKS. Ovo je i SIGURNOSNI GATE za Fazu 6.
  **✅ 5.6a GOTOVO (2026-08, `metric.py` + `_metric56.py` → `REPORTS/metric56.txt`) — GENERALNOST s PRAVOM
  metrikom (ne-detekcija):** novi `metric.py` = generički scorer za ne-detekciju (pareani val-readeri +
  evaluatori r2/rmse, mIoU, f1/accuracy; detekcija mAP ostaje morphology `A.evaluate`). Quality-gate NADOGRAĐEN
  na **real-metric** (`morph_loop.metric_fn`/`metric_tol`): best = najmanji GFLOPs čija PRAVA metrika ≥
  tol×baseline; early-stop kad padne 3× zaredom. **GT ULAZI SAMO u gate/izvještaj, NIKAD u loss** ([[kd-only-no-gt]]).
  VALIDIRANO na voc (segmentation): **mIoU zadržan 92.1%** (0.486→0.448) uz 5% ušteda (skromno = kratak smoke-FT,
  ne mana; gate ispravno staje kad kvaliteta padne). TRI nalaza usput (svi u §5.6-nalazi ispod).
  **🔬 5.6-NALAZI (rješenja + otvorene odluke):**
  1. **Seg treba per-piksel KL** (RIJEŠENO): `_out_kind` "kl" i za segmentation; `loss.kd_terms` reshapea [B,K,H,W]
     →[B·H·W,K] pa KL. Globalni `_rel` MSE je favorizirao pozadinu → foreground IoU kolabirao.
  2. **Dead-removal OFF po defaultu** (RIJEŠENO): v. 5.4 REVIDIRANO — nije KD-siguran/census-fragilan; KD-grad prune ga subsumira.
  3. **Real-metric quality-gate** (RIJEŠENO): KD-proxy prag se teško kalibrira (KD raste s kompresijom); prava metrika u gate-u.
  **✅ 5.6 ODLUKE RIJEŠENE (2026-08, korisnik):**
  - **A) Teacher-agreement gate** (`metric.teacher_agreement` + auto-fallback u `full_cycle`): label-free gate za
    nepoznat task / bez oznaka — slaganje s TEACHEROM (referenca umjesto GT): kl=argmax-agreement, mse=R²-vs-teacher.
    Ograničeno/kalibrabilno ("zadrži ≥97% slaganja"). GATE-ljestvica: prava metrika > teacher-agreement > KD-proxy.
    Validirano (`_agreegate.py`): m5(kl)+midas(mse) — daje signal, sigurno bira/čuva original (kad kompresija padne
    ispod praga). GT nikad u lossu.
  - **B) Baked-normalizacija u model** (odluka: modeli su self-contained, sirovi ulaz→izlaz; NEMA preprocessing-ravnine
    u pipelineu): `model_mlp.NormalizedRegressor` (bufferi μ_x/σ_x/μ_y/σ_y; standardizacija ulaza + de-std izlaza kao
    dio modela; classify ih vidi kao passthrough). housing re-wrapan+re-saveran (`_wraphousing.py`, backup
    `model_prewrap.pt.bak`): goli r2=−2155→umotani **r2=0.7565** na sirovom X. GENERALITY: full_cycle **r2 zadržan
    100%** (0.7565→0.7564) uz **−27.3% params**. Zoo-ugovor: modeli koji trebaju normalizaciju nose je interno.
  - **C) Detection gap IZMJEREN** (`_detdiag.py`/`_yolorun.py`; odluka "prvo izmjeri"): detekcija NIJE monolitna.
    **yolo (single-stage):** raw izlaz = tenzor [1,189,6] → generic output-KD (raw-head MSE) MEHANIČKI RADI; naš
    engine kompresira yolo (reže 75-89 kan/korak, forward-safe, 1 ban). **fasterrcnn (two-stage):** izlaz = lista
    dict-ova (boxes/labels/scores) → `_main_out` nije tenzor → generic output-KD NE RADI (samo feature-KD).
    KLJUČNO: raw-head R²-agreement pada brzo (0.27 na 1% reza) ALI to NIJE mAP (glava se mijenja u MSE dok mAP nakon
    NMS/decode ostaje) → **generic KD/agreement je SLAB proxy za detekciju**. ZAKLJUČAK: detekcija treba dedicirani
    plan — morphology mAP-scorer (gate) + decode-svjesni enhaneri (5.7 box_giou/dense_cls) za pravi paritet; ne može
    se skratiti generic jezgrom. → 5.6b/5.7 su STVARNO potrebni za detekciju (potvrđeno mjerenjem).
- **5.6b Detection paritet (yolo/frcnn)** — SLJEDEĆE: reuse morphology mAP-scorer + 5.7 enhaneri; single-stage
  (yolo) prvo (generic KD djelomično primjenjiv), two-stage (frcnn) treba feature-only ili puni enhancer-plan.
  **✅ 5.6b korak-1 GOTOVO (2026-08): PRAVI mAP-gate integriran.** `A.eval_map(mdl, A.pick_adapter(orig), val_loader)[0]['map']`
  proslijeđen kao `metric_fn` u `full_cycle` → detekcijski gate je sad PRAVI mAP (decode+NMS), ne lažni raw-head R².
  MJERENO na yolo: baseline mAP **0.427**; 1% GFLOPs reza (75 kan) → mAP **0.355** (83%), dalje pada (0.34, 0.32);
  gate (≥90%) → čuva original. ZAKLJUČAK (kvantitativan): generic KD (feature + raw-head MSE, BEZ enhancera, kratak
  FT, **@96px probe-res**) NE čuva detekcijski mAP — pada strmo. DVA detekcijski-specifična uzroka: (a) nema
  enhancera (dense_cls/box_giou/objectness) → KD ne destilira strukturu glave; (b) KD @96px (probe-minimalno) vs
  mAP @native — detekcija je rezolucijski osjetljiva (mali objekti). Oba su decode/detection-svjesni = dedicirani
  plan 5.7. Gate radi ispravno (štiti). → ODLUKA za korisnika: uložiti u 5.7 (enhaneri + native-res KD) da yolo
  smisleno kompresira kroz naš engine, ILI hibrid (detekcija ostaje morphology domena, generički vlasnik ostalog).
  **✅ NATIVE-RES RIJEŠEN GENERIČKI (2026-08):** `probe_adapter` sad bira NAJVEĆU radnu veličinu (ljestvica SILAZNO
  640→96), ne najmanju — jedno pravilo za sve (nema per-model veličine). Zoo bez regresije: yolo/frcnn/voc/midas 640,
  schoolcnn 320 (fiksni FC, silazna pretraga ga nađe), housing 8, m5 48000, sst2 64. GFLOPs sad PRAVI (yolo
  0.134@96→5.96@640). MJERENO: native-res pomaže MALO (yolo mAP retention 83%→89% na 1% reza), mAP i dalje pada strmo
  → **rezolucija NIJE glavni uzrok; ENHANERI (dense_cls/box_giou) jesu.** Enhaneri su inherentno per-obitelj
  (decode) → po principu "ništa per-model u jezgri" moraju biti IZOLIRAN opcijski plug, ne u generičkoj jezgri.
- **5.7 detection enhaneri (4.4):** dense_cls/box_giou/objectness vs teacher, za yolo/frcnn kvalitetu.
  **✅ GRADIVO GOTOVO (2026-08, `enhancers.py` + `loss_fn` threading):** `loss = KD-core + task-uvjetni enhaneri`.
  `enhancers.detection_kd` = REUSE morphology detekcijskog KD-a (feature+dense_cls+box_giou preko `profiles.pick_profile`
  → `_dense_decode`+`kd._LOSS`; decode po obitelji AUTO unutar pick_profile). `enhancer_loss_fn(ctx, teacher)` bira
  plug ISKLJUČIVO po `ctx['task']` (auto), nikad po imenu modela → jezgra agnostična. Integrirano: `kd_importance`/
  `prune_ft_recover`/`morph_loop`/`full_cycle` primaju `loss_fn`; `full_cycle` ga auto-gradi kad task ima enhanere →
  enhaneri i u VAŽNOSTI (koje kanale rezati) i u FT-u. LEKCIJA: enhancer-FT SAM (nakon generic-važnost reza) NE
  pomaže (0.330→0.318) — šteta je učinjena generic-važnošću; enhaneri MORAJU voditi i važnost. VALIDIRANO na yolo
  (pravi mAP gate): s enhancerima korak-1 mAP **93.2%** zadržano (vs generic 89%) — skroman plus pri laganoj kompr.;
  gate bira korak-1. Dublja kompresija i dalje pada brže (kratak FT + BN-u-važnosti refinement: morphology
  `_dead_decode(train_bn=True)` toggle-a BN → buduci fix = snapshot/restore). ZAKLJUČAK: plug radi + integrira se
  čisto (agnostično, task-uvjetno); detekcija kompresira BOLJE s enhancerima; puni agresivni paritet treba još FT/BN-tuninga.

## 6. FAZA 6 — merge u glavni Streamlit GUI + čišćenje 🔜 DETALJAN PLAN

CILJ: arch_agnostic postaje AKTIVAN pipeline iza `morphology/gui.py` — GUI radi na **bilo kojem modelu + bilo
kojem tasku** (ne samo yolo/frcnn detekcija). Stari morphology = kralježnica (dokazane mehanike prune/grow/dead/
align/budžet/rollback/verzije) + detekcijski decode (scorer/enhaner plug); novi generički slojevi
(probe/classify/position/task/dataset/loss/enhancers/engine) preuzimaju vođenje.

KADA: Faza 5 gotova (svih 5 taskova + detekcija s enhancerima + BN-fix + frcnn, dokazano). Izolacija
(`morphology` read-only) vrijedi DO 6.1 (backend swap) — od tada se morphology smije uređivati.

STRATEGIJA: **strangler-fig** (novi raste unutar, stari se šuplji), NE big-bang. Flip na novi default TEK nakon
paritet-provjere (6.3). Svaki korak je zaseban commit, reverzibilan.

### Coupling-točke koje merge dira (iz analize morphology GUI-ja, 2026-08)
| gdje | sada (detection-hardkodirano) | postaje (generičko) |
|---|---|---|
| `gui.py:40` | `st.selectbox("Model", ["fasterrcnn", YOLO_PATH])` | tekst-unos putanje na BILO KOJI `.pt` + putanja dataseta |
| `prep_worker.py:52` | `A.pick_adapter(model)` (per-obitelj adapter) | `classify.probe_adapter` + `pipeline.prepare` → `ctx` |
| `prep_worker.py:57` | `C.autobatch(model, adapter, …)` | `engine.autobatch(model, adapter, ctx, path)` |
| `prep_worker.py:106` | `C.precompute_teacher` (uvijek) | `engine.precompute_teacher` — **preskoči kad loss_fn (detekcija)** |
| `prep_worker.py:114` | `A.baseline_perf` (mAP) | per-task baseline: `metric.evaluate` (ne-det) \| `A.eval_map` (det) |
| `worker.py:66,70` | `C.run_dead_ft` + `C.run_morph` | `engine.full_cycle(ctx, metric_fn)` |
| `config.py:27,47,52` | `MODEL_SPEC`/`FT_METRICS=["map"]`/`PHASE2_STOP_METRIC` | job.json: `model_path`+`dataset_path`; task AUTO; metrika iz SUPPORTED_TASKS |
| GUI metrika-kartica | map/mar birači (hardkod) | `SUPPORTED_TASKS[task]["metrics"]` (dinamički, 5 taskova) |
| `analysis.py:444,787` | `TASK_METRICS`/`_EVALUATORS` (2 taska) | SUPPORTED_TASKS (5) + `metric.py` evaluatori |
| `analysis.py:429` + `profiles.py` | `pick_adapter`/`ProfileAdapter` kao POGON pipelinea | **ostaje SAMO kao detekcijski decode-plug** (enhancers/scorer), ne kao pogon |

### Sub-koraci

- **6.0 Pred-merge paritet-harness (offline, prije diranja GUI-ja):** skripta koja na ISTOM modelu vrti (a) stari
  `morphology.run_morph` i (b) novi `engine.full_cycle`, pa usporedi finalne GFLOPs + metriku. Za yolo/frcnn
  (detekcija, aktivni produkt) mora biti PARITET (nema regresije mAP-a); za voc/housing/m5 dokaz generalnosti.
  Ovo je SIGURNOSNI GATE — bez zelenog ovdje, ne diramo GUI.
- **6.1 Backend swap (worker/prep_worker → engine/ctx):** `prep_worker` gradi `ctx = pipeline.prepare(model,
  probe_adapter, dev, dataset_path)`, autobatch preko `engine.autobatch(…, ctx, …)`, baseline metrika per-task;
  `worker` zove `engine.full_cycle(…, metric_fn=<per-task>)` umjesto `run_dead_ft`/`run_morph`. `config.json`
  dobiva `model_path` + `dataset_path` (pravi korisnički ulaz); task/metrika/enhaneri se AUTO-detektiraju (ne u
  configu). Engine-mehanike i dalje importane iz morphology-a (kralježnica). **Izolacija pada ovdje** (smijemo
  editirati morphology zbog integracije).
- **6.2 GUI generalizacija:** (a) unos = putanja modela + putanja dataseta (ne selectbox 2 modela); (b) task se
  prikaže AUTO (read-only, iz `detect_task`) + `mode` (full/core_kd_only); (c) metrika-kartica dinamička iz
  `SUPPORTED_TASKS[task]["metrics"]` (npr. seg→mIoU, reg→r2/rmse, cls→f1/acc, det→mAP/mAR); (d) NOVA
  capability-kartica: što pipeline zna na OVOM modelu (iz `classify`/`position`/LAYER_REGISTER — prunable/tapovi/
  terminali/kd_mode) + format dataseta (SUPPORTED_DATASET_FORMATS); (e) trajektorija-graf task-generičan (ime
  metrike iz taska, ne hardkod mAP).
- **6.3 Paralelni rad + paritet-gate + flip:** u GUI-ju (ili harnessu) kratko oba puta na istom modelu → potvrdi
  paritet (detekcija) i smislenost (ne-detekcija) → **flip default** na novi engine. Stari put ostaje dostupan
  iza zastavice dok se ne obriše u 6.4.
- **6.4 ČIŠĆENJE (puno, tek nakon zelenog flipa):** obriši SUPERSEDED: hardkodirani `TASK_METRICS`/`_EVALUATORS`
  (2 taska → SUPPORTED_TASKS+metric.py), detection-centrične GUI-birače, `config.MODEL_SPEC`/`FT_METRICS` hardkod,
  `pick_adapter` kao pogon. **VAŽNA NIJANSA — NE briši detekcijski decode:** `profiles.py` (ProfileAdapter/
  pick_profile/`_dense_decode`), `YoloAdapter`/`FrcnnAdapter` decode + `eval_map` OSTAJU jer ih `enhancers.py`
  (detekcijski KD) i mAP-scorer (gate) REUSE-aju kao IZOLIRAN per-obitelj plug (jedini priznati per-model dio).
  Zadrži REUSE-mehanike (dead/prune/grow/coupled-cost/align/rollback/autobatch/verzije/`kd._LOSS`).
- **6.5 Konsolidacija foldera:** arch_agnostic = glavni pipeline modul; zadržane morphology-mehanike + detekcijski
  decode-plug se presele u arch_agnostic (npr. `engine/` submodul za mehanike, `decode/` za detekcijski plug)
  ILI ostanu kao tanki uvezeni `backbone`. `helper/` ostaje dev-alat. Jedan čist modul-stablo, jasna granica
  jezgra (agnostično) / plug (decode).

### Otvoreni zaostaci iz F1–F5 koji se rješavaju TIJEKOM mergea
(Nijedan ne BLOKIRA merge-scaffolding; rješavaju se kad ih GUI-tok prirodno izloži.)
- **Format-readeri (§4f #2):** čim GUI prima PROIZVOLJNE korisničke datasete, izranjaju formati bez readera
  (coco/seg_masks/nlp **label**-readeri; **parquet-dense** za nyu/midas input+label; **tokenizer** za tekst; **VOC
  txt-list splitovi** u `dataset.splits()`). Dodaju se INKREMENTALNO, po formatu, kako se pojave (isti obrazac kao
  `detect_format` kaskada). Do tada: nepoznat format → `core_kd_only`/fallback (glasno, ne tiho-krivo).
- **Split-politika + group-leakage (§4f #4):** GUI koji radi auto-split treba surfacati split-plan; grupni
  leakage (govornik/pacijent) traži grupni ključ iz dataseta — dodati kad GUI izloži split-konfiguraciju.
- **Final multi-term KD težine (§4f #3):** ako GUI izloži KD-težine (feature/output/enhaneri), tu se sredi
  politika (sad jednake, O(1)); inače ostaje default. Nije blokada.
- **Metrika-registar dovršetak:** SUPPORTED_TASKS metrike moraju sve imati `metric.py` evaluator (reg→r2/rmse,
  seg→mIoU, cls→f1/acc GOTOVI; multilabel→mAP, detection→mAP preko morphology — provjeriti pokrivenost pri 6.2).
- **m5 @48000 native-overshoot (§4f #5):** minor; opc. native-iz-podataka rezolucija; ne dira merge.

### Načelo + rizik
NAČELO: sve per-model/detection-HARDKODIRANO nestaje iz JEZGRE; mjereno-generičko preuzima; detekcijski **decode**
preživljava SAMO kao izoliran, auto-biran (po arhitekturi) plug. RIZIK: ne regresirati aktivni detection-produkt →
zato **6.0 offline paritet-harness + 6.3 paralelna usporedba PRIJE flipa**, i cleanup (6.4) TEK nakon zelenog.

## 7. Bit

Teži (strukturni) dio agnostičnosti je razbijen. Preostaje ravnina gubitka — nju čine agnostičnom
**hook-KD + KD-grad**, dok se dokazani **engine i redoslijed iz `morphology` preuzimaju gotovi**.
Rezultat: pipeline koji na novom modelu radi *odmah* (generic feature+output KD), a decode/mAP je jedini
opcijski, izolirani plug.
