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

NAPOMENA (organizacija): `slinn/` = pravi moduli (`classify`/`position`/`task`/`dataset`) + registri
(3 JSON-a). `slinn/helper/` = svi `_*` harnessi + ovaj PLAN.md. `slinn/REPORTS/` (PARALELNO s
helper, ne pod njim) = `*_report.txt`. Harnessi imaju slinn apsolutno na `sys.path` (importi rade iz
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

## 6. FAZA 6 — merge u novi `slinn/` folder (jedinstveni proizvod) ✅ ZAVRSENA (2026-08-23)

> **FAZA 6 JE ZAVRSENA.** Svi podkoraci (6.0, 6.0.1, 6.1, 6.4, 6.2, 6.3, 6.5, 6.6) su gotovi i
> verificirani. `slinn/` je jedini zivi proizvod: agnosticna jezgra + izolirani detekcijski plug +
> GUI. `legacy/` je cista arhiva koja se smije obrisati bez posljedica.
> Detalji svakog koraka su nize; sazetak na kraju poglavlja ("ZAVRSNO STANJE").


CILJ: složiti **NOVI top-level folder `slinn/`** kao jedinstveni proizvod — generički compression pipeline s
GUI-jem koji radi na **bilo kojem modelu + bilo kojem tasku** (ne samo yolo/frcnn detekcija). `slinn/` = jezgra
iz `arch_agnostic/` (prioritet, vodi) + preuzete morphology-mehanike (dokazane prune/grow/dead/align/budžet/
rollback/verzije) + detekcijski decode (scorer/enhaner plug). Novi generički slojevi (probe/classify/position/
task/dataset/loss/enhancers/engine) preuzimaju vođenje; sve per-model/detection-hardkodirano nestaje iz jezgre.

**ODLUKA O FOLDERU (korisnik, 2026-08): NOVI `slinn/`, građen KOPIRANJEM — NE `git mv`, NE merge-u-morphology.**
- `slinn/` je novi dom. U njega se **kopira** `arch_agnostic/*` kao jezgra; kako se generaliziraju GUI i detekcijski
  decode, kopiraju/prilagođavaju se potrebni dijelovi iz `morphology/` (npr. `slinn/gui/`, `slinn/plugins/detection/`).
- Povijest renamea nije bitna (zato ne `git mv`). Ime `morphology` je zavaravajuće (nastalo detekcijski-only) →
  proizvod NE živi pod njim.
- **ANTI-DRIFT (jedino pravilo):** za vrijeme mergea `morphology/` i `arch_agnostic/` su **ZAMRZNUTI** (nula novih
  izmjena ondje) — sav rad ide u `slinn/`. Time je duplikacija privremena i ograničena, ne troizvorni kaos. Oba
  ostaju netaknuta i radna kao referenca/fallback dok paritet-gate (6.0/6.3) ne prođe; nakon flipa se OBA PREMJESTAJU u `legacy/` (NE brisu se — odluka korisnika, v. 6.5).

KADA: Faza 5 gotova (svih 5 taskova + detekcija s enhancerima + BN-fix + frcnn, dokazano).

STRATEGIJA: **strangler-fig** (novi `slinn/` raste, stari se ostavlja radnim dok se ne isprazni), NE big-bang.
Flip na `slinn/` kao default TEK nakon paritet-provjere (6.3). Svaki korak je zaseban commit, reverzibilan.
Pošto se KOPIRA (ne editira morphology in-place), izolacija starog koda vrijedi cijelo vrijeme — do arhiviranja u `legacy/` (6.5).

### Coupling-točke koje merge dira (iz analize morphology GUI-ja, 2026-08)
(Lijeva strana = izvor u `morphology/` koji se KOPIRA+prilagođava u `slinn/`; `morphology/` original ostaje zamrznut.)
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

**IZMJENA REDOSLIJEDA (korisnik, 2026-08): 6.4 → 6.3 → 6.2** (čišćenje → flip → GUI), umjesto izvornog
6.2 → 6.3 → 6.4. Razlog: GUI je vrh hrpe; građen nad jezgrom koja se u 6.4 seli, gradio bi se dvaput.
Nalaz iz 6.1 da je `import worker` dvosmislen između `morphology/` i `slinn/gui/` je simptom istoga —
nestaje sam kad morphology ode. Preseljenje je jeftino osigurati: postoji `_parity60.py`, pa je
petlja `preseli → paritet → zeleno` dovoljna garancija da ništa nije puklo.

**NAČELO PRESELJENJA (korisnik, 2026-08):** `arch_agnostic`/`slinn` jezgra je smjer — općenitija je i
uvijek pobjeđuje u tehničkim odlukama. `morphology/` je izgrađeniji, ali mu jezgra ne generalizira.
Iz njega seli SAMO ono čega u `slinn/` nema; gdje postoje obje verzije, morphology se odbacuje.

- **6.0 Pred-merge paritet-harness (offline, prije diranja GUI-ja):** skripta koja na ISTOM modelu vrti (a) stari
  `morphology.run_morph` i (b) novi `engine.full_cycle`, pa usporedi finalne GFLOPs + metriku. Za yolo/frcnn
  (detekcija, aktivni produkt) mora biti PARITET (nema regresije mAP-a); za voc/housing/m5 dokaz generalnosti.
  Ovo je SIGURNOSNI GATE — bez zelenog ovdje, ne diramo GUI.
  **✅ GOTOVO (2026-08, `slinn/helper/_parity60.py`→`REPORTS/parity60.txt`):** yolo26n, 5 koraka, oba iz originala,
  bez dead-removala, isti step 1.5%/reinvest 0.30, OBA finalna mjerena istim mAP-om (morphology ProfileAdapter
  decode). Baseline GFLOPs 5.958/mAP 0.4174. **A stari:** rez 4.2%, mAP 0.3057 (73.2%). **B novi:** rez 4.0%,
  mAP 0.3839 (**92.0%**). VERDICT PARITET OK — novi NE regresira (čak bolji: nosi enhanere+BN-fix iz 5.6b/5.7).
  Novi `full_cycle` odvrtio pun engine end-to-end IZ `slinn/` na najtežem slučaju (detekcija) → nova lokacija radi.
  Napomena: DEV_DATA_SUBSET=200 (niska vjernost apsolutnih brojki; svrha = paritet). Generalnost ne-detekcije već
  dokazana u F5 (voc/housing/m5) + scaffold `prepare` iz `slinn/`.
- **6.0.1 Skele `slinn/`:** napravi prazan `slinn/`, **kopiraj `arch_agnostic/*` → `slinn/`** (jezgra). Potvrdi da
  smokovi/lanac probe→task→loss→importance rade iz nove lokacije (importi, `sys.path`, putanje do REPORTS-a).
  Od ovog trena SVE novo ide u `slinn/`; `arch_agnostic/` i `morphology/` su zamrznuti.
  **✅ GOTOVO (2026-08):** `slinn/` = kopija `arch_agnostic` (bez `__pycache__`/`tmp`); self-putanje
  `code/dipl/arch_agnostic`→`code/dipl/slinn` (sed; `_MORPH`/`baseline_models` netaknuti, 23 morphology-ref
  očuvane). Verifikacija (`_slinn_scaffold_check`): svi core moduli importaju IZ `slinn/`, `REGISTER_PATH`→
  `slinn/…`, `prepare` radi (housing→regression). SHIM-GENERALIZACIJA (bonus, nužna za paritet): `install_sizing_
  shims` sad pamti ORIGINALNE morphology funkcije i `_ag_layer_table`/`_ag_forward_ok`/`_ag_try_grow_layer`
  DELEGIRAJU na original kad adapter nema `forward_example` (morphology `ProfileAdapter`) → probe-adapter i
  morphology-adapter koegzistiraju bez sudara (točno što merge treba).
- **6.1 Backend swap (engine-vođeni GUI backend, autoriran u `slinn/`):** u `slinn/` se **kopira+prilagodi**
  morphology-ov `prep_worker`/`worker` tako da: `prep_worker` gradi `ctx = pipeline.prepare(model, probe_adapter,
  dev, dataset_path)`, autobatch preko `engine.autobatch(…, ctx, …)`, baseline metrika per-task; `worker` zove
  `engine.full_cycle(…, metric_fn=<per-task>)` umjesto `run_dead_ft`/`run_morph`. `config.json` dobiva `model_path`
  + `dataset_path` (pravi korisnički ulaz); task/metrika/enhaneri se AUTO-detektiraju (ne u configu). Dokazane
  engine-mehanike se preuzimaju u `slinn/` (kopirane/prilagođene, ne editiran morphology in-place). `morphology/`
  ostaje netaknut i radan kao paritet-referenca.
  **✅ GOTOVO (2026-08, `slinn/gui/` + `slinn/helper/_run61.py`→`REPORTS/run61.txt`):** novi engine-vođeni backend:
  `backend.py` (dijeljeno: `load_ctx`=eager+probe+`prepare`; `build_metric_fn`=PER-TASK gate — detection→morphology
  mAP plug, seg→mIoU, reg→r2, ostalo/bez-oznaka→teacher-agreement; `frozen_teacher`; `baseline_report`),
  `prep_worker.py` (semafor gpu/task/batch/perf iz `{model_path,dataset_path}`, `engine.autobatch`), `worker.py`
  (`engine.full_cycle` + `on_step`→`trajectory.jsonl`, best→`compressed.pt`, `status.json`). Isti JSON-protokol kao
  stari GUI (prep_status/status/trajectory). VALIDIRANO na housing (ne-detekcija) kroz SUBPROCESS-launch (kako GUI
  stvarno pokreće): prep `ready` (task=regression AUTO, r2 baseline 0.7565), compress `done` (6 koraka, params
  110721→99512, r2 zadržan 0.749, best_step 6, compressed.pt spremljen). NALAZ: `import worker`/`import prep_worker`
  je DVOSMISLEN (morphology/ i slinn/gui/ dijele imena) → workeri se pokreću SAMO kao subprocess-po-putanji (kako
  gui.py ionako radi `subprocess.Popen`); kolizija nestaje kad morphology ode u `legacy/` (6.5). Detekcija kroz worker:
  isti kod-put (`build_metric_fn`→mAP); engine na yolo već dokazan u 6.0 (odvojena yolo-kroz-worker provjera čeka
  slobodan GPU — trenutno zauzet korisnikovim `train_gt.py`).
- **6.2 GUI generalizacija:** (a) unos = putanja modela + putanja dataseta (ne selectbox 2 modela); (b) task se
  prikaže AUTO (read-only, iz `detect_task`) + `mode` (full/core_kd_only); (c) metrika-kartica dinamička iz
  `SUPPORTED_TASKS[task]["metrics"]` (npr. seg→mIoU, reg→r2/rmse, cls→f1/acc, det→mAP/mAR); (d) NOVA
  capability-kartica: što pipeline zna na OVOM modelu (iz `classify`/`position`/LAYER_REGISTER — prunable/tapovi/
  terminali/kd_mode) + format dataseta (SUPPORTED_DATASET_FORMATS); (e) trajektorija-graf task-generičan (ime
  metrike iz taska, ne hardkod mAP).
- **6.3 Paralelni rad + paritet-gate + flip:** u GUI-ju (ili harnessu) kratko oba puta na istom modelu (stari
  `morphology` vs novi `slinn`) → potvrdi paritet (detekcija) i smislenost (ne-detekcija) → **flip default** na
  `slinn/`. Stari `morphology/` ostaje radan kao fallback dok ne ode u `legacy/` (6.5).
- **6.4 Dovršetak `slinn/` jezgre (nakon zelenog flipa):** unutar `slinn/` počisti superseded obrasce naslijeđene
  kopiranjem: hardkodirani `TASK_METRICS`/`_EVALUATORS` (2 taska → SUPPORTED_TASKS+metric.py), detection-centrične
  GUI-birače, `config.MODEL_SPEC`/`FT_METRICS` hardkod, `pick_adapter` kao pogon. **VAŽNA NIJANSA — NE briši
  detekcijski decode:** `profiles.py` (ProfileAdapter/pick_profile/`_dense_decode`), `YoloAdapter`/`FrcnnAdapter`
  decode + `eval_map` žive u `slinn/plugins/detection/` jer ih `enhancers.py` (detekcijski KD) i mAP-scorer (gate)
  REUSE-aju kao IZOLIRAN per-obitelj plug (jedini priznati per-model dio). Zadrži REUSE-mehanike (dead/prune/grow/
  coupled-cost/align/rollback/autobatch/verzije/`kd._LOSS`) — sad kao `slinn/` moduli.

  **📐 IZMJERENO (2026-08, `_move64.py` → `REPORTS/move64.txt`):** tranzitivno zatvorenje iz 30 ulaznih točaka
  koje `slinn/` stvarno zove = **85 simbola, 1198 od 2486 redaka (48%)**. Ostalih 1288 redaka je mrtvo za nas
  i NE seli. Klasifikacija je po **dosežnosti** (dvije zatvorenosti: detekcijski ulazi `pick_adapter`/`eval_map`/
  `make_gt_loader` naspram ostalih), ne po imenu.
  - **Kolizije → 32 retka se odbacuju:** `weighted_leaves` (imamo `classify.py`), `evaluate` +
    `eval_classification` (imamo `metric.py`).
  - **Samo detekcija → `slinn/plugins/detection/`: 35 simbola, 508 redaka.** `YoloAdapter` (141) ·
    `FrcnnAdapter` (74) · `ProfileAdapter` (47) · `ModelAdapter` (37) · `_DetDataset` (32) · cijeli
    `profiles.py` (10/10 simbola) · gotovo cijeli `kd.py` (9/10 — detekcijski KD članovi).
  - **Samo jezgra → `slinn/`: 45 simbola, 639 redaka.** Dokazane mehanike: `_try_grow_layer` (83) ·
    `coupled_unit_cost` (58) · `remove_dead_neardead` (50) · `_grow_decide` (48) · `_select_grow_plan` (41) ·
    `_select_prune_plan` (36) · `_apply_prune_plan` (33) · `align_factors` · `_widen_*` (depthwise guard).
  - **21 `config` konstanta** — otud `slinn/config.py`.

  **TRI TOČKE ZAGAĐENJA koje analiza otkriva (riješiti TIJEKOM preseljenja, ne poslije):**
  1. `analysis.load_any` (generički eager loader, zove ga i `classify.py` i `position.py`) interno zove
     `build_fasterrcnn` → detekcija se provlači u jezgru. Razdvojiti: čisti loader ostaje, frcnn-grana ide u plug.
  2. `analysis._EVALUATORS` (hardkod 2 taska) povlači `eval_map` u jezgrenu zatvorenost — zato `eval_map`
     ispada „dijeljen". Kad `_EVALUATORS` umre (zamjena: SUPPORTED_TASKS + `metric.py`), `eval_map` postaje
     čisto detekcijski. **Ne seliti ga — obrisati.**
  3. `config.COCO_IDS`/`NUM_CLASSES`/`CLASS_NAMES`/`DATASET_ROOT` su konstante KONKRETNOG detekcijskog
     dataseta, ne globalne postavke → u plug (ili u job-config), ne u `slinn/settings.py`.

  **✅ KORAK 1 — JEZGRA PRESELJENA (2026-08, `_extract64.py`):** mehanicka AST-ekstrakcija (izvorni rasponi
  simbola, NE prepisivanje rukom) → 4 nova modula, svi bez ijedne morphology ovisnosti:
  - `slinn/settings.py` — kompresijski hiperparametri (ALIGN_*/PHASE2_*/FT_*), nula dataset/detekcijskih konstanti
  - `slinn/introspect.py` — 9 simbola iz `analysis.py`; **ne koristi NIJEDNU konstantu** (cista introspekcija)
  - `slinn/morph.py` — 21 simbol iz `compress.py`; treba samo 3 konstante + `config.ALIGN_*` dinamicki
  - `slinn/kdterms.py` — cijeli `kd.py`

  Rewiring uz ZADRZANE aliase (`A`/`C`/`CFG`) → 24 pozivna mjesta nepromijenjena. Preseljenje je
  SEMANTICKI NEUTRALNO; jedina namjerna izmjena = `load_any` gubi `spec=="fasterrcnn"` string-precac.
  Verifikacija: pun lanac (engine/loss/enhancers/pipeline) uvozi se uz **NULA ucitanih morphology modula**.
  Jedini preostali most je `gui/backend.py` (alias `DET`) na detekcijskom gate-u — tamo i pripada.

  **ISPRAVAK move64 analize:** `kd.py` je bio svrstan kao 90% detekcijski. KRIVO — modul je po vlastitom
  opisu genericki KD po tipu tapa, a `loss.py` ga zove iz AGNOSTICNE grane. Promaklo jer `_move64.py` skenira
  prefikse `A.`/`C.`/`K.`, a poziv ide kao `kd.`. Cijeli modul ide u jezgru.

  **⚠️ ZAMKA — SJENANJE IMENA MODULA (potrosen jedan ciklus):** dok su `slinn/` i `morphology/` oba na
  `sys.path`, ISTOIMENI modul u `slinn/` zasjenjuje morphologyjev za CIJELI proces. `slinn/config.py` je
  oborio paritet-harness (`morphology/analysis.py` → `from config import DATASET_ROOT` → dobio nas config).
  Rjeseno preimenovanjem u **`slinn/settings.py`**. Isto pravilo je vec bilo primijenjeno na `kdterms.py`
  (umjesto `kd.py`) i na `import worker` nalaz iz 6.1. Preostali sudar `introspect.py` je BEZOPASAN —
  provjereno: morphology nigdje ne uvozi vlastiti `introspect`. **Pravilo do 6.5 (dok je morphology jos u korijenu): novi modul u `slinn/`
  NE SMIJE nositi ime koje postoji u `morphology/`.**

  **🔜 KORAK 2 — UNIFIKACIJA `weighted_leaves` (ODLUKA KORISNIKA, 2026-08: OBAVEZNO, ne opcija).**
  Postoje dvije verzije: `introspect.weighted_leaves` (preseljena morphology, dim **2/4**) i
  `classify.weighted_leaves` (dim **2/3/4**, vidi Conv1d). **ULTIMATIVNO POBJEDUJE 2/3/4** — 2/4 verzija
  mora NESTATI, ne smije prezivjeti u jezgri ni kao fallback. Razlog: bez dim-3 je citav 1D lanac (M5,
  audio, sekvence) nevidljiv grafu ovisnosti → nema tapova ni terminala → jezgra nije agnosticna.
  Trenutno stanje: `engine.py` NAMJERNO drzi obje (`A.weighted_leaves` 7×, `_WL_AG` = classify verzija).
  Zasto NIJE napravljeno u istom koraku s preseljenjem: unifikacija je promjena PONASANJA (1D convovi
  postaju vidljivi ondje gdje prije nisu), pa bi pad pariteta postao dvosmislen — greska preseljenja ili
  namjerna izmjena. Redoslijed je zato: preseli (neutralno) → paritet zelen → TEK ONDA unificiraj →
  paritet ponovno. DEFINICIJA GOTOVOG: u `slinn/` postoji SAMO `classify.weighted_leaves`, `introspect`
  je vise ne definira, svih 7 poziva u `engine.py` ide na 2/3/4, `_WL_AG` alias nestaje kao suvisan,
  paritet zelen I 1D model (m5) i dalje prolazi pun ciklus.

  **✅ KORAK 2 GOTOV (2026-08).** `weighted_leaves` ima JEDNU definiciju: `classify.py:195` (dim 2/3/4).
  `introspect` je re-exporta (pozivna mjesta `A.weighted_leaves` nepromijenjena), `_WL_AG` alias i
  monkeypatch `A.weighted_leaves = _WL_AG` uklonjeni iz `install_sizing_shims`.
  NALAZ: ta je zakrpa bila JEDINA od cetiri BEZ fallbacka (ostale tri provjeravaju `forward_example` i
  inace zovu original) → 2/3/4 je i prije uvijek pobjedivao, pa unifikacija NIJE promjena ponasanja
  nego uklanjanje obmane.
  VERIFICIRANO: (a) paritet `REPORTS/parity60.txt` — B novi 5.721 GFLOPs / 4.0% / mAP 0.3834 (91.9%),
  strukturno IDENTICNO pred-unifikacijskom runu (5.721 / 4.0); mAP razlika u 4. decimali je FT-sum
  (stara strana varirala 0.2971→0.3167 izmedu ista dva pokretanja). (b) 1D `REPORTS/prune1d55.txt` —
  PROLAZI: prune 61 Conv1d kanala + forward-ok, grow 3 Conv1d sloja function-preserving |Δ|=0.00e+00.

  **⚠️ NUSNALAZ — ZASTARJELI HARNESSI (rijeseno):** preseljenje je utisalo dev-harnesse. Oni su uvozili
  `compress`/`analysis` iz morphology, a engine od 6.4 zakrpa `morph`/`introspect` → harnessi bi testirali
  NEZAKRPANI stari kod, i to BEZ greske pri uvozu (tiho krivo). Prvi simptom: `_prune1d55.py` grow-test
  pao s CUDA OOM 25.75 GiB jer je dosao u `morphology/compress.py` hardkod `rand(3, sz, sz)` nad audio
  modelom. Preusmjereno 16 harnessa na `morph`/`introspect`; **`_parity60.py` NAMJERNO ostaje na oba**
  (po naravi usporeduje stari i novi put). Ponovno pokrenut samo `_prune1d55.py`; ostalima status nije
  provjeren — pokrenuti po potrebi.

  **✅ KORAK 3 GOTOV — DETEKCIJSKI PLUG (2026-08, `_extractplug64.py`):** `slinn/plugins/detection/`
  - `adapters.py` — 16 simbola (`ModelAdapter`/`YoloAdapter`/`FrcnnAdapter`/`_DetDataset`/`eval_map`/`pick_adapter`)
  - `profiles.py` — cijeli modul (svih 10 simbola je detekcijskih)
  - `dsconfig.py` — 10 konstanti KONKRETNOG dataseta (razredi, COCO remap, DATASET_ROOT); jezgra ih ne cita
  - `__init__.py` — JAVNI OTVOR: `pick_adapter`, `make_gt_loader`, `eval_map`, `set_bn_eval`

  ZASTO BAS DETEKCIJA: kod regresije/klasifikacije/segmentacije sirovi izlaz modela VEC JEST odgovor.
  Kod detekcije nije — yolo izbacuje tisuce brojeva vezanih uz sidra i DFL, iz kojih se okviri tek moraju
  izracunati (decode), pa procistiti (NMS), pa upariti s GT-om po IoU (mAP). Ta tri koraka se NE MOGU
  izmjeriti probanjem; traze unaprijed poznatu konvenciju OBITELJI (yolo != frcnn). Sve ostalo u jezgri
  (koji kanal rezati, kako rasti, KD gubitak) radi jednako na svim taskovima — zato jedino ovo ide iza ograde.

  Rucni popravci nakon ekstrakcije (oba bi u paketu tiho promasila): `pick_adapter` je radio `import profiles`
  → `from . import profiles`, i `import kd` → `import kdterms as kd`.
  VERIFICIRANO na STVARNOM modelu (ne samo uvozom): yolo26n → `ProfileAdapter(kind=yolo)` → `eval_map`
  mAP 0.4270 (48 slika).

  **✅ RAZDVAJANJE POTPUNO:** `morphology` maknut sa `sys.path` u CIJELOJ jezgri i GUI-ju
  (classify/enhancers/loss/metric/pipeline/position/gui.backend). Test s nasilno ocisenom putanjom:
  jezgra + plug + `gui/backend.py` uvoze se uz **NULA morphology modula**. `backend.py` je JEDINO mjesto
  gdje jezgra dodiruje plug (`build_metric_fn`) → obrisi `plugins/detection/` i jezgra radi dalje
  (gate padne na teacher-agreement). Karantena je time stvarna, ne deklarativna.

  **✅ KORAK 4 (ciscenje hardkoda) — NIJE BILO POTREBNO.** `_EVALUATORS`, `TASK_METRICS`, `MODEL_SPEC`,
  `FT_METRICS` NIKAD nisu preseljeni: ekstrakcija je bila selektivna pa su ostali u morphology, a slinn ga
  vise ne uvozi. Provjera: nula pogodaka u `slinn/**.py` (osim spomena u komentaru `metric.py:7`).
  `pick_adapter` vise nije pogon nego samo plug-simbol. Preostala stavka iz izvornog 6.4 popisa —
  "detection-centricni GUI-birace" — pripada 6.2 jer `gui.py` jos ne postoji.

  **VERIFIKACIJA CIJELE 6.4:** `parity60` B novi = 5.721 GFLOPs / 4.0% / mAP 0.3839 / 92.0% — **TRECI put
  zaredom ZNAMENKU PO ZNAMENKU isto**, kroz preseljenje jezgre, unifikaciju `weighted_leaves` I izdvajanje
  pluga. Novi put je deterministican; stara strana pluta (0.3057 → 0.2971 → 0.3167 → 0.3242), sto potvrduje
  da varijacija dolazi od nje. `run61` (ne-detekcija, housing/regresija, subprocess): prep `ready`
  (autobatch TRAIN=64), compress `done`, 6 koraka, r2 0.7565 → 0.7492, `compressed.pt` spremljen.

  **⚠️ REDOSLIJED — NAPOMENA ZA 6.3/6.5:** flip i selidba u `legacy/` ne mogu se DOVRSITI prije 6.2, jer je
  `morphology/gui.py` jos uvijek JEDINI radni frontend. Praktican rasplet: flip ENGINE-a je vec dokazan
  (3× paritet + run61) → napravi 6.2 (`slinn/gui/gui.py`) → pa 6.3 flip i 6.5 selidbu u `legacy/` zajedno.

  **✅ KORAK 5 — PREPOZNAVANJE FORMATA IZLAZA (2026-08, `outfmt.py` + `_outfmt64.py` → `REPORTS/outfmt64.txt`).**
  NALAZ KOJI JE OVO POKRENUO: `FrcnnAdapter.matches` je strukturan (`hasattr(model, "roi_heads")`), ali
  `YoloAdapter.matches` je NJUSIO IME PAKETA (`type(model).__module__.startswith("ultralytics")`).
  To je jedino mjesto u lancu koje je uzimalo model po imenu.

  ODLUKA KORISNIKA: ultralytics ostaje PRIMARNA provjera (brza i sigurna za nase modele), a ispod nje
  ide fallback po FORMATU IZLAZA. Cilj: model koji nikad nismo vidjeli mora i dalje proci kroz slinn.

  `pick_adapter(model, sample_input=None, strict=False)` — tri razine:
    1. poznata obitelj (`matches`) · 2. `outfmt.describe` -> posudi decode koji taj OBLIK zna citati ·
    3. None + glasno upozorenje -> jezgra degradira na KD-only (gate = teacher-agreement) i KOMPRIMIRA DALJE.
  `strict=True` pretvara 3. u tvrdi prekid (samo kad je korisnik izricito trazio mAP-gated kompresiju).

  PREPOZNATE OBITELJI FORMATA (`FORMATS`), sve mjerenjem oblika:
  `boxes_dicts` (torchvision liste) → FrcnnAdapter · `dense_nc` [B,4+K,N] (yolov8+) i `dense_cn` [B,N,4+K]
  (yolov5) → YoloAdapter · `set_pred` (DETR dict) i `multilevel` (4D po FPN razini) → prepoznati ali BEZ
  decode-a (posten "znam sto je, ne znam procitati") · `unknown`.

  **OGRANICENJE KOJE SE PRIJAVLJUJE, NE PRESUCUJE:** oblik otkriva OBITELJ, ali NE i konvenciju okvira
  (`xyxy`/`xywh`/`cxcywh`, px ili normalizirano). Zato `probe_box_layout` mjeri: pretvori po svim
  pretpostavkama i uzmi onu s najvise VALJANIH okvira. KLJUCNO: `xywh` i `cxcywh` razlikuju se SAMO po
  tome probije li okvir granicu — ako su svi okviri duboko unutar slike, razlika je NEMJERLJIVA. Tada
  funkcija vraca `confident=False` + popis `ambiguous` umjesto tiho pogodjenog odgovora. Kod pravih
  gustih izlaza (tisuce sidara preko cijele slike) okviri dodiruju rub pa se razlucuje.
  (Prva verzija je tiho birala `xywh` za `cxcywh` ulaz — uhvatio test, popravljeno.)

  VERIFICIRANO (`_outfmt64.py`, 6 izmisljenih mreza koje NISU iz ultralytics/torchvision — ime ne pomaze):
  formati 6/6 tocno · degradacija na None bez pada · konvencija okvira: 3/3 tocno kad okviri dodiruju rub,
  a kod okvira-svi-unutra ispravno prijavljuje `ambiguous=['cxcywh/px','xywh/px']`.
  Regresija: pravi yolo26n i dalje ide PRIMARNIM putem (`ProfileAdapter kind=yolo`), fallback se ne poziva.

  **✅ KORAK 6 — DECODE ZA SVE PREPOZNATE FORMATE (2026-08, `outfmt.decode` + `_outfmtreal64.py`
  → `REPORTS/outfmt_real64.txt`).** Prosireno sa 6 na 9 formata, svi osim `dense_cn`/`dense_nc`
  potvrdjeni na STVARNIM tezinama. Svaki decode svodi izlaz na ISTI oblik
  `[{boxes xyxy PIKSELI, scores, labels}]` — ono sto torchmetrics mAP jede, pa jezgra dalje ne mora
  znati odakle je doslo.

  | format | izvor potvrde | decode |
  |---|---|---|
  | `nms_out` [B,D,6] | yolo26n eval (end2end) | builtin — vec dekodirano |
  | `dense_split` {boxes,scores} | yolo26n train `one2one` | builtin — sidro-relativno |
  | `set_pred` {logits,pred_boxes} | **yolos-tiny** (HF, 6.5M) | builtin — cxcywh norm, bez NMS |
  | `boxes_dicts` | fasterrcnn (torchvision) | FrcnnAdapter |
  | `feat_pyramid` | yolo26n train `feats` | NEMA (i ne smije) |
  | `dense_nc` / `dense_cn` | sinteticki | YoloAdapter |
  | `multilevel`, `unknown` | — | NEMA |

  **DVIJE GRESKE KOJE JE OVAJ KORAK UHVATIO NA KORISNIKOVOM VLASTITOM MODELU:**
  1. **`nms_out` nije postojao** — yolo26n eval vraca `[B,300,6]`, sto bi stari klasifikator proglasio
     gustom glavom "s ~2 razreda". Razlucuje se MJERENJEM: ch5 je CJELOBROJAN (id razreda), ch4 u [0,1]
     (conf) → vec dekodiran NMS izlaz.
  2. **`feat_pyramid` naspram `multilevel`** — yolo26 `feats` su `[64,128,256]` kanala po razini, dakle
     RAZLICITI → piramida ZNACAJKI, ne predikcije. Prava glava ima ISTI broj kanala na svim razinama.
     Ta razlika sprjecava da se piramida znacajki proglasi detekcijskim izlazom.

  **SIDRO-RELATIVNI DECODE (`anchor_grid` + `_ltrb_to_xyxy`).** Nalaz: yolo26 train `boxes` NISU
  koordinate — raspon [-0.6, 13.7], svih 6 izravnih hipoteza dobiva score ~0.006. To su DFL udaljenosti
  (l,t,r,b) od sidra u jedinicama koraka. Resetka se IZVODI, ne hardkodira: za N sidara i ulaz HxW
  provjeri pogadja li `sum((H//s)*(W//s))` za korake (8,16,32) tocno N (640² → 6400+1600+400 = 8400).
  Ako da, dekodiraj i OCIJENI rezultat kroz `probe_box_layout`; prihvati samo ako je udio valjanih okvira
  >= 0.2. **KRIZNA POTVRDA:** tako dekodirana gusta glava daje top-score **0.749**, a modelov vlastiti
  end2end NMS na ISTOJ slici **0.751** — dva neovisna puta do istog okvira.

  **`_dense_to_dets` sada ODBIJA umjesto da izmislja:** ako nijedna hipoteza (izravna ni sidro-relativna)
  ne prijedje `min_score`, vraca None → `decode` vraca None → degradacija na KD-only. Prva verzija je
  koristila `lay["layout"] or "xyxy"` bez obzira na pouzdanost i proizvodila smece.

  **TEST MORA IC NA STVARNU SLIKU, NE SUM.** Prvo mjerenje je dalo "0% valjanih okvira" na `torch.rand`
  ulazu — model na sumu izbacuje besmislene okvire pa provjera valjanosti nema sto mjeriti. Sa stvarnom
  slikom iz val skupa: 5/5 formata, svi 100% valjanih.

  yolos-tiny je nakon prolaza OBRISAN (25 MB, `~/.cache/huggingface/hub/models--hustvl--yolos-tiny`) —
  test ga preskace uz poruku ako ga nema, ostali slucajevi i dalje rade.
- **✅ 6.2 GUI GENERALIZACIJA — GOTOVO (2026-08).** `slinn/gui/gui.py` (Streamlit), tri stranice.
  Odluke korisnika: sve tri stranice (puna zamjena) · SAMO rucne putanje (bez popisa modela) ·
  align/kvantizacija SE PRENOSI iz morphology.

  **Novi moduli / izmjene:**
  - `slinn/overview.py` — GENERICKA zamjena za `analysis.analyze_report` (koji je bio detekcijski).
    `summary` (velicina, tipovi, task/mode/format/tapovi/kd_mode), `layer_rows` (per-layer + spregnuta
    cijena iz JEDNOG izvora `morph.prune_costs`), `top_prune`, `worst_aligned`, `capabilities` (About).
    `report(deep=True)` doda KD-vaznost. NULA grananja po modelu ni tasku.
  - `slinn/morph.py` += `model_align_score`, `best_align_score` (preseljeno bez izmjena).
  - `engine.py` trajektorija += `size_mb`, `gflops_freed`, `gflops_reinvested`, `align_score`,
    `align_best`; `traj[0]` (BASELINE) se sad EMITIRA kroz `on_step` — prije ga GUI nije imao kao
    referentnu tocku.
  - `gui/worker.py` — primjenjuje `align_m` iz configa PRIJE `full_cycle` (`settings.ALIGN_M` +
    `PHASE2_MIN_ALIVE = M//2`), jer ih `morph` cita dinamicki.

  **Sto je nestalo u odnosu na morphology GUI:** `selectbox("Model", ["fasterrcnn", YOLO_PATH])`,
  hardkodirane mAP/mAR kartice i biraci metrika, `config.MODEL_SPEC`/`FT_METRICS`. Metrika se sad
  ISPISUJE (auto iz taska), ne bira. Ako je `teacher_agreement`, GUI to glasno kaze.

  **VERIFICIRANO U PRAVOM PREGLEDNIKU** (headless streamlit + klikanje), nula gresaka u logu:
  - Overview na yolo26n: 2.572M params · 5.9584 GFLOPs (poklapa se s paritet baselineom) · 9.8 MB ·
    poravnanje 83.7% · **sve auto-detektirano**: task=detection, mode=full, format=yolo, 8371 uzoraka,
    kd_mode=feature+logit, 3 tapa, 93/126 prunable, enhaneri=da. Tablice najjeftinijeg reza i najgore
    poravnatih slojeva se crtaju.
  - Overview na housing (regresija, tablicni podaci): isti ekran, task=regression, format=tabular —
    dokaz da nema detekcijskog grananja.
  - About: svih 6 taskova, 10 formata dataseta, 9 formata IZLAZA s pripadnim decode-om, 27 tipova slojeva.
  - Compress: priprema/semafor renderira; pun run nije pokretan (dugotrajno).

  **NALAZ (uhvacen pri prvom pokretanju):** `overview.py` je zvao `A.gflops_total`, a sizing-shimovi se
  instaliraju tek pri `import engine` → mjerenje je padalo na hardkodirani `rand(3,640,640)` i pucalo na
  tablicnom modelu. Ispravak: `overview` uvozi `engine` i koristi `E.gflops` (adapter-svjestan).

- **6.5 Arhiviranje starih foldera + finalna struktura.**

  **⛔ ODLUKA KORISNIKA (2026-08): `morphology/` i `arch_agnostic/` se NE BRIŠU. NIKAD.**
  Umjesto brisanja: napravi `legacy/` i **premjesti** oba foldera u njega →
  `legacy/morphology/` i `legacy/arch_agnostic/`. Kôd ostaje sačuvan kao referenca; miče se samo iz
  korijena da `slinn/` bude jedini živi proizvod. Zamjena za "brisanje" u svakoj ranijoj rečenici ovog
  plana (§ODLUKA O FOLDERU, 6.1, 6.3, 6.4) glasi: **premještanje u `legacy/`**.

  **ČIŠĆENJE PRI PREMJEŠTANJU:** iz `morphology/` obriši regenerabilne artefakte da arhiva ostane čista i
  prazna — **precomputane datasetove/cacheve (`morphology/tmp/`) i spremljene modele (`morphology/models/`)**.
  To su izlazi, ne izvor; ponovno se generiraju iz koda. Prije brisanja provjeriti veličinu i sadržaj i
  potvrditi s korisnikom (jednosmjerno).

  Ciljno stablo:
  `slinn/` (jezgra: engine/loss/task/dataset/classify/position/pipeline/metric/enhancers/morph/introspect/
  kdterms/settings + registri) · `slinn/plugins/detection/` (izolirani decode-plug) · `slinn/gui/` (Streamlit) ·
  `slinn/helper/` (dev-alat) · `slinn/REPORTS/` · `legacy/` (arhiva, izvan puta izvođenja).
  Jedno čisto modul-stablo, jasna granica jezgra (agnostično) / plug (decode); u korijenu nema više
  `morphology`/`arch_agnostic` duplikata.

  Nuspojava koja time nestaje: sudari imena modula (`introspect`, ranije `config`) prestaju biti opasni čim
  `legacy/` nije na `sys.path`.

### ✅ 6.3 FLIP-GATE PROSAO (2026-08-23) — pun run KROZ NOVI GUI

Zadnji neprovjereni put: kompresija pokrenuta iz `slinn/gui/gui.py` (Compress stranica), yolo26n,
INT8/M=32, tolerancija 0.75, cilj 15%, `DEV_DATA_SUBSET=200`.

    kor    GFLOPs     params      mAP    align      MB    rez%
      0    5.9584  2,572,280   0.4270    0.837    9.81    0.0%
      6    5.5529  2,465,985   0.3345    0.847    9.41    6.8%   <- best
      9    5.3183  2,338,064   0.2371    0.823    8.92   10.7%

`state=done` · `best_step=6` · `compressed.pt` spremljen (9.9 MB).

**QUALITY-GATE RADI KAKO TREBA:** prag = 0.75 x 0.4270 = 0.3202. Koraci 7-9 probili ga TRI puta
zaredom (0.3192 / 0.2832 / 0.2371) -> petlja stala i zadrzala KORAK 6 kao najmanji model koji jos
drzi kvalitetu. Nije uzet zadnji nego najbolji.

**POKRIVENO OVIM RUNOM (nijedan raniji test to nije imao zajedno):**
worker pokrenut IZ GUI-ja · `align_m=32` iz kartice kvantizacije stvarno primijenjen (align_score
prati ×32 skalu, poklapa se s Overview 83.7%) · svih pet novih trajektorijskih vrijednosti
(`size_mb`, `gflops_freed/reinvested`, `align_score`, `align_best`) · baseline tocka (step 0) ·
`compressed.pt`. Time je zatvoren i zaostatak iz 6.1 ("yolo kroz novi worker ceka slobodan GPU").

**⚠️ BUG UHVACEN OVIM RUNOM — `enhancers.py:31 import profiles as PF`.** Prvi pokusaj pao s
`ModuleNotFoundError: No module named 'profiles'`. Uvoz je bio UNUTAR funkcije (lazy), pa ga provjera
"jezgra se uvozi uz nula morphology modula" nije mogla uhvatiti. Dok je morphology bio na `sys.path`,
tiho je povlacio NJEGOV `profiles` — detekcijski KD je isao MIMO pluga, a sve je izgledalo ispravno.
Ispravak: `from plugins.detection import profiles as PF` unutar `try/except ImportError` -> bez pluga
se enhancer sam iskljuci umjesto da srusi run.
POUKA: skidanje morphology sa `sys.path` (6.4) nije kozmetika — ovo je prvi dokaz da karantena hvata
ono sto bi inace proslo nezapazeno. Provjeriti LAZY uvoze, ne samo modul-razinske.

**OGRADA OKO BROJKI:** mAP pada strmije nego u paritet testu (tamo 92% zadrzano na 4% reza) jer
`DEV_DATA_SUBSET=200` daje oporavku samo 200 slika, a gate mjeri na 48. **Ovo su brojke INSTALACIJE,
ne rezultat.** Za rad: `DEV_DATA_SUBSET=None` + ozbiljniji `ft_steps`.

### ✅ 6.5 SELIDBA U `legacy/` GOTOVA (2026-08-23) — FAZA 6 ZAVRSENA

Korijen projekta sada: `slinn/` (zivi proizvod) + `legacy/` (arhiva, izvan puta izvodjenja).

**OBRISANO:** `morphology/tmp/` — **12 GB** teacher cachea (yolo26n 11G, fasterrcnn 1.2G,
yolo26l 106M). Regenerabilno.
**ZADRZANO (odluka korisnika):** `morphology/models/` (174 MB — 3 rezultata ranijih kompresija:
frcnn 75M, yolo26l 96M, yolo26n 9.7M) i `arch_agnostic/tmp/` (527 MB).
`legacy/` = 702 MB · `legacy/morphology` 175 MB · `legacy/arch_agnostic` 527 MB.

**ZASTO JE UOPCE BILO 24 REFERENCE NA `morphology/`** (pitanje korisnika — razvrstano mjerenjem,
`scratchpad/why_morph.py`):
1. **16 harnessa — LAZNA UZBUNA.** Uvoze `introspect`, ali to je NAS `slinn/introspect.py` (nastao u
   6.4). Nista nije dolazilo iz morphology; ostala im je samo mrtva `sys.path.insert` linija. Maknuto.
2. **2 harnessa — STVARAN BUG (moj).** `_grow53.py`/`_spike50.py` rade `import config as CFG`. Kad sam
   `slinn/config.py` preimenovao u `settings.py` (zbog sjenanja imena), oni su TIHO pali na
   morphologyjev `config`. Ista klasa buga kao `enhancers.py`. Popravljeno -> `import settings as CFG`.
3. **`_parity60.py` — NAMJERNO.** Usporeduje stari i novi engine, treba oba. Putanja -> `legacy/`.
4. **5 `baseline_models/*/_verify.py` — stari kod** koji prethodi slinnu i koristi `analysis`.
   Putanja -> `legacy/`. Jedini stvarni preostali korisnici arhive.

**VERIFICIRANO NAKON SELIDBE:** jezgra + plug + `gui/backend` se uvoze cisto · `pick_adapter` na
yolo26n vraca `ProfileAdapter` · **nula referenci na staru putanju** u cijelom projektu ·
stara putanja nije na `sys.path`.

**POUKA (druga potvrda istog obrasca):** dvije od tri stvarne ovisnosti bile su TIHE — kod se oslanjao
na to da je morphology na `sys.path`, pa je pogresan modul ulazio bez ijedne greske. Prvi put
`enhancers.py` (lazy `import profiles`), drugi put `import config`. Sjenanje imena modula je glavni
izvor te klase gresaka; nakon selidbe vise nije moguca jer `legacy/` nije na putanji.

### ✅ 6.6 `legacy/` VISE NIJE UVJET ZA RAD (2026-08-23)

ZAHTJEV KORISNIKA: "ako sutra obrisem legacy, sve mora i dalje raditi". Arhiva smije biti referenca,
nikad ovisnost. Odvezano svih 12 preostalih datoteka.

**A. 5x `baseline_models/*/_verify.py` — POTPUNO na slinn.** Iz `analysis` su koristili SAMO
`load_any` (postoji u `slinn/introspect.py`). NALAZ: pokazivali su i na `arch_agnostic`, koji je
TAKODJER u `legacy/` — dakle bili su dvostruko slomljeni, ne samo jednom. Sada `_AA` -> `slinn`,
`import analysis as A` -> `import introspect as A`, legacy linija maknuta.

**B. 3x mrtve path linije** (`_sweep.py`, `_populate.py`, `_spike50.py`) — imale su
`sys.path.insert(legacy)` ali uvoze SLINN module. Promasilo ih je ranije ciscenje jer koriste druga
imena varijabli (`_M`, `_MORPH`). Maknuto.

**C. 4x POVIJESNI alati — graciozno odustajanje umjesto pada.** `_parity60.py` (usporeduje stari i
novi engine) i `_move64/_extract64/_extractplug64` (citaju iz morphology jer su njime PRESELILI kod).
Ti po naravi trebaju arhivu i ne mogu se portati — posao im je odradjen. Sada provjere postoji li
`legacy/morphology` i, ako ne, ispisu sto su bili i zavrse s kodom 0.

**VERIFICIRANO S POTPUNO SAKRIVENOM ARHIVOM** (`mv legacy _legacy_HIDDEN`):
- jezgra + plug + `gui/backend` uvoz OK; `pick_adapter` -> `ProfileAdapter`
- `baseline_models/housing_mlp/_verify.py` odradi pun probe+classify+position (13 leafova, 1 tap)
- svih 26 harnessa u `slinn/helper/` sintaksno prolazi
- 4 povijesna alata uredno odustanu (exit 0, jasna poruka)
- **`_run61.py` odvrti PUN GUI-backend lanac** (prep -> compress -> `compressed.pt`) bez arhive

Arhiva je od sada cisto povijesna: `legacy/` se moze obrisati bez ijedne posljedice za rad.

### ✅ 6.7 TERMINAL-LOG U GUI-ju (2026-08-23)

ZAHTJEV: stari morphology GUI je uz grafove prikazivao i terminal-ispis napretka; isto se trazi i ovdje.

**Mehanika (preneseno iz morphology/worker.py):** `backend._Tee` + `backend.tee_log(name)` preusmjere
`stdout`/`stderr` u terminal **I** u `JOB/<name>`. `worker.py` -> `worker.log`, `prep_worker.py` ->
`prep.log`. Pokretanje iz terminala ostaje nepromijenjeno; log je samo dodatni zapis.
GUI: `_render_log()` (expander + `st.code`, zadnjih 8000 znakova) na tri mjesta — trening log otvoren
DOK RUN TRAJE, log pripreme otvoren pri gresci. `_startup_cleanup` brise oba pri novom pokretanju servera.

**NALAZ: sam tee nije bio dovoljan.** Prvi test dao je log od **0 bajtova** — `slinn/engine.py` ima
svega 4 `print`-a (samo poruke o cacheu), dok je morphologyjev `compress.py` bio brbljav. Bez dodatnog
ispisa kartica bi bila prazna.
RJESENJE: ispis napretka dodan u `worker.py`, NE u engine — `on_step` ionako vec dobiva svaki zapis
trajektorije. Tako je log task-genericki i engine ostaje tih.
  - zaglavlje: model · task/mode/enhaneri/metrika · cilj/tolerancija/batch/FT/max · align M · baseline -> prag
  - tablica po koraku: GFLOPs · params · metrika · align · MB · rez% · KD, uz oznaku `<-- ISPOD praga`
  - podredak: rezano kanala · naraslo (ime sloja skraceno na 2 dijela) · broj zabranjenih slojeva
  - zavrsetak: najbolji korak, rez%, putanja spremljenog modela
  - `prep_worker`: svaki semafor-korak ispisan JEDNOM kad dobije boju

**DVIJE GRESKE UHVACENE PRI PRVOM ISPISU (housing):**
1. GFLOPs stupac je pokazivao `0.0002` za SVE korake — 4 decimale nisu dovoljne za mali model.
   Popravak: `_g()` bira `%.4f` iznad 0.01, inace `%.3e` (yolo ~6, housing ~2e-4).
2. Zavrsni redak je tvrdio "zadnji korak ispod praga, zato se ne uzima" i kad je najbolji korak BIO
   zadnji. Sada se ispisuje samo kad se `best_gflops` i `final_gflops` stvarno razlikuju.
(Usput ocisen besmislen ostatak `n.split(".")[-2:] and n` u retku za rast.)

**VERIFICIRANO:** `_run61.py` pun lanac -> `prep.log` i `worker.log` sadrze tocno ocekivano
(v. ispis u razgovoru); sve 4 GUI datoteke sintaksno prolaze. Vizualni prikaz expandera nije
potvrdjen u pregledniku (Streamlit text_input ne prima sinteticke evente) — provjera je trivijalna
u vlastitom pregledniku.

### ✅ 6.8 CETIRI BUGA IZ PRVE PRIMJENE NA CIJELOM ZOOU (2026-08-23)

Korisnik je uocio dva simptoma ("staje na 20", "rezovi nisu 1.5%") i pitao za f1. Dijagnoza je dala
CETIRI odvojena uzroka; svaki je prvo izmjeren, pa popravljen, pa provjeren.

**1. Petlja je stajala na 20 koraka.** `engine.py:444` je imao HARDKODIRAN literal `max_steps=20`,
dok morphology koristi `PHASE2_MAX_STEPS = 200`. Zato je ta konstanta izgledala "mrtvo" — nije bila
suvisna, nego NESPOJENA. To je ujedno odgovor na pitanje "zasto neke konstante nisu implementirane":
prekopirane su u `settings.py`, a u kodu je pisao literal.
FIX: `max_steps=None` -> razrjesava se u `CFG.PHASE2_MAX_STEPS`; GUI default 200 (uz help da je to
sigurnosna granica, NE cilj).

**2. Rezovi 0.07% umjesto 1.5% (DistilBERT).** `coupled_unit_cost` je broj ULAZNIH kanala citao iz
`ish[1]` — dimenzije 1 IZMJERENOG oblika ulaza. To vrijedi za NCHW conv i 2D linear, ali kod
sekvencijskog ulaza `(B, S, C)` to je DULJINA SEKVENCE. DistilBERT: ulaz `(1,64,768)` -> dijelilo se
sa 64 umjesto 768 (lin2: 64 umjesto 3072).
DOKAZ: cijena/kanal napuhana **49x** (lin1) i **134x** (lin2); `pre_classifier` je bio JEDINI tocan
(1.0x) jer je jedini 2D. Planer je zato rezao ~20 kanala umjesto tisuca.
FIX: `in_features`/`in_channels` iz SAMOG MODULA (egzaktno, neovisno o rasporedu).
PROVJERA: 49x -> **2.0x**, sto je i tocan iznos (rez kanala `lin1` uklanja i stupac iz `lin2`).
yolo26n nakon popravka: rez **1.46% / 1.19% / 0.99%** — u rangu morphologyjevih 1.58/1.51/1.57 na vocu.

**3. Klasifikacija je uvijek padala na `teacher_agreement`.** `metric.py` ima `eval_classification`,
ali NEMA citac parova, a `build_metric_fn` nije imao granu (u kodu je stajao komentar "6.2 backlog").
PROVJERENO U `legacy/arch_agnostic/metric.py`: ista rupa — nije bilo rijeseno ni tamo
(`REPORTS/metric56.txt` pokriva samo segmentaciju). Dakle nov posao, ne ponovno otkrivanje.
FIX: `metric.pairs_classification` za `folder_per_class` (slike i audio).
DVIJE MOJE GRESKE UHVACENE PRI PROVJERI: (a) uzimao prvih N sortiranih datoteka = SVE iz jednog
razreda -> f1 0.0000; (b) indeks razreda gradio iz uzorka umjesto iz strukture foldera.
**TUDJA ZAMKA KOJA JE VAZNIJA:** M5 ima **36 foldera ali 12 izlaza** — `data.py` mapira 25 foldera u
`unknown`, `_background_noise_` u `silence`. IME FOLDERA NIJE OZNAKA. Bez provjere bi metrika bila
uvjerljiva a kriva (izmjereno f1 0.011 = razina slucajnog pogadjanja).
FIX: `n_classes` (sirina izlaza modela) se prosljedjuje citacu; nepoklapanje -> vrati [] i degradiraj
na teacher-agreement uz glasnu poruku. Bolje priznati neznanje nego izmisliti brojku.

**4. Spregnutost tapova (otkrio ju je popravak #2).** Cim je planer poceo rezati normalno, feature-KD
je pukao: `student 758 vs teacher 768`. `position.py` tap oznaci `morph=False`, ali to ga stiti samo
kao KORIJEN reza — sirina mu ostaje spregnuta kroz tp-grupu.
Morphology to nije morao rjesavati: kod yola/frcnn su tapovi u vratu a rezivo u okosnici (strukturno
odvojeno), a `adapter.protect_prefixes` je RUCNO pokrivao cijela podstabla.
FIX: `morph.tap_coupled` — isto nacelo, ali IZMJERENO iz tp-grafa, pa vrijedi za bilo koji model.
`pipeline.prepare` izbaci iz `prunable` sve sto dijeli grupu s tapom.
| model | prunable prije -> poslije |
|---|---|
| yolo26n | 93 -> 85 |
| voc_deeplabv3 | 34 -> 30 |
| DistilBERT | 8 -> **0** (uz upozorenje) |
DistilBERT-ova nula je TOCAN odgovor: cijeli rezidualni tok je na 768, pa se feature-KD i strukturni
rez medjusobno iskljucuju. Prije bi to puknulo usred treninga.

**TAP CAP 5 -> 3 (odluka korisnika).** `TAP_CAP_ABS` je zapravo bio DONJA granica
(`max(5, 10% morphabilnih)`), pa su veliki modeli dobivali 9+ tapova. Sada je TVRDA gornja:
`max(1, min(3, 10% morphabilnih))`. Manje tapova = manje spregnutog = vise rezivog; nusucinak je
grublji feature-KD signal. Ucinak: voc 5->3 tapa i prunable **25 -> 30**; yolo nepromijenjen (imao 3);
m5 dobiva 1 tap; DistilBERT i dalje 0 (njemu tapovi nisu uzrok).

### ✅ 6.9 KOLICINA PODATAKA — AUTO UMJESTO PITANJA (2026-08-23)

Korisnik: "ako proces sam pronalazi batch size, zasto ja moram definirati `n_batches`? Sto manje pitati,
sto vise sigurno zakljuciti." Provjereno u legacy: **morphology NIKAD nije pitao** —
`teacher_mem_plan` uzima `n_batches = len(loader)` (CIJELI train split), izmjeri jedan batch, izracuna
cache i odustane ako ne stane. arch_agnostic je taj plan ispustio i zabio `n_batches=8`.

**SADA: nista se ne pita.** KD = cijeli train split, metrika = cijeli val split.

**ZID NIJE BIO DISK NEGO RAM.** `materialize_train_batches` je drzala SVE dekodirane uzorke u memoriji:
yolo 5860 x 4.7 MB (fp32 640x640) = **26.8 GB**, dostupno 12 GB. Zato je `n_batches=8` i bio default —
nije bila lijenost nego posljedica dizajna. morphology to nije imao jer je isao DataLoaderom (ulazi se
dekodiraju po batchu i odbacuju; na disk idu SAMO teacher signali).
FIX: `LazyBatches` — batchevi drze PUTANJE, dekodiraju se pri pristupu. Ponasa se kao lista pa
`to_device(batches[i])` i `for b in batches` rade nepromijenjeno.
Uz to: `_fingerprint` racuna otisak iz PUTANJA (inace bi dekodirao 1 uzorak po batchu = 367 dekodiranja),
a `_batch_size_of` cita `bsize` bez dekodiranja (`max(len(b) for b in batches)` bi procitao SVE).

IZMJERENO na punom yolo train splitu:
    367 batcheva · 5872 uzoraka · materijalizacija 2.1s · RAM +0.01 GB  (prije bi 26.8 GB)
    teacher cache 7.86 GB · slobodno 636 GB · stane
Smoke (lijeni batchevi kroz precompute + FT): rez 1.45% / 0.95% / 0.91%, KD racuna normalno.

**NOVO: `engine.plan_teacher_cache`** (port morphology `teacher_mem_plan`) — izmjeri jedan batch
teacher-signala, procijeni cijeli cache, provjeri disk. Prep dobiva karticu **"Plan podataka i teacher
cachea"** (koju je stari GUI imao, a 6.1 ispustio): broj uzoraka, velicina cachea, slobodan disk.
Prikaz, ne pitanje.

**BUG USPUT: DVA `DEV_DATA_SUBSET`.** Ekstrakcija pluga (6.4) je u `plugins/detection/dsconfig.py`
kopirala VLASTITU vrijednost 200. Gasenje `settings.DEV_DATA_SUBSET` je tiho ostavljalo detekcijski GT
loader kapiran na 200 — mAP gate je davao ISTI rezultat (0.4217) za n_gate 200/400/837.
FIX: `dsconfig` uvozi konstantu iz `settings` — jedan prekidac.
POTVRDA NAKON FIXA: gate skalira (200 -> 0.4217 / 400 -> 0.4133 / 837 -> **0.4126**), i **0.4126 je
tocno yolo26n val mAP iz tablice u radu** — slinn reproducira poznati baseline u decimalu.
Cijena: 13.7s po koraku na punom valu.

**STANJE PARAMETARA (odgovor na pitanje korisnika):**
- `n_batches` — nema ga u `settings.py`, nema ga u `config.json`, nema ga u GUI-ju. U `engine.py` je
  parametar s `None` (= cijeli split) u tri potpisa. Jedini preostali broj je `FALLBACK_BATCHES = 8`,
  koji vrijedi SAMO kad nema citljivih podataka (token/parquet bez readera) — ondje su uzorci nasumicni
  pa "sve" nema znacenje. `plan["n_batches"]` i meta cachea su ispis/identitet, ne postavka.
- `n_gate` — NIJE broj batcheva nego broj UZORAKA na kojima se mjeri metrika nakon svakog morph koraka
  (quality-gate). Koristi se u `backend.build_metric_fn`: detekcija `eval_map(max_images=)`,
  segmentacija `pairs_segmentation(n=)`, klasifikacija `pairs_classification(n=)`. Default je sada
  `None` = cijeli val split; worker to prosljeduje eksplicitno; GUI ne pita. Ostao kao parametar samo
  za smoke-testove. To je JEDINA prava vremenska cijena po koraku.

**Korisniku u GUI-ju ostaju samo prave odluke:** kvantizacija (M), tolerancija, ciljano smanjenje,
FT koraci, max koraka, dead-rez.

### ✅ 6.10 REVIZIJA `settings.py` — TRI POLUSTRGANE KONSTANTE (2026-08-23)

Revizija svih 23 konstante (tko ih stvarno cita): **14 radi · 7 mrtvo · 3 POLUSTRGANE**.
Mrtve se na zahtjev korisnika NE MICU. Polustrgane su popravljene jer su aktivno lagale.

**1. `TMP_ROOT` — dvije definicije.** Izgledalo je aktivno (6 citanja u `engine.py`), ali engine je
imao VLASTITU definiciju `TMP_ROOT = _AA/tmp` i nikad nije gledao u settings. Promjena u settings nije
imala nikakav ucinak → cache se nije dao preseliti na drugi disk (a rijec je o 7.86 GB).
FIX: `engine.TMP_ROOT = CFG.TMP_ROOT`. Sto je TMP_ROOT: dom svega regenerabilnog — teacher cache
(`<model>/train/sig_*.pt`) i GUI job (`gui_job/`: config, status, trajektorija, compressed.pt, logovi).

**2. `DEV_DATA_SUBSET` — vrijedio je SAMO za detekciju.** U jezgri se koristio iskljucivo za ISPIS
upozorenja; stvarno rezanje radio je jedino detekcijski GT loader u plugu. Za regresiju, segmentaciju
i klasifikaciju prekidac nije radio NISTA.
FIX: cap u `engine._candidate_files` (jedino usko grlo kroz koje jezgra cita medijske ulaze) i u
`count_train_samples` za tabularni put.
PROVJERA (KD ulazi): yolo 5860→200 · voc 17125→200 · m5 105835→200 (prije je rezala samo detekcija).

  **DOPUNA (6.10b): cap vrijedi i za MJERENJE metrike.** Prvi popravak je hvatao KD ulaze i detekcijski
  GT loader, ali `metric.pairs_*` citaci su si SAMI listali datoteke i zaobilazili to usko grlo —
  pa je kod segmentacije i klasifikacije dev-podskup rezao TRENING, a metrika se mjerila na
  CIJELOM val skupu. Nekonzistentno i sporo.
  FIX: `metric._cap(n)` na jednom mjestu, primijenjen u sva tri citaca (`pairs_segmentation`,
  `pairs_regression`, `pairs_classification`).
  PROVJERA: segmentacija (voc) 1449→200 · regresija (housing) 3096→200.
  ZNACENJE (kao morphology): najvise N uzoraka **PO SPLITU** — dakle 200 za train I 200 za val, zasebno.

**3. `FT_RECOVERY_FRAC` — nije bila tolerancija nego samo default klizaca.** U kodu su postojale TRI
vrijednosti: settings 0.75, hardkodirano `metric_tol=0.90` u `morph_loop`, i `0.97` u `full_cycle`.
Pozovi engine izravno bez `metric_tol` i dobio bi 0.90, ne 0.75.
FIX: `morph_loop(metric_tol=None)` -> uzima `CFG.FT_RECOVERY_FRAC` (task-metrika) ili novi
`CFG.AGREEMENT_TOL = 0.97` (kad je gate teacher-agreement — on se krece oko 1.0 pa bi 0.75 bilo
besmisleno labavo). Ponasanje GUI-ja NEPROMIJENJENO: klizac i dalje POCINJE od `FT_RECOVERY_FRAC` i
korisnik ga pomice; ta vrijednost ide u config i engine je koristi. Settings = default, klizac nadglasava.

**MRTVE (7, ostaju po odluci korisnika):** `FT_PATIENCE`, `FT_MAX_EPOCHS` (slinn nema epohe ni patience —
`ft_steps` je fiksan broj gradijentnih koraka) · `PHASE2_PRUNE_PATIENCE` (zamijenjeno pravilom "3
uzastopna koraka ispod praga", hardkodiranim u `engine.py`) · `TRAIN_BATCH`, `EVAL_BATCH` (autobatch) ·
`VAL_CAP` (metrika mjeri na cijelom valu) · `MODELS_DIR` (worker sprema u `gui_job/compressed.pt`).

### ✅ 6.11 PRECIZAN REZ — ZATVORENA PETLJA plan→rez→IZMJERI→doplaniraj (2026-08-23)

**SIMPTOM.** Korisnik: "cini mi se da ne radi rezove od 1.5% pocetnih parametara". Tocno. Novi
dijagnosticki redak (cilj -> procjena -> stvarno, dodan u `worker.py`) dao je nedvosmislen nalaz na
yolo26n / 4 koraka:

    kor  kanala  cilj      procjena          stvarno            banned
     1     83    0.0894    0.0899 (101%)     0.0875 ( 97%)        0
     2     68    0.0894    0.0907 (101%)     0.0668 ( 74%)        1
     3     57    0.0894    0.0898 (100%)     0.0564 ( 63%)        2
     4     60    0.0894    0.0903 (101%)     0.0489 ( 54%)        3

Planer UVIJEK nade dovoljno kandidata (procjena pogada cilj 100-101% svaki korak). Ne valja
IZVRSENJE: stvarni rez pada na 54% procjene.

**PRVO SAM PROVJERIO LEGACY** (korisnikova pretpostavka: morphology je to vec rijesio re-planiranjem).
NIJE. Usporedba funkcija `legacy/morphology/compress.py` vs `slinn/morph.py`:

    coupled_unit_cost   RAZLIKA — samo in_ch fix iz 6.7; za NCHW conv ponasanje IDENTICNO
    prune_costs         IDENTICAL
    _select_prune_plan  IDENTICAL
    _apply_prune_plan   IDENTICAL

`while True:` u morphology petlji NIJE korekcija preciznosti — vrti se samo dok je n_rem == 0
(`if n_rem > 0 or not bad: break`), tj. parcijalni uspjeh se prihvaca. Bug je star; NOVO je samo to
sto ga sada MJERIMO. Morphology ga nikad nije ispisao: njegov jedini redak je
"GFLOPs 5.874->5.807 (97.5% orig.)" — a to je koliko je OSTALO, ne koliko je promasen cilj.

**TRI CURENJA** (sva u istom smjeru — procjena veca od stvarnosti):

  a) **Banani listovi ulaze u procjenu, a ne u rez.** `est_freed` je procjena CIJELOG plana;
     `_apply_prune_plan` list koji padne na forward-checku preskoci. Kandidati koji padaju su C2f/concat
     cvorista — a bas njih planer bira PRVE jer imaju najvecu spregnutu cijenu po kanalu. Zato procjena
     po kanalu raste 1.08e-3 -> 1.33e-3 -> 1.58e-3 dok stvarna stoji na ~1.0e-3.
  b) **Drugi red spregnutosti.** Kad su producent A i potrosac B OBA u planu, usteda a*b se broji
     dvaput: procjena je linearna (in*out - a*out - b*in), stvarnost je (in-a)*(out-b). To objasnjava
     manjak od 3% vec u koraku 1, gdje nema nijednog bana.
  c) **Tihi preskoci** u `_apply_prune_plan` (floor/cap, len(idx2) >= C).

**FIX** (`engine.morph_loop`, ~40 redaka): jedan prolaz zamijenjen zatvorenom petljom unutar KORAKA —

    remaining = step_target
    dok remaining > SLACK*step_target  i  krugova < PHASE2_PRUNE_ROUNDS:
        (od 2. kruga) prekalkuliraj prune_costs na VEC SMANJENOM modelu
        plan  = _select_prune_plan(target=remaining, exclude=banned|touched|cooldown)
        model = _apply_prune_plan(plan)                 # banovi se skupljaju kao i prije
        remaining -= (gflops_prije - gflops_poslije)    # MJERENO, ne procijenjeno

Sva tri curenja nestaju odjednom: banani list u iducem krugu ispada iz elig, drugi red spregnutosti
nestaje jer se cijena racuna na vec rezanom modelu, preskoceni kanali se nadoknade drugdje.

**Kljucni detalj — `touched`.** Sloj dirnut u ranijem krugu se IZUZIMA do kraja koraka. Dva razloga:
(1) imp/order su za njega zastarjeli — tp prepakira indekse kanala pa bi se u 2. krugu rezali KRIVI
kanali; (2) usput to cuva `PHASE2_PRUNE_LAYER_CAP` — bez izuzeca bi se isti sloj mogao rezati 15% PO
KRUGU, tj. do 50% u jednom koraku, sto je upravo kolaps od kojeg cap stiti. `touched` se racuna
MJERENJEM sirina (`_widths`), pa hvata i spregnute siblinge koje tp orezi usput, ne samo kljuceve plana.

**Sto se NIJE mijenjalo:** imp (`kd_importance`) se racuna JEDNOM po koraku — najskuplji dio, a za
nedirnute slojeve i dalje vrijedi. GROW i dalje dobiva flops_per0/units0 s POCETKA koraka (kao prije
6.11), da mu se ponasanje ne pomakne.

**Novo u `settings.py`:** PHASE2_PRUNE_ROUNDS = 4 (1 = staro ponasanje) i PHASE2_PRUNE_SLACK = 0.02.
**Novo u trajektoriji:** prune_rounds. **Novo u `worker.py`:** ispis "cilj -> stvarno (% cilja) u N
krug(a)" + koliko je zatvorena petlja nadoknadila nad 1. krugom.


**REZULTAT** (isti job: yolo26n, 4 koraka, DEV_DATA_SUBSET=400 radi brzine):

    kor   PRIJE 6.11              POSLIJE 6.11                       krugova   1. krug
     1    0.0875 ( 97% cilja)     0.0880 ( 98% cilja)   rez 1.5%        3        97% procjene
     2    0.0668 ( 74% cilja)     0.0879 ( 98% cilja)   rez 2.9%        3        77% procjene
     3    0.0564 ( 63% cilja)     0.0888 ( 99% cilja)   rez 4.4%        1        99% procjene
     4    0.0489 ( 54% cilja)     0.0884 ( 99% cilja)   rez 5.8%        1        98% procjene

Kumulativni rez 4.3% -> **5.8%**, a neto po koraku 1.46/1.43/1.48/1.47% naspram cilja 1.50%.

**POTVRDA UZROKA (a).** Koraci 1-2 trebali su 3 kruga jer je 1. krug isporucio 97%/77% procjene;
koraci 3-4 pogode iz PRVOG kruga (99%/98%). Razlika je `banned`: do kraja 2. koraka skupi se 6 trajno
banananih slojeva i vise se ne biraju. Dakle glavno curenje JEST bilo (a) — precijenjeni C2f/concat
listovi koji padnu na forward-checku — a zatvorena petlja ih otkriva brze (6 bana do 2. koraka umjesto
3 bana do 4. koraka) jer u istom koraku odmah proba sljedece kandidate.

**Nuspojava koju treba znati:** buduci da se sada REALNO reze 1.5% po koraku, kvaliteta pada brze po
koraku nego prije (mAP 0.4133 -> 0.3213 u 4 koraka umjesto -> 0.4020). To NIJE regresija — to je isti
model na 5.8% reza umjesto 4.3%. Quality-gate ce zato pragom zagristi u ranijem KORAKU, ali na istoj
razini GFLOPs-a; trajektorija je samo gusca.

### ⏱ GDJE PRIPREMA TROSI VRIJEME (izmjereno, yolo26n)

Korisnik je posumnjao na plan-korak. Nije on:

    load_any / probe_adapter            1.5s
    prepare (classify+position+task)    5.0s
    plan_teacher_cache (cijeli korak)   3.1s
    autobatch                          60.1s   <- glavni trosak
    baseline metrika (pun val)         35.8s   <- drugi trosak

Autobatch traje jer proba PUNE FT korake pri sve vecim batchevima dok ne nadje granicu VRAM-a.
PRIJEDLOG (nije implementirano): rezultat ovisi samo o (model, GPU) i ne mijenja se izmedu runova —
da se kesirati kao teacher cache, pa bi drugi put bio trenutacan. Baseline se ne moze izbjeci i tocan je.

### STANJE NA DAN ZAUSTAVLJANJA (2026-08-23)

**KAKO POKRENUTI GUI** (iz WSL-a):

    cd /home/tomi/code/dipl/slinn/gui
    /home/tomi/miniconda3/envs/dipl/bin/python -m streamlit run gui.py

Otvori http://localhost:8501. U lijevoj traci upisi putanju modela (`.pt`, pun eager modul) i putanju
dataseta; ostalo je AUTO. Za modele cija klasa nije uvoziva (npr. housing) dodaj kod-putanju.
Primjer koji radi: model `baseline_models/yolo26n/yolo26n.pt` + dataset
`datasets/mini_set/sub10k_open_images_v7`.
NAPOMENA: workeri se pokrecu kao subprocess po putanji, pa GUI mora ostati pokrenut iz `slinn/gui/`.

**GOTOVO U FAZI 6:**

| korak | stanje |
|---|---|
| 6.0 paritet-harness | ✅ |
| 6.0.1 skele `slinn/` | ✅ |
| 6.1 backend swap | ✅ |
| 6.4 ciscenje (7 koraka) | ✅ jezgra preseljena · `weighted_leaves` unificiran 2/3/4 · detekcijski plug · hardkodi nikad preseljeni · outfmt prepoznavanje · decode za 9 formata · align preseljen |
| 6.2 GUI | ✅ tri stranice, provjereno u pregledniku |
| 6.3 flip | ✅ gate prosao (pun run kroz GUI, v. gore) |
| 6.5 selidba u `legacy/` | ✅ gotovo |
| 6.6 odvez od `legacy/` | ✅ arhiva se smije obrisati — FAZA 6 ZAVRSENA |

**SVE JE GOTOVO.** Nista iz Faze 6 ne ceka.

**PRIJE PRAVIH RUNOVA (jedina obavezna radnja):** `slinn/settings.py` → `DEV_DATA_SUBSET = 200`
mora na `None`. GUI to prikazuje kao upozorenje na About stranici.

**ZAOSTALO, NE BLOKIRA NISTA:**
- 16 dev-harnessa preusmjereno na `morph`/`introspect`, ali ponovno su pokrenuti samo `_prune1d55.py`
  i `_run61.py` — ostalima status nije provjeren.
- `multilevel` format izlaza (ista sirina kanala po razini) je PREPOZNAT ali bez decode-a: nema
  stvarnog modela tog oblika za provjeru, a pogadjanje podjele kanala bez validacije se ne radi.
  Obrazac za dodavanje kad se pojavi: skini najmanji takav model → izmjeri → implementiraj → obrisi
  model (tako je dodan `set_pred` preko yolos-tiny).

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

### ZAVRSNO STANJE FAZE 6 (2026-08-23)

Stablo:

    slinn/                      jezgra — agnosticna, ne zna nista o tasku ni obitelji modela
      settings.py               kompresijski hiperparametri (nula dataset/detekcijskih konstanti)
      introspect.py             layer-tablica, census, GFLOPs, eager load
      morph.py                  prune/grow/dead mehanike, coupled cost, GradMax, align
      kdterms.py                genericki KD po tipu tapa
      classify/position/task/dataset/loss/metric/pipeline/engine/enhancers/overview
      plugins/detection/        JEDINI per-obitelj dio: decode + NMS + mAP + outfmt
      gui/                      gui.py (3 stranice) + backend/prep_worker/worker
      helper/ · REPORTS/        dev-alat i izvjestaji
    legacy/                     arhiva, IZVAN puta izvodjenja, smije se obrisati

Brojke koje stoje iza toga:
- **paritet 4x zaredom znamenku po znamenku** (B novi 5.721 GFLOPs / 4.0% / mAP 0.3839 / 92.0%),
  kroz preseljenje jezgre, unifikaciju `weighted_leaves` I izdvajanje pluga
- **pun run kroz GUI** na yolo26n: best_step 6, 6.8% reza uz mAP iznad praga, `compressed.pt`
- **9 formata izlaza** prepoznato mjerenjem, 5 potvrdjeno na stvarnim tezinama
- **1D lanac** (Conv1d): prune 61 kanal + grow function-preserving |Δ|=0
- **ne-detekcija** (housing/regresija) kroz isti kod-put, bez ijednog grananja

## 7. Bit

Agnosticnost je postignuta na sve tri ravnine: strukturno-pozicijskoj (classify/position mjerenjem),
gubitka (hook-KD + KD-grad), i evaluacije (per-task metrike + teacher-agreement fallback). Dokazane
prune/grow mehanike su PRESELJENE, ne prepisane. Detekcijski decode je prezivio kao izoliran,
auto-biran plug — i sam se bira mjerenjem OBLIKA IZLAZA kad model nije poznat.

Rezultat: pipeline koji na novom modelu radi *odmah*. Ako nesto ne prepozna, kaze to naglas i
degradira na KD-only umjesto da pogadja — i nastavi komprimirati.
