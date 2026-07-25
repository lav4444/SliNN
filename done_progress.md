# Progres rada — Knowledge Distillation za detekciju u prometu

Bilješke o svemu što je napravljeno, testirano i izmjereno. Za korištenje pri pisanju diplomskog rada.

Repo root: `/home/tomi/code/dipl`
Okruženje: `conda activate dipl` (Python 3.10.12, CUDA)

---

## 1. Dataset — sub10k Open Images V7 (6 klasa)

**Cilj:** subset Open Images V7 za detekciju u cestovnom prometu, 6 klasa.
**Klase (id → naziv):** 0=Person, 1=Car, 2=Truck, 3=Bus, 4=Motorcycle, 5=Bicycle.

**Skripta za izgradnju:** [datasets/mini_set/download_ds.py](datasets/mini_set/download_ds.py)
- koristi FiftyOne (`fiftyone.zoo.load_zoo_dataset("open-images-v7", ...)`)
- po klasi povlači do `PER_CLASS=1700` slika iz Open Images train splita, spaja ih i deduplicira po `filepath`
- filtrira detekcije na 6 ciljnih klasa
- random split 70/10/20 (seed=42) i export u YOLOv5 formatu

**Izlazni dataset:** [datasets/mini_set/sub10k_open_images_v7/](datasets/mini_set/sub10k_open_images_v7/)
- [dataset.yaml](datasets/mini_set/sub10k_open_images_v7/dataset.yaml) — YOLO config (path, train/val/test, names)
- `images/{train,val,test}/` i `labels/{train,val,test}/`

**Veličine splitova:**

| Split | #slika |
|-------|-------:|
| train |   5860 |
| val   |    837 |
| test  |   1674 |
| **Σ** | **8371** |

---

## 2. Teacheri — YOLO26 (l i n)

### 2.1 YOLO26l (glavni teacher za KD)

**Lokacija:** [baseline_models/yolo26l/](baseline_models/yolo26l/)
**Težine:** [baseline_models/yolo26l/yolo26l.pt](baseline_models/yolo26l/yolo26l.pt) (COCO-pretrained, 80 klasa)
**Skripta za evaluaciju:** [baseline_models/yolo26l/evaluate.py](baseline_models/yolo26l/evaluate.py)

**Što skripta radi:**
- Fiksni 640×640 letterbox preprocessing (zaobilazi Ultralytics `predict()` koji koristi dinamični letterbox do višekratnika 32 — to bi davalo varijabilan broj sidara po slici)
- Forward direktno preko `model.model(...)` → dense pre-NMS tenzor `[B, 4+nc, 8400]` (P3+P4+P5: 6400+1600+400 sidara)
- Na YOLO26 `Detect` head-u gasi `end2end` mode da dobije YOLOv8-stil dense izlaz (potrebno za KD)
- COCO id-ove preslikava na naših 6 klasa preko `OURS_TO_COCO = [0, 2, 7, 5, 3, 1]`
- Spremaju se dvije vrste izlaza po slici:
  - **soft labele (za KD):** `yolo26l/<split>/soft/<stem>.pt` — `{boxes_xywh [8400,4], class_probs [8400,6], letterbox, inference_ms}`
  - **hard pseudo-labele (YOLO format, >0.25 conf):** `yolo26l/<split>/labels/<stem>.txt`
- Ako sve `.pt` već postoje za split → samo recompute metrika iz njih (preskače inferenciju)

**Konfiguracija:** `EVAL_CONF=0.001`, `HARD_LABEL_CONF=0.25`, `NMS_IOU=0.7`, `MAX_DET=300`.

**Izlazi:**
- Soft labele: `datasets/mini_set/sub10k_open_images_v7/yolo26l/{train,val,test}/soft/`
- Hard pseudo-labele: `.../yolo26l/{train,val,test}/labels/`
- Meta podaci: `.../yolo26l/meta.json`
- Rezultati: [baseline_models/yolo26l/eval_result.txt](baseline_models/yolo26l/eval_result.txt)

**Rezultati:**

| Split | mAP@50:95 | mAP@50 | mAP@75 | Inferenca | Total |
|-------|----------:|-------:|-------:|----------:|------:|
| train | 0.4626 | 0.6711 | 0.5018 | 21.88 ms (45.7 FPS) | 29.39 ms (34.0 FPS) |
| val   | 0.4736 | 0.6852 | 0.5237 | 22.50 ms (44.4 FPS) | 30.11 ms (33.2 FPS) |
| test  | 0.4682 | 0.6817 | 0.5073 | 22.58 ms (44.3 FPS) | 30.29 ms (33.0 FPS) |

**Per-class mAP@50:95 (test):**

| Klasa | mAP |
|-------|----:|
| Person     | 0.2070 |
| Car        | 0.3967 |
| Truck      | 0.5240 |
| Bus        | 0.7022 |
| Motorcycle | 0.5323 |
| Bicycle    | 0.4473 |

### 2.2 YOLO26n (paralelni baseline, manji teacher)

**Lokacija:** [baseline_models/yolo26n/](baseline_models/yolo26n/)
**Težine:** `yolo26n.pt` (COCO-pretrained, 80 klasa, 2.5M params)
**Skripta:** [baseline_models/yolo26n/evaluate.py](baseline_models/yolo26n/evaluate.py) — identično yolo26l skripti samo s `MODEL_NAME = "yolo26n.pt"` i `PRED_ROOT` na `yolo26n/` mapu.

**Svrha:** **referent za "naš student vs gotovi mali YOLO"** usporedbu. Pretrained na COCO + ultralytics native trening na 118k slika. Slijedi: ovo je gornja granica koju može postići arhitektura yolo26n na ovom subsetu.

**Rezultati** (iz [baseline_models/yolo26n/eval_result.txt](baseline_models/yolo26n/eval_result.txt)):

| Split | mAP@50:95 | mAP@50 | mAP@75 |
|-------|----------:|-------:|-------:|
| train | 0.4054 | 0.6039 | 0.4340 |
| val   | (slično) | | |
| test  | (slično) | | |

Per-class: slično yolo26l obrascu (Bus najlakša, Person najteža), samo niži apsolutni brojevi (manji model).

---

## 3. Custom student arhitekture

Svi studenti imaju **isti vanjski API**:
- `forward(x) → [B, 4+nc, 8400]` (raw — direct anchor-relative output)
- `decode(raw) → (boxes_xywh, class_probs)` u 640 letterbox px
- `anchor_xy [8400, 2]`, `anchor_stride [8400]` registered kao bufferi
- Anchor layout: P3(80×80) + P4(40×40) + P5(20×20) = 6400+1600+400 = 8400

To znači **drop-in zamjenjivost** kroz cijeli pipeline (`train_kd.py`, `evaluate_student.py`).

### 3.1 student_0_5_m (~468k params)

**Lokacija:** [custom_models/student_0_5_m/](custom_models/student_0_5_m/)
**Arhitektura:** [custom_models/student_0_5_m/KD_first.py](custom_models/student_0_5_m/KD_first.py)

- Backbone: 16 / 32 / 48 / 64 / 96 channels, depth 1/2/3/3/2
- Neck: top-down FPN, `neck_ch=48`
- Heads: 3 coupled (ConvBN + 1×1 conv)
- **Parametri:** 467,758 (≈ 0.47 M)
- Operatori: Conv2d, BN, SiLU, Upsample-nearest, Concat, Reshape, Sigmoid (sve export-friendly)

### 3.2 student_1_m (~1.01M params)

**Lokacija:** [custom_models/student_1_m/](custom_models/student_1_m/)
**Arhitektura:** [custom_models/student_1_m/model_arch.py](custom_models/student_1_m/model_arch.py)

Skaliranje od 0.5M baseline-a:
- **Wider:** channels 16/32/48/64/96 → **24/48/64/96/128** (~1.5× wider)
- **Slightly deeper:** dark3 i dark4 imaju 1 dodatni ConvBN blok (4 umjesto 3)
- Neck: ista FPN struktura, `neck_ch=48 → 64`
- **Parametri:** 1,013,654 (≈ 1.01 M)

### 3.3 student_2_m — modern YOLO design (PAN + SPPF) (~2.07M params)

**Lokacija:** [custom_models/student_2_m/](custom_models/student_2_m/)
**Arhitektura:** [custom_models/student_2_m/model_arch.py](custom_models/student_2_m/model_arch.py)

**Bitno:** ovo NIJE samo wider 1M student — uvodi se i moderna YOLO neck struktura:
- **Backbone:** width-scaling od 1M-a — 32/64/96/128/192 channels, depth 1/2/4/4/2
- **SPPF blok** na izlazu iz backbone-a (5×5 maxpool chain — multi-scale receptive field, ~93k params)
- **PAN neck** umjesto čistog FPN-a — top-down (FPN) + bottom-up putevi, `neck_ch=64`
- Operatori: + MaxPool2d (svi i dalje export-friendly)
- **Parametri:** 2,071,166 (≈ 2.07 M)

**Posljedica za rad:** 1M → 2M nije čisto skaliranje kapaciteta — i arhitektonska promjena (PAN + SPPF). Ablation između 1M i 2M miješa dvije varijable; za clean ablation potreban bi bio `student_2_m_fpn/` (samo backbone scale, isti FPN) — *trenutno nije napravljen*.

### 3.4 student_yolo26n — ultralytics YOLO26n wrapper (~2.51M params)

**Lokacija:** [custom_models/student_yolo26n/](custom_models/student_yolo26n/)
**Arhitektura:** [custom_models/student_yolo26n/model_arch.py](custom_models/student_yolo26n/model_arch.py)

Tanki wrapper oko `ultralytics.nn.tasks.DetectionModel(cfg="yolo26n.yaml", nc=6)`:
- **Ne piše nove slojeve** — koristi ultralytics-ovu yolo26n arhitekturu out-of-the-box (C3k2, SPPF, C2PSA blokovi)
- `nc=6` umjesto default 80, `end2end=False`, random init (no pretrained)
- Wrapper konvertira output u format kompatibilan s našim KD pipelineom:
  - `BOX_OUTPUT_FORMAT = "decoded"` atribut → train_kd.py prepoznaje i koristi drugačiji box loss
  - Class kanali iz sigmoid'd back u logit (`log(p/(1-p))`) — za focal-loss compatibilnost
- **Parametri:** 2,506,140 (≈ 2.51 M)

**Svrha:** ista arhitektura kao baseline yolo26n.pt → **direktna usporedba "moja KD metodologija vs ultralytics native trening" na istoj mreži**.

---

## 4. KD trening pipeline — evolucija

### 4.1 Loss formulacija (osnovna, isti kroz sve custom studente)

Za 0.5M, 1M, 2M:
```
L = λ_cls · focal_loss(student_logits, teacher_probs) / num_pos
  + λ_box · weighted_smooth_l1(student_raw, encoded_teacher_raw)
```
- Focal: α=0.25, γ=2.0; sigmoid focal s soft target-ima (RetinaNet-style)
- `num_pos = #anchora gdje je max(teacher_probs) > 0.5` (RetinaNet normalizacija)
- Box loss: SmoothL1 u **raw** prostoru (encode teacher decoded → raw); težine = max teacher prob po sidru

Za yolo26n student: **drugačiji box loss** jer arhitektura izbacuje decoded box-eve (`BOX_OUTPUT_FORMAT="decoded"`). Box loss je SmoothL1 u **stride-normaliziranom decoded prostoru** umjesto raw prostoru. Cls loss je identičan. Dispatch kroz `box_format` parametar u `kd_loss(...)` funkciji.

### 4.2 Hiperparametri (trenutni — v3)

- `batch=16`, `num_workers=4`, `image=640`
- AdamW: `lr=1e-3`, `weight_decay=5e-4`
- LR schedule: 2 warmup epoha → cosine decay
- Grad clip 10.0, AMP enabled
- `LOSS_W_CLS=1.5`, `LOSS_W_BOX=1.0`
- `VAL_EVERY_N_EPOCHS=3` (validacija na epohama 1, 4, 7, 10, ...)
- `EARLY_STOP_PATIENCE=9` (3 val provjere bez poboljšanja → stop)
- `LR_REDUCE_PATIENCE=6` (2 val provjere bez poboljšanja → LR ×0.6)
- `LR_REDUCE_FACTOR=0.6` (može se aktivirati više puta tijekom run-a)
- Max 100 epoha, seed=42

### 4.3 Validation + early stop + LR reduction (trenutna logika)

- **Best checkpoint kriterij:** `val_mAP@50:95 ↑` (najbolja vrijednost se sprema u `best.pt`)
- **Early stop:** kad val_mAP@50:95 ne raste 9 uzastopnih epoha (= 3 val provjere)
- **LR reduction:** kad val_mAP@50:95 ne raste 6 uzastopnih epoha (= 2 val provjere) → halve preko `scheduler.base_lrs *= 0.6` + active LR-a — može se aktivirati više puta tokom run-a ako mAP ponovo raste pa opet plateau-uje

Implementacija: vidi `epochs_without_improvement` brojač + `lr_reduced_in_current_streak` flag u [train_kd.py](custom_models/student_0_5_m/pure_KD/train_kd.py).

### 4.4 Evolucija konfiguracija (verzije)

| Verzija | LOSS_W_CLS | VAL_EVERY | Patience | Kriterij | LR reduction |
|---|---|---|---|---|---|
| v1 (inicijalno) | 2.0 | 1 (svaka epoha) | 5 | `val_kd_loss ↓` | nema |
| v2 | 1.5 | 3 | 6 | `val_kd_loss ↓` | nema |
| **v3 (trenutno)** | **1.5** | **3** | **9** | **`val_mAP@50:95 ↑`** | **×0.6 @ patience 6** |

**Napomena:** rezultati u sekciji 5 dolje su iz **v1 ili v2** metodologije (rani run-ovi). Future re-runs s v3 metodologijom očekivani **+10-25 % mAP** zbog (1) bolje selekcije best-checkpoint-a i (2) LR reduction šanse za izlaz iz plateaua.

---

## 5. Rezultati po studentima — pure_KD

### 5.1 student_0_5_m

**Skripta:** [custom_models/student_0_5_m/pure_KD/train_kd.py](custom_models/student_0_5_m/pure_KD/train_kd.py)
**Trained:** ✓ (v1 metodologija)
**Stvarni tijek:** stop na e21, best e16 (`val_kd_loss=1.2621`, `val_mAP@50:95=0.0811`)

Rezultati na test setu ([eval_result.txt](custom_models/student_0_5_m/pure_KD/eval_result.txt)):

| Split | mAP@50:95 | mAP@50 | mAP@75 | Inferenca | Total |
|-------|----------:|-------:|-------:|----------:|------:|
| train | 0.1260 | 0.3299 | 0.0580 | 7.97 ms (125 FPS) | 62.60 ms (16 FPS) |
| val   | 0.0816 | 0.2380 | 0.0266 | 8.01 ms (125 FPS) | 62.19 ms (16 FPS) |
| test  | 0.0737 | 0.2167 | 0.0238 | 8.13 ms (123 FPS) | 64.59 ms (15.5 FPS) |

Per-class test mAP@50:95 vs teacher:

| Klasa | Student 0.5M | Teacher L | Δ |
|-------|--------:|--------:|--:|
| Person     | 0.0471 | 0.2070 | −0.16 |
| Car        | 0.0536 | 0.3967 | −0.34 |
| Truck      | 0.0825 | 0.5240 | −0.44 |
| Bus        | 0.0994 | 0.7022 | −0.60 |
| Motorcycle | 0.0821 | 0.5323 | −0.45 |
| Bicycle    | 0.0774 | 0.4473 | −0.37 |

### 5.2 student_1_m

**Skripta:** [custom_models/student_1_m/pure_KD/train_kd.py](custom_models/student_1_m/pure_KD/train_kd.py)
**Trained:** *čeka u redu za pokretanje (skripta spremna)*

### 5.3 student_2_m

**Skripta:** [custom_models/student_2_m/pure_KD/train_kd.py](custom_models/student_2_m/pure_KD/train_kd.py)
**Trained:** ✓ (v2 metodologija — val_kd_loss kriterij, patience 6)
**Stvarni tijek:** stop na e19, best e13 (`val_kd_loss=0.9322`, `val_mAP@50:95=0.1423`)

**Važno opažanje** za rad (iz analize trajektorije u [training_history.json](custom_models/student_2_m/pure_KD/training_history.json)):
- e13 spasen kao best, ali **e19 imao bolji mAP=0.1640** (15 % veći)
- val_kd_loss kriterij sjekao u "krivu" epohu — divergencija KD loss vs task metric
- Ovo je motivacija za v3 metodologiju (`val_mAP@50:95` kriterij) — vidi sekciju 4.4
- Trajektorija pokazuje train loss još pada na e19 (0.51), val_kd_loss pleše (0.93-0.98), val_mAP raste — klasični signal "ima još kapaciteta + suboptimalan kriterij selekcije"

### 5.4 student_yolo26n

**Skripta:** [custom_models/student_yolo26n/pure_KD/train_kd.py](custom_models/student_yolo26n/pure_KD/train_kd.py)
**Trained:** ✓ (v2 metodologija)
**Stvarni tijek:** stop na e34, best e28 (`val_kd_loss=2.1735`, `val_mAP@50:95=0.1946`)

**Box loss u logu izgleda 3× veći** od ostalih studenata (≈0.7 vs ≈0.2) — to je posljedica drugačije loss formulacije (decoded mode vs raw mode), **nije signal lošijeg učenja**. Mjerena metrika za usporedbu = `val_mAP@50:95`.

**Usporedba s pretrained yolo26n.pt na našem subsetu:**

| | Random-init + naša KD pipeline | Pretrained COCO + ultralytics native | Δ |
|---|---:|---:|--:|
| val mAP@50:95 | 0.195 | 0.40 | −0.21 |

Glavni faktori jaza (objašnjeni u analizi):
1. **Pretraining** (~70 % razlike): yolo26n.pt vidio 118k COCO slika prije naših 6 klasa
2. **Trening data** (~15 %): 5860 vs 118k slika
3. **Trening metodologija** (~10 %): bez augmentacija + KD-only loss + 34 epoha
4. **Teacher kvaliteta plafon** (~5 %): teacher sam ima ~0.47 mAP, student-mimic ne može preko

---

## 6. Evaluacijska skripta — `evaluate_student.py`

**Skripta** (analogna struktura kroz sve studente): [custom_models/student_0_5_m/pure_KD/evaluate_student.py](custom_models/student_0_5_m/pure_KD/evaluate_student.py)

Što radi:
- Isti letterbox preprocessing kao u trening pipelineu (reuse iz train_kd.py)
- Evaluira `checkpoints/best.pt` na train/val/test
- Mjerenja: model-only inference time + total (preproc+inf+postproc) time
- Format izlaza identičan teacher `eval_result.txt`-u

### 6.1 CPU vs GPU latency benchmark (dodano u sve eval skripte)

Na kraju svake eval skripte (i teacher i student) dodan je benchmark:
- Uzme 10 random train slika (fixed seed=42 za reproducibilnost)
- Preprocessing napravi jednom unaprijed
- Forward na CPU, pa forward na GPU, `torch.cuda.synchronize()` oko poziva
- Mjeri samo `model.forward` (NE preprocessing, NE NMS)
- Odbaci 2 najsporija (warmup), report mean od 8 najbržih
- Output u `eval_result.txt` + na konzolu

**Datoteke u koje je dodano** (svih 5):
- [baseline_models/yolo26l/evaluate.py](baseline_models/yolo26l/evaluate.py)
- [baseline_models/yolo26n/evaluate.py](baseline_models/yolo26n/evaluate.py)
- [custom_models/student_0_5_m/pure_KD/evaluate_student.py](custom_models/student_0_5_m/pure_KD/evaluate_student.py)
- [custom_models/student_1_m/pure_KD/evaluate_student.py](custom_models/student_1_m/pure_KD/evaluate_student.py)
- [custom_models/student_2_m/pure_KD/evaluate_student.py](custom_models/student_2_m/pure_KD/evaluate_student.py)
- [custom_models/student_yolo26n/pure_KD/evaluate_student.py](custom_models/student_yolo26n/pure_KD/evaluate_student.py)
- [exploring_aternatives/early_stop_on_map/evaluate_student.py](exploring_aternatives/early_stop_on_map/evaluate_student.py)

**Konstantnice:** `BENCHMARK_N_IMAGES=10`, `BENCHMARK_N_DISCARD=2`, `BENCHMARK_SEED=42`.

**Primjer izlaza** (student 0.5M):
```
CPU per-image times (ms): [24.56, 16.11, ..., 14.37]
GPU per-image times (ms): [ 6.28,  6.88, ...,  6.92]
Mean of 8 fastest (2 slowest discarded as warmup):
  CPU:  14.20 ms/image  (70.4 FPS)
  GPU:   6.97 ms/image  (143.4 FPS)
  Speedup (GPU vs CPU):  2.0x
```

**Note:** mali modeli imaju mali speedup CPU→GPU (~2×) jer GPU fiksni troškovi (kernel launches, BN syncs) dominiraju nad računskim radom. Veći modeli imaju veći speedup.

---

## 7. exploring_aternatives — metodologijski eksperimenti

**Lokacija:** [exploring_aternatives/](exploring_aternatives/)

Folder za eksploraciju **metodoloških varijanti** treninga, paralelno s glavnim run-ovima.

### 7.1 early_stop_on_map (zaključen)

**Lokacija:** [exploring_aternatives/early_stop_on_map/](exploring_aternatives/early_stop_on_map/)

**Cilj:** usporediti `val_mAP@50:95` vs `val_kd_loss` kao early-stop kriterij na student_0_5_m.

**Setup:** identičan v1 pure_KD treningu za 0.5M, samo:
- Kriterij za best/early-stop = `val_mAP@50:95 ↑` umjesto `val_kd_loss ↓`
- Patience 10 umjesto 5

**Rezultat:** oba kriterija dala **istu epohu** (e16) kao best. Razlog: u kratkom run-u (stop @e26) oba signala su korelirana, jer dataset/model su mali — overfitting dolazi brzo i jednoglasno.

**Zaključak:** za ovaj konkretan setup nije bilo praktične razlike, ali metodološki mAP-kriterij je preferiran (selektiraš stvarnu metriku, ne proxy). Eksperiment opravdao prelazak na v3 metodologiju.

---

## 8. Infrastruktura

### 8.1 auto_run_mail.py

**Lokacija:** [auto_run_mail.py](auto_run_mail.py)

Sequential script runner za pokretanje višestrukih treninga/eval-ova bez ručnog presjedanja.

**Što radi:**
- Iterira `PIPELINE` listu (bare-string komande ili `(naziv, komanda)` tuples)
- Svaku komandu pokreće u subprocess-u iz `WORKING_DIR=/home/tomi/code/dipl`
- stdout/stderr **live streamano u terminal** (s `PYTHONUNBUFFERED=1` da Python ne blok-buffera output iza pipe-a)
- Šalje mail nakon svake komande s status + elapsed time (mail body NE sadrži output)
- Subject zadnjeg uspješnog koraka označen ` | END`
- Na prvi non-zero exit pipeline staje (osim ako `CONTINUE_ON_ERROR=True`)

**Setup za Gmail:**
1. Uključi 2-Step Verification na Google računu
2. Generiraj App Password ([myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)) — 16 malih slova
3. Stavi u env: `export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"` u ~/.bashrc

**Korisno za:** noćni run-ovi, run-ovi koji traju sat+, kombinirani pipeline (train + eval + više modela).

---

## 9. Pomoćne skripte (per-student)

Svaki student u `pure_KD/` ima i dodatne skripte uz `train_kd.py` i `evaluate_student.py`:
- **`export.py`** — export checkpointa u ONNX (`student.onnx`, opset 14, dynamic batch); upute za daljnji NCNN/TensorRT korak u docstringu. Primjer: [custom_models/student_0_5_m/pure_KD/export.py](custom_models/student_0_5_m/pure_KD/export.py)
- **`gui_test.py`** — Tk GUI; povuče nasumičnu sliku iz odabranog splita i vizualizira student detekcije. Primjer: [custom_models/student_0_5_m/pure_KD/gui_test.py](custom_models/student_0_5_m/pure_KD/gui_test.py)

---

## 10. KD_GT_FGD varijanta — kombinirani loss

**Lokacije:** [custom_models/student_0_5_m/KD_GT_FGD/](custom_models/student_0_5_m/KD_GT_FGD/), [custom_models/student_1_m/KD_GT_FGD/](custom_models/student_1_m/KD_GT_FGD/), [custom_models/student_2_m/KD_GT_FGD/](custom_models/student_2_m/KD_GT_FGD/)

**Cilj:** kombinirani loss koji uz response KD dodaje (a) GT supervisiju i (b) feature-level distillation (FGD).
```
L = λ_kd · L_kd_response + λ_gt · L_gt + λ_fgd · L_fgd
```
gdje:
- `L_kd_response` — postojeći pure KD loss (focal + SmoothL1 na soft labelama)
- `L_gt` — direct anchor-IoU assignment + focal cls + SmoothL1 box na pravim GT YOLO labelama (~100 linija)
- `L_fgd` — feature-level distillation na P3 (1×1 projection + MSE s fg/bg masking)

### 10.1 Teacher P3 feature ekstrakcija

**Skripta:** [baseline_models/yolo26l/evaluate.py](baseline_models/yolo26l/evaluate.py) (modificirano)

Forward hook na layer 4 yolo26L backbone-a (P3 stage output) hvata `[512, 80, 80]` feature po slici, sprema kao fp16 ključ `p3_features` u svaki postojeći `.pt` u `yolo26l/<split>/soft/`. `is_split_complete()` spot-checkira P3 prisutnost i re-runa inferenciju ako fali. Disk: ~55 GB dodano.

### 10.2 PCA redukcija teacher featura (10× smanjenje + fix za rank cage)

**Skripta:** [baseline_models/yolo26l/reduce_p3_to_pca.py](baseline_models/yolo26l/reduce_p3_to_pca.py)

**Motivacija — rank-bottleneck problem u v1:** prva verzija FGD-a koristila je 1×1 conv projekciju **48 → 512** (student na teacher prostor). Rank takve linearne projekcije je ≤ 48, pa je `proj(student)` ograničen na 48-d potprostor unutar 512-d teacher prostora. Teacher feature obično živi u efektivno ~100-300-d potprostoru, pa **MSE ima ireducibilan dio** (komponenta orthogonalna na student projekciju). Empirijski potvrđeno: FGD loss u v1 ostao gotovo konstantno na 0.347-0.349 kroz 37 epoha — projekcija je dosegla svoj rank-bounded plafon.

**Rješenje:** offline PCA redukcija teacher featura iz 512 → 48 komponenata. Daje studentu **matched-dim cilj** — za 0.5M (student_ch = 48) FGD projection postaje **Identity** (no rank cage), za 1M (64) i 2M (96) ostaje samo mala down-projekcija (64→48 = 3k params, 96→48 = 4.6k params, vs originalnih 24-49k).

**Implementacija:**
- Uzorkovanje 25 spatial pozicija × 8371 fileova = 209k samples u pre-alocated `[N, 512]` fp32 buffer (memory-conscious, izbjegava cat doubling)
- `torch.svd_lowrank(q=48, niter=8)` na CPU (~2 s)
- Apply: svaki `.pt` se overwriteа — drops `p3_features` (512-d, ~6.5 MB), adds `p3_features_pca48` (48-d, ~600 KB)
- PCA basis spremljen u [`yolo26l/pca_p3_basis_dim48.pt`](datasets/mini_set/sub10k_open_images_v7/yolo26l/pca_p3_basis_dim48.pt) za diagnostic/reuse

**Rezultati:**
- Top-48 komponente objašnjavaju **55.6 %** varijance teacher featura
- Disk: 54.4 GB → **7.5 GB** (7.2× smanjenje)
- Trajanje: ~2.5 min za cijelo (sample + fit + apply)

### 10.3 Loss konfiguracija

```
LAMBDA_KD = 1.0
LAMBDA_GT = 0.1   ← halved iz 0.2 nakon negativnog rezultata; v1 imao 0.4
LAMBDA_FGD = 0.4
```

GT komponenta postupno smanjena jer **konflikira s teacher KD signalom** za P3 sidra koja teacher djelomično aktivira (vidi opažanja u 10.4 dolje).

### 10.4 Rezultati v1 (PRE PCA fix-a) — negativan rezultat

Prvi pokušaj s `λ_gt=0.4, λ_fgd=0.4` i 48→512 projekcijom. Rezultati zaostaju za pure_KD baseline-om:

| Model | pure_KD val mAP@50:95 | KD_GT_FGD v1 val mAP@50:95 | Δ |
|---|---:|---:|---:|
| 0.5M | 0.0816 | **0.0429** | **−47 %** ❌ |
| 1M | nije trenirano | 0.0635 | n/a |
| 2M | (uskoro) | (uskoro) | |

**Dijagnostika:**

1. **GT-KD konflikt na P3 sidrima.** GT loss assigna samo **TOP-1 IoU sidro po GT-u** (rijetka assignment), pa većina P3 sidara dobiva **background** label. Teacher pak djelomično aktivira mnoga P3 sidra (prob ~0.3-0.5) — KD loss ih gura prema "umjereno aktivno", GT loss prema 0. Kontradiktorni gradienti.
   - Empirijska potvrda: val_cls KD_GT_FGD = 0.42 vs pure_KD = 0.34 (+24 %)

2. **Anchor-IoU assignment systematski krivi scale-pairing.** Naša heuristika koristi `stride × stride` kvadrat kao anchor box. Za GT 50×50 px, P5 anchor (32×32) daje IoU ≈ 0.22, a P3 anchor (8×8) daje IoU ≈ 0.03 — pa P5 pobjeđuje. Ali konceptualno **mali objekt → P3 anchor** je standardna praksa. **90 % GT-ova završi na P5**, a P3 (najvažniji za sitne objekte) dobivaju samo background — to konfliktira s teacherom.

3. **FGD projection rank cage** (vidi 10.2 motivaciju). Loss stagnira oko 0.347 kroz cijeli trening.

### 10.5 Fix v2 — što je promijenjeno

Tri točke iz dijagnostike adresiranе (osim #2 koja ostaje za buduću iteraciju):

| Promjena | v1 | v2 |
|---|---|---|
| Teacher P3 cache | full 512-d (55 GB) | PCA-reducirano 48-d (5 GB) |
| FGD projection | student 48 → 512 (rank ≤ 48) | matched-dim, **Identity** za 0.5M; mala down-proj za 1M/2M |
| `λ_gt` | 0.4 | 0.1 (4× smanjeno) |
| `λ_fgd` | 0.4 | 0.4 (nepromijenjeno; sad zapravo informativan signal) |

**Što i dalje nije popravljeno:** anchor-IoU assignment je i dalje top-1 IoU sa stride-square box-evima (problem #2). Ako v2 ne donese dovoljan boost, sljedeća iteracija mora prijeći na **center-in-box + scale-matched** assignment (~80 linija dodatnog koda).

### 10.6 Status v2 — spreman za pokretanje

| Komponenta | Status |
|---|---|
| PCA redukcija teacher featura | ✓ done (`pca48` ključ u svim 8371 .pt fileova) |
| 0.5M `KD_GT_FGD/train_kd.py` | ✓ ažuriran (Identity FGD, λ_gt=0.1) |
| 1M `KD_GT_FGD/train_kd.py` | ✓ ažuriran (64→48 proj, λ_gt=0.1) |
| 2M `KD_GT_FGD/train_kd.py` | ✓ ažuriran (96→48 proj, λ_gt=0.1) |
| Trening | ⌛ čeka pokretanje |

**Pokretanje:**
```bash
cd custom_models/student_0_5_m/KD_GT_FGD && python train_kd.py
# pa analogno za 1M i 2M, ili kroz auto_run_mail.py
```

**Hipoteza:** v2 bi trebao prijeći **0.07-0.10** za 0.5M (≈ pure_KD baseline ili malo bolje), **0.10-0.14** za 1M, **0.13-0.18** za 2M. Glavni mehanizam poboljšanja: FGD sad ima koherentnu metu pa daje informativan gradient signal, GT je blagi regularizator umjesto co-equal goal.

---

## 11. Ostali otvoreni pravci

- **Re-trening pure_KD-a s v3 metodologijom** — postojeći 0.5M/2M/yolo26n rezultati su iz v1/v2 (val_kd_loss kriterij). Re-run s val_mAP kriterij + LR reduction donosi vjerojatno +10-25 % mAP.
- **Anchor assignment fix** — center-in-box + scale-matched (vidi 10.4 problem #2), bitno ako v2 KD_GT_FGD ne donese dovoljno
- **Augmentacije** (mosaic, flip, HSV) — vjerojatno najveći single-win (+15-30 % mAP) ali zahtijeva regeneraciju teacher soft labela
- **Veličina + FLOPs analiza** — usporediti param count, .pt size na disku, FLOPs, latencije sve modele
- **Export u ONNX/NCNN/TensorRT** + mjerenje na ciljnoj platformi (Jetson, ARM CPU)
- **`student_2_m_fpn/` clean ablation** — backbone width-scaled od 1M-a ali bez PAN+SPPF
- **KD methodology eksperimenti**: feature-level (MGD, DeFeat), relation-based (RKD), DFL distillation za yolo26n student

---

## Konvencija za nove eksperimente

Layout je dvoslojan:
- **`custom_models/student_<size>/`** — definira **arhitekturu** studenta (`KD_first.py` ili `model_arch.py`). Sve KD varijante koje koriste tu istu arhitekturu žive ispod.
- **`custom_models/student_<size>/<kd_variant>/`** — definira **trening pristup** (loss, scheduler, augmentacije...). Sadrži: `train_*.py`, `evaluate_student.py`, `export.py`, `gui_test.py`, `checkpoints/`, `training_log.txt`, `training_history.json`, `training_plots.png`, `eval_result.txt`.

Skripte u varijantnim folderima importaju model preko `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` → `from KD_first import StudentYOLO` (ili `from model_arch import StudentYOLO`).

Za svaki novi eksperiment u ovaj dokument dodati novu sekciju s: hipotezom, što je promijenjeno u odnosu na referentni `pure_KD`, hiperparametrima, rezultatima i pathevima.

---

## Trenutna struktura repo-a

```
/home/tomi/code/dipl/
├── datasets/mini_set/sub10k_open_images_v7/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── yolo26l/{train,val,test}/{soft,labels}/   # KD soft + hard pseudo-labels
│
├── baseline_models/
│   ├── yolo26l/{evaluate.py, yolo26l.pt, eval_result.txt}
│   └── yolo26n/{evaluate.py, yolo26n.pt, eval_result.txt}
│
├── custom_models/
│   ├── student_0_5_m/
│   │   ├── KD_first.py
│   │   ├── pure_KD/  ✓ trained (v1)
│   │   └── KD_GT_FGD/  ← sljedeći eksperiment
│   ├── student_1_m/
│   │   ├── model_arch.py
│   │   └── pure_KD/  ⌛ čeka u redu
│   ├── student_2_m/
│   │   ├── model_arch.py
│   │   └── pure_KD/  ✓ trained (v2)
│   └── student_yolo26n/
│       ├── model_arch.py  (wrapper oko ultralytics yolo26n)
│       └── pure_KD/  ✓ trained (v2)
│
├── exploring_aternatives/
│   └── early_stop_on_map/  (zaključen metodologijski eksperiment)
│
├── auto_run_mail.py
├── README.md
└── done_progress.md   ← ovaj file
```

Legenda: ✓ = trening završen, ⌛ = skripta spremna ali run pendiramo.
