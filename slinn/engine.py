"""
engine.py — FAZA 5 engine: split-materijalizacija + teacher-cache (5.1); prune/grow petlja (5.2+).

5.1 daje dvije gradivne cigle koje petlja treba:
  materialize_train_batches  — FIKSNI skup train-batcheva (isti kroz SVE epohe) -> teacher-cache alignment.
                               split_plan.train realno ime -> ti fajlovi; 'AUTO' -> pool (split=None) + seeded
                               subset. KD je bez oznaka pa TRAIN ulazi NE trebaju stratifikaciju; `stratified_split`
                               (dataset.py) ostaje za val/test METRIKU u 5.6. Vrati CPU tenzore (reuse-safe).
  TeacherSigCache/precompute — teacher tap+terminal signali (loss.teacher_signals) po batchu, PREDRACUNATI jednom
                               (disk fp16 + opc. RAM), reuse kroz epohe -> bez ponovnog vrtenja teachera svaki FT
                               korak. batch_size/n_batches/tapovi/fingerprint u META -> promjena INVALIDIRA cache
                               (kao morphology TeacherCache: batch-2 cache vs batch-8 student = tihi broadcast bug).

Izolacija: morphology se NE dira. Reuse je KONCEPTUALNI (obrazac disk+RAM+meta), signali su NASI (hook-KD).
"""
import collections.abc
import copy
import hashlib
import json
import math
import os
import sys

_AA = "/home/tomi/code/dipl/slinn"
if _AA not in sys.path:
    sys.path.insert(0, _AA)

import torch                                                 # noqa: E402
import torch.nn as nn                                        # noqa: E402
import introspect as A                                       # noqa: E402  (6.4: bivsi morphology.analysis, genericki dio)
import morph as C                                            # noqa: E402  (6.4: bivsi morphology.compress mehanike)
import settings as CFG                                       # noqa: E402
import dataset as DS                                         # noqa: E402
import loss as L                                             # noqa: E402
# (6.4) `_WL_AG` alias uklonjen: `weighted_leaves` ima JEDNU definiciju (classify.py, dim 2/3/4),
# a `introspect` je re-exporta -> `A.weighted_leaves` je vec ta verzija, bez monkeypatcha.

# Dom svega regenerabilnog: teacher cache (`<model>/train/sig_*.pt`) + GUI job (`gui_job/`).
# JEDAN izvor istine — `settings.TMP_ROOT`. (Do 6.9 je engine imao vlastitu definiciju, pa promjena
# u settings.py nije imala nikakav ucinak; cache se nije dao preseliti na drugi disk.)
TMP_ROOT = CFG.TMP_ROOT


# =========================== PREP: adapter-size shim (5.0 nalaz-a) =========================== #
# Morphology mjerne funkcije koje trebaju forward HARDKODIRAJU ulaz (layer_table: rand(3,640,640);
# _forward_ok: rand(3,320,320)) — pretpostavka "svaki ulaz je slika ~640". Pukne na ne-640 / ne-slikovnim
# modelima (schoolcnn 320, housing vektor, m5 1D, DistilBERT token). ISPRAVAK: jedini izvor ulaza je ADAPTER
# (probom izmjeren ugovor) preko `adapter.forward_example`. Dok izolacija vrijedi (Faza 5) ovo je runtime
# monkeypatch (morphology fajlovi NETAKNUTI); Faza 6 to pretvara u direktan edit morphology-a (isti ispravak).
def _ag_layer_table(model, adapter, device):
    if not hasattr(adapter, "forward_example"):              # ne-probe adapter (morphology ProfileAdapter) -> original
        return _ORIG_LT(model, adapter, device)
    leaves = A.weighted_leaves(model)
    by_id = {id(m): (name, tn, w) for name, m, tn, w in leaves}
    rec, handles = [], []

    def mk(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and o:
                o = o[0]
            name, tn, w = by_id[id(mod)]
            ishape = tuple(inp[0].shape) if inp and isinstance(inp[0], torch.Tensor) else None
            oshape = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() >= 3:
                    ksize = math.prod(w.shape[2:]); spatial = math.prod(o.shape[2:])
                    flops = 2 * w.shape[0] * w.shape[1] * ksize * spatial
                elif w.dim() == 2:
                    out_f, in_f = w.shape
                    flops = 2 * out_f * in_f * (o.numel() // out_f if out_f else 0)
            rec.append({"name": name, "type": tn, "role": "neuron" if w.dim() == 2 else "filter",
                        "units": int(w.shape[0]), "params": sum(p.numel() for p in mod.parameters(recurse=False)),
                        "gflops": flops / 1e9, "in": ishape, "out": oshape})
        return hook

    for name, m, tn, w in leaves:
        handles.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad():
        adapter.forward(model, adapter.forward_example(device))   # ADAPTEROV ulaz, ne hardkod 640
    for h in handles:
        h.remove()
    return rec


def _ag_forward_ok(model, adapter, device):
    if not hasattr(adapter, "forward_example"):              # ne-probe adapter -> original morphology _forward_ok
        return _ORIG_FO(model, adapter, device)
    was = model.training
    try:
        model.eval()
        with torch.no_grad():
            adapter.forward(model, adapter.forward_example(device))   # ADAPTEROV ulaz, ne hardkod 320
        return True
    except BaseException:
        return False
    finally:
        model.train(was)


def _ag_try_grow_layer(model, adapter, device, name, k, init_filters=None):
    """Adapter-input verzija morphology `_try_grow_layer` (5.5): JEDINA izmjena = referentni ulaz iz
    `adapter.forward_example` umjesto hardkodiranog `rand(3,sz,sz)` (koji radi samo za 3ch-2D). Ostatak
    (coupled tp-grupa widening + function-preserving commit diff<1e-3) NETAKNUT; widen-helperi su dim-generički
    pa Conv1d/1D rastu jednako kao 2D. Uz `A.weighted_leaves`=(2,3,4) patch → grow radi na 1D (M5). REUSE `C._*`."""
    import torch_pruning as tp
    if not hasattr(adapter, "forward_example"):              # ne-probe adapter -> original morphology _try_grow_layer
        return _ORIG_GROW(model, adapter, device, name, k, init_filters)
    ref_imgs = adapter.forward_example(device)                # ADAPTEROV ispravan ulaz (image/seq/vector/token)
    try:
        with torch.no_grad():
            ref_out = adapter.teacher_outputs(model, ref_imgs)
    except BaseException:
        return None
    trial = copy.deepcopy(model)
    leaves = {nm: (mm, w.dim()) for nm, mm, _, w in A.weighted_leaves(trial)}
    if name not in leaves:
        return None
    Lm, dim = leaves[name]
    if getattr(Lm, "groups", 1) > 1:
        return None
    old_L = Lm.weight.shape[0]
    try:
        for p in trial.parameters():
            p.requires_grad_(True)
        fn = (tp.function.prune_conv_out_channels if dim >= 3 else tp.function.prune_linear_out_channels)
        DG = tp.DependencyGraph().build_dependency(trial, example_inputs=adapter.tp_example(device))
        group = DG.get_pruning_group(Lm, fn, idxs=[0])
        out_mods, bns, fbns, dws, cons = [], [], [], [], {}
        for dep, idxs in group:
            tgt = getattr(getattr(dep, "target", None), "module", None)
            if tgt is None:
                continue
            hn = getattr(dep.handler, "__name__", type(dep.handler).__name__).lower()
            if isinstance(tgt, nn.modules.batchnorm._BatchNorm):
                bns.append(tgt)
            elif type(tgt).__name__ == "FrozenBatchNorm2d" or (hasattr(tgt, "running_mean") and hasattr(tgt, "weight")
                                                               and not isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear))):
                fbns.append(tgt)
            elif C._is_depthwise(tgt):
                dws.append(tgt)
            elif isinstance(tgt, nn.Conv2d) and tgt.groups > 1:
                return None
            elif "in_channel" in hn or "in_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    cons.setdefault(tgt, []).append(int(min(idxs)))
            elif "out_channel" in hn or "out_feature" in hn:
                if isinstance(tgt, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    out_mods.append(tgt)
        if Lm not in out_mods:
            out_mods.append(Lm)
        seen = set()
        for mod in out_mods:
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            if mod is Lm and init_filters is not None:
                init = init_filters
            else:
                wabs = mod.weight.detach().abs().flatten(1).mean(1)
                order = torch.argsort(wabs, descending=True)
                idx = [int(order[i % len(order)]) for i in range(k)]
                cl = mod.weight.detach()[idx].clone()
                init = cl + torch.randn_like(cl) * 0.02 * cl.abs().mean().clamp(min=1e-6)
            C._widen_out(mod, k, init)
        for dw in {id(d): d for d in dws}.values():
            C._widen_depthwise(dw, k)
        for bn in {id(b): b for b in bns}.values():
            C._widen_bn(bn, k)
        for fb in {id(f): f for f in fbns}.values():
            C._widen_frozen_bn(fb, k)
        for cmod, offs in cons.items():
            C._insert_in_zeros(cmod, k, [o + old_L for o in offs])
        if not _ag_forward_ok(trial, adapter, device):
            return None
        with torch.no_grad():
            after = adapter.teacher_outputs(trial, ref_imgs)
        if C._max_abs_diff(ref_out, after) >= 1e-3:
            return None
        return trial
    except BaseException:
        return None


_SHIMS_INSTALLED = False
_ORIG_LT = _ORIG_FO = _ORIG_GROW = None                      # originalne morphology funkcije (delegacija za ne-probe adaptere)


def install_sizing_shims():
    """Zamijeni morphology 2D-vezane dijelove adapter/generic verzijom (idempotentno). Pokupe ih i INTERNI
    pozivatelji (bare-name kroz module-globale):
      A.layer_table      -> adapter-input mjerenje (ne hardkod 640); vide `gflops_total`/`coupled_unit_cost`
      C._forward_ok      -> adapter-input (ne hardkod 320); vidi `_apply_prune_plan`
      C._try_grow_layer  -> adapter-input + 1D-svjestan; vidi `_grow_decide`
    2D modeli nepromijenjeni (nemaju dim-3 slojeve; forward_example daje isti 3ch-2D ulaz).
    (6.4) `A.weighted_leaves` VISE NIJE ovdje: bila je jedina zakrpa bez fallbacka, tj. 2/3/4 je i onako
    uvijek pobjedivao -> sada je to jedina definicija (classify.py), pa zakrpa nema sto raditi."""
    global _SHIMS_INSTALLED, _ORIG_LT, _ORIG_FO, _ORIG_GROW
    if _SHIMS_INSTALLED:
        return
    _ORIG_LT, _ORIG_FO, _ORIG_GROW = A.layer_table, C._forward_ok, C._try_grow_layer   # zapamti originale
    A.layer_table = _ag_layer_table
    C._forward_ok = _ag_forward_ok
    C._try_grow_layer = _ag_try_grow_layer
    _SHIMS_INSTALLED = True


install_sizing_shims()                                       # engine = genericki engine -> shim aktivan cim se uveze


def gflops(model, adapter, device):
    """Model-level GFLOPs na ADAPTEROVOM ulazu (preko zakrpanog layer_table)."""
    return A.gflops_total(model, adapter, device)


def _widths(model):
    """{sloj: broj IZLAZNIH jedinica} — kanali za conv, neuroni za linear. Sluzi zatvorenoj prune petlji
    (6.11) da vidi KOJI su se slojevi stvarno suzili, ukljucujuci spregnute siblinge koje tp orezi usput."""
    return {nm: int(w.shape[0]) for nm, _, _, w in A.weighted_leaves(model)}


# =========================== PREP: autobatch (port na ctx) =========================== #
def autobatch(model, adapter, device, ctx, path, free_frac=0.9, cap=64, cands=(1, 2, 4, 8, 16, 32, 64)):
    """Auto TRAIN batch: probaj kandidate UZLAZNO mimicirajuci CACHIRANI FT korak (student fwd + KD-loss vs
    PREDRACUNATI teacher-signal + backward + Prodigy.step = najtezi mod, glavni OOM rizik; teacher se NE vrti
    jer petlja koristi cache), izmjeri vrsni VRAM. Uzmi najveci ciji peak <= free_frac × raspolozivi-VRAM.
    Snapshot/restore state_dict -> model NETAKNUT. Samo CUDA (CPU -> najmanji). Vrati train_bs.
    Port morphology `compress.autobatch` na nas ctx: `_DetDataset`->`input_batch`, adapter.kd_loss->`loss.kd_loss`."""
    cands = [b for b in cands if b <= cap]
    if device.type != "cuda":
        return cands[0]
    import gc

    import prodigyopt
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    split = ctx["split_plan"]["train"] if ctx.get("split_plan") else None
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rg = [p.requires_grad for p in model.parameters()]
    chosen = cands[0]
    try:
        for bs in cands:
            torch.cuda.empty_cache(); gc.collect()
            free, _ = torch.cuda.mem_get_info()
            usable = free + torch.cuda.memory_allocated()             # koliko torch SMIJE narasti
            try:
                imgs, _ = DS.input_batch(path, adapter, device, split=split, n=bs)
                with torch.no_grad():                                 # teacher-signal REZIDENTAN na GPU = kao cache.get
                    tsig = L.teacher_signals(model, adapter, imgs, taps)
                for p in model.parameters():
                    p.requires_grad_(True)
                opt = prodigyopt.Prodigy([p for p in model.parameters() if p.requires_grad], lr=1.0)
                model.eval()                                          # zrcali FT (BN-eval disciplina); peak ~isti
                torch.cuda.reset_peak_memory_stats(device)            # (aktivacije za backward se cuvaju i u eval);
                # + izbjegava bs=1 BN-crash na global-pool granama (npr. voc ASPP [1,C,1,1])
                loss, _ = L.kd_loss(model, model, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=tsig)
                loss.backward(); opt.step()
                peak = torch.cuda.max_memory_allocated(device)
                del loss, tsig, imgs, opt
                for p in model.parameters():
                    p.grad = None
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache(); gc.collect(); break        # OOM -> stani, zadrzi prethodni
            if peak <= free_frac * usable:
                chosen = bs
            else:
                break                                                # probio prag -> stani
    finally:
        model.load_state_dict(sd); model.to(device)                  # vrati netaknut model
        for p, r in zip(model.parameters(), rg):
            p.requires_grad_(r)
        torch.cuda.empty_cache(); gc.collect()
    return chosen


# =========================== SPLIT / BATCH MATERIJALIZACIJA =========================== #
def _candidate_files(root, mode, split, cap, split_key=None, carve=False):
    # DEV_DATA_SUBSET vrijedi za SVE taskove, ne samo detekciju: ovo je jedino usko grlo kroz koje
    # jezgra cita medijske ulaze. (Do 6.9 je konstanta rezala SAMO detekcijski GT loader u plugu,
    # a u jezgri je sluzila iskljucivo za ispis upozorenja — gasenje/paljenje nije mijenjalo nista
    # za regresiju, segmentaciju i klasifikaciju.)
    #
    # `carve=True` (split_plan kaze 'AUTO', tj. dataset nema gotove foldere): uzmi CIJELI popis pa ga
    # deterministicki podijeli 70/15/15 i vrati trazeni dio. Do 6.13 se AUTO tumacio kao split=None ->
    # "cijelo stablo za SVAKI split", pa su train i val bili ISTE datoteke.
    # Cap se primjenjuje TEK NAKON podjele -> DEV_DATA_SUBSET znaci N po splitu, ne N na pool.
    lim = 10 ** 9 if carve else cap
    if mode == "image":
        files = DS._media_files(root, DS._IMG, split, lim, skip_masks=True)
    elif mode == "seq":
        files = DS._media_files(root, DS._AUD, split, lim)
    else:
        return []
    if carve and split_key:
        files = DS.auto_carve(files, split_key)
    if CFG.DEV_DATA_SUBSET:
        cap = min(cap, int(CFG.DEV_DATA_SUBSET))
    return files[:cap]


def to_device(batch, device):
    """Pomakni jedan batch (lista CPU uzoraka) na uredjaj."""
    return [im.to(device) for im in batch]


FALLBACK_BATCHES = 8                                         # bez citljivih podataka nema sto "sve" znaciti


class LazyBatches(collections.abc.Sequence):
    """Batchevi kao PUTANJE; dekodira se tek pri pristupu.

    Zasto: eager verzija je drzala SVE dekodirane uzorke u RAM-u — za yolo train split to je
    5860 x 4.7 MB = **26.8 GB** (dostupno 12 GB). Zato je `n_batches=8` i bio default; nije bila
    lijenost nego posljedica dizajna. morphology to nije imao jer je isao DataLoaderom: ulazi se
    dekodiraju po batchu i odbacuju, a na disk idu SAMO teacher signali.
    Ovako je RAM konstantan (~jedan batch) i cijeli split je upotrebljiv.

    Ponasa se kao lista pa `to_device(batches[i])` i `for b in batches` rade nepromijenjeno."""

    def __init__(self, groups, dec, in_ch, size, sr=None):
        self.groups, self._dec, self._in_ch, self._size = groups, dec, in_ch, size
        self._sr = sr                                        # deklarirana frekvencija (audio); None = ne resampliraj

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return [self._dec(f, self._in_ch, self._size, self._sr) if self._sr is not None
                else self._dec(f, self._in_ch, self._size) for f in self.groups[i]]

    @property
    def bsize(self):                                         # bez dekodiranja (inace bi max(len(b)) sve procitao)
        return max((len(g) for g in self.groups), default=0)

    @property
    def paths(self):
        return [f for g in self.groups for f in g]


def _batch_size_of(batches):
    """Velicina batcha bez dekodiranja lijenih uzoraka."""
    bs = getattr(batches, "bsize", None)
    return bs if bs is not None else max((len(b) for b in batches), default=0)


def count_train_samples(path, adapter, split_plan, split_key="train"):
    """Koliko uzoraka split UOPCE ima. 0 = nema readera (token/parquet) -> frozen-random put."""
    mode = getattr(adapter, "_mode", "image")
    root = DS.detect_format(path).get("root", path)
    tr = (split_plan or {}).get(split_key) if isinstance(split_plan, dict) else split_plan
    split = None if (tr in (None, "AUTO")) else tr
    carve = (tr == "AUTO")                                   # nema gotovih foldera -> podijeli sami
    try:
        if mode in ("image", "seq"):
            return len(_candidate_files(root, mode, split, 10 ** 9, split_key, carve))
        if mode == "vector":
            X = DS._tabular_matrix(root, adapter._in_ch, split=split_key)
            n = len(X) if X is not None else 0
            return min(n, int(CFG.DEV_DATA_SUBSET)) if CFG.DEV_DATA_SUBSET else n
    except BaseException:
        pass
    return 0


def materialize_train_batches(path, adapter, device, split_plan, batch_size=8, n_batches=None, seed=0,
                              split_key="train"):
    """FIKSNI batchevi jednog splita kao CPU tenzori (reuse kroz epohe). Vrati (batches, source).

    `split_key` bira split: "train" za KD ulaze, "val" za mjerenje kvalitete. Isti citac za oboje —
    slaganje s uciteljem se mjeri na VAL splitu, kao i svaka druga metrika kvalitete.

    `n_batches=None` (zadano) = **CIJELI train split**, kao morphology (`n_batches = len(loader)`).
    Nema pitanja korisniku ni magicnog broja; koliko podataka ima, toliko se uzme. Eksplicitni
    `n_batches` postoji samo za testove/smoke.
    Deterministicno (seed); fallback-random se generira JEDNOM i drzi (frozen) -> teacher-cache
    alignment vrijedi i bez readera (token/parquet)."""
    import random
    mode = getattr(adapter, "_mode", "image")
    root = DS.detect_format(path).get("root", path)
    tr = (split_plan or {}).get(split_key) if isinstance(split_plan, dict) else split_plan
    split = None if (tr in (None, "AUTO")) else tr
    carve = (tr == "AUTO")                                   # 'AUTO' -> sami dijelimo 70/15/15

    if n_batches is None:                                    # SVE sto ima
        n_avail = count_train_samples(path, adapter, split_plan, split_key)
        n_batches = max(1, -(-n_avail // batch_size)) if n_avail else FALLBACK_BATCHES
    need = batch_size * n_batches

    samples, source = [], "fallback-random"
    try:
        if mode in ("image", "seq"):
            files = _candidate_files(root, mode, split, need * 3, split_key, carve)
            if files:
                rng = random.Random(seed); rng.shuffle(files)
                files = (files * (need // len(files) + 1))[:need]        # cikliraj ako je uzoraka premalo
                dec = DS._decode_image if mode == "image" else DS._decode_audio
                groups = [files[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
                groups = [g for g in groups if g]
                return (LazyBatches(groups, dec, adapter._in_ch, adapter.imgsz,
                                    getattr(adapter, "sr", None)),                 # RAM = jedan batch
                        "image-files" if mode == "image" else "audio-files")
        elif mode == "vector":
            X = DS._tabular_matrix(root, adapter._in_ch, split=split_key)
            if X is not None and len(X):
                import numpy as np
                idx = np.random.default_rng(seed).integers(0, len(X), need)
                samples = [torch.from_numpy(X[i]) for i in idx]; source = "tabular"
    except BaseException:
        samples = []
    if not samples:                                          # token bez tokenizera / image-u-parquetu -> frozen random
        samples = [adapter._one(torch.device("cpu")) for _ in range(need)]
        source = "fallback-random"
    batches = [samples[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
    return [b for b in batches if b], source


def agreement_metrics(teacher, adapter, device, ctx, path, frac=None, seed=0):
    """SLAGANJE S UCITELJEM kao OBICNA metrika kvalitete. Vrati (metric_fn, monitor_fn).

    Do 6.13 je ovo bio special-case: 64 uzorka izvucena iz TRAIN batcheva, bez monitora. Dvije greske
    odjednom — kvaliteta se mjerila na podacima na kojima se trenira, i to na uzorku tako malom da je
    rezolucija bila 1/64 = 1.6 pb (uz prag 0.97 nije postojala dostizna vrijednost izmedju 0.9688 i
    0.9844). Sada vrijede ISTA PRAVILA kao za mAP/f1/r2:
      * mjeri se na VAL splitu, ne na train ulazima
      * `metric_fn` ide na CIJELI val, `monitor_fn` na FIKSNU (seedanu) polovicu — isto kao svugdje
      * prag se racuna iz ODGOVARAJUCE baseline vrijednosti (v. phase1_loop floor_full / gate_floor)
    Nema zasebnog gatea ni zasebne tolerancije."""
    import metric as _M
    frac = CFG.METRIC_MONITOR_FRAC if frac is None else frac
    kind = ctx["out_kind"]
    vb, _src = materialize_train_batches(path, adapter, device, ctx["split_plan"],
                                         batch_size=16, seed=seed, split_key="val")
    full = [x for b in vb for x in b]
    mon = None
    if full and 0.0 < frac < 1.0 and len(full) > 1:
        import random
        k = max(1, int(round(frac * len(full))))
        idx = sorted(random.Random(seed).sample(range(len(full)), k))    # FIKSAN, seedano nasumican
        sub = [full[i] for i in idx]
        mon = lambda mdl: _M.teacher_agreement(mdl, teacher, adapter, sub, device, kind=kind)["agreement"]
    return (lambda mdl: _M.teacher_agreement(mdl, teacher, adapter, full, device, kind=kind)["agreement"],
            mon)


# =========================== TEACHER-SIGNAL CACHE =========================== #
def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def _to_half_cpu(sig):
    return {"feat": {k: v.detach().half().cpu() for k, v in sig["feat"].items()},
            "out": sig["out"].detach().half().cpu()}


def _to_dev_float(sig, device):
    return {"feat": {k: v.float().to(device) for k, v in sig["feat"].items()},
            "out": sig["out"].float().to(device)}


def _fingerprint(batches):
    """Jeftin sadrzajno-osjetljiv otisak ulaza (oblik + prorijedena suma po batchu) -> promjena ulaza
    (drugi seed/split/velicina) invalidira cache iako se meta-brojevi poklope.
    Lijeni batchevi: otisak iz PUTANJA (inace bi dekodirao po jedan uzorak iz svakog batcha)."""
    h = hashlib.sha1()
    paths = getattr(batches, "paths", None)
    if paths is not None:
        for f in paths:
            h.update(str(f).encode())
        return h.hexdigest()[:16]
    for b in batches:
        if not b:
            continue
        t = b[0].detach().float().flatten()
        h.update(str(tuple(b[0].shape)).encode())
        step = max(1, t.numel() // 64)
        h.update(str(round(float(t[::step].sum()), 3)).encode())
    return h.hexdigest()[:16]


class TeacherSigCache:
    """Teacher signali po batchu: disk (perzistentno, reuse kroz RUN-ove) + opc. RAM-rezident (brzina)."""
    def __init__(self, cache_dir, n, in_ram):
        self.dir = cache_dir; self.n = n; self.in_ram = in_ram
        self.ram = [None] * n if in_ram else None

    def _path(self, i):
        return os.path.join(self.dir, f"sig_{i}.pt")

    def has_all(self):
        return all(os.path.exists(self._path(i)) for i in range(self.n))

    def put(self, i, sig):
        h = _to_half_cpu(sig)
        torch.save(h, self._path(i))
        if self.in_ram:
            self.ram[i] = h

    def warm_ram(self):
        if self.in_ram:
            for i in range(self.n):
                if self.ram[i] is None:
                    self.ram[i] = torch.load(self._path(i), map_location="cpu")

    def get(self, i, device):
        h = self.ram[i] if (self.in_ram and self.ram[i] is not None) else torch.load(self._path(i), map_location="cpu")
        return _to_dev_float(h, device)


def _nbytes(o):
    """Bajtovi tenzora u ugnijezdenoj strukturi, u fp16 (tako se i sprema)."""
    if torch.is_tensor(o):
        return o.numel() * 2
    if isinstance(o, dict):
        return sum(_nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_nbytes(v) for v in o)
    return 0


def plan_teacher_cache(teacher, adapter, batches, taps, model_name, split="train", disk_frac=0.8):
    """Sizing PRIJE racunanja: izmjeri JEDAN batch teacher-signala pa procijeni cijeli cache.
    Nista ne pise. Vrati dict za GUI/odluku (port morphology `teacher_mem_plan`).

    Zasto: `precompute_teacher` inace samo pise dok disk ne pukne. morphology je ovo imao i GUI je
    plan prikazivao u pripremi — korisnik vidi koliko ce zauzeti PRIJE nego krene."""
    dev = next(teacher.parameters()).device
    n = len(batches)
    cache_dir = os.path.join(TMP_ROOT, _safe(model_name), _safe(split))
    per = 0
    if n:
        sig = L.teacher_signals(teacher, adapter, to_device(batches[0], dev), taps)
        per = _nbytes(sig)
    total = per * n
    try:
        st = os.statvfs(TMP_ROOT if os.path.isdir(TMP_ROOT) else os.path.dirname(TMP_ROOT) or "/")
        free = st.f_bavail * st.f_frsize
    except BaseException:
        free = 0
    fits = bool(free) and total <= free * disk_frac
    return {"n_batches": n, "batch_size": _batch_size_of(batches),
            "bytes_per_batch": per, "total_gb": total / C.GB, "free_gb": free / C.GB,
            "fits_disk": fits, "cache_dir": cache_dir,
            "n_fit": int(free * disk_frac / per) if per else 0}


def precompute_teacher(teacher, adapter, batches, taps, model_name, split="train", in_ram=True, verbose=False):
    """Predracunaj teacher signale za sve batcheve -> TeacherSigCache. META (model/n_batches/batch_size/tapovi/
    fingerprint) invalidira cache pri promjeni. Reuse valjanog diskovnog cachea (preskoci racun)."""
    dev = next(teacher.parameters()).device
    n = len(batches)
    bs = _batch_size_of(batches)
    cache_dir = os.path.join(TMP_ROOT, _safe(model_name), _safe(split))
    meta_path = os.path.join(cache_dir, "meta.json")
    meta = {"model": model_name, "n_batches": n, "batch_size": bs,
            "taps": sorted(taps), "fingerprint": _fingerprint(batches)}
    cache = TeacherSigCache(cache_dir, n, in_ram)
    if os.path.exists(meta_path) and _load_json(meta_path) == meta and cache.has_all():
        cache.warm_ram()
        if verbose:
            print(f"  reuse teacher-cache (meta valjan): {cache_dir}")
        return cache
    os.makedirs(cache_dir, exist_ok=True)
    for f in os.listdir(cache_dir):                          # ISPOCETKA: ocisti stari (drugi batch/taps) cache
        if f.startswith("sig_") and f.endswith(".pt"):
            os.remove(os.path.join(cache_dir, f))
    for i, b in enumerate(batches):
        cache.put(i, L.teacher_signals(teacher, adapter, to_device(b, dev), taps))
    json.dump(meta, open(meta_path, "w"))
    if verbose:
        print(f"  precompute teacher-cache: {n} batcheva -> {cache_dir}")
    return cache


def _load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except BaseException:
        return None


# =========================== PRUNE-PETLJA (5.2) =========================== #
def prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, steps, clip=5.0, loss_fn=None,
                     offset=0):
    """KD-FT oporavak nakon reza: Prodigy (auto-LR), BN-eval disciplina (5.0 nalaz-b: eval NE gasi grad, ali
    zamrzne BN running-stats -> konzistentno s teacher.eval(), bez train/eval napuhavanja lossa), teacher-cache.
    `loss_fn(student, imgs)->(loss,info)` (enhancer-KD) ZAMJENJUJE genericki kd_loss (cache se tad ne koristi).
    Vrati zadnji KD-loss."""
    import prodigyopt
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    opt = prodigyopt.Prodigy([p for p in student.parameters() if p.requires_grad],
                             lr=1.0, d_coef=0.9, safeguard_warmup=True)
    last, nb = None, len(batches)
    for s in range(steps):
        # POMICNI kursor: bez `offset` bi svaki morph korak krenuo od batcha 0, pa bi student vidio
        # SAMO prvih `steps` batcheva kroz cijelu kompresiju (pri ft_steps=6 i batchu 16 = 96 slika,
        # uvijek istih) — i podizanje `n_batches` ne bi promijenilo nista. (BUG do 6.8.)
        i = (offset + s) % nb
        imgs = to_device(batches[i], device)
        sig = cache.get(i, device) if cache is not None else None
        opt.zero_grad(set_to_none=True)
        loss, _ = (loss_fn(student, imgs) if loss_fn is not None
                   else L.kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=sig))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], clip)
        opt.step()
        last = float(loss)
    return last


# =========================== FAZA 1: PUNA KD EPOHA + PERZISTENTAN PRODIGY =========================== #
# (PLAN_KOMPRESIJA.md §1.5 i §1.6.) Do sada je oporavak bio `prune_ft_recover(steps=6)`: 6 gradijentnih
# koraka i NOVI Prodigy pri svakom pozivu. Dva problema koja to nosi:
#   (a) pri batchu 16 to je 96 uzoraka po morph koraku — morphology je vrtio PUNU epohu (~10 000);
#   (b) Prodigy treba desetke koraka da procijeni `d`, pa ga 6 koraka baci prije nego se stabilizira.
# Novo: jedna epoha preko SVIH batcheva, uz JEDAN optimizer koji zivi kroz cijelu fazu.

def _new_prodigy(model):
    """Prodigy po morphology receptu (`growth_rate` je jedina razlika od starog `prune_ft_recover`).
    Rebuilda se SAMO na arhitekturnu promjenu — prune/grow zamijene tenzore pa stari optimizer state
    pokazuje na mrtve parametre. `gstep` se pritom NE resetira (warmup se ne restarta)."""
    import prodigyopt
    return prodigyopt.Prodigy([p for p in model.parameters() if p.requires_grad],
                              lr=1.0, d_coef=0.9, growth_rate=1.1, safeguard_warmup=True)


def lr_eff(opt):
    """Prodigy: efektivni LR = d x lr, gdje je `lr` ovdje samo warmup-multiplikator."""
    pg = opt.param_groups[0]
    return float(pg.get("d", 1.0)) * float(pg["lr"])


def kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep, warmup,
             clip=5.0, loss_fn=None, on_batch=None):
    """JEDNA PUNA KD epoha preko svih `batches`. Vrati (mean_loss, gstep).

    Razlike od `prune_ft_recover` (koji ostaje dok se stara petlja ne makne):
      * ide kroz SVE batcheve, ne `steps` komada
      * `opt` i `gstep` dolaze IZVANA -> optimizer i warmup prezive vise epoha
      * linearni warmup `lr = min(1, (gstep+1)/warmup)` — kao morphology, ali samo JEDNOM na pocetku
        faze (gstep ne pada na 0 pri rebuildu), pa se ne restarta nakon svakog reza.

    BN ostaje ZAMRZNUT (`student.eval()`) — [[bn-eval-detection-trainmode]]. Morphology je ovdje
    pustao BN da trenira; to je bio izvor lazno niskih mAP-ova i NE kopiramo ga.
    `loss_fn(student, imgs)->(loss, info)` (enhancer-KD) zamjenjuje genericki `kd_loss`; tada se
    teacher cache ne koristi. Bez GT-a u lossu ([[kd-only-no-gt]])."""
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    student.eval()
    for p in student.parameters():
        p.requires_grad_(True)
    warmup = max(int(warmup), 1)
    tot, n = 0.0, 0
    for i in range(len(batches)):
        for g in opt.param_groups:
            g["lr"] = min(1.0, (gstep + 1) / warmup)         # warmup je multiplikator, Prodigy drzi `d`
        imgs = to_device(batches[i], device)
        sig = cache.get(i, device) if cache is not None else None
        opt.zero_grad(set_to_none=True)
        loss, _ = (loss_fn(student, imgs) if loss_fn is not None
                   else L.kd_loss(student, teacher, adapter, imgs, taps, kd_mode, out_kind, teacher_sig=sig))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], clip)
        opt.step()
        tot += float(loss); n += 1; gstep += 1
        if on_batch is not None:                             # napredak (epoha na punom splitu traje) —
            on_batch(i + 1, len(batches), tot / n)           # engine sam NE ispisuje, samo javlja
    return (tot / max(n, 1)), gstep


# =========================== MORPH DOGADJAJ (prune + grow) =========================== #
# (6.12) Tijelo jednog morph koraka izdvojeno iz `morph_loop` u `MorphState` + `morph_step`, da ga
# `phase1_loop` (PLAN_KOMPRESIJA) i stara `morph_loop` DIJELE. Ista logika kao kod coupled costa:
# prune/grow mehanika smije imati samo JEDAN dom ([[coupled-cost-single-source]]) — dvije kopije bi
# se razisle cim jednu popravimo.

class MorphState:
    """Stanje koje prezivljava KROZ korake jedne faze: ban-lista, churn-cooldown, reinvest pool.

    `banned`   — slojevi cija tp-grupa razbije forward (C2f split/concat); jednom bananі, trajno vani.
    `grown_at` / `pruned_at` — {sloj: morph_idx} za anti-churn cooldown ([[grow-prune-churn-rootcause]]).
                 Mjeri se u MORPH dogadjajima, NE u FT epohama — recovery epohe ne troše cooldown.
    `total_pruned` / `total_grown` — reinvest pool = reinvest_frac x total_pruned - total_grown.
    `step_target` — apsolutan i konstantan: step_frac x GFLOPs na POCETKU faze."""

    def __init__(self, prunable, g0, step_frac=None, reinvest_frac=None, cooldown=None):
        self.prunable = set(prunable)
        self.g0 = float(g0)
        self.step_target = (step_frac or CFG.PHASE2_PRUNE_STEP_FRAC) * self.g0
        self.reinvest_frac = CFG.PHASE2_REINVEST_FRAC if reinvest_frac is None else reinvest_frac
        self.cooldown = CFG.PHASE2_CHURN_COOLDOWN if cooldown is None else cooldown
        self.banned = set()
        self.grown_at, self.pruned_at = {}, {}
        self.morph_idx = 0
        self.total_pruned = self.total_grown = 0.0

    def align_best(self, model):
        return C.best_align_score(model, {nm: True for nm in self.prunable}, self.banned)


def morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=None, grow=True,
               step_target=None):
    """JEDAN morph dogadjaj: KD-importance -> ZATVORENA prune petlja -> grow iz reinvest poola.
    Mijenja `st` na mjestu (banned/cooldown/pool/morph_idx). Vrati (student, info).

    `grow=False` gasi rast za ovaj korak (Faza 2, zadnja precka: `g_min` je mjeren cistim rezom pa je
    uz aktivan grow nedostizan). `step_target` nadglasa `st.step_target` (Faza 2 suzava zadnji korak
    da ne prebaci meducilj)."""
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    target = st.step_target if step_target is None else float(step_target)
    st.morph_idx += 1
    mi = st.morph_idx
    grow_protected = {l for l, k in st.grown_at.items() if mi - k <= st.cooldown}    # svjež narastao -> ne rezi
    prune_protected = {l for l, k in st.pruned_at.items() if mi - k <= st.cooldown}  # svjež rezan -> ne rastri
    elig_all = st.prunable - st.banned
    imp, gavg = L.kd_importance(student, teacher, adapter, imp_dev, taps, kd_mode, out_kind,
                                prunable=elig_all, loss_fn=loss_fn)
    cost, flops_per, units = C.prune_costs(student, adapter, device, elig_all)
    info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in elig_all}
    flops_per0, units0 = dict(flops_per), dict(units)    # GROW koristi cijenu s POCETKA koraka (kao prije 6.11);
                                                         # prune je ispod smije preracunati na suzenom elig setu

    # ---- PRUNE: ZATVORENA PETLJA (plan -> primijeni -> IZMJERI -> doplaniraj OSTATAK) ----
    # Do 6.11 je ovo bio JEDAN prolaz i procjena planera se uzimala zdravo za gotovo. Mjerenje je
    # pokazalo da stvarni rez padne na 54% procjene (yolo26n, 4. korak) iz tri razloga:
    #   (a) banani listovi (C2f/concat cvorista) ulaze u procjenu, a _apply_prune_plan ih preskoci —
    #       a bas njih planer bira PRVE jer imaju najvecu spregnutu cijenu po kanalu;
    #   (b) drugi red spregnutosti: kad su producent i potrosac OBA u planu, usteda a*b se broji dvaput
    #       (procjena je linearna, stvarnost je (in-a)*(out-b));
    #   (c) tihi preskoci u _apply_prune_plan (floor/cap, len(idx2)>=C).
    # Sva tri nestaju ako se ostatak MJERI pa doplanira: svaki iduci krug racuna cijenu na VEC SMANJENOM
    # modelu, a banani listovi su iz njega ispali. Slojevi dirnuti u ranijem krugu se IZUZIMAJU do kraja
    # koraka — imp/order su za njih zastarjeli (tp prepakira indekse kanala), a usput to cuva i
    # PHASE2_PRUNE_LAYER_CAP (bez izuzeca bi se isti sloj mogao rezati 15% po krugu).
    g_before = g_cur = gflops(student, adapter, device)
    w_step0 = _widths(student)                           # sirine na POCETKU koraka -> detekcija dirnutih slojeva
    cd_override, n_rem, pruned_names, touched = False, 0, set(), set()
    est_freed = r1_freed = 0.0                           # PRVI krug: procjena i STVARNI rez
                                                         # (razlika = koliko cost model precjenjuje)
    remaining, rounds, stall = target, 0, 0
    for _rnd in range(CFG.PHASE2_PRUNE_ROUNDS):
        if remaining <= CFG.PHASE2_PRUNE_SLACK * target or stall >= 2:
            break                                        # cilj pogoden (unutar slacka) ili dva prazna kruga
        excl = (st.banned | touched) if cd_override else (st.banned | touched | grow_protected)
        elig_r = elig_all - excl
        if not elig_r:
            if not cd_override and grow_protected:
                cd_override = True; continue             # cooldown blokira SVE -> rez nužan: SOFT override
            break
        if rounds:                                       # 2. krug nadalje: cijena na SMANJENOM modelu
            cost, flops_per, units = C.prune_costs(student, adapter, device, elig_r)
            info = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in elig_r}
        plan, _r = C._select_prune_plan({nm: True for nm in elig_r}, imp, cost, flops_per, units, info,
                                        remaining, CFG.PHASE2_PRUNE_LAYER_CAP, CFG.PHASE2_MIN_ALIVE_FRAC,
                                        CFG.PHASE2_MIN_ALIVE, exclude=excl)
        if not plan:
            if not cd_override and grow_protected:
                cd_override = True; continue
            break
        if not rounds:
            est_freed = float(_r)                        # koliko planer MISLI da ce osloboditi (1. krug)
        student, k_rem, n_lay, n_bad, bad = C._apply_prune_plan(student, adapter, device, plan)
        st.banned |= bad
        pruned_names |= (set(plan) - bad)
        n_rem += k_rem
        rounds += 1
        w_now = _widths(student)                         # sve cije se sirine promijenilo (i spregnuti siblinzi)
        touched |= {nm for nm, w0 in w_step0.items() if w_now.get(nm, w0) != w0}
        g_new = gflops(student, adapter, device)
        got = g_cur - g_new
        g_cur = g_new
        if rounds == 1:
            r1_freed = max(got, 0.0)                     # sto bi stari jednoprolazni kod isporucio
        remaining -= max(got, 0.0)
        stall = stall + 1 if got <= 1e-9 else 0
    freed = g_before - g_cur
    st.total_pruned += max(freed, 0.0)
    for nm in pruned_names:
        st.pruned_at[nm] = mi

    # ---- GROW (reinvest pool, GradMax iz gavg) ----
    pool = st.reinvest_frac * st.total_pruned - st.total_grown
    grown_info = []
    if grow and pool > 0:
        info_g = {nm: (m, w.dim()) for nm, m, _, w in A.weighted_leaves(student) if nm in st.prunable}
        struct_g = {nm: (nm in st.prunable and nm not in pruned_names and nm not in prune_protected
                         and nm not in st.banned) for nm in st.prunable}
        g2, ginfos, spent = C._grow_decide(student, adapter, device, struct_g, gavg, flops_per0, units0,
                                           info_g, pool)
        if g2 is not None:
            student = g2; st.total_grown += max(spent, 0.0); grown_info = ginfos
            for gi in ginfos:
                st.grown_at[gi["layer"]] = mi
    return student, {"n_rem": n_rem, "grown": grown_info, "cd_override": cd_override,
                     "step_target": target, "est_freed": est_freed, "r1_freed": r1_freed,
                     "act_freed": max(freed, 0.0), "prune_rounds": rounds,
                     "grow_protected": len(grow_protected)}



# =========================== PRUNE + GROW PETLJA (5.3) =========================== #
# (Napomena: raniji `continuous_prune` (5.2, prune-only) uklonjen kao redundantan — `morph_loop(reinvest_frac=0)`
#  daje isti čisti prune jer pool = 0·pruned − grown ≤ 0 → grow nikad ne opali.)
def morph_loop(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, step_frac=None, reinvest_frac=None, max_steps=None, ft_steps=6,
               batch_size=8, n_batches=None, imp_batches=3, seed=0, cache=None, batches=None,
               cooldown=None, on_step=None,
               metric_fn=None, metric_tol=None, metric_baseline=None, loss_fn=None):
    """Kontinuirani PRUNE + GROW pod GFLOPs budžetom, s churn-cooldownom ([[grow-prune-churn-rootcause]]) i
    reinvest-poolom (grow troši <= reinvest_frac × oslobođenih GFLOPs -> net i dalje pada). Svaki morph korak:
      1) kd_importance -> imp (PRUNE rang) + gavg (GROW smjer)  [jedan prolaz, dva signala]
      2) PRUNE (elig = prunable − banned − grow_protected; soft-override ako cooldown blokira SVE) pod step budžetom
      3) GROW (reinvest pool; elig = growable − upravo_rezano − prune_protected) GradMax  [REUSE `_grow_decide`]
      4) prune_ft_recover (KD-FT, teacher-cache).
    Cooldown se mjeri u MORPH dogadajima (ne FT epohama). `_grow_decide._fresh` je dodatni guard (rezani sloj ima
    stari gavg-oblik -> preskočen za grow).

    QUALITY-GATE (best-model selekcija): `metric_fn(model)->broj` (prava metrika ako ima oznaka, inače
    teacher-agreement; `full_cycle` ga UVIJEK postavi). best = najmanji GFLOPs čija metrika >= `metric_tol ×
    metric_baseline`; early-stop kad padne ispod (ta i iduće 2). GT/labeli SAMO u gate-u, nikad u lossu
    ([[kd-only-no-gt]]). Bez metric_fn (izravni poziv) -> nema best-trackinga, vrati se trajektorija+zadnji.
    Vrati {g0, trajectory, final_gflops, banned, student, best_model, ...}."""
    taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]
    prunable = set(ctx["prunable"])
    step_frac = step_frac or CFG.PHASE2_PRUNE_STEP_FRAC
    reinvest_frac = CFG.PHASE2_REINVEST_FRAC if reinvest_frac is None else reinvest_frac
    # (6.7) prije je stajao hardkodirani literal 20, pa je `settings.PHASE2_MAX_STEPS` (=200) izgledao
    # kao mrtva konstanta. Petlja se zaustavljala na 20 koraka i nikad ne bi dosegla `target_frac`.
    max_steps = CFG.PHASE2_MAX_STEPS if max_steps is None else max_steps
    if metric_tol is None:                                   # JEDAN izvor tolerancije (v. settings)
        metric_tol = CFG.FT_RECOVERY_FRAC        # JEDAN prag, bez obzira na vrstu metrike (v. settings)
    cooldown = CFG.PHASE2_CHURN_COOLDOWN if cooldown is None else cooldown
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, n_batches, seed)
    if cache is None and loss_fn is None:                    # enhaner-loss (detekcija) ne koristi generic cache
        cache = precompute_teacher(teacher, adapter, batches, taps, model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches]]

    g0 = gflops(student, adapter, device)
    st = MorphState(prunable, g0, step_frac, reinvest_frac, cooldown)   # ban + cooldown + pool (6.12)
    ft_cursor = 0                                            # tekuci batch za FT (v. prune_ft_recover offset)
    best_model, best_gflops, best_step = None, float("inf"), 0     # quality-gate: best = najmanji GFLOPs koji prolazi
    if metric_fn is not None and metric_baseline is None:
        metric_baseline = metric_fn(student)                       # PRAVA metrika originala (baseline za tol)
    metric_floor = (metric_tol * metric_baseline) if metric_fn is not None else None
    below = 0                                                       # uzastopni koraci ispod metric_floor -> early-stop
    _p0 = A.count_params(student)
    traj = [{"step": 0, "gflops": g0, "params": _p0, "kd": None, "removed_ch": 0, "grown": [],
             "metric": metric_baseline, "size_mb": _p0 * 4 / (1024 ** 2),
             "gflops_freed": 0.0, "gflops_reinvested": 0.0,
             "align_score": C.model_align_score(student), "align_best": st.align_best(student)}]
    if on_step:
        on_step(traj[0])                                   # step 0 = BASELINE tocka (GUI je crta kao referencu)
    for step in range(1, max_steps + 1):
        if (g0 - traj[-1]["gflops"]) >= target_frac * g0:
            break
        student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=loss_fn)
        n_rem, grown_info = mi["n_rem"], mi["grown"]

        kd = prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, ft_steps,
                              loss_fn=loss_fn, offset=ft_cursor)
        ft_cursor += ft_steps                                # nastavi gdje si stao -> prolazi kroz SVE batcheve
        cur_metric = metric_fn(student) if metric_fn is not None else None
        n_par = A.count_params(student)
        rec = {"step": step, "gflops": gflops(student, adapter, device), "params": n_par,
               "kd": kd, "metric": cur_metric, "removed_ch": n_rem,
               "grown": [(gi["layer"], gi["k"]) for gi in grown_info],
               "cd_override": mi["cd_override"], "banned": len(st.banned),
               # --- za GUI trajektoriju (6.2): velicina, bilanca GFLOPs, HW-poravnanje ---
               "size_mb": n_par * 4 / (1024 ** 2),           # fp32 tezine
               "step_target": mi["step_target"],              # cilj reza za ovaj korak
               "est_freed": mi["est_freed"],                  # PROCJENA planera (cost model)
               "r1_freed": mi["r1_freed"],                    # STVARNI rez 1. kruga (= staro ponasanje)
               "act_freed": mi["act_freed"],                  # STVARNO oslobodeno (mjereno)
               "prune_rounds": mi["prune_rounds"],            # koliko krugova plan->rez->mjeri je trebalo
               "gflops_freed": st.total_pruned,               # kumulativno oslobodeno rezom
               "gflops_reinvested": st.total_grown,           # kumulativno potroseno rastom
               "align_score": C.model_align_score(student),
               "align_best": st.align_best(student)}
        traj.append(rec)
        if on_step:
            on_step(rec)
        # QUALITY-GATE: best = najmanji GFLOPs čija PRAVA metrika (ili teacher-agreement) >= floor; early-stop 3× ispod
        if metric_fn is not None:
            if cur_metric >= metric_floor:
                below = 0
                if rec["gflops"] < best_gflops:
                    best_model = copy.deepcopy(student); best_gflops = rec["gflops"]; best_step = step
            else:
                below += 1
                if below >= 3:                                        # metrika probila floor 3 koraka zaredom -> stani
                    break
        if n_rem == 0 and not grown_info and not mi["grow_protected"]:   # nista rezano ni naraslo, bez cooldowna -> iscrpljeno
            break
    return {"g0": g0, "trajectory": traj, "final_gflops": traj[-1]["gflops"], "banned": sorted(st.banned),
            "grown_at": st.grown_at, "pruned_at": st.pruned_at, "student": student, "cache": cache,
            "best_model": best_model, "best_gflops": best_gflops, "best_step": best_step,
            "metric_baseline": metric_baseline}


# =========================== FAZA 1 — NAJMANJI MODEL IZNAD PRAGA =========================== #
def phase1_loop(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                monitor_fn=None, metric_baseline=None, monitor_baseline=None, metric_tol=None,
                step_frac=None, reinvest_frac=None, cooldown=None, max_steps=None,
                batch_size=8, imp_batches=3, seed=0, cache=None, batches=None,
                loss_fn=None, on_step=None, on_batch=None):
    """FAZA 1 (PLAN_KOMPRESIJA §1) — nadji NAJMANJI model koji je JOS iznad praga kvalitete.

    DVOMODALNI AUTOMAT, za razliku od `morph_loop` koji reze svaki korak bez obzira na stanje:
      * metrika >= prag  -> MORPH: rez (`morph_step`) + rast + JEDNA PUNA KD epoha
      * metrika <  prag  -> FT-RECOVERY: BEZ reza i rasta, samo epohe dok se ne vrati
    Povratak iznad praga resetira oba broaca (`F1_FT_MAX_EPOCHS`, `F1_FT_PATIENCE` vrijede PO EPIZODI).

    IZLAZ JE ZAJAMCEN: `LAST_GOOD` je deepcopy studenta u trenutku kad je metrika IZMJERENA iznad
    praga, i to onaj s najmanje GFLOPs-a. U najgorem slucaju to je pocetni model (njegova metrika je
    100% sebe pa uvijek prolazi prag < 1.0). Nikad se ne isporucuje model koji nije izmjeren dobrim.

    DVIJE RAZINE MJERENJA (§1.7): `monitor_fn` (fiksni podskup, brzo) odlucuje prune/recovery i
    patience; `metric_fn` (pun skup) se vrti SAMO kad model kandidira za LAST_GOOD. Pragovi su
    ODVOJENI — `floor_mon` iz monitor-baselinea, `floor_full` iz punog — jer to nisu iste brojke.
    Ako puna potvrda padne ispod svog praga, korak se tretira kao pad (monitor je bio optimistican).

    Vrati {model, gflops, metric, step, g0, trajectory, reason, banned, ...}."""
    taps = ctx["taps"]
    prunable = set(ctx["prunable"])
    max_steps = CFG.PHASE2_MAX_STEPS if max_steps is None else max_steps
    metric_tol = CFG.FT_RECOVERY_FRAC if metric_tol is None else metric_tol
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if cache is None and loss_fn is None:                    # enhaner-loss (detekcija) ne koristi generic cache
        cache = precompute_teacher(teacher, adapter, batches, taps, model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches]]

    g0 = gflops(student, adapter, device)
    st = MorphState(prunable, g0, step_frac, reinvest_frac, cooldown)
    opt, gstep, warmup = _new_prodigy(student), 0, len(batches)   # JEDAN optimizer; warmup = 1 epoha

    if metric_baseline is None:
        metric_baseline = float(metric_fn(student))
    if monitor_fn is not None and monitor_baseline is None:
        monitor_baseline = float(monitor_fn(student))
    floor_full = metric_tol * metric_baseline
    gate_fn = monitor_fn if monitor_fn is not None else metric_fn
    gate_floor = (metric_tol * monitor_baseline) if monitor_fn is not None else floor_full

    last_good = copy.deepcopy(student)                       # ★ zajamceni izlaz (pocetni model)
    lg_gflops, lg_metric, lg_step = g0, metric_baseline, 0
    ft_used = no_imp = 0
    curr_best = None                                         # najbolja metrika U TEKUCOJ recovery epizodi
    above = True                                             # start = original -> uvijek iznad praga
    prev = monitor_baseline if monitor_fn is not None else metric_baseline
    _p0 = A.count_params(student)
    # `metric` = vrijednost kojom petlja ODLUCUJE (monitor ako postoji, inace puna) -> graf i linija
    # praga su tada na ISTOJ skali. `metric_full` je rijedja potvrda na punom skupu (samo kandidati).
    traj = [{"step": 0, "phase": "baseline", "gflops": g0, "params": _p0, "kd": None,
             "metric": prev, "metric_full": metric_baseline, "monitor": prev,
             "removed_ch": 0, "grown": [],
             "size_mb": _p0 * 4 / (1024 ** 2), "gflops_freed": 0.0, "gflops_reinvested": 0.0,
             "align_score": C.model_align_score(student), "align_best": st.align_best(student),
             "is_best": True}]
    if on_step:
        on_step(traj[0])
    reason = "max_steps ({})".format(max_steps)

    for step in range(1, max_steps + 1):
        mode = "morph" if above else "ft"
        mi = None
        if mode == "morph":
            student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev, loss_fn=loss_fn)
            opt = _new_prodigy(student)                      # arh. promijenjena -> stari state gleda mrtve tenzore
            if mi["n_rem"] == 0 and not mi["grown"] and not mi["grow_protected"]:
                reason = "nema vise rezivih kandidata (off-limits/banned/floor/cap)"
                break
        kd, gstep = kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep,
                             warmup, loss_fn=loss_fn, on_batch=on_batch)
        m = float(gate_fn(student))
        g_now = gflops(student, adapter, device)

        above = m >= gate_floor
        full = None
        if above and g_now < lg_gflops - 1e-9:               # kandidat za LAST_GOOD -> POTVRDI na punom skupu
            full = m if monitor_fn is None else float(metric_fn(student))
            if full >= floor_full:
                last_good = copy.deepcopy(student)
                lg_gflops, lg_metric, lg_step = g_now, full, step
            else:
                above = False                                # monitor je bio optimistican -> tretiraj kao pad

        # ---- knjigovodstvo oporavka (PLAN_KOMPRESIJA §1.3) ----
        # `curr_best` = najbolja metrika u TEKUCOJ epizodi. Zasijava se vrijednoscu ODMAH NAKON REZA
        # (kvaliteta tada padne) i pomice se samo kad je epoha bolja od nje. Brojac se resetira samo na
        # NOVI curr_best — ne na "bolje od prethodne epohe". Ta razlika nije kozmeticka: pilasti oporavak
        # koji ide dolje-gore-dolje oko iste razine beskonacno bi resetirao brojac.
        stop = None
        if above:                                            # vratili smo se -> epizoda gotova
            ft_used = no_imp = 0
            curr_best = None
        elif mode == "morph":                                # rez nas je oborio -> pocetak nove epizode
            curr_best, ft_used, no_imp = m, 0, 0
        else:                                                # cista FT epoha
            ft_used += 1
            if curr_best is None or m > curr_best + 1e-4:
                curr_best, no_imp = m, 0                     # napredak
            else:
                no_imp += 1
            if no_imp >= CFG.F1_FT_PATIENCE:
                stop = "oporavak stagnira ({} FT epohe bez novog najboljeg)".format(CFG.F1_FT_PATIENCE)
            elif ft_used >= CFG.F1_FT_MAX_EPOCHS:
                stop = "oporavak nije uspio u {} FT epoha".format(CFG.F1_FT_MAX_EPOCHS)

        n_par = A.count_params(student)
        rec = {"step": step, "phase": mode, "gflops": g_now, "params": n_par, "kd": kd,
               "metric": m, "monitor": m, "metric_full": full,
               "removed_ch": (mi["n_rem"] if mi else 0),
               "grown": [(gi["layer"], gi["k"]) for gi in (mi["grown"] if mi else [])],
               "banned": len(st.banned), "cd_override": bool(mi and mi["cd_override"]),
               "size_mb": n_par * 4 / (1024 ** 2),
               "step_target": (mi["step_target"] if mi else None),
               "est_freed": (mi["est_freed"] if mi else None),
               "r1_freed": (mi["r1_freed"] if mi else None),
               "act_freed": (mi["act_freed"] if mi else None),
               "prune_rounds": (mi["prune_rounds"] if mi else 0),
               "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
               "align_score": C.model_align_score(student), "align_best": st.align_best(student),
               "lr": lr_eff(opt), "ft_used": ft_used, "no_imp": no_imp, "curr_best": curr_best,
               "is_best": lg_step == step}
        traj.append(rec)
        if on_step:
            on_step(rec)

        if stop:
            reason = stop
            break
        prev = m

    return {"model": last_good, "gflops": lg_gflops, "metric": lg_metric, "step": lg_step,
            "g0": g0, "metric_baseline": metric_baseline, "monitor_baseline": monitor_baseline,
            "floor_full": floor_full, "floor_monitor": gate_floor, "trajectory": traj,
            "reason": reason, "banned": sorted(st.banned), "student": student, "cache": cache,
            "batches": batches, "state": st}


def run_phase1(student, teacher, adapter, device, ctx, path, model_name, metric_fn=None,
               monitor_fn=None, batch_size=8, imp_batches=3, seed=0, batches=None, cache=None, **kw):
    """Runner za FAZU 1: ista priprema kao `full_cycle` (batchevi, task-uvjetni enhancer-loss,
    teacher cache, gate-ljestvica), pa `phase1_loop`. BEZ dead-removala — izbacen (PLAN_KOMPRESIJA §0):
    census-fragilan je i nije KD-siguran, a KD-grad prune ga subsumira.

    GATE-ljestvica je ista kao u `full_cycle`: prava task-metrika ako je dana, inace label-free
    teacher-agreement ([[kd-only-no-gt]]). Kod agreementa NEMA monitora — on je ionako jeftin
    (64 uzorka), pa bi podskup samo unio sum bez ustede."""
    if batches is None:                                      # pozivatelj ih moze dati (lancani run)
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)

    if not kw.get("loss_fn"):                                # TASK-UVJETNI enhaneri, biran po ctx['task']
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    if cache is None and not kw.get("loss_fn"):
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")

    if metric_fn is None:                                    # bez oznaka -> slaganje s uciteljem
        metric_fn, monitor_fn = agreement_metrics(teacher, adapter, device, ctx, path, seed=seed)
    return phase1_loop(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                       monitor_fn=monitor_fn, batches=batches, cache=cache,
                       imp_batches=imp_batches, seed=seed, **kw)


# =========================== FAZA 2 — LJESTVICA DO STRUKTURNOG MINIMUMA =========================== #
def ft_until(student, teacher, adapter, device, ctx, batches, cache, opt, gstep, warmup, gate_fn,
             max_epochs, patience, loss_fn=None, on_epoch=None, on_batch=None):
    """FT dok se metrika popravlja: stani nakon `patience` uzastopnih epoha bez popravka ili na
    `max_epochs`. Za razliku od Faze 1 ovdje NEMA praga koji se lovi — kvaliteta nije gate, samo se
    uzima koliko se jeftino da vratiti prije iduceg reza. Vrati (gstep, zadnja_metrika, n_epoha)."""
    best, no_imp, m, ep = -float("inf"), 0, None, 0
    for ep in range(1, int(max_epochs) + 1):
        kd, gstep = kd_epoch(student, teacher, adapter, device, ctx, batches, cache, opt, gstep,
                             warmup, loss_fn=loss_fn, on_batch=on_batch)
        m = float(gate_fn(student))
        if m > best + 1e-4:
            best, no_imp = m, 0
        else:
            no_imp += 1
        if on_epoch is not None:
            on_epoch(ep, kd, m, no_imp)
        if no_imp >= int(patience):
            break
    return gstep, m, ep


def phase2_ladder(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                  monitor_fn=None, out_dir=None, n_ckpt=None, g_min=None, banned=(),
                  step_frac=None, reinvest_frac=None, cooldown=None, batch_size=8, imp_batches=3,
                  seed=0, cache=None, batches=None, loss_fn=None,
                  on_step=None, on_batch=None, on_probe=None, max_steps_per_rung=200):
    """FAZA 2 (PLAN_KOMPRESIJA §2) — od izlaza Faze 1 do IZMJERENOG strukturnog minimuma, uz
    `n_ckpt` verzija na linearno rasporedjenim razinama GFLOPs-a.

    `g_min` se mjeri `probe_min_gflops` (ako nije dan). delta = (g_start − g_min)/n_ckpt; ciljevi su
    g_start−delta, g_start−2·delta, ..., g_min. ZADNJI cilj JE minimum.

    GROW pada na ZADNJOJ precki: `g_min` je mjeren CISTIM rezom, pa je uz aktivan grow nedostizan —
    blizu dna se korak reza suzava na (gflops − t), a grow svejedno smije vratiti dio oslobodjenog,
    pa bi petlja oscilirala oko mete (PLAN_KOMPRESIJA §2.5).

    Kvaliteta NIJE gate. `monitor_fn` (ili `metric_fn`) sluzi samo FT patienceu; puna `metric_fn` se
    vrti JEDNOM po checkpointu, za manifest. Checkpointi se spremaju full-eager
    ([[save-models-full-eager]]). Vrati {g_start, g_min, checkpoints[], trajectory, ...}."""
    n_ckpt = int(CFG.F2_CHECKPOINTS if n_ckpt is None else n_ckpt)
    out_dir = out_dir or os.path.join(CFG.TMP_ROOT, "gui_job")
    os.makedirs(out_dir, exist_ok=True)
    prunable = set(ctx["prunable"])
    if batches is None:
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if cache is None and loss_fn is None:
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")
    imp_dev = [to_device(b, device) for b in batches[:imp_batches]]
    gate_fn = monitor_fn if monitor_fn is not None else metric_fn

    g_start = gflops(student, adapter, device)
    if g_min is None:                                        # IZMJERI dno (usput napuni banned)
        g_min, banned = C.probe_min_gflops(student, adapter, device, prunable, banned=banned,
                                           gflops_fn=lambda m: gflops(m, adapter, device),
                                           on_progress=on_probe)
    g_min = min(float(g_min), g_start)
    delta = (g_start - g_min) / max(n_ckpt, 1)
    targets = [g_start - delta * (i + 1) for i in range(n_ckpt - 1)] + [g_min]

    st = MorphState(prunable, g_start, step_frac, reinvest_frac, cooldown)
    st.banned |= set(banned)
    opt, gstep, warmup = _new_prodigy(student), 0, len(batches)
    traj, ckpts = [], []
    step = 0

    def _emit(rec):
        traj.append(rec)
        if on_step:
            on_step(rec)

    _emit({"step": 0, "phase": "baseline", "rung": 0, "gflops": g_start,
           "params": A.count_params(student), "kd": None, "metric": None,
           "g_min": g_min, "target": None, "removed_ch": 0, "grown": [],
           "size_mb": A.count_params(student) * 4 / (1024 ** 2),
           "gflops_freed": 0.0, "gflops_reinvested": 0.0,
           "align_score": C.model_align_score(student), "align_best": st.align_best(student)})

    exhausted = None
    for i, t in enumerate(targets):
        last_rung = (i == len(targets) - 1)
        for _ in range(max_steps_per_rung):
            g_now = gflops(student, adapter, device)
            if g_now <= t + 1e-9:
                break
            step += 1
            # zadnji korak do precke se SUZAVA da je ne prebaci
            student, mi = morph_step(student, teacher, adapter, device, ctx, st, imp_dev,
                                     loss_fn=loss_fn, grow=not last_rung,
                                     step_target=min(st.step_target, g_now - t))
            if mi["n_rem"] == 0 and not mi["grown"] and not mi["grow_protected"]:
                exhausted = "nema vise rezivih kandidata na {:.4f} GFLOPs".format(g_now)
                break
            opt = _new_prodigy(student)                      # arh. promijenjena
            _ep = {"n": 0}

            def _on_ep(ep, kd, m, no_imp):
                _ep["n"] = ep
                n_par = A.count_params(student)
                _emit({"step": step, "phase": "morph" if ep == 1 else "ft", "rung": i + 1,
                       "gflops": gflops(student, adapter, device), "params": n_par, "kd": kd,
                       "metric": m, "g_min": g_min, "target": t, "ft_epoch": ep, "no_imp": no_imp,
                       "removed_ch": mi["n_rem"] if ep == 1 else 0,
                       "grown": [(gi["layer"], gi["k"]) for gi in mi["grown"]] if ep == 1 else [],
                       "banned": len(st.banned), "size_mb": n_par * 4 / (1024 ** 2),
                       "step_target": mi["step_target"], "est_freed": mi["est_freed"],
                       "r1_freed": mi["r1_freed"], "act_freed": mi["act_freed"],
                       "prune_rounds": mi["prune_rounds"],
                       "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
                       "align_score": C.model_align_score(student), "align_best": st.align_best(student),
                       "lr": lr_eff(opt)})

            gstep, _m, _n = ft_until(student, teacher, adapter, device, ctx, batches, cache, opt,
                                     gstep, warmup, gate_fn, CFG.F2_FT_MAX_EPOCHS,
                                     CFG.F2_FT_PATIENCE, loss_fn=loss_fn, on_epoch=_on_ep,
                                     on_batch=on_batch)
        # --- checkpoint (i kad je precka promasena zbog iscrpljenosti — sprema se sto JEST) ---
        g_ck = gflops(student, adapter, device)
        m_full = float(metric_fn(student)) if metric_fn is not None else None
        f = os.path.join(out_dir, "ckpt_{}.pt".format(i + 1))
        torch.save(student, f)                               # full-eager
        ckpts.append({"i": i + 1, "path": f, "target": t, "gflops": g_ck,
                      "params": int(A.count_params(student)), "metric": m_full,
                      "reached": g_ck <= t + 1e-9, "grow": not last_rung})
        if on_step:
            on_step({"step": step, "phase": "checkpoint", "rung": i + 1, "gflops": g_ck,
                     "params": int(A.count_params(student)), "metric": m_full, "target": t,
                     "g_min": g_min, "path": f, "reached": g_ck <= t + 1e-9,
                     "size_mb": A.count_params(student) * 4 / (1024 ** 2),
                     "gflops_freed": st.total_pruned, "gflops_reinvested": st.total_grown,
                     "align_score": C.model_align_score(student), "align_best": st.align_best(student)})
        if exhausted:
            break

    man = os.path.join(out_dir, "ladder.json")
    json.dump({"g_start": g_start, "g_min": g_min, "delta": delta, "targets": targets,
               "checkpoints": ckpts, "exhausted": exhausted}, open(man, "w"), indent=1)
    return {"g_start": g_start, "g_min": g_min, "delta": delta, "targets": targets,
            "checkpoints": ckpts, "manifest": man, "trajectory": traj, "exhausted": exhausted,
            "student": student, "banned": sorted(st.banned), "state": st}


def run_phase2(student, teacher, adapter, device, ctx, path, model_name, metric_fn=None,
               monitor_fn=None, batch_size=8, imp_batches=3, seed=0, batches=None, cache=None, **kw):
    """Runner za FAZU 2: ista priprema kao `run_phase1`, pa `phase2_ladder`."""
    if batches is None:                                      # pozivatelj ih moze dati (lancani run)
        batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, None, seed)
    if not kw.get("loss_fn"):
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    if cache is None and not kw.get("loss_fn"):
        cache = precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")
    if metric_fn is None:                                    # bez oznaka -> slaganje s uciteljem
        metric_fn, monitor_fn = agreement_metrics(teacher, adapter, device, ctx, path, seed=seed)
    return phase2_ladder(student, teacher, adapter, device, ctx, path, model_name, metric_fn,
                         monitor_fn=monitor_fn, batches=batches, cache=cache,
                         imp_batches=imp_batches, seed=seed, **kw)


# =========================== DEAD-REMOVAL + PUN CIKLUS (5.4) =========================== #
def dead_removal(student, adapter, device, ctx, batches, census_max=None):
    """Faza-1 ekvivalent: one-shot rez SVIH dead+near-dead izlaznih kanala (samo prunabilni, forward-safe).
    REUSE `compress.remove_dead_neardead`; loader = naši materijalizirani batchevi (`(imgs, None)`).
    Vrati (student, n_removed_kanala, n_layers, bad)."""
    struct = {nm: True for nm in ctx["prunable"]}
    loader = [(b, None) for b in batches]                    # remove_dead_neardead cita `for imgs,_ in loader`
    return C.remove_dead_neardead(student, adapter, device, loader, struct, census_max=census_max)


def full_cycle(student, teacher, adapter, device, ctx, path, model_name,
               target_frac=0.15, ft_steps=6, dead_ft_steps=8,
               batch_size=8, n_batches=None, imp_batches=3, seed=0, max_steps=None, on_step=None,
               dead=False, **kw):
    """PUN quality-gated ciklus: [opc. dead-removal + oporavak] → morph_loop (prune+grow) s REAL-METRIC gateom.

    LOSS = KD-core + TASK-UVJETNI enhaneri (auto po ctx['task'], v. dolje). GATE-ljestvica: prava task-metrika
    (metric_fn dana) > TEACHER-AGREEMENT (auto, label-free) — GT/labeli SAMO u gate-u, nikad u lossu ([[kd-only-no-gt]]).

    `dead=False` (DEFAULT, 5.6 nalaz): activation-frequency dead/near-dead removal je CENSUS-FRAGILAN i NIJE
    KD-siguran — na densnom izlazu (segmentacija) reže rijetko-opaljujuće kanale kritične za rijetke foreground
    klase (voc mIoU 0.47→0.10 od dead-removala samog, dok KD-grad prune petlja zadrži 74%). KD-grad prune GA
    SUBSUMIRA (truly-dead ima ~0 KD-grad → svejedno se reže prvi, ali output-sigurno). Uključi `dead=True` samo za
    ne-densne taskove gdje aktivacijska frekvencija pouzdano znači 'mrtav' i uz dovoljno velik census."""
    batches, _ = materialize_train_batches(path, adapter, device, ctx["split_plan"], batch_size, n_batches, seed)

    # LOSS = KD-core + TASK-UVJETNI enhaneri: ako task ima enhanere (SUPPORTED_TASKS) i postoji plug (enhancers.py),
    # auto-gradi loss_fn (npr. detekcija: dense_cls+box_giou+feature, decode preko morphology profila). Biran po
    # ctx['task'] (auto), NIKAD po imenu modela; ulazi i u VAZNOST i u FT (inace prune rezuje task-kriticne kanale).
    if not kw.get("loss_fn"):
        import enhancers as _ENH
        efn = _ENH.enhancer_loss_fn(ctx, teacher)
        if efn is not None:
            kw["loss_fn"] = efn
    # generic teacher-cache SAMO kad nema enhaner-loss-a (detekcijski loss_fn ima vlastite teacher signale + izlaz
    # frcnn-a je lista dict-ova pa generic cache ionako ne stoji)
    cache = None if kw.get("loss_fn") else precompute_teacher(teacher, adapter, batches, ctx["taps"], model_name, split="train")

    # (opcijski) dead-removal + oporavak — default OFF (nije KD-siguran; KD-grad prune ga subsumira)
    n_dead = n_lay = 0
    if dead:
        student, n_dead, n_lay, _bad = dead_removal(student, adapter, device, ctx, batches)
        if dead_ft_steps:
            prune_ft_recover(student, teacher, adapter, device, ctx, batches, cache, dead_ft_steps, loss_fn=kw.get("loss_fn"))

    # GATE-ljestvica: prava metrika (metric_fn dana) > TEACHER-AGREEMENT (auto, label-free). Bez metric_fn ->
    # auto-agreement (slaganje s učiteljem na sirovim ulazima, bez oznaka).
    if not kw.get("metric_fn"):
        kw["metric_fn"] = agreement_metrics(teacher, adapter, device, ctx, path)[0]
        # NE namecemo drugi prag: `metric_tol` ostaje ono sto je korisnik zadao (ili FT_RECOVERY_FRAC).
        # Kad nema task-metrike GUI SUGERIRA strozi (AGREEMENT_SUGGEST), ali odluka je korisnikova.

    res = morph_loop(student, teacher, adapter, device, ctx, path, model_name,
                     target_frac=target_frac, ft_steps=ft_steps, batches=batches, cache=cache,
                     imp_batches=imp_batches, max_steps=max_steps, on_step=on_step, **kw)
    res.update({"n_dead": n_dead, "n_dead_layers": n_lay})
    if res["best_model"] is None:                            # ništa nije prošlo gate (rijetko) -> uzmi zadnji
        res["best_model"] = res["student"]; res["best_gflops"] = res["final_gflops"]
    return res
