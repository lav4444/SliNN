"""
run_pipeline_pruning.py (pipeline_experiment2) — PRUNING, A-PARADIGMA, PURE-KD (BEZ GT) + RPN-KD.
Usporedba: pure LOGIT-KD vs pure FEATURE+LOGIT-KD (sve ostalo fiksno) — sada s RPN-KD u OBJE.

Model: fasterrcnn_mobilenet_v3_large_320_fpn (COCO-mapped). Ucitelj = zamrznuti nepruned self.

PURE-KD (bez GT, bez det lossa):
  * logit-KD: KL(ROI cls-logiti) + box-L1, na UCITELJEVIM proposalima (trenira glave + backbone).
  * feature-KD (opcionalno, povrh): MSE na FPN mapama (sidri backbone). SAM ne bi trenirao glave!
  * RPN-KD (NOVO, u obje): destilira teacher RPN glavu (objectness BCE + box-L1) na ISTIM
    anchorima/gridovima -> studentov RPN dobiva gradijent i prati pomaknute FPN mape nakon reza.
    RPN se NE pruna (i dalje u ignored_layers), ali se SADA fine-tuna zajedno s ostatkom.
  * Zato je LOGIT u OBJE varijante; feature-only nema smisla (glave bez signala).
  Studentovi izlazi se vade RUCNO (student.transform + backbone + rpn.head + roi_heads), bez targeta.

CISTI GRADIENT PRUNING (gate samo tranzientno za mjerenje):
  * kriterij reza = |d(KD)/dgate| (gradijent stvarnog KD cilja te varijante).
  * stvarni tp rez svaku epohu (~3%/ep do 30%), pa konsolidacija.

A-PARADIGMA: jedan kontinuirani run (SGD+warmup-cosine), optimizer rebuild samo na rez-eventima.
KD ucitelj se kesira (compute 1. epohu, reuse svaku). BEZ BN reseta u evalu (BN se sam azurira train-modom).

Izlaz: results.txt + summary.json + pipeline.png + pruned_models/{logit,featlogit}.pt (eager)
Pokretanje:  conda activate dipl && python run_pipeline_pruning.py
"""

import sys
import copy
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

HERE = Path(__file__).parent
EXP3_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment3"
for _p in (str(HERE), EXP3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common3 as C                  # noqa: E402
import run_experiment3 as R          # noqa: E402
import inloss_lib as IL              # noqa: E402


# =========================== KONFIGURACIJA =========================== #
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "pruned_models"
# (ime, use_feature, use_logit, use_rpn)  — logit u obje (glave); feature samo u prvoj; RPN-KD u obje
KD_VARIANTS = [("featlogit", True, True, True), ("logit", False, True, True)]
DEVICE = R.DEVICE

KEEP_FINAL = 0.70                # zadrzi 70% GFLOPs (izbaci 30% GFLOPs) — budzet u GFLOPs
PRUNE_EPOCHS = 10               # faza reza: ~3%/ep, stvarni tp rez/epohi
TOTAL_EPOCHS = 20
CONSOLIDATION_PATIENCE = 5
BATCH_SIZE = 16
NUM_WORKERS = 2
LR_MODEL = 0.005
W_FEAT = 1.0                    # tezina feature-MSE (kad use_feature)
W_RPN = 1.0                     # tezina RPN-KD (objectness BCE + box-L1) — destilira teacher RPN glavu
LOGIT_T = 1.0                  # temperatura logit-KD
LOGIT_BOX_W = 1.0             # tezina box-L1 u logit-KD
GRAD_CLIP = 5.0
GRAD_CALIB_BATCHES = 8        # za gradijentni kriterij |d(KD)/dgate|
MON_VAL_IMAGES = 400
SAVE_MODELS = True


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def move_imgs(imgs):
    return [im.to(DEVICE, non_blocking=True) for im in imgs]


def gflops(model):
    """GFLOPs = 2*MAC (konvencija kroz cijeli projekt). Za fasterrcnn = backbone GFLOPs
    (puni RPN/ROI su dinamicni pa se ne broje); backbone je statacki proxy."""
    return 2.0 * C.backbone_gmacs(model, DEVICE)


def build_fixed_train_loader():
    import random as _random
    from torch.utils.data import DataLoader
    ds = C.DetDataset("train", drop_empty=True)
    _random.Random(42).shuffle(ds.items)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=True, collate_fn=C.det_collate, persistent_workers=NUM_WORKERS > 0)


# =========================== PURE-KD (manualni forward, bez GT) =========================== #
class KD:
    """Pure-KD: feature (FPN-MSE) i/ili logit (ROI cls-KL + box-L1) prema zamrznutom
    ucitelju. Studentovi izlazi se racunaju RUCNO (transform+backbone+roi), bez targeta,
    pa NE treba GT. Ucitelj nepromjenjiv -> izlazi se kesiraju."""
    def __init__(self, teacher, use_feature, use_logit, use_rpn=False, T=1.0, box_w=1.0, w_feat=1.0, w_rpn=1.0):
        self.teacher = teacher; self.use_feature = use_feature; self.use_logit = use_logit
        self.use_rpn = use_rpn
        self.T = T; self.box_w = box_w; self.w_feat = w_feat; self.w_rpn = w_rpn

    @torch.no_grad()
    def compute_teacher(self, imgs):
        t_imgs, _ = self.teacher.transform(imgs)
        t_feat = self.teacher.backbone(t_imgs.tensors)
        out = {"sizes": list(t_imgs.image_sizes)}
        if self.use_feature:
            out["feat"] = {k: v.detach() for k, v in t_feat.items()}
        if self.use_rpn:
            # sirovi izlazi RPN glave po FPN razini: objectness [N,A,H,W], box-deltas [N,A*4,H,W]
            t_obj, t_box = self.teacher.rpn.head(list(t_feat.values()))
            out["rpn_obj"] = [o.detach() for o in t_obj]; out["rpn_box"] = [b.detach() for b in t_box]
        if self.use_logit:
            props, _ = self.teacher.rpn(t_imgs, t_feat)
            bf = self.teacher.roi_heads.box_roi_pool(t_feat, props, t_imgs.image_sizes)
            cls, box = self.teacher.roi_heads.box_predictor(self.teacher.roi_heads.box_head(bf))
            out["props"] = [p.detach() for p in props]; out["cls"] = cls.detach(); out["box"] = box.detach()
        return out

    def store(self, td):
        e = {"sizes": td["sizes"]}
        if self.use_feature:
            e["feat"] = {k: v.half().cpu() for k, v in td["feat"].items()}
        if self.use_rpn:
            e["rpn_obj"] = [o.half().cpu() for o in td["rpn_obj"]]; e["rpn_box"] = [b.half().cpu() for b in td["rpn_box"]]
        if self.use_logit:
            e["props"] = [p.cpu() for p in td["props"]]; e["cls"] = td["cls"].half().cpu(); e["box"] = td["box"].half().cpu()
        return e

    def load(self, e):
        d = {"sizes": e["sizes"]}
        if self.use_feature:
            d["feat"] = {k: v.to(DEVICE) for k, v in e["feat"].items()}
        if self.use_rpn:
            d["rpn_obj"] = [o.float().to(DEVICE) for o in e["rpn_obj"]]; d["rpn_box"] = [b.float().to(DEVICE) for b in e["rpn_box"]]
        if self.use_logit:
            d["props"] = [p.to(DEVICE) for p in e["props"]]; d["cls"] = e["cls"].float().to(DEVICE); d["box"] = e["box"].float().to(DEVICE)
        return d

    def loss(self, student, imgs, td):
        """RUCNI studentov forward (s gradom) -> feature MSE i/ili RPN-KD i/ili logit KL+L1. Vrati (total, info)."""
        s_imgs, _ = student.transform(imgs)
        s_feat = student.backbone(s_imgs.tensors)
        total = torch.zeros((), device=DEVICE)
        info = {}
        if self.use_feature:
            keys = list(s_feat.keys()) if hasattr(s_feat, "keys") else range(len(s_feat))
            fl = sum(F.mse_loss(s_feat[k], td["feat"][k].to(s_feat[k].dtype)) for k in keys) / max(len(keys), 1)
            total = total + self.w_feat * fl; info["feat"] = float(fl)
        if self.use_rpn:
            # studentova RPN glava na istim FPN gridovima -> isti shape kao teacher (anchori fiksni).
            s_obj, s_box = student.rpn.head(list(s_feat.values()))
            obj_l = sum(F.binary_cross_entropy_with_logits(so, torch.sigmoid(to.to(so.dtype)))
                        for so, to in zip(s_obj, td["rpn_obj"])) / max(len(s_obj), 1)
            box_l = sum(F.smooth_l1_loss(sb, tb.to(sb.dtype))
                        for sb, tb in zip(s_box, td["rpn_box"])) / max(len(s_box), 1)
            rpn_l = obj_l + box_l
            total = total + self.w_rpn * rpn_l; info["rpn"] = float(rpn_l)
        if self.use_logit:
            s_bf = student.roi_heads.box_roi_pool(s_feat, td["props"], s_imgs.image_sizes)
            s_cls, s_box = student.roi_heads.box_predictor(student.roi_heads.box_head(s_bf))
            kl = F.kl_div(F.log_softmax(s_cls / self.T, -1), F.softmax(td["cls"] / self.T, -1),
                          reduction="batchmean") * (self.T * self.T)
            l1 = F.smooth_l1_loss(s_box, td["box"])
            total = total + kl + self.box_w * l1; info["logit"] = float(kl + self.box_w * l1)
        return total, info


# =========================== Gradijentni kriterij |d(KD)/dgate| =========================== #
def measure_gate_grad(student, gc, kd, calib):
    """Akumuliraj |d(KD)/dgate| (gradijent KD cilja varijante) preko calib batcheva.
    Gate (tranzientno attach-an) mnozi izlaze -> dL/dgate_c = sum(akt*grad)."""
    student.train(); gc.train()
    accum = {key: torch.zeros_like(gc.gate[key]) for key in gc.apply_mod}
    nb = 0
    for imgs, _targets in calib:
        imgs = move_imgs(imgs)
        for p in student.parameters():
            p.grad = None
        for key in gc.gate:
            gc.gate[key].grad = None
        td = kd.compute_teacher(imgs)
        L, _ = kd.loss(student, imgs, td)
        L.backward()
        for key in gc.apply_mod:
            g = gc.gate[key].grad
            if g is not None:
                accum[key] += g.detach().abs()
        nb += 1
    for p in student.parameters():
        p.grad = None
    return {key: (accum[key] / max(nb, 1)).detach() for key in accum}


def materialize_to(student, target_gflops):
    target = target_gflops      # APSOLUTNI ciljni GFLOPs (relativno na original); binarno trazi tp ratio

    def trial_ratio(ratio):
        m = copy.deepcopy(student)
        pr = tp.pruner.MetaPruner(
            m, example_inputs=R.EXAMPLE(),
            importance=tp.importance.MagnitudeImportance(p=1, normalizer=None, group_reduction="mean",
                                                         target_types=[nn.BatchNorm2d, nn.Linear]),
            pruning_ratio=ratio, global_pruning=R.GLOBAL_PRUNING, round_to=R.ROUND_TO,
            ignored_layers=C.prunable_ignored_layers(m))
        pr.step()
        p = gflops(m); del m, pr
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return p

    lo, hi, best_r, best_d = 0.0, 0.95, 0.0, float("inf")   # lo=0 -> moze i sitne rezove (3% GFLOPs)
    for _ in range(R.RATIO_SEARCH_ITERS):
        mid = (lo + hi) / 2
        p = trial_ratio(mid)
        if abs(p - target) < best_d:
            best_d = abs(p - target); best_r = mid
        if p > target:
            lo = mid
        else:
            hi = mid
    pruner = tp.pruner.MetaPruner(
        student, example_inputs=R.EXAMPLE(),
        importance=tp.importance.MagnitudeImportance(p=1, normalizer=None, group_reduction="mean",
                                                     target_types=[nn.BatchNorm2d, nn.Linear]),
        pruning_ratio=best_r, global_pruning=R.GLOBAL_PRUNING, round_to=R.ROUND_TO,
        ignored_layers=C.prunable_ignored_layers(student))
    pruner.step()
    student.eval()
    return best_r


def make_optim(student):
    return torch.optim.SGD([p for p in student.parameters() if p.requires_grad],
                           lr=LR_MODEL, momentum=0.9, weight_decay=5e-4)


# =========================== JEDNA KD VARIJANTA (pure-KD A-pruning) =========================== #
def run_variant(name, use_feature, use_logit, use_rpn, train_loader, mon_val, eval_loaders, calib, base_params, base_gflops):
    append_block("=" * 80)
    tag = ("feature+logit" if use_feature else "logit-only") + ("+rpn" if use_rpn else "")
    append_block(f"KD VARIJANTA: {name.upper()}  (pure-KD: {tag}, BEZ GT | kriterij=gradient |d(KD)/dgate|)")

    teacher = C.build_model(pretrained=True, coco_map=True).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = C.build_model(pretrained=True, coco_map=True).to(DEVICE)
    kd = KD(teacher, use_feature, use_logit, use_rpn=use_rpn, T=LOGIT_T, box_w=LOGIT_BOX_W, w_feat=W_FEAT, w_rpn=W_RPN)

    optim = make_optim(student)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    steps_per_epoch = max(1, len(train_loader))
    total_steps = TOTAL_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch
    tcache = []

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    start_eval = C.evaluate(student, mon_val, DEVICE)
    append_block(f"  START params={base_params:,} val mAP={start_eval['map']:.4f}")

    best_map = -1.0; best_state = None; no_imp = 0; gstep = 0
    history = [{"epoch": 0, "keep_frac": 1.0, "val_map": start_eval["map"], "pruned": False}]

    for epoch in range(1, TOTAL_EPOCHS + 1):
        pruned_now = False
        if epoch <= PRUNE_EPOCHS:
            target_frac = (1.0 - KEEP_FINAL) * epoch / PRUNE_EPOCHS    # kumulativna frakcija GFLOPs za maknuti
            gc = IL.GateController(student, mode="l1").to(DEVICE)
            IL.compute_costs(gc, student, DEVICE); gc.attach()
            total_macs = sum(float(getattr(gc, f"macs__{k}").sum()) for k in gc.apply_mod)
            score = measure_gate_grad(student, gc, kd, calib[:GRAD_CALIB_BATCHES])
            gate_pool = 1.5 * ((1.0 - KEEP_FINAL) / PRUNE_EPOCHS) * total_macs   # kandidat-pool za OVAJ korak (+50% rezerva)
            gc.prune_by_external_to_removed(score, gate_pool, metric="macs")     # gate budzet u FLOPs (macs)
            gc.reset_survivor_gates(); gc.fold_into_weights(); gc.remove(); del gc
            target_gflops = base_gflops * (1.0 - target_frac)
            ratio = materialize_to(student, target_gflops)
            optim = make_optim(student)                              # rebuild (tp promijenio tenzore)
            cur = C.count_params(student); gf = gflops(student); pruned_now = True
            append_block(f"  [REZ ep{epoch:2d}] tp ratio={ratio:.3f} -> GFLOPs {gf:.3f} "
                         f"(cilj {target_gflops:.3f}, {(1-target_frac)*100:.0f}%) | params {cur:,} ({cur/base_params*100:.1f}%)")

        student.train()
        t_ep = time.time(); rf = rl = rr = 0.0; nb = 0
        for bi, (imgs, _targets) in enumerate(train_loader):
            imgs = move_imgs(imgs)
            if bi < len(tcache):
                td = kd.load(tcache[bi])
            else:
                td = kd.compute_teacher(imgs)
                tcache.append(kd.store(td))
            f = lr_factor(gstep)
            for grp in optim.param_groups:
                grp["lr"] = LR_MODEL * f
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss, info = kd.loss(student, imgs, td)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optim); scaler.update()
            rf += info.get("feat", 0.0); rl += info.get("logit", 0.0); rr += info.get("rpn", 0.0); nb += 1
            gstep += 1

        val = C.evaluate(student, mon_val, DEVICE)
        cur = C.count_params(student); gf = gflops(student)
        phase = "rez" if epoch <= PRUNE_EPOCHS else "konsolidacija"
        track = epoch >= PRUNE_EPOCHS
        improved = track and val["map"] > best_map + 1e-4
        append_block(f"  [ep{epoch:2d}/{TOTAL_EPOCHS} {phase:13s}] feat={rf/nb:.3f} rpn={rr/nb:.3f} logit={rl/nb:.3f} "
                     f"| params={cur/1e6:.2f}M ({cur/base_params*100:.1f}%) GFLOPs={gf:.3f} | val_mAP={val['map']:.4f} "
                     f"| lr={LR_MODEL*lr_factor(gstep):.2e} | {time.time()-t_ep:.0f}s{'  *best' if improved else ''}")
        history.append({"epoch": epoch, "keep_frac": cur / base_params, "gflops": gf, "val_map": val["map"], "pruned": pruned_now})
        if improved:
            best_map = val["map"]; best_state = copy.deepcopy(student.state_dict()); no_imp = 0
        elif track:
            no_imp += 1
        if epoch > PRUNE_EPOCHS and no_imp >= CONSOLIDATION_PATIENCE:
            append_block(f"  >>> early-stop (konsolidacija): val mAP ne raste {CONSOLIDATION_PATIENCE} ep")
            break

    if best_state is not None:
        student.load_state_dict(best_state)
    fin = R.eval_all(student, eval_loaders)
    bench = C.benchmark_latency(student)
    dead = C.count_dead(student, calib, DEVICE)
    fin_params = C.count_params(student); fin_gflops = gflops(student)
    append_block(f"  --- FINALNO ({name}) params={fin_params:,} ({fin_params/base_params*100:.1f}%) GFLOPs={fin_gflops:.3f} ---")
    append_block(R.fmt_eval(fin))
    append_block(f"  GPU={bench['cuda']:.2f} ms/img CPU={bench['cpu']:.2f} ms/img "
                 f"mrtvi filteri={dead['dead_filters']}/{dead['total_filters']}")
    append_block("")

    if SAVE_MODELS:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(student, MODELS_DIR / f"{name}.pt")

    out = {"params": fin_params, "gflops": fin_gflops, "val_map": fin["val"]["map"],
           "test_map": fin["test"]["map"], "gpu_ms": bench["cuda"], "cpu_ms": bench["cpu"],
           "dead_filters": dead["dead_filters"], "total_filters": dead["total_filters"],
           "best_val_map": best_map, "history": history}
    del teacher, student, kd
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


# =========================== MAIN =========================== #
def main():
    t0 = time.time()
    RESULTS_FILE.write_text("")
    append_block("=" * 80)
    append_block("PRUNING pipeline_experiment2 — A-PARADIGMA, PURE-KD (BEZ GT) + RPN-KD | pure logit vs pure feat+logit")
    append_block("Model: fasterrcnn_mobilenet_v3_large_320_fpn | sub10k_open_images_v7 (6 kl)")
    append_block("=" * 80)
    append_block(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Rez: izbaci {(1-KEEP_FINAL)*100:.0f}% GFLOPs, ~3%/ep kroz {PRUNE_EPOCHS} ep (budzet u GFLOPs), konsolidacija do {TOTAL_EPOCHS} ep | "
                 f"SGD lr={LR_MODEL}+cosine | kriterij=gradient |d(KD)/dgate| | logit T={LOGIT_T} box_w={LOGIT_BOX_W} w_feat={W_FEAT} w_rpn={W_RPN}")
    append_block(f"RPN-KD: u obje varijante (RPN se NE pruna, ali se fine-tuna: objectness BCE + box-L1 prema teacher RPN glavi)")
    append_block(f"Varijante (pure-KD, bez GT): {', '.join(n for n,*_ in KD_VARIANTS)}")
    append_block("")

    train_loader = build_fixed_train_loader()
    mon_val = C.make_loader("val", BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, max_images=MON_VAL_IMAGES)
    eval_loaders = {s: C.make_loader(s, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                     max_images=R.MAX_EVAL_IMAGES) for s in R.EVAL_SPLITS}
    from torch.utils.data import DataLoader as _DL
    calib_ds = C.DetDataset("train", max_images=GRAD_CALIB_BATCHES * BATCH_SIZE, drop_empty=True)
    calib = list(_DL(calib_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                     pin_memory=False, collate_fn=C.det_collate))

    base = C.build_model(pretrained=True, coco_map=True).to(DEVICE).eval()
    base_params = C.count_params(base); base_gflops = gflops(base)
    base_eval = R.eval_all(base, eval_loaders)
    append_block("-" * 80)
    append_block(f"BASELINE (COCO-mapped, nepruned): params={base_params:,} GFLOPs={base_gflops:.3f}")
    append_block(R.fmt_eval(base_eval))
    append_block("")
    summary = {"baseline": {"params": base_params, "gflops": base_gflops,
                            "val_map": base_eval["val"]["map"], "test_map": base_eval["test"]["map"]}}
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for name, uf, ul, ur in KD_VARIANTS:
        print(f"\n>>> KD: {name}")
        summary[name] = run_variant(name, uf, ul, ur, train_loader, mon_val, eval_loaders, calib, base_params, base_gflops)

    append_block("=" * 80)
    append_block("USPOREDBA — pure-KD + RPN-KD: logit-only vs feature+logit")
    append_block("=" * 80)
    append_block(f"{'pristup':<22}{'params':>9}{'GFLOPs':>9}{'val mAP':>9}{'test mAP':>9}{'GPUms':>7}{'deadF':>10}")
    append_block("-" * 80)
    b = summary["baseline"]
    append_block(f"{'baseline':<22}{b['params']/1e6:>8.2f}M{b['gflops']:>9.3f}{b['val_map']:>9.4f}{b['test_map']:>9.4f}{'-':>7}{'-':>10}")
    for name, _, _, _ in KD_VARIANTS:
        s = summary[name]
        append_block(f"{'pure '+name:<22}{s['params']/1e6:>8.2f}M{s['gflops']:>9.3f}{s['val_map']:>9.4f}"
                     f"{s['test_map']:>9.4f}{s['gpu_ms']:>7.2f}{s['dead_filters']:>4}/{s['total_filters']:<5}")
    append_block("-" * 80)

    plot_compare(summary)
    (HERE / "summary.json").write_text(json.dumps(
        {k: ({kk: vv for kk, vv in v.items() if kk != "history"} if isinstance(v, dict) else v)
         for k, v in summary.items()} | {"history": {n: summary[n]["history"] for n, *_ in KD_VARIANTS}}, indent=2))
    append_block(f"\nUKUPNO VRIJEME: {(time.time()-t0)/60:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'} | {HERE/'pipeline.png'}")


def plot_compare(summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"logit": "C1", "featlogit": "C0"}
    for name, *_ in KD_VARIANTS:
        h = summary[name]["history"]
        eps = [x["epoch"] for x in h]; vm = [x["val_map"] for x in h]
        ax.plot(eps, vm, "-o", color=colors.get(name, "C2"), ms=4, label=f"pure {name} val mAP")
    ax.axvspan(0.5, PRUNE_EPOCHS + 0.5, color="C2", alpha=0.06, label="prune phase")
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP@[.5:.95]")
    ax.set_title("A-paradigm pure-KD + RPN-KD pruning — logit-only vs feature+logit")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = HERE / "pipeline.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    main()
