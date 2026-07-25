"""
run_pipeline_experiment.py (pipeline_experiment1) — NETWORK GROWING, A-PARADIGMA.

Samo GradMax (gradient), StudentYOLO, start = pruned taylor.pt (50%).

A-PARADIGMA (kontinuirani rast, za razliku od B = grow->FT-do-platoa po ciklusu):
  * JEDAN kontinuirani trening, JEDAN LR-raspored (warmup+cosine), bez reset-a/platoa.
  * Rast je DOGADJAJ u petlji: tijekom prvih GROW_EPOCHS epoha, na pocetku svake,
    izmjeri GradMax benefit na ZIVOM modelu -> alociraj MALI prirast (benefit/FLOP,
    per-layer cap) -> grow_layer (function-preserving) -> nastavi trenirati.
  * Nakon faze rasta: KONSOLIDACIJA (cist trening pune mreze do kraja / early-stop).

Ogranicenja rasta (dogovoreno):
  * cilj +30% pocetnih GFLOPs (TARGET_FACTOR=1.3)
  * ~+3%/epohi (MAX_GROW_PCT), linearna rampa do cilja kroz GROW_EPOCHS (~10)
  * per-layer cap +25% trenutne sirine po eventu (PER_LAYER_CAP)

LR/optimizer: AdamW + warmup-cosine (originalni student recept iz train_kd). LR se
racuna iz GLOBALNOG koraka (lr_at) i postavlja svaki korak -> kontinuiran i kad se
optimizer rebuilda. Optimizer se rebuilda SAMO na growth-eventima (jer grow_layer
zamjenjuje module); konsolidacija ima i puni momentum-kontinuitet.

Izlaz: results.txt + summary.json + pipeline.png + grown_models/GradMax-pipeline.pt
Pokretanje:  conda activate dipl && python run_pipeline_experiment.py
"""

import sys
import io
import copy
import json
import math
import time
import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

HERE = Path(__file__).parent
STUDENT_DIR = "/home/tomi/code/dipl/custom_models/student_2_m"
PURE_KD_DIR = "/home/tomi/code/dipl/custom_models/student_2_m/pure_KD"
EXP1_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment"
for _p in (str(HERE), PURE_KD_DIR, STUDENT_DIR, EXP1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_kd as TK                 # noqa: E402
import evaluate_student as EV         # noqa: E402
import pruning_lib as PL              # noqa: E402
import growing_lib as G               # noqa: E402
import criteria as C                  # noqa: E402


# =========================== KONFIGURACIJA =========================== #
START_CKPT = Path(EXP1_DIR) / "pruned_models" / "taylor.pt"
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "grown_models"
TECHNIQUE = "gradient"               # GradMax
DISP = "GradMax-pipeline"

TARGET_FACTOR = 1.3                  # cilj = 1.3x pocetni GFLOPs (+30%)
TOTAL_EPOCHS = 35                    # ukupni budzet treninga (rast + konsolidacija)
GROW_EPOCHS = 10                     # faza rasta: 1 event na pocetku svake od prvih 10 epoha
MAX_GROW_PCT = 0.03                  # tvrda granica +3% GFLOPs po eventu
PER_LAYER_CAP = 0.25                 # sloj smije +25% trenutne sirine po eventu
CALIB_BATCHES = 16                   # za GradMax benefit (svaki event, na zivom modelu)

LR = TK.LR                           # 1e-3 (isti recept kao train_kd)
WEIGHT_DECAY = TK.WEIGHT_DECAY       # 5e-4
WARMUP_EPOCHS = TK.WARMUP_EPOCHS     # 2
CONSOLIDATION_PATIENCE = 8           # early-stop SAMO nakon faze rasta (val mAP ne raste)

MAX_EVAL_IMAGES = 400
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNPRUNED_REF = (11.553, 0.1858)


# cap slika u EV.eval_split / benchmark
_orig_list_images = EV.list_images
EV.list_images = lambda d: _orig_list_images(d)[:MAX_EVAL_IMAGES] if MAX_EVAL_IMAGES else _orig_list_images(d)


# =========================== POMOCNE =========================== #
def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def val_map(model):
    with quiet():
        return EV.eval_split(model, "val", DEVICE)["map"]


def make_loaders():
    train_ds = TK.KDTrainDataset(TK.TRAIN_IMG_DIR, TK.TEACHER_SOFT_DIR)
    val_ds = TK.ValDataset(TK.VAL_IMG_DIR, TK.VAL_LBL_DIR, TK.VAL_TEACHER_SOFT_DIR)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=TK.BATCH_SIZE, shuffle=True, num_workers=TK.NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=TK.NUM_WORKERS > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=TK.BATCH_SIZE, shuffle=False, num_workers=TK.NUM_WORKERS,
        pin_memory=True, collate_fn=TK.val_collate, persistent_workers=TK.NUM_WORKERS > 0)
    return train_loader, val_loader


@torch.no_grad()
def count_dead(model, calib, device, eps=1e-6):
    acts = {}; handles = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.SiLU):
            def hook(mm, i, o, key=nm):
                ch = o.amax(dim=(0, 2, 3)) if o.dim() == 4 else o.amax(0)
                prev = acts.get(key)
                acts[key] = ch.detach() if prev is None else torch.maximum(prev, ch.detach())
            handles.append(m.register_forward_hook(hook))
    model.eval()
    for b in calib:
        model(b["image"].to(device, non_blocking=True))
    for h in handles:
        h.remove()
    dead = tot = 0
    for _, mx in acts.items():
        dead += int((mx <= eps).sum()); tot += mx.numel()
    return dead, tot


# =========================== A-paradigma: jedan growth event =========================== #
def grow_one_event(model, budget_flops, calib, loss_fn, device, cap_frac):
    """Jedan diskretni rast na ZIVOM modelu: GradMax benefit -> allocate(budget) ->
    per-layer cap -> grow_layer (function-preserving). Vrati (plan, overhead)."""
    benefit, imp, ov = C.layer_benefit(model, TECHNIQUE, calib, loss_fn, device)
    init_fn = C.make_init_fn(imp, TECHNIQUE)
    cost = C.flops_per_filter(model, device)
    plan = C.allocate(benefit, cost, budget_flops)
    _, by_name = PL.build_plan(model)
    widths = G.current_widths(by_name)
    plan = {L: min(k, max(1, int(round(cap_frac * widths[L])))) for L, k in plan.items()}
    plan = {L: k for L, k in plan.items() if k > 0}
    if plan:
        G.grow_many(model, plan, init_fn)
    return plan, ov


# =========================== LR raspored (global, neovisan o optimizeru) =========================== #
def make_lr_fn(total_steps, warmup_steps):
    def lr_at(step):
        if step < warmup_steps:
            return LR * (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return LR * 0.5 * (1.0 + math.cos(math.pi * prog))
    return lr_at


def new_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)


# =========================== MAIN =========================== #
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    RESULTS_FILE.write_text("")
    append_block("=" * 78)
    append_block("NETWORK GROWING pipeline_experiment1 — GradMax, A-PARADIGMA (kontinuirani rast)")
    append_block("Polaziste: pruned taylor.pt (50%) | dataset: sub10k_open_images_v7")
    append_block("=" * 78)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Optimizer: AdamW lr={LR:.1e} wd={WEIGHT_DECAY:.0e} | warmup {WARMUP_EPOCHS} ep + cosine (kontinuiran)")
    append_block(f"Rast: cilj +{(TARGET_FACTOR-1)*100:.0f}% GFLOPs | faza rasta {GROW_EPOCHS} ep "
                 f"(<= +{MAX_GROW_PCT*100:.0f}%/ep, per-layer cap +{PER_LAYER_CAP*100:.0f}%/event) | "
                 f"konsolidacija do {TOTAL_EPOCHS} ep (early-stop patience {CONSOLIDATION_PATIENCE})")
    append_block("")

    model = G.load_pruned_model(START_CKPT, DEVICE)
    f0 = PL.count_flops(model, DEVICE); p0 = PL.count_params(model)
    target = f0 * TARGET_FACTOR
    train_loader, val_loader = make_loaders()
    calib_eval = C.make_calib_batches(8, DEVICE)
    loss_fn = C.make_kd_loss_fn(DEVICE)

    start_map = val_map(model)
    with quiet():
        test0 = EV.eval_split(model, "test", DEVICE)["map"]
    append_block("-" * 78)
    append_block("START (pruned taylor 50%)")
    append_block(f"  params={p0:,}  GFLOPs={f0/1e9:.3f}  val mAP={start_map:.4f}  test mAP={test0:.4f}  "
                 f"cilj GFLOPs={target/1e9:.3f}")
    append_block("")

    iters_per_epoch = len(train_loader)
    total_steps = TOTAL_EPOCHS * iters_per_epoch
    warmup_steps = WARMUP_EPOCHS * iters_per_epoch
    lr_at = make_lr_fn(total_steps, warmup_steps)
    optimizer = new_optimizer(model, lr_at(0))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    axy, ast = model.anchor_xy, model.anchor_stride

    best_map = -1.0
    best_model = copy.deepcopy(model)
    no_imp = 0
    gstep = 0
    # trajektorija po epohi: (epoch, gflops, val_map)
    traj = [{"epoch": 0, "gflops": f0 / 1e9, "val_map": start_map, "grew": False}]

    for epoch in range(1, TOTAL_EPOCHS + 1):
        grew = False
        # --- growth event (pocetak epohe, faza rasta) ---
        if epoch <= GROW_EPOCHS:
            cur = PL.count_flops(model, DEVICE)
            if cur < 0.98 * target:
                events_left = GROW_EPOCHS - (epoch - 1)
                remaining = target - cur
                budget = min(remaining / max(events_left, 1), MAX_GROW_PCT * cur)
                calib = C.make_calib_batches(CALIB_BATCHES, DEVICE)
                plan, ov = grow_one_event(model, budget, calib, loss_fn, DEVICE, PER_LAYER_CAP)
                if plan:
                    grew = True
                    optimizer = new_optimizer(model, lr_at(gstep))   # rebuild (moduli zamijenjeni)
                    gf = PL.count_flops(model, DEVICE); pr = PL.count_params(model)
                    axy, ast = model.anchor_xy, model.anchor_stride
                    append_block(f"  [GROW ep{epoch:2d}] +{sum(plan.values())} filtera u {len(plan)} slojeva "
                                 f"-> GFLOPs {cur/1e9:.3f}->{gf/1e9:.3f} (cilj {target/1e9:.3f}) "
                                 f"params {pr:,} | budget {budget/1e9:.3f} | overhead {ov['time_ms_per_image']:.1f} ms/img")

        # --- trening epohe (kontinuirano) ---
        model.train()
        t_ep = time.time()
        run_loss = run_cls = run_box = 0.0; nb = 0
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            tb = batch["teacher_boxes"].to(DEVICE, non_blocking=True)
            tp = batch["teacher_probs"].to(DEVICE, non_blocking=True)
            lr_now = lr_at(gstep)
            for grp in optimizer.param_groups:
                grp["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                raw = model(imgs)
                loss, lcls, lbox = TK.kd_loss(raw, tb, tp, axy, ast)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), TK.GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            run_loss += float(loss); run_cls += float(lcls); run_box += float(lbox); nb += 1
            gstep += 1

        vm = val_map(model)
        gf = PL.count_flops(model, DEVICE)
        phase = "rast" if epoch <= GROW_EPOCHS else "konsolidacija"
        improved = vm > best_map + 1e-4
        append_block(f"  [ep{epoch:2d}/{TOTAL_EPOCHS} {phase:13s}] loss={run_loss/nb:.4f} "
                     f"(cls={run_cls/nb:.4f} box={run_box/nb:.4f}) | val_mAP={vm:.4f} "
                     f"| GFLOPs={gf/1e9:.3f} | lr={lr_at(gstep):.2e} | {time.time()-t_ep:.0f}s"
                     f"{'  *best' if improved else ''}")
        traj.append({"epoch": epoch, "gflops": gf / 1e9, "val_map": vm, "grew": grew})

        if improved:
            best_map = vm; best_model = copy.deepcopy(model); no_imp = 0
        else:
            no_imp += 1
        if epoch > GROW_EPOCHS and no_imp >= CONSOLIDATION_PATIENCE:
            append_block(f"  >>> early-stop (konsolidacija): val mAP ne raste {CONSOLIDATION_PATIENCE} ep")
            break

    # ---- finalni eval na NAJBOLJEM modelu ----
    model = best_model
    with quiet():
        ev_val = EV.eval_split(model, "val", DEVICE)
        ev_test = EV.eval_split(model, "test", DEVICE)
        bench = EV.benchmark_cpu_vs_gpu_latency(model, EV.DATASET_ROOT / "images" / "train")
    gpu_ms = bench["cuda"]["fast_mean_ms"] if bench.get("cuda") else float("nan")
    cpu_ms = bench["cpu"]["fast_mean_ms"] if bench.get("cpu") else float("nan")
    dead, tot = count_dead(model, calib_eval, DEVICE)
    fin_flops = PL.count_flops(model, DEVICE); fin_params = PL.count_params(model)

    append_block("")
    append_block("=" * 78)
    append_block(f"FINALNO ({DISP}) — best val mAP")
    append_block("=" * 78)
    append_block(f"  params={fin_params:,}  GFLOPs={fin_flops/1e9:.3f}  "
                 f"val mAP={ev_val['map']:.4f}  test mAP={ev_test['map']:.4f}")
    append_block(f"  GPU={gpu_ms:.2f} ms/img  CPU={cpu_ms:.2f} ms/img  mrtvi filteri={dead}/{tot}")
    append_block(f"  reference: start(50%) {f0/1e9:.3f} GFLOPs/test {test0:.4f} | "
                 f"unpruned {UNPRUNED_REF[0]:.3f}/{UNPRUNED_REF[1]:.4f}")

    if MODELS_DIR:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, MODELS_DIR / f"{DISP}.pt")     # cijeli eager modul

    # ---- plot ----
    plot_pipeline(traj, start_map, test0)
    summary = {
        "start": {"params": p0, "gflops": f0 / 1e9, "val_map": start_map, "test_map": test0},
        "final": {"params": fin_params, "gflops": fin_flops / 1e9,
                  "val_map": ev_val["map"], "test_map": ev_test["map"],
                  "gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "dead": dead, "total": tot,
                  "best_val_map": best_map},
        "config": {"technique": DISP, "paradigm": "A-continuous", "target_factor": TARGET_FACTOR,
                   "total_epochs": TOTAL_EPOCHS, "grow_epochs": GROW_EPOCHS,
                   "max_grow_pct": MAX_GROW_PCT, "per_layer_cap": PER_LAYER_CAP,
                   "optimizer": "AdamW", "lr": LR},
        "trajectory": traj,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    append_block(f"\nUKUPNO VRIJEME: {(time.time()-t0)/60:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'} | {HERE/'pipeline.png'}")


def plot_pipeline(traj, start_map, test0):
    eps = [t["epoch"] for t in traj]
    gf = [t["gflops"] for t in traj]
    vm = [t["val_map"] for t in traj]
    grow_eps = [t["epoch"] for t in traj if t["grew"]]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val mAP@[.5:.95]", color="C0")
    ax1.plot(eps, vm, "-o", color="C0", label="val mAP", markersize=4)
    ax1.axhline(start_map, ls="--", color="gray", alpha=0.6)
    ax1.text(eps[-1], start_map, " pruned start val", fontsize=8, va="bottom", color="gray", ha="right")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.set_ylabel("GFLOPs", color="C3")
    ax2.plot(eps, gf, "-s", color="C3", label="GFLOPs", markersize=3, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="C3")
    for ge in grow_eps:
        ax1.axvline(ge, color="C2", alpha=0.18)
    if grow_eps:
        ax1.axvspan(min(grow_eps) - 0.5, max(grow_eps) + 0.5, color="C2", alpha=0.06, label="growth phase")
    ax1.set_title("A-paradigm continuous growing (GradMax) — val mAP & GFLOPs vs epoch")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / "pipeline.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    main()
