"""
run_grow_experiment.py (exp3) — NETWORK GROWING: GradMax vs Hessian-split, ONE-SHOT.

Po uzoru na exp1 (StudentYOLO, polaziste pruned taylor.pt 50%), ali:
  * samo 2 kriterija: gradient (GradMax) i hessian (Hessian-split)
  * ONE-SHOT grow (criteria.grow_one_shot): jedna alokacija + jedan grow, BEZ
    iterativnog re-mjerenja cost-a (za razliku od exp1 grow_to_budget)
  * 1 ciklus, cilj +30% pocetnih GFLOPs (zbog one-shot compoundinga realni GFLOPs
    obicno premase cilj — to je dio onoga sto exp3 testira)

Po tehnici:
  baseline (start) -> [ ONE-SHOT GROW do +30% GFLOPs -> FT do platoa (patience) ] x 1 -> eval

Mjeri: params/GFLOPs, val/test mAP@50:95, brzina (CPU/GPU), mrtve jedinice (proxy).
Izlaz: results.txt + summary.json + gflops_vs_map.png.

Pokretanje:  conda activate dipl && python run_grow_experiment.py
"""

import sys
import io
import copy
import json
import time
import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from prodigyopt import Prodigy

HERE = Path(__file__).parent
STUDENT_DIR = "/home/tomi/code/dipl/custom_models/student_2_m"
PURE_KD_DIR = "/home/tomi/code/dipl/custom_models/student_2_m/pure_KD"
EXP1_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment"
for _p in (PURE_KD_DIR, STUDENT_DIR, EXP1_DIR):
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

TECHNIQUES = ["gradient", "hessian"]   # exp3: samo GradMax + Hessian-split
# prefiks s pravim imenom metode (za ispis/tablicu/graf/spremanje)
PREFIX = {"magnitude": "Net2Net", "gradient": "GradMax",
          "taylor": "Taylor", "hessian": "Hessian-split"}
def disp(tech):
    return f"{PREFIX[tech]}-{tech}"

N_CYCLES = 1                         # exp3: ONE-SHOT -> samo 1 ciklus
GROW_TOTAL_FACTOR = 1.3              # cilj = 1.3x pocetni GFLOPs (+30%)
WANTED_MAP = 0.17                    # rani izlaz (irelevantno uz 1 ciklus)
PATIENCE_FT = 5                      # inner: val mAP ne raste 5 epoha -> kraj FT-a (isto kao exp1)
MAX_EP_PER_CYCLE = 35                # sigurnosna kapica epoha po ciklusu
STALE_CYCLES = 2                     # outer (irelevantno uz 1 ciklus)

CALIB_BATCHES = 16                   # za benefit (grad/taylor/hessian); magnitude=0
MAX_EVAL_IMAGES = 400                # cap slika u eval/benchmark (brze)
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNPRUNED_REF = (11.553, 0.1858)      # (GFLOPs, test mAP) nepruned StudentYOLO — referenca

# cap slika u EV.eval_split / benchmark (ne dira evaluate_student.py)
_orig_list_images = EV.list_images
EV.list_images = lambda d: _orig_list_images(d)[:MAX_EVAL_IMAGES] if MAX_EVAL_IMAGES else _orig_list_images(d)


# =========================== POMOCNE =========================== #
def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


@contextlib.contextmanager
def quiet():
    """Ugasi stdout spam iz EV.eval_split / benchmarka ([val] running inference...)."""
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
    """Proxy mrtvih filtera: SiLU izlazi (conv->bn->act) koji nikad ne predu ~0
    preko calib batcheva. StudentYOLO koristi ConvBNAct (act=SiLU)."""
    import torch.nn as _nn
    acts = {}
    handles = []
    for nm, m in model.named_modules():
        if isinstance(m, _nn.SiLU):
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


# =========================== FT do platoa =========================== #
def ft_until_plateau(model, train_loader, val_loader, patience, max_ep, label):
    """KD trening dok val mAP ne stagnira `patience` epoha (ili max_ep).
    LR automatski preko Prodigy (lr=1.0, sam procjenjuje korak `d`).
    Vrati (best_state, best_map, epochs_run)."""
    optim = Prodigy(model.parameters(), lr=1.0, weight_decay=TK.WEIGHT_DECAY, decouple=True)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    axy, ast = model.anchor_xy, model.anchor_stride
    best_map = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_imp = ep = 0
    while ep < max_ep and no_imp < patience:
        ep += 1
        model.train()
        t0 = time.time()
        run_loss = run_cls = run_box = 0.0; nb = 0
        for batch in train_loader:
            imgs = batch["image"].to(DEVICE, non_blocking=True)
            tb = batch["teacher_boxes"].to(DEVICE, non_blocking=True)
            tp = batch["teacher_probs"].to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                raw = model(imgs)
                loss, lcls, lbox = TK.kd_loss(raw, tb, tp, axy, ast)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), TK.GRAD_CLIP)
            scaler.step(optim); scaler.update()
            run_loss += float(loss); run_cls += float(lcls); run_box += float(lbox); nb += 1
        eff_lr = optim.param_groups[0].get("d", float("nan"))   # Prodigy procijenjeni korak
        vm = val_map(model)
        improved = vm > best_map + 1e-4
        append_block(f"      [{label} ep{ep:2d}/{max_ep}] loss={run_loss/nb:.4f} "
                     f"(cls={run_cls/nb:.4f} box={run_box/nb:.4f}) | val_mAP={vm:.4f} "
                     f"| lr(d)={eff_lr:.2e} | {time.time()-t0:.0f}s{'  *best' if improved else ''}")
        if improved:
            best_map = vm; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
    return best_state, best_map, ep


# =========================== JEDNA TEHNIKA =========================== #
def run_technique(tech, f0_flops, start_map, loaders, calib_eval):
    train_loader, val_loader = loaders
    append_block("=" * 78)
    append_block(f"TEHNIKA: {disp(tech)}")
    model = G.load_pruned_model(START_CKPT, DEVICE)
    loss_fn = C.make_kd_loss_fn(DEVICE)

    best_overall = -1.0
    best_model = copy.deepcopy(model)        # CIJELI model (oblik se mijenja svaki ciklus!)
    stale = 0
    traj = [(f0_flops / 1e9, start_map)]      # (GFLOPs, val mAP) trajektorija
    cyc_times = []

    for cycle in range(1, N_CYCLES + 1):
        target = f0_flops * (1.0 + (GROW_TOTAL_FACTOR - 1.0) * cycle / N_CYCLES)
        calib = C.make_calib_batches(CALIB_BATCHES, DEVICE) if tech != "magnitude" else []
        t_cyc = time.time()
        # --- ONE-SHOT GROW do +30% GFLOPs (function-preserving) ---
        # NE radimo BN rekalib: stari kanali vec imaju ispravne stats (function-preserving),
        # novi doprinose 0 dok ih FT ne otkljuca; FT (train mode) sam azurira BN.
        added, ov = C.grow_one_shot(model, tech, calib, loss_fn, target, DEVICE)
        gf = PL.count_flops(model, DEVICE)
        pr = PL.count_params(model)
        append_block(f"  >>> CIKLUS {cycle}: grow -> GFLOPs {gf/1e9:.3f} (cilj {target/1e9:.3f}) "
                     f"params {pr:,} | dodano filtera={sum(added.values())} u {len(added)} slojeva "
                     f"| benefit overhead={ov['time_ms_per_image']:.2f} ms/img")
        # --- FT do platoa ---
        bstate, bmap, eps = ft_until_plateau(model, train_loader, val_loader,
                                             PATIENCE_FT, MAX_EP_PER_CYCLE, f"{disp(tech)} c{cycle}")
        model.load_state_dict(bstate)
        traj.append((gf / 1e9, bmap))
        cyc_times.append(time.time() - t_cyc)
        append_block(f"      ciklus {cycle} best val mAP={bmap:.4f} ({eps} ep, {cyc_times[-1]/60:.1f} min)")
        # --- outer kriteriji ---
        if bmap > best_overall + 1e-4:
            best_overall = bmap; best_model = copy.deepcopy(model); stale = 0
        else:
            stale += 1
        if bmap >= WANTED_MAP:
            append_block(f"      >>> rani izlaz: val mAP {bmap:.4f} >= {WANTED_MAP}")
            break
        if stale >= STALE_CYCLES:
            append_block(f"      >>> stop: {STALE_CYCLES} ciklusa bez napretka")
            break

    model = best_model       # reportiramo NAJBOLJI model (moze biti raniji/manji ciklus)
    # finalni eval (val+test) + brzina + mrtve  (eval/bench spam ugasen)
    with quiet():
        ev_val = EV.eval_split(model, "val", DEVICE)
        ev_test = EV.eval_split(model, "test", DEVICE)
        bench = EV.benchmark_cpu_vs_gpu_latency(model, EV.DATASET_ROOT / "images" / "train")
    gpu_ms = bench["cuda"]["fast_mean_ms"] if bench.get("cuda") else float("nan")
    cpu_ms = bench["cpu"]["fast_mean_ms"] if bench.get("cpu") else float("nan")
    dead, tot = count_dead(model, calib_eval, DEVICE)
    fin_flops = PL.count_flops(model, DEVICE); fin_params = PL.count_params(model)
    append_block(f"  --- FINALNO ({disp(tech)}) ---")
    append_block(f"    params={fin_params:,}  GFLOPs={fin_flops/1e9:.3f}  "
                 f"val mAP={ev_val['map']:.4f}  test mAP={ev_test['map']:.4f}")
    append_block(f"    GPU={gpu_ms:.2f} ms/img  CPU={cpu_ms:.2f} ms/img  "
                 f"mrtvi filteri={dead}/{tot}")
    append_block("")

    if MODELS_DIR:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, MODELS_DIR / f"{disp(tech)}.pt")   # cijeli eager modul (analyzer-kompatibilno)

    return {"params": fin_params, "gflops": fin_flops / 1e9,
            "val_map": ev_val["map"], "test_map": ev_test["map"],
            "gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "dead": dead, "total": tot,
            "traj": traj, "cyc_min": sum(cyc_times) / 60.0}


def plot_results(summary, start_pt):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"magnitude": "C0", "gradient": "C1", "taylor": "C2", "hessian": "C3"}
    for tech in TECHNIQUES:
        if tech not in summary:
            continue
        traj = summary[tech]["traj"]
        xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
        ax.plot(xs, ys, "-o", color=colors[tech], label=disp(tech))
    ax.scatter([start_pt[0]], [start_pt[1]], marker="D", s=80, color="gray",
               zorder=5, label="pruned start (taylor 50%)")
    ax.scatter([UNPRUNED_REF[0]], [UNPRUNED_REF[1]], marker="*", s=200, color="k",
               zorder=5, label="unpruned StudentYOLO")
    ax.axhline(WANTED_MAP, ls="--", color="gray", alpha=0.5)
    ax.text(ax.get_xlim()[0], WANTED_MAP, f" target {WANTED_MAP}", fontsize=8, va="bottom", color="gray")
    ax.set_xlabel("backbone GFLOPs"); ax.set_ylabel("val mAP@[.5:.95]")
    ax.set_title("Network growing: GFLOPs vs performance (4 criteria)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "gflops_vs_map.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# =========================== MAIN =========================== #
def main():
    t_script = time.time()
    torch.manual_seed(SEED)
    RESULTS_FILE.write_text("")
    append_block("=" * 78)
    append_block("NETWORK GROWING exp3 — GradMax vs Hessian-split, ONE-SHOT (StudentYOLO)")
    append_block("Polaziste: pruned taylor.pt (50%) | dataset: sub10k_open_images_v7")
    append_block("=" * 78)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Kriteriji: {', '.join(TECHNIQUES)} | benefit/FLOP alokacija, function-preserving grow")
    append_block(f"{N_CYCLES} ciklus ONE-SHOT grow do {GROW_TOTAL_FACTOR:.1f}x pocetnih GFLOPs (+{(GROW_TOTAL_FACTOR-1)*100:.0f}%, bez iterativnog re-mjerenja)")
    append_block(f"FT do platoa: Prodigy (auto LR), inner patience {PATIENCE_FT}, max {MAX_EP_PER_CYCLE} ep | "
                 f"rani izlaz val mAP>={WANTED_MAP} | outer stale {STALE_CYCLES}")
    append_block(f"eval cap {MAX_EVAL_IMAGES}/split | benefit calib {CALIB_BATCHES} batcheva")
    append_block("")

    # baseline (pruned start)
    base = G.load_pruned_model(START_CKPT, DEVICE)
    f0 = PL.count_flops(base, DEVICE); p0 = PL.count_params(base)
    start_map = val_map(base)
    with quiet():
        test0 = EV.eval_split(base, "test", DEVICE)["map"]
    append_block("-" * 78)
    append_block("START (pruned taylor 50%)")
    append_block(f"  params={p0:,}  GFLOPs={f0/1e9:.3f}  val mAP={start_map:.4f}  test mAP={test0:.4f}")
    append_block(f"  ciljevi po ciklusu (GFLOPs): " +
                 ", ".join(f"{f0*(1+(GROW_TOTAL_FACTOR-1)*c/N_CYCLES)/1e9:.2f}" for c in range(1, N_CYCLES+1)))
    append_block("")
    loaders = make_loaders()
    calib_eval = C.make_calib_batches(8, DEVICE)        # za BN rekalib (magnitude) + dead-count
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    summary = {"start": {"params": p0, "gflops": f0 / 1e9, "val_map": start_map, "test_map": test0}}
    for tech in TECHNIQUES:
        print(f"\n>>> TEHNIKA: {disp(tech)}")
        summary[tech] = run_technique(tech, f0, start_map, loaders, calib_eval)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # tablica
    append_block("=" * 78)
    append_block("USPOREDNA TABLICA (head-to-head)")
    append_block("=" * 78)
    append_block(f"{'tehnika':<22}{'params':>10}{'GFLOPs':>8}{'val mAP':>9}{'test mAP':>9}"
                 f"{'GPUms':>7}{'CPUms':>7}{'deadF':>10}")
    append_block("-" * 86)
    s = summary["start"]
    append_block(f"{'start(50%)':<22}{s['params']/1e6:>9.3f}M{s['gflops']:>8.2f}"
                 f"{s['val_map']:>9.4f}{s['test_map']:>9.4f}{'-':>7}{'-':>7}{'-':>10}")
    append_block(f"{'unpruned':<22}{2.071:>9.3f}M{UNPRUNED_REF[0]:>8.2f}{'-':>9}{UNPRUNED_REF[1]:>9.4f}"
                 f"{'-':>7}{'-':>7}{'-':>10}")
    for tech in TECHNIQUES:
        t = summary[tech]
        append_block(f"{disp(tech):<22}{t['params']/1e6:>9.3f}M{t['gflops']:>8.2f}"
                     f"{t['val_map']:>9.4f}{t['test_map']:>9.4f}{t['gpu_ms']:>7.2f}{t['cpu_ms']:>7.2f}"
                     f"{t['dead']:>4}/{t['total']:<5}")
    append_block("-" * 86)
    graf = plot_results(summary, (f0 / 1e9, start_map))
    (HERE / "summary.json").write_text(json.dumps(
        {k: ({kk: vv for kk, vv in v.items() if kk != "traj"} if isinstance(v, dict) else v)
         for k, v in summary.items()}, indent=2))
    append_block(f"\nGraf: {graf}")
    append_block(f"UKUPNO VRIJEME: {(time.time()-t_script)/60:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
