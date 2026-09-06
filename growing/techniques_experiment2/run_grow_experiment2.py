
import sys
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from prodigyopt import Prodigy

HERE = Path(__file__).parent
EXP2_DIR = "/home/tomi/code/dipl/pruning/critereum_experiment2"
for _p in (str(HERE), EXP2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common                          # noqa: E402
import train_baseline as TB            # noqa: E402
import pruning_lib2 as PL              # noqa: E402
from model_cnn import INPUT_SIZE       # noqa: E402
import growing_lib2 as G               # noqa: E402
import criteria2 as C                  # noqa: E402


START_CKPT = Path(EXP2_DIR) / "pruned_models" / "taylor.pt"
RESULTS_FILE = HERE / "results.txt"
MODELS_DIR = HERE / "grown_models"

TECHNIQUES = ["magnitude", "gradient", "taylor", "hessian"]
PREFIX = {"magnitude": "Net2Net", "gradient": "GradMax",
          "taylor": "Taylor", "hessian": "Hessian-split"}
def disp(tech):
    return f"{PREFIX[tech]}-{tech}"

N_CYCLES = 3
GROW_TOTAL_FACTOR = 2.0
WANTED_MAP = 0.615
PATIENCE_FT = 4
MAX_EP_PER_CYCLE = 30
STALE_CYCLES = 2

CALIB_BATCHES = 16
MAX_EVAL_IMAGES = 2000
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNPRUNED_REF = (2.661, 0.6158)
BASELINE_PARAMS = 3_699_622
BASELINE_VAL_MAP = 0.6165


def append_block(text):
    with RESULTS_FILE.open("a") as f:
        f.write(text + "\n")
    print(text)


def make_loaders():
    train_loader = common.make_loader("train", TB.BATCH_SIZE, shuffle=True, num_workers=TB.NUM_WORKERS)
    val_loader = common.make_loader("val", TB.BATCH_SIZE, shuffle=False, num_workers=TB.NUM_WORKERS,
                                    max_images=MAX_EVAL_IMAGES)
    test_loader = common.make_loader("test", TB.BATCH_SIZE, shuffle=False, num_workers=TB.NUM_WORKERS,
                                     max_images=MAX_EVAL_IMAGES)
    return train_loader, val_loader, test_loader


def val_map(model, val_loader):
    return common.evaluate(model, val_loader, DEVICE)["map"]


@torch.no_grad()
def count_dead(model, calib, device, eps=1e-6):
    acts = {}
    handles = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            def hook(mm, i, o, nm=nm):
                ch = o.amax(dim=(0, 2, 3)) if o.dim() == 4 else o.amax(0)
                key = (nm, ch.shape[0])
                prev = acts.get(key)
                acts[key] = ch.detach() if prev is None else torch.maximum(prev, ch.detach())
            handles.append(m.register_forward_hook(hook))
    model.eval()
    for b in calib:
        model(b[0].to(device, non_blocking=True))
    for h in handles:
        h.remove()
    dead = tot = 0
    for _, mx in acts.items():
        dead += int((mx <= eps).sum()); tot += mx.numel()
    return dead, tot


def ft_until_plateau(model, train_loader, val_loader, patience, max_ep, label):
    optim = Prodigy(model.parameters(), lr=1.0, weight_decay=0.0, decouple=True)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    best_map = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_imp = ep = 0
    while ep < max_ep and no_imp < patience:
        ep += 1
        model.train()
        t0 = time.time()
        run_loss = 0.0; nb = 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optim); scaler.update()
            run_loss += float(loss); nb += 1
        eff_lr = optim.param_groups[0].get("d", float("nan"))
        vm = val_map(model, val_loader)
        improved = vm > best_map + 1e-4
        append_block(f"      [{label} ep{ep:2d}/{max_ep}] loss={run_loss/nb:.4f} "
                     f"| val_mAP={vm:.4f} | lr(d)={eff_lr:.2e} | {time.time()-t0:.0f}s"
                     f"{'  *best' if improved else ''}")
        if improved:
            best_map = vm; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
    return best_state, best_map, ep


def run_technique(tech, f0_flops, start_map, loaders, calib_eval):
    train_loader, val_loader, test_loader = loaders
    append_block("=" * 78)
    append_block(f"TEHNIKA: {disp(tech)}")
    model = G.load_pruned_model(START_CKPT, DEVICE)
    loss_fn = C.make_bce_loss_fn(DEVICE)

    best_overall = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    traj = [(f0_flops / 1e9, start_map)]
    cyc_times = []

    for cycle in range(1, N_CYCLES + 1):
        target = f0_flops * (1.0 + (GROW_TOTAL_FACTOR - 1.0) * cycle / N_CYCLES)
        calib = C.make_calib_batches(CALIB_BATCHES, DEVICE) if tech != "magnitude" else []
        t_cyc = time.time()
        added, ov = C.grow_to_budget(model, tech, calib, loss_fn, target, DEVICE)
        gf = PL.count_flops(model, DEVICE, INPUT_SIZE)
        pr = PL.count_params(model)
        append_block(f"  >>> CIKLUS {cycle}: grow -> GFLOPs {gf/1e9:.4f} (cilj {target/1e9:.4f}) "
                     f"params {pr:,} | dodano jedinica={sum(added.values())} u {len(added)} slojeva "
                     f"| benefit overhead={ov['time_ms_per_image']:.2f} ms/img")
        append_block(f"      [dodano] " + "  ".join(f"{n}:+{k}" for n, k in added.items()))
        bstate, bmap, eps = ft_until_plateau(model, train_loader, val_loader,
                                             PATIENCE_FT, MAX_EP_PER_CYCLE, f"{disp(tech)} c{cycle}")
        model.load_state_dict(bstate)
        traj.append((gf / 1e9, bmap))
        cyc_times.append(time.time() - t_cyc)
        append_block(f"      ciklus {cycle} best val mAP={bmap:.4f} ({eps} ep, {cyc_times[-1]/60:.1f} min)")
        if bmap > best_overall + 1e-4:
            best_overall = bmap; best_state = copy.deepcopy(bstate); stale = 0
        else:
            stale += 1
        if bmap >= WANTED_MAP:
            append_block(f"      >>> rani izlaz: val mAP {bmap:.4f} >= {WANTED_MAP}")
            break
        if stale >= STALE_CYCLES:
            append_block(f"      >>> stop: {STALE_CYCLES} ciklusa bez napretka")
            break

    model.load_state_dict(best_state)
    ev_val = common.evaluate(model, val_loader, DEVICE)
    ev_test = common.evaluate(model, test_loader, DEVICE)
    bench = common.benchmark_latency(model)
    gpu_ms = bench["cuda"] if bench.get("cuda") else float("nan")
    cpu_ms = bench["cpu"] if bench.get("cpu") else float("nan")
    dead, tot = count_dead(model, calib_eval, DEVICE)
    fin_flops = PL.count_flops(model, DEVICE, INPUT_SIZE); fin_params = PL.count_params(model)
    append_block(f"  --- FINALNO ({disp(tech)}) ---")
    append_block(f"    params={fin_params:,}  GFLOPs={fin_flops/1e9:.4f}  "
                 f"val mAP={ev_val['map']:.4f}  test mAP={ev_test['map']:.4f}  "
                 f"test F1={ev_test['f1']:.4f}  test acc={ev_test['acc']:.4f}")
    append_block(f"    GPU={gpu_ms:.2f} ms/img  CPU={cpu_ms:.2f} ms/img  "
                 f"mrtve jedinice={dead}/{tot}")
    append_block("")

    if MODELS_DIR:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"technique": disp(tech), "model": best_state}, MODELS_DIR / f"{disp(tech)}.pt")

    return {"params": fin_params, "gflops": fin_flops / 1e9,
            "val_map": ev_val["map"], "test_map": ev_test["map"],
            "test_f1": ev_test["f1"], "test_acc": ev_test["acc"],
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
               zorder=5, label="pruned start (taylor 5%)")
    ax.scatter([UNPRUNED_REF[0]], [UNPRUNED_REF[1]], marker="*", s=200, color="k",
               zorder=5, label="unpruned SchoolCNN")
    ax.axhline(WANTED_MAP, ls="--", color="gray", alpha=0.5)
    ax.text(ax.get_xlim()[0], WANTED_MAP, f" target {WANTED_MAP}", fontsize=8, va="bottom", color="gray")
    ax.set_xlabel("GFLOPs"); ax.set_ylabel("val macro mAP")
    ax.set_title("Network growing: GFLOPs vs performance (SchoolCNN, 4 criteria)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = HERE / "gflops_vs_map.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    t_script = time.time()
    torch.manual_seed(SEED)
    RESULTS_FILE.write_text("")
    append_block("=" * 78)
    append_block("NETWORK GROWING — usporedba 4 kriterija dodavanja jedinica (SchoolCNN)")
    append_block("Polaziste: pruned taylor.pt (~5%) | dataset: sub10k_open_images_v7 | multi-label")
    append_block("=" * 78)
    append_block(f"Device: {DEVICE} "
                 f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    append_block(f"Kriteriji: {', '.join(TECHNIQUES)} | benefit/FLOP alokacija (per-layer cap), function-preserving grow")
    append_block(f"{N_CYCLES} simetrična ciklusa do {GROW_TOTAL_FACTOR:.1f}x pocetnih GFLOPs (+{(GROW_TOTAL_FACTOR-1)*100:.0f}%)")
    append_block(f"FT do platoa: Prodigy (auto LR), inner patience {PATIENCE_FT}, max {MAX_EP_PER_CYCLE} ep | "
                 f"rani izlaz val mAP>={WANTED_MAP} | outer stale {STALE_CYCLES}")
    append_block(f"eval cap {MAX_EVAL_IMAGES}/split | benefit calib {CALIB_BATCHES} batcheva | metrika: macro mAP")
    append_block("")

    base = G.load_pruned_model(START_CKPT, DEVICE)
    f0 = PL.count_flops(base, DEVICE, INPUT_SIZE); p0 = PL.count_params(base)
    loaders = make_loaders()
    train_loader, val_loader, test_loader = loaders
    start_map = val_map(base, val_loader)
    test0 = common.evaluate(base, test_loader, DEVICE)["map"]
    append_block("-" * 78)
    append_block("START (pruned taylor ~5%)")
    append_block(f"  params={p0:,}  GFLOPs={f0/1e9:.4f}  val mAP={start_map:.4f}  test mAP={test0:.4f}")
    append_block(f"  ciljevi po ciklusu (GFLOPs): " +
                 ", ".join(f"{f0*(1+(GROW_TOTAL_FACTOR-1)*c/N_CYCLES)/1e9:.3f}" for c in range(1, N_CYCLES+1)))
    append_block(f"  reference: unpruned SchoolCNN {UNPRUNED_REF[0]:.3f} GFLOPs / test mAP {UNPRUNED_REF[1]:.4f}")
    append_block("")
    calib_eval = C.make_calib_batches(8, DEVICE)
    del base
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    summary = {"start": {"params": p0, "gflops": f0 / 1e9, "val_map": start_map, "test_map": test0}}
    for tech in TECHNIQUES:
        print(f"\n>>> TEHNIKA: {disp(tech)}")
        summary[tech] = run_technique(tech, f0, start_map, loaders, calib_eval)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    append_block("=" * 78)
    append_block("USPOREDNA TABLICA (head-to-head)")
    append_block("=" * 78)
    append_block(f"{'tehnika':<22}{'params':>10}{'GFLOPs':>8}{'val mAP':>9}{'test mAP':>9}"
                 f"{'testF1':>8}{'GPUms':>7}{'CPUms':>7}{'deadU':>10}")
    append_block("-" * 90)
    s = summary["start"]
    append_block(f"{'start(~5%)':<22}{s['params']/1e6:>9.3f}M{s['gflops']:>8.3f}"
                 f"{s['val_map']:>9.4f}{s['test_map']:>9.4f}{'-':>8}{'-':>7}{'-':>7}{'-':>10}")
    append_block(f"{'unpruned':<22}{BASELINE_PARAMS/1e6:>9.3f}M{UNPRUNED_REF[0]:>8.3f}"
                 f"{BASELINE_VAL_MAP:>9.4f}{UNPRUNED_REF[1]:>9.4f}{'-':>8}{'-':>7}{'-':>7}{'-':>10}")
    for tech in TECHNIQUES:
        t = summary[tech]
        append_block(f"{disp(tech):<22}{t['params']/1e6:>9.3f}M{t['gflops']:>8.3f}"
                     f"{t['val_map']:>9.4f}{t['test_map']:>9.4f}{t['test_f1']:>8.4f}"
                     f"{t['gpu_ms']:>7.2f}{t['cpu_ms']:>7.2f}{t['dead']:>4}/{t['total']:<5}")
    append_block("-" * 90)
    graf = plot_results(summary, (f0 / 1e9, start_map))
    (HERE / "summary.json").write_text(json.dumps(
        {k: ({kk: vv for kk, vv in v.items() if kk != "traj"} if isinstance(v, dict) else v)
         for k, v in summary.items()}, indent=2))
    append_block(f"\nGraf: {graf}")
    append_block(f"UKUPNO VRIJEME: {(time.time()-t_script)/60:.1f} min")
    append_block(f"Spremljeno: {RESULTS_FILE} | {HERE/'summary.json'}")


if __name__ == "__main__":
    main()
