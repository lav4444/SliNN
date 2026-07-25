"""
analyze.py — generic model analyzer (torch).

Postavke se zadaju U KODU (blok "POSTAVI OVDJE" na vrhu), ne kao terminal argumenti.
Pokretanje:  python analyze.py

PODRZAN JE ISKLJUCIVO cijeli eager modul:
    torch.save(model, "m.pt")        # arhitektura (klasa) + tezine
Bilo koji drugi oblik (state_dict, TorchScript, ...) -> javi "FORMAT NIJE PODRZAN", bez analize.

Ispisuje: tip izlaza (klasifikacija/lokalizacija/oboje; binarno/multiclass),
#parametara, #filtera (conv), #neurona (linear), #slojeva, ukupno GFLOPs,
CPU i GPU vrijeme (warmup odbacen). Vizualizira slojeve (layers.png) s
imenom/vrstom, #tezina, #filtera/neurona i GFLOPs po sloju.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn

try:
    import cv2
except Exception:
    cv2 = None


# =================== POSTAVI OVDJE (umjesto terminal argumenata) =================== #
MODEL_PATH = "/home/tomi/code/dipl/growing/techniques_experiment/grown_models/GradMax-gradient.pt"        # put do modela spremljenog s  torch.save(model, "m.pt")  (cijeli eager modul)
DATA_DIR   = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7/images/test"        # folder slika za mjerenje brzine (.jpg/.png/.bmp)
INPUT_SIZE = 640       # ulazna strana (kvadrat): 224 klasifikacija, 640 detekcija
CODE_DIRS  = ["/home/tomi/code/dipl/custom_models/student_2_m"]   # folder(i) s kodom klase modela (model_arch.py za StudentYOLO)
N_IMAGES   = 10        # broj slika za timing
DEVICE     = "cuda"    # "cuda" ili "cpu"
OUT_DIR    = str(Path(__file__).parent / "report")
# (Ako je MODEL_PATH prazan -> pokrece se self-check na malom modelu.)
# ================================================================================== #


# --------------------------------------------------------------------------- #
# Ucitavanje modela (TorchScript ili cijeli eager modul)
# --------------------------------------------------------------------------- #
def load_model(path, device, code_dirs=None):
    """Ucitaj CIJELI eager modul (torch.save(model, ...)). Jedini podrzani format."""
    for d in (code_dirs or []):
        if d not in sys.path:
            sys.path.insert(0, d)
    obj = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval().to(device).float()        # fp32 referenca (neki ckpt su fp16)
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k].eval().to(device).float()
    # bilo sto drugo (state_dict / dict tenzora / ...) -> NIJE podrzano, BEZ analize
    what = "state_dict (samo tezine, nema arhitekture)" if isinstance(obj, dict) else type(obj).__name__
    raise SystemExit(
        f"FORMAT NIJE PODRZAN: dobiveno '{what}'.\n"
        f"Podrzan je ISKLJUCIVO cijeli eager modul:  torch.save(model, 'm.pt')\n"
        f"(NE state_dict, NE TorchScript). Analiza se NE izvodi.")


# --------------------------------------------------------------------------- #
# Introspekcija slojeva
# --------------------------------------------------------------------------- #
def _get_weight(m):
    w = getattr(m, "weight", None)
    if isinstance(w, torch.Tensor):
        return w
    try:
        d = dict(m.named_parameters(recurse=False))
        w = d.get("weight")
        return w if isinstance(w, torch.Tensor) else None
    except Exception:
        return None


def _type_name(m):
    return getattr(m, "original_name", None) or type(m).__name__


def weighted_leaves(model):
    """Vrati [(name, module, type_name, weight)] za leaf module s conv/linear weightom."""
    out = []
    for name, m in model.named_modules():
        if len(list(m.children())) > 0:        # ne-leaf
            continue
        w = _get_weight(m)
        if w is not None and w.dim() in (2, 4):
            out.append((name, m, _type_name(m), w))
    return out


# --------------------------------------------------------------------------- #
# FLOPs + per-layer statistika (forward s hookovima)
# --------------------------------------------------------------------------- #
def analyze_layers(model, sample, device):
    """Forward hookovi -> per-layer GFLOPs (2*MAC), #tezina, #filtera, #neurona,
    u FORWARD redoslijedu. Vrati (rec, out)."""
    layers = weighted_leaves(model)
    by_mod = {id(m): (name, tn, w) for name, m, tn, w in layers}
    rec = []          # popunjava se hookom u forward redoslijedu
    handles = []

    def mk_hook(m):
        def hook(mod, inp, out):
            o = out
            while isinstance(o, (list, tuple)) and len(o) > 0:
                o = o[0]
            name, tn, w = by_mod[id(mod)]
            flops = 0
            if isinstance(o, torch.Tensor):
                if w.dim() == 4:                      # conv
                    Cout, Cin_g, kh, kw = w.shape
                    Hout, Wout = o.shape[-2], o.shape[-1]
                    flops = 2 * Cout * Cin_g * kh * kw * Hout * Wout
                elif w.dim() == 2:                    # linear
                    out_f, in_f = w.shape
                    n = o.numel() // out_f if out_f else 0
                    flops = 2 * out_f * in_f * n
            n_w = sum(p.numel() for p in mod.parameters(recurse=False))
            rec.append({
                "name": name, "type": tn,
                "weights": int(n_w),
                "filters": int(w.shape[0]) if w.dim() == 4 else 0,
                "neurons": int(w.shape[0]) if w.dim() == 2 else 0,
                "gflops": flops / 1e9,
            })
        return hook

    for name, m, tn, w in layers:
        handles.append(m.register_forward_hook(mk_hook(m)))
    model.eval()
    with torch.no_grad():
        out = model(sample.to(device))
    for h in handles:
        h.remove()
    return rec, out


# --------------------------------------------------------------------------- #
# Tip izlaza (heuristika)
# --------------------------------------------------------------------------- #
def _tensors(o, acc):
    if isinstance(o, torch.Tensor):
        acc.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            _tensors(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _tensors(v, acc)
    return acc


def infer_task(out):
    if isinstance(out, dict) and {"boxes", "labels"} & set(out.keys()):
        return "lokalizacija + klasifikacija (detekcija; dict boxes/labels/scores)"
    ts = _tensors(out, [])
    shapes = [tuple(t.shape) for t in ts]

    # jedan 2D tenzor -> klasifikacija
    if len(ts) == 1 and ts[0].dim() == 2:
        K = ts[0].shape[1]
        kind = "binarna" if K <= 2 else f"multiclass ({K} klasa)"
        return f"klasifikacija — {kind}  | izlaz {tuple(ts[0].shape)}"

    # detekcijski uzorci (YOLO i sl.): [B,4,A] box + [B,nc,A] klase, ili [B,N,6/7] finalne
    has_box = final_det = False
    ncls = None
    for t in ts:
        if t.dim() == 3:
            _, d1, d2 = t.shape
            lo, hi = min(d1, d2), max(d1, d2)
            if lo == 4 and hi >= 64:
                has_box = True
            elif lo in (6, 7) and hi >= 10:
                final_det = True
            elif 1 < lo <= 1000 and hi >= 64:
                ncls = lo
    if final_det or has_box:
        c = f", ~{ncls} klasa" if ncls else ""
        return f"lokalizacija + klasifikacija (detekcija{c}) | izlazi: {shapes}"

    if len(ts) == 1 and ts[0].dim() == 3:
        b, d1, d2 = ts[0].shape
        C, N = min(d1, d2), max(d1, d2)
        if C >= 5:
            return (f"lokalizacija + klasifikacija (detekcija raw) | izlaz {tuple(ts[0].shape)} "
                    f"~ {N} sidra x ({C} = 4 box + {C-4} klasa)")
    if len(ts) == 1 and ts[0].dim() == 4:
        return f"feature-mapa / segmentacija? | izlaz {tuple(ts[0].shape)}"
    return f"nepoznato/slozeno | izlazi: {shapes}"


# --------------------------------------------------------------------------- #
# Mjerenje brzine (CPU/GPU; warmup odbacen)
# --------------------------------------------------------------------------- #
def load_images(data_dir, n, input_size):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(p for p in Path(data_dir).iterdir() if p.suffix.lower() in exts)[:n]
    imgs = []
    for p in files:
        if cv2 is not None:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (input_size, input_size))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        else:
            t = torch.rand(3, input_size, input_size)
        imgs.append(t)
    return imgs


def bench(model, imgs, device, n_warmup=2):
    model.to(device).eval()
    times = []
    with torch.no_grad():
        for i, im in enumerate(imgs):
            x = im.unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    kept = times[n_warmup:] if len(times) > n_warmup else times
    return sum(kept) / len(kept) if kept else float("nan")


# --------------------------------------------------------------------------- #
# Vizualizacija slojeva
# --------------------------------------------------------------------------- #
def visualize(rec, out_path, title):
    n = len(rec)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * n + 1.2)))
    cmap = {}
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ys = list(range(n))[::-1]                 # prvi sloj na vrhu
    maxg = max((r["gflops"] for r in rec), default=1e-9) or 1e-9
    for r, y in zip(rec, ys):
        t = r["type"]
        if t not in cmap:
            cmap[t] = palette[len(cmap) % len(palette)]
        ax.barh(y, r["gflops"], color=cmap[t], edgecolor="black", height=0.7)
        fn = (f"F={r['filters']}" if r["filters"] else
              (f"N={r['neurons']}" if r["neurons"] else ""))
        label = (f"{r['name']} [{r['type']}]  W={r['weights']:,}  {fn}  "
                 f"{r['gflops']:.3f} GFLOPs")
        ax.text(maxg * 0.01, y, label, va="center", ha="left", fontsize=7.5,
                color="black")
    ax.set_yticks([])
    ax.set_xlabel("GFLOPs (bar length)")
    ax.set_xlim(0, maxg * 1.02)
    ax.set_title(title, fontsize=11)
    handles = [mpatches.Patch(color=c, label=t) for t, c in cmap.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Glavna analiza
# --------------------------------------------------------------------------- #
def run(model_path, data_dir, input_size, code_dirs, n_images, device, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = load_model(model_path, dev, code_dirs)

    # sample za shape/flops
    sample = torch.rand(1, 3, input_size, input_size)
    rec, out = analyze_layers(model, sample, dev)

    total_params = sum(p.numel() for p in model.parameters())
    total_flops = sum(r["gflops"] for r in rec)
    flops_note = f"(@ {input_size}px, batch 1; 2*MAC)"
    total_filters = sum(r["filters"] for r in rec)
    total_neurons = sum(r["neurons"] for r in rec)
    n_conv = sum(1 for r in rec if r["filters"])
    n_lin = sum(1 for r in rec if r["neurons"])
    task = infer_task(out)

    # timing
    cpu_ms = gpu_ms = float("nan")
    if data_dir:
        imgs = load_images(data_dir, n_images, input_size)
        if imgs:
            cpu_ms = bench(model, imgs, torch.device("cpu"))
            if torch.cuda.is_available():
                gpu_ms = bench(model, imgs, torch.device("cuda"))

    # ---- izvjestaj ----
    lines = []
    def P(s): lines.append(s); print(s)
    P("=" * 70)
    P(f"ANALIZA MODELA: {model_path}")
    P(f"  format: eager full module | ulaz: 1x3x{input_size}x{input_size} | device: {dev}")
    P("=" * 70)
    P(f"Tip izlaza: {task}")
    P("-" * 70)
    P(f"Parametri ukupno : {total_params:,}  (~{total_params*4/1e6:.2f} MB fp32)")
    P(f"Slojevi (tezinski): {len(rec)}  (conv={n_conv}, linear={n_lin})")
    P(f"Filteri (conv out): {total_filters:,}")
    P(f"Neuroni (linear)  : {total_neurons:,}")
    P(f"GFLOPs ukupno     : {total_flops:.3f}  {flops_note}")
    P(f"CPU vrijeme       : {cpu_ms:.2f} ms/img  ({1000/cpu_ms:.1f} FPS)" if cpu_ms == cpu_ms else "CPU vrijeme: n/a")
    P(f"GPU vrijeme       : {gpu_ms:.2f} ms/img  ({1000/gpu_ms:.1f} FPS)" if gpu_ms == gpu_ms else "GPU vrijeme: n/a")
    P("-" * 70)
    P(f"{'#':>3} {'sloj':<26}{'tip':<10}{'tezine':>10}{'filt':>6}{'neur':>6}{'GFLOPs':>9}")
    for i, r in enumerate(rec):
        P(f"{i:>3} {r['name'][:26]:<26}{r['type'][:10]:<10}{r['weights']:>10,}"
          f"{r['filters']:>6}{r['neurons']:>6}{r['gflops']:>9.3f}")
    P("=" * 70)

    report = outdir / "analysis_report.txt"
    report.write_text("\n".join(lines))
    png = outdir / "layers.png"
    visualize(rec, png, f"Layers — {Path(model_path).name} (@{input_size}px)")
    print(f"\nSpremljeno: {report}\nSpremljeno: {png}")

    return {
        "rec": rec,
        "task": task,
        "model_path": str(model_path),
        "model_name": Path(model_path).name,
        "input_size": input_size,
        "device": str(dev),
        "summary": {
            "total_params": total_params,
            "params_mb": total_params * 4 / 1e6,
            "total_flops": total_flops,
            "total_filters": total_filters,
            "total_neurons": total_neurons,
            "n_layers": len(rec),
            "n_conv": n_conv,
            "n_lin": n_lin,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
        },
        "report_path": str(report),
        "png_path": str(png),
    }


# --------------------------------------------------------------------------- #
class TinyNet(nn.Module):       # module-level (picklable) za self-check
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 2, 1); self.b1 = nn.BatchNorm2d(16); self.a = nn.ReLU()
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1); self.b2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 5)
    def forward(self, x):
        x = self.a(self.b1(self.c1(x)))
        x = self.a(self.b2(self.c2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _selfcheck():
    """Bez --model: napravi mali CNN klasifikator, analiziraj (eager + TorchScript)."""
    import tempfile
    print(">>> SELF-CHECK (mali CNN klasifikator -> analiza)")
    m = TinyNet().eval()
    tmp = Path(tempfile.mkdtemp())
    m = TinyNet().eval()
    ddir = tmp / "imgs"; ddir.mkdir()
    if cv2 is not None:
        for i in range(6):
            cv2.imwrite(str(ddir / f"{i}.png"), (np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    data = str(ddir) if cv2 is not None else None
    ep = tmp / "tiny_eager.pt"; torch.save(m, str(ep))     # cijeli eager modul
    run(str(ep), data, 64, None, 6, "cpu", tmp / "out_eager")
    print("SELF-CHECK OK")


if __name__ == "__main__":
    if not MODEL_PATH:
        _selfcheck()                      # MODEL_PATH prazan -> self-check
    else:
        run(MODEL_PATH, DATA_DIR or None, INPUT_SIZE, CODE_DIRS or None,
            N_IMAGES, DEVICE, OUT_DIR)
