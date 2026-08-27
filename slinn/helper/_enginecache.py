"""_enginecache.py — smoke za Fazu 5.1 (split-materijalizacija + teacher-cache). Za image/vector/seq model:
  1) materialize_train_batches -> FIKSNI batchevi, DETERMINISTICNI (dva poziva = isti fingerprint + oblici)
  2) precompute_teacher -> cache; cached kd_loss ~ inline kd_loss (fp16 tolerancija) na razidjenom studentu
  3) REUSE: drugi precompute (isti args) = valjan meta -> brzo, iste vrijednosti
  4) INVALIDACIJA: promjena tapova -> meta se prepise (recompute)
  5) SPEEDUP: cached vs inline kroz N koraka
-> REPORTS/engine_cache.txt
"""
import copy
import json
import os
import sys
import time

import torch

_AA = "/home/tomi/code/dipl/slinn"
sys.path.insert(0, _AA)
BM = "/home/tomi/code/dipl/baseline_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "engine_cache.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import introspect as A                                          # noqa: E402
import loss as L                                              # noqa: E402
import engine as E                                            # noqa: E402
from classify import probe_adapter                            # noqa: E402
from pipeline import prepare                                  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("voc_deeplabv3", f"{BM}/voc_deeplabv3/model.pt", [f"{BM}/voc_deeplabv3"], f"{BM}/voc_deeplabv3/data"),
    ("housing_mlp", f"{BM}/housing_mlp/model.pt", [f"{BM}/housing_mlp"], f"{BM}/housing_mlp/data"),
    ("speechcommands_m5", f"{BM}/speechcommands_m5/model.pt", [f"{BM}/speechcommands_m5"], f"{BM}/speechcommands_m5/data"),
]

LINES = []


def emit(s):
    print(s, flush=True)
    LINES.append(s)


emit("===== FAZA 5.1 — split-materijalizacija + teacher-cache =====")
for name, spec, cd, data in MODELS:
    try:
        for d in cd:
            sys.path.insert(0, d)
        m = A.load_any(spec, dev, code_dirs=cd)
        ad = probe_adapter(m, dev, verbose=False)
        ctx = prepare(m, ad, dev, data)
        taps, kd_mode, out_kind = ctx["taps"], ctx["kd_mode"], ctx["out_kind"]

        teacher = copy.deepcopy(m).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        student = copy.deepcopy(m).to(dev)                    # razidji studenta -> KD>0 (kao Faza 1/2)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(0.02 * torch.randn_like(p))

        # 1) materijalizacija + determinizam
        b1, src = E.materialize_train_batches(data, ad, dev, ctx["split_plan"], batch_size=4, n_batches=4, seed=0)
        b2, _ = E.materialize_train_batches(data, ad, dev, ctx["split_plan"], batch_size=4, n_batches=4, seed=0)
        det = E._fingerprint(b1) == E._fingerprint(b2) and [len(b) for b in b1] == [len(b) for b in b2]

        # 2) cache + cached ~ inline
        t0 = time.time()
        cache = E.precompute_teacher(teacher, ad, b1, taps, name, split="train", verbose=False)
        t_build = time.time() - t0
        maxrel = 0.0
        for i, b in enumerate(b1):
            imgs = E.to_device(b, dev)
            lc, _ = L.kd_loss(student, teacher, ad, imgs, taps, kd_mode, out_kind, teacher_sig=cache.get(i, dev))
            li, _ = L.kd_loss(student, teacher, ad, imgs, taps, kd_mode, out_kind)
            maxrel = max(maxrel, abs(float(lc) - float(li)) / (abs(float(li)) + 1e-9))

        # 3) reuse (isti args -> valjan meta, brzo)
        t0 = time.time()
        cache2 = E.precompute_teacher(teacher, ad, b1, taps, name, split="train", verbose=False)
        t_reuse = time.time() - t0
        reuse_ok = cache2.has_all() and t_reuse < max(t_build * 0.6, 0.05)

        # 4) invalidacija: drugaciji tapovi -> meta se prepise
        mod_taps = taps[:-1] if len(taps) > 1 else taps + ["__probe_dummy__"]
        E.precompute_teacher(teacher, ad, b1, mod_taps, name, split="train", verbose=False)
        meta = json.load(open(os.path.join(E.TMP_ROOT, E._safe(name), "train", "meta.json")))
        invalidated = meta["taps"] == sorted(mod_taps)
        E.precompute_teacher(teacher, ad, b1, taps, name, split="train", verbose=False)  # vrati na ispravan cache

        # 5) speedup cached vs inline (N koraka na batchu 0)
        imgs0 = E.to_device(b1[0], dev)
        sig0 = cache.get(0, dev)
        N = 10
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            L.kd_loss(student, teacher, ad, imgs0, taps, kd_mode, out_kind, teacher_sig=sig0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_cached = time.time() - t0
        t0 = time.time()
        for _ in range(N):
            L.kd_loss(student, teacher, ad, imgs0, taps, kd_mode, out_kind)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_inline = time.time() - t0

        emit(f"\n[{name}]  mode={ad._mode}  taps={len(taps)}  out_kind={out_kind}  source={src}")
        emit(f"  materijalizacija: {len(b1)} batcha x {[len(b) for b in b1]}  determ.={'DA' if det else 'NE'}")
        emit(f"  cached~inline: max rel-diff={maxrel:.2e} ({'OK' if maxrel < 2e-2 else 'PREVISOK'})  build={t_build:.2f}s")
        emit(f"  reuse: {'DA' if reuse_ok else 'NE'} ({t_reuse:.3f}s vs build {t_build:.2f}s)  invalidacija={'DA' if invalidated else 'NE'}")
        emit(f"  speedup: inline={t_inline:.3f}s cached={t_cached:.3f}s  ->  {t_inline/max(t_cached,1e-6):.2f}x")
        del m, teacher, student
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        emit(f"\n[{name}] ERR {type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n[zapisano] -> {OUT}", flush=True)
