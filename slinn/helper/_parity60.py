"""6.0 — OFFLINE PARITET-HARNESS (Faza 6 sigurnosni gate, prije diranja GUI-ja).
Na ISTOM yolo modelu vrti (A) STARI morphology.run_morph i (B) NOVI slinn.engine.full_cycle,
oba iz ORIGINALA, bez dead-removala, isti broj koraka + isti step_frac(1.5%)/reinvest(0.30),
pa OBA finalna modela mjeri ISTIM mAP-om + GFLOPs. Cilj: novi engine NE regresira mAP naspram starog.
-> REPORTS/parity60.txt
"""
import copy
import os
import sys

import torch

# POVIJESNI HARNESS. Usporeduje STARI (morphology) i NOVI (slinn) engine, pa mu stari treba.
# Stari je nakon flipa (6.3) arhiviran u `legacy/`. Ako legacy vise ne postoji, ovaj harness NEMA
# sto usporedjivati -> uredno odustaje umjesto da pukne. Njegov posao je odradjen: paritet potvrdjen
# 4x (zadnji: B novi 5.721 GFLOPs / 4.0% / mAP 0.3839 / 92.0%), v. REPORTS/parity60.txt.
_MORPH = "/home/tomi/code/dipl/legacy/morphology"
_SLINN = "/home/tomi/code/dipl/slinn"
if not os.path.isdir(_MORPH):
    print("[parity60] `legacy/morphology` ne postoji — stari engine je arhiviran/obrisan.")
    print("[parity60] Ovaj harness je POVIJESNI; paritet je potvrdjen prije flipa (REPORTS/parity60.txt).")
    print("[parity60] Za provjeru NOVOG enginea koristi: _run61.py (GUI backend) ili _cycle54.py.")
    raise SystemExit(0)
sys.path.insert(0, _MORPH)
sys.path.insert(0, _SLINN)
BM = "/home/tomi/code/dipl/baseline_models"
DATA = "/home/tomi/code/dipl/datasets/mini_set/sub10k_open_images_v7"
SPEC = f"{BM}/yolo26n/yolo26n.pt"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS", "parity60.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

import analysis as A                                          # noqa: E402
import compress as C                                          # noqa: E402
import engine as E                                            # noqa: E402
import pipeline as PP                                         # noqa: E402
from classify import probe_adapter                            # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 5
MAP_IMGS = 64
LINES = []


def emit(s=""):
    print(s); LINES.append(s)


def main():
    orig = A.load_any(SPEC, dev)
    ad_m = A.pick_adapter(orig)                               # morphology decode-adapter (mAP scorer, oba puta isti)
    vloader = A.make_gt_loader("val", bs=4)
    mapf = lambda mdl: float(A.eval_map(mdl, ad_m, vloader, dev, max_images=MAP_IMGS)[0]["map"])
    g0 = E.gflops(orig, probe_adapter(orig, dev, verbose=False), dev)
    map0 = mapf(orig)
    emit("===== 6.0 PARITET-HARNESS (yolo26n, max_steps=%d, step 1.5%%, reinvest 0.30) =====" % MAX_STEPS)
    emit("baseline: GFLOPs=%.3f  mAP=%.4f  (DEV_DATA_SUBSET=%s)" % (g0, map0, A.dev_subset_note() and "on"))
    emit("")

    # ---------- (A) STARI: morphology.run_morph iz ORIGINALA, 5 koraka ----------
    emit("[A] STARI morphology.run_morph ...")
    old_start = copy.deepcopy(orig).to(dev)                   # run_morph ocekuje trenirljiv start (inace dolazi iz run_dead_ft);
    for p in old_start.parameters():                          # spremljeni yolo je frozen (requires_grad=False) -> ukljuci grad
        p.requires_grad_(True)
    old_best, _ = C.run_morph(SPEC, dev, metrics=["map"], start_model=old_start, max_steps=MAX_STEPS)
    old_g = E.gflops(old_best, probe_adapter(old_best, dev, verbose=False), dev)
    old_map = mapf(old_best)
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---------- (B) NOVI: slinn.engine.full_cycle iz ORIGINALA, 5 koraka, isti knobs ----------
    emit("[B] NOVI slinn.engine.full_cycle ...")
    ad = probe_adapter(orig, dev, verbose=False)
    ctx = PP.prepare(orig, ad, dev, DATA)
    teacher = copy.deepcopy(orig).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(orig).to(dev)
    res = E.full_cycle(student, teacher, ad, dev, ctx, DATA, "yolo26n",
                       target_frac=0.50, ft_steps=6, imp_batches=3, max_steps=MAX_STEPS,
                       dead=False, metric_fn=mapf, metric_tol=0.90)
    new_best = res["best_model"]
    new_g = E.gflops(new_best, probe_adapter(new_best, dev, verbose=False), dev)
    new_map = mapf(new_best)

    # ---------- USPOREDBA ----------
    emit("")
    emit("===== PARITET =====")
    emit("%-10s %10s %10s %10s %10s" % ("strana", "GFLOPs", "rez%", "mAP", "zadrzano%"))
    emit("%-10s %10.3f %10s %10.4f %10s" % ("baseline", g0, "-", map0, "100.0"))
    emit("%-10s %10.3f %10.1f %10.4f %10.1f" % ("A stari", old_g, 100 * (g0 - old_g) / g0, old_map, 100 * old_map / map0))
    emit("%-10s %10.3f %10.1f %10.4f %10.1f" % ("B novi", new_g, 100 * (g0 - new_g) / g0, new_map, 100 * new_map / map0))
    emit("")
    verdict = "PARITET OK (novi ne regresira)" if new_map >= 0.97 * old_map else "PAZI: novi ispod starog"
    emit("VERDICT: %s  (novi mAP %.4f vs stari %.4f)" % (verdict, new_map, old_map))
    open(OUT, "w").write("\n".join(LINES) + "\n")
    emit("-> " + OUT)


if __name__ == "__main__":
    main()
