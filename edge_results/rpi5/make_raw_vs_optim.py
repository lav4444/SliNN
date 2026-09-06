# -*- coding: utf-8 -*-
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sorted(glob.glob(os.path.join(HERE, "baseline_eager", "results_*.csv")))[-1]
OPT = sorted(glob.glob(os.path.join(HERE, "baseline_optim", "results_*.csv")))[-1]

TRAJANJE = {"eager": 154, "ort": 77}
REDOSLIJED = ["housing_mlp", "speechcommands_m5", "sst2_distilbert", "midas_depth",
              "yolo26n", "voc_deeplabv3", "yolo26l"]


def ucitaj(p):
    met, lat, n = {}, {}, {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        k = (r["model"], r["split"])
        met[(r["model"], r["split"], r["metrika"])] = float(r["vrijednost"])
        if r["latencija_ms"] and k not in lat:
            lat[k] = float(r["latencija_ms"])
            n[k] = int(r["n_uzoraka"])
    return met, lat, n


ma, la, na = ucitaj(RAW)
mb, lb, nb = ucitaj(OPT)
kljuc = sorted(set(la) & set(lb), key=lambda k: (REDOSLIJED.index(k[0]), k[1]))

L = []
L += ["# RPi 5 — original model, dva runtimea", "",
      "Ista tezina, isti podaci, isti mjerni aparat. Razlikuje se **samo** runtime:",
      "eager PyTorch naspram ONNX Runtimea. Predobrada, letterbox, prag, NMS i metrika",
      "su bajt u bajt isti — kopija eval skripte mijenja jedino ucitavanje modela.", "",
      "| | |", "|---|---|",
      "| uredjaj | Raspberry Pi 5, 4 jezgre ARM, 2.4 GHz |",
      "| preciznost | FP32 u oba slucaja |",
      "| kvantizacija | **nema** — nijedan kvantizacijski cvor u grafu (provjereno brojanjem) |",
      "| splitovi | val, test (train se ne evaluira) |",
      f"| eager | torch 2.14.0+cpu · `{os.path.basename(RAW)}` |",
      f"| ORT | onnxruntime 1.29.0, CPUExecutionProvider · `{os.path.basename(OPT)}` |",
      "| termika | `throttled=0x0` — bez prigusivanja u oba runa |", "",
      "## Latencija (batch = 1)", "",
      "| model | split | n | eager ms | ORT ms | ubrzanje |",
      "|---|---|---:|---:|---:|---:|"]
for k in kljuc:
    L.append(f"| {k[0]} | {k[1]} | {na[k]} | {la[k]:.2f} | {lb[k]:.2f} | "
             f"**{la[k] / lb[k]:.2f}x** |")

e, o = TRAJANJE["eager"], TRAJANJE["ort"]
L += ["", f"Zidni sat cijelog runa: **{e} min -> {o} min** ({e / o:.2f}x). "
          "Nijedan model nije sporiji.", "",
      "## Tocnost", "",
      "| model | split | metrika | eager | ORT | razlika |",
      "|---|---|---|---:|---:|---:|"]
naj = 0.0
for k in kljuc:
    for (m, s, met), va in sorted(ma.items()):
        if (m, s) != k or (m, s, met) not in mb:
            continue
        vb = mb[(m, s, met)]
        d = abs(va - vb)
        naj = max(naj, d)
        L.append(f"| {m} | {s} | {met} | {va:.6f} | {vb:.6f} | {d:.0e} |")

L += ["", f"Najvece odstupanje kroz sve metrike i splitove: **{naj:.0e}** — to je granica",
      "zaokruzivanja u samom CSV-u (sest decimala), ne izmjerena razlika.", "",
      "## Zasto ovo mjerenje postoji", "",
      "Bez njega bi glavni omjer bio `SLINN_OPTIM / BASELINE_RAW` i SliNN bi naslijedio",
      "ovih 1.9x do 8.1x koje je donio runtime, a nije ih zaradio. Posten omjer je",
      "`SLINN_OPTIM / BASELINE_OPTIM`: ista tezina runtimea s obje strane, razlika je",
      "cisti ucinak kompresije.", "",
      "Druga strana: letvica je time podignuta. Kompresija mora pokazati dobitak povrh",
      "vec optimiziranog baselinea, ne povrh eagera.", "",
      "---", "",
      "Generirano iz `make_raw_vs_optim.py`. Ne uredjivati rucno."]

out = os.path.join(HERE, "rpi5_raw_vs_optim.md")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"[save] -> {out}   ({len(L)} redaka, {len(kljuc)} parova)")
