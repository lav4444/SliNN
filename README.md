# SliNN

**Slimming via Imitation (Neural Networks)** — stanjivanje neuronskih mreža iterativnim podrezivanjem i destilacijom znanja.

SliNN je arhitekturno-agnostičan alat za kompresiju neuronskih mreža. Automatski prepoznaje zadatak, komponente i tipove slojeva, pa kroz kontinuiranu petlju **podrezivanja** (primarno) i uvjetnog **doraštanja** kanala uz očuvanje funkcije traži manje modele — sve vođeno **destilacijom znanja** iz zamrznutog učitelja (bez oznaka), pod zadanim GFLOPs-budžetom i s hardverskim poravnanjem kanala. Rezultat je cijela Pareto putanja kompromisa kvaliteta↔složenost, uz završnu **kvantizaciju** (PTQ/QAT).

## Struktura

| Folder | Sadržaj |
|---|---|
| `morphology/` | Jezgra: kontinuirano podrezivanje + doraštanje + KD (faza 1 i 2) |
| `pruning/` | Eksperimenti s kriterijima podrezivanja |
| `growing/` | Function-preserving doraštanje (GradMax i sl.) |
| `custom_models/` | Studentske arhitekture, KD varijante (pure vs feat+logit) |
| `pareto_sweep/` | Prolazak Pareto krivulje (detekcija + klasifikacija) |
| `quantization/` | PTQ i QAT (PyTorch fbgemm/x86, OpenVINO, TensorRT) |
| `custom_framework/`, `Analyze_Net/` | Pomoćni alati i analiza mreže |

## Okruženje

Python 3.10.12:

```bash
conda activate dipl
```

## Napomena o težinama i podatcima

Repozitorij sadrži **samo kod**. Težine modela (`.pt`, `.onnx`, `.engine`), datasetovi i cache namjerno su izostavljeni (vidi `.gitignore`) — GitHub ionako odbija velike datoteke, a podatci se generiraju/preuzimaju skriptama.

## Kontekst

Nastalo u sklopu diplomskog rada na FER-u (Sveučilište u Zagrebu).

## Reference

- Open Images V7 (Ultralytics): https://docs.ultralytics.com/datasets/detect/open-images-v7/
- https://www.nature.com/articles/s42005-023-01364-0
