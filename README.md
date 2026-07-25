# SliNN

**Slimming via Imitation (Neural Networks)** — slimming neural networks through iterative pruning and knowledge distillation.

SliNN is an architecture-agnostic neural network compression tool. It automatically detects the task, components, and layer types, then searches for smaller models through a continuous loop of **pruning** (primary) and conditional function-preserving **growing** of channels — all guided by **knowledge distillation** from a frozen teacher (no labels), under a target GFLOPs budget and with hardware-aware channel alignment. The result is a full Pareto trajectory of the quality↔complexity trade-off, followed by final **quantization** (PTQ/QAT).

## Structure

| Folder | Contents |
|---|---|
| `morphology/` | Core: continuous pruning + growing + KD (phase 1 and 2) |
| `pruning/` | Pruning-criterion experiments |
| `growing/` | Function-preserving growing (GradMax, etc.) |
| `custom_models/` | Student architectures, KD variants (pure vs feat+logit) |
| `pareto_sweep/` | Pareto-curve sweep (detection + classification) |
| `quantization/` | PTQ and QAT (PyTorch fbgemm/x86, OpenVINO, TensorRT) |
| `custom_framework/`, `Analyze_Net/` | Helper utilities and network analysis |

## Environment

Python 3.10.12:

```bash
conda activate dipl
```

## Note on weights and data

This repository contains **code only**. Model weights (`.pt`, `.onnx`, `.engine`), datasets, and caches are intentionally excluded (see `.gitignore`) — GitHub rejects large files anyway, and the data is generated/downloaded via the scripts.

## Context

Developed as part of a master's thesis at FER (University of Zagreb).

## References

- Open Images V7 (Ultralytics): https://docs.ultralytics.com/datasets/detect/open-images-v7/
- https://www.nature.com/articles/s42005-023-01364-0
