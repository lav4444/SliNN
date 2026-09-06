# SliNN

**Slimming via Imitation (Neural Networks)** — slimming neural networks through iterative pruning and knowledge distillation.

SliNN is an architecture-agnostic neural network compression tool. It automatically detects the task, components, and layer types, then searches for smaller models through a continuous loop of **pruning** (primary) and conditional function-preserving **growing** of channels — all guided by **knowledge distillation** from a frozen teacher (no labels), under a target GFLOPs budget and with hardware-aware channel alignment. The result is a full Pareto trajectory of the quality↔complexity trade-off, followed by final **quantization** (PTQ/QAT).

## Structure

| Folder | Contents |
|---|---|
| `slinn/` | **Core**: pruning + growing + KD (phase 1 and 2), quantization, GUI |
| `slinn/gui/` | Streamlit interface (`gui.py`) and headless workers |
| `slinn/helper/` | Batch runners and self-checks |
| `baseline_models/` | Reference models (teachers) with their eval scripts |
| `edge/`, `edge_results/` | Deployment to Raspberry Pi 5 / Jetson Orin Nano, and measurements |
| `pruning/`, `growing/` | Pruning-criterion and function-preserving growing experiments |
| `custom_models/`, `pareto_sweep/` | Student architectures, KD variants, Pareto sweeps |
| `quantization/` | PTQ and QAT (PyTorch fbgemm/x86, ONNX Runtime, TensorRT) |
| `custom_framework/`, `Analyze_Net/` | Helper utilities and network analysis |
| `legacy/` | Previous iteration of the core, kept for reference |

## Environment

Python **3.10.12**, CUDA 12.4 build of PyTorch. A GPU is not required, but compression on CPU is slow.

```bash
conda create -n dipl python=3.10.12
conda activate dipl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install streamlit==1.58.0 torch-pruning==1.6.1 torchmetrics==1.9.0 \
            transformers==5.15.0 datasets==5.0.1 timm==1.0.28 ultralytics==8.4.48 \
            onnx==1.22.0 onnxruntime==1.23.2 pycocotools==2.0.11 pyarrow==24.0.0 \
            numpy==2.1.3 pandas==2.3.3 scikit-learn==1.7.2 pillow==12.2.0 \
            matplotlib==3.10.9 psutil==7.2.2
```

Not every package is needed for every task: `transformers` and `datasets` only for text
models, `ultralytics` and `pycocotools` only for detection, `onnx*` only for export.

## Running the GUI

```bash
conda activate dipl
cd slinn
streamlit run gui/gui.py
```

It opens at `http://localhost:8501`. If the port is taken, add `--server.port 8600`.

The input is **two paths**, entered in the sidebar — there is no model picker, because the
core is architecture-agnostic and so is the input:

| Field | Meaning |
|---|---|
| Model path | `.pt` file holding a **full eager module** (`torch.save(model)`, not a `state_dict`) |
| Dataset path | Root folder of the dataset; the format is auto-detected |

The model must be saved as a complete module — the analyzer attaches per-layer hooks, which a
`state_dict` cannot provide. Ready-made examples live under `baseline_models/`.

Pages: **Overview** (analysis of the loaded model — size, capabilities, per-layer breakdown,
cost of a cut, alignment), **Compress** (preparation with a readiness check, settings, then
compression with a live trajectory view), **About** (what the tool supports, read from the
registries).

## Running without the GUI

The same core runs headless; the settings are read from `slinn/settings.py`, which is the
single source of truth for every tunable value.

```bash
cd slinn
BATCH_ONLY=housing_mlp python helper/batch_slinn.py
```

Results land in `slinn/runs/<model>_<timestamp>/`: `best_quality_model.pt`,
`best_quality_model_qat.pt`, the trajectory, and the run log.
