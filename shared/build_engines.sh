#!/bin/bash
# ONNX -> TensorRT FP16 engine. Pokrenuti NA UREDJAJU, nakon shared/export.py.
#   bash shared/build_engines.sh                    -> BASELINE_OPTIM
#   EVAL_DIR=SLINN_OPTIM bash shared/build_engines.sh
set -u
source "$HOME/dipl-venv/bin/activate"
exec python "$(dirname "$0")/build_engines.py"
