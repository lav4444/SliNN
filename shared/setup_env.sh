#!/bin/bash
# Instalacija okruzenja za BASELINE EVALUACIJU na edge uredjaju.
#
# ZASTO NE `pip install -r <freeze s laptopa>`:
#   Laptop je x86_64 s torch==2.6.0+cu124 i 247 paketa. Nijedan uredjaj to ne moze
#   instalirati. Torch mora doci iz izvora specificnog za platformu:
#     Jetson  -> NVIDIA-in indeks, wheel gradjen protiv konkretne JetPack/CUDA verzije
#     RPi 5   -> obicni PyPI aarch64 CPU wheel
#   Paketi openvino / nncf / onnxruntime-gpu NE IDU ni na jedan od njih u tim verzijama.
#
# Za baseline eval treba 14 paketa, ne 247.
#
# UPORABA:  bash setup_env.sh          (sam prepoznaje uredjaj)

set -euo pipefail

VENV="$HOME/dipl-venv"

# ---------------------------------------------------------------- prepoznaj uredjaj
if [ -f /etc/nv_tegra_release ] || [ -d /usr/lib/aarch64-linux-gnu/tegra ]; then
  DEV="jetson"
elif grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
  DEV="rpi5"
else
  DEV="nepoznato"
fi
echo "### uredjaj: $DEV   ($(uname -m))"

if [ "$DEV" = "nepoznato" ]; then
  echo "!!! Nisam prepoznao uredjaj. Postavi DEV rucno u skripti i pokreni ponovno."
  exit 1
fi

# ---------------------------------------------------------------- sustavne biblioteke
echo; echo "### sustavni paketi"
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-dev libopenblas-dev libsndfile1 \
                        libjpeg-dev zlib1g-dev ffmpeg

# ---------------------------------------------------------------- venv
if [ ! -d "$VENV" ]; then
  # --system-site-packages je OBAVEZAN na Jetsonu: torch dolazi kao sustavni paket
  # iz JetPacka i u izoliranom venvu se ne vidi.
  if [ "$DEV" = "jetson" ]; then
    python3 -m venv --system-site-packages "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# ---------------------------------------------------------------- torch
echo; echo "### torch"
if [ "$DEV" = "jetson" ]; then
  echo "Na Jetsonu torch NE dolazi s PyPI-ja."
  echo "Provjeri je li vec tu (JetPack ga cesto donese):"
  if python3 -c "import torch" 2>/dev/null; then
    python3 -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"
  else
    cat <<'EOF'
  NEMA TORCHA. Instaliraj s NVIDIA-inog indeksa, verzija MORA odgovarati JetPacku:
      # provjeri JetPack:  cat /etc/nv_tegra_release
      pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v<VER>/pytorch/<wheel>
  Popis wheelova: https://developer.download.nvidia.com/compute/redist/jp/
  torchvision i torchaudio se na Jetsonu obicno GRADE IZ IZVORA protiv tog torcha —
  paziti da se verzije poklapaju s torchom, inace pucaju pri uvozu.
EOF
    exit 1
  fi
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# ---------------------------------------------------------------- ostalo
echo; echo "### paketi za evaluaciju"
pip install -r "$(dirname "$0")/requirements-edge.txt"

echo
echo "### GOTOVO. Aktivacija:  source $VENV/bin/activate"
echo "### Provjera:            python $(dirname "$0")/smoke_test.py"
