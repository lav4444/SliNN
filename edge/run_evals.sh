#!/bin/bash
# Pokrece baseline evaluaciju svih 7 modela, sekvencijalno, svaki u ZASEBNOM procesu.
#
# EVAL_SPLITS je tvrdo postavljen na "val,test" — train se NIKAD ne evaluira. Nije stvar
# zaborava po pozivu: train se na uredjaj ni ne salje, a i da se salje, baselineu ne treba.
#
# Sekvencijalno jer se latencija ne moze mjeriti dok nesto drugo radi: Pi 5 ima 4 jezgre,
# torch uzme sve, pa bi dva modela istovremeno mjerila raspored a ne model.
#
# UPORABA:
#   bash run_evals.sh 5      mini-test, 5 uzoraka po splitu -> eval_result_mini.txt
#   bash run_evals.sh        puni run                       -> eval_result.txt

set -u
ROOT="$HOME/Documents/code/dipl/EVAL"
BM="$ROOT/baseline_models"
LIM="${1:-}"

export EVAL_SPLITS="val,test"
export WRITE_CACHE=0                    # yolo: ne generiraj KD kes (na 26l bi bilo ~55 GB)
[ -n "$LIM" ] && export EVAL_LIMIT="$LIM"

source "$HOME/dipl-venv/bin/activate"

if [ -n "$LIM" ]; then
  echo "### MINI-TEST — $LIM uzoraka po splitu, splitovi: $EVAL_SPLITS"
else
  echo "### PUNI RUN — splitovi: $EVAL_SPLITS"
fi
echo "### $(date '+%H:%M:%S')"

run() {   # run <mapa> <skripta>
  echo
  echo "==================== $1  ($(date '+%H:%M:%S'))"
  cd "$BM/$1" || return 1
  if timeout 21600 python "$2" > /tmp/eval_$1.log 2>&1; then
    grep -E "^=====|mAP@50:95 = |R2   =|Accuracy  =|macro-F1  =|mIoU |AbsRel |acc |Inference \(batch=1\)|CPU: " \
         /tmp/eval_$1.log | head -14
    echo "  [OK] puni log: /tmp/eval_$1.log"
  else
    echo "  [PAD] izlazni kod $? — zadnjih 8 redaka:"
    tail -8 /tmp/eval_$1.log | sed 's/^/     /'
  fi
}

# redom od najbrzeg: ako nesto sistemski ne valja, vidi se za nekoliko sekundi
run housing_mlp        eval_baseline.py
run speechcommands_m5  eval_baseline.py
run midas_depth        eval_baseline.py
run sst2_distilbert    eval_baseline.py
run yolo26n            evaluate.py
run voc_deeplabv3      eval_baseline.py
run yolo26l            evaluate.py          # zadnji: sam traje ~110 min na punom runu

echo
echo "### GOTOVO  $(date '+%H:%M:%S')"
