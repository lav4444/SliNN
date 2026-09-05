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
# Mjerna mapa: zasad samo BASELINE_RAW (original + eager PyTorch).
# Kad dodju ostale, ovo postaje argument ili posebna skripta po mapi.
BM="$ROOT/${EVAL_DIR:-BASELINE_RAW}"
LIM="${1:-}"

export EVAL_SPLITS="val,test"
export WRITE_CACHE=0                    # yolo: ne generiraj KD kes (na 26l bi bilo ~55 GB)
if [ -n "$LIM" ]; then
  export EVAL_LIMIT="$LIM"
else
  unset EVAL_LIMIT        # naslijedjen iz ljuske bi tiho skratio "puni" run
fi

source "$HOME/dipl-venv/bin/activate"

echo "############################################################"
if [ -n "$LIM" ]; then
  echo "###  MINI-TEST — $LIM uzoraka po splitu"
  echo "###  zapis -> eval_result_mini.txt"
else
  echo "###  PUNI RUN — svi uzorci"
  echo "###  zapis -> eval_result.txt"
fi
echo "###  EVAL_SPLITS=$EVAL_SPLITS   WRITE_CACHE=$WRITE_CACHE"
echo "###  EVAL_LIMIT=${EVAL_LIMIT:-<nije postavljen>}"
echo "############################################################"
echo "### $(date '+%H:%M:%S')"

N_OK=0
N_TOTAL=0
FAILED=""
T0=$(date +%s)

run() {   # run <mapa> <skripta>
  echo
  echo "==================== $1  ($(date '+%H:%M:%S'))"
  cd "$BM/$1" || return 1
  N_TOTAL=$((N_TOTAL + 1))
  if timeout 21600 python "$2" > /tmp/eval_$1.log 2>&1; then
    N_OK=$((N_OK + 1))
    grep -E "^=====|mAP@50:95 = |R2   =|Accuracy  =|macro-F1  =|mIoU |AbsRel |acc |Inference \(batch=1\)|CPU: " \
         /tmp/eval_$1.log | head -14
    echo "  [OK] puni log: /tmp/eval_$1.log"
  else
    FAILED="$FAILED $1"
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
echo "### GOTOVO  $(date '+%H:%M:%S')   ($N_OK/$N_TOTAL proslo)"

# --- neobavezna obavijest mailom ---------------------------------------------
# ~/edge_notify.py NIJE u repozitoriju (ondje su adresa i app-lozinka). Ako ga nema,
# run zavrsava tiho. Postavljanje: v. komentar na vrhu te skripte.
NOTIFY="$HOME/edge_notify.py"
if [ -f "$NOTIFY" ]; then
  MINS=$(( ($(date +%s) - T0) / 60 ))
  if [ "$N_OK" -eq "$N_TOTAL" ]; then
    STATUS="done"
  else
    STATUS="done s greskama ($N_OK/$N_TOTAL)"
  fi
  if [ -n "$LIM" ]; then
    KIND="mini-test ($LIM uzoraka)"
  else
    KIND="puni run"
  fi
  python3 "$NOTIFY" "$STATUS" \
    "$KIND, splitovi $EVAL_SPLITS
proslo: $N_OK/$N_TOTAL   trajanje: ${MINS} min
palo:${FAILED:- nista}
logovi: /tmp/eval_<model>.log"
fi
