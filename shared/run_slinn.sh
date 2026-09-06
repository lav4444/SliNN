#!/bin/bash
# Pokrece evaluaciju SVIH celija u SLINN_OPTIM, sekvencijalno, svaku u ZASEBNOM procesu.
#
# ZASTO ZASEBNA SKRIPTA OD run_evals.sh: ondje je popis od sedam modela tvrdo upisan, jer
# BASELINE mape imaju tocno jednu celiju po modelu. SliNN proizvodi trajektoriju checkpointa,
# pa je celija `<model>__<checkpoint>` i popis se ne moze znati unaprijed — cita se s diska.
# Sve ostalo je isto: isti mjerni aparat, isti splitovi, isti pecat kroz cijeli run.
#
# Sekvencijalno jer se latencija ne moze mjeriti dok nesto drugo radi.
#
# UPORABA:
#   bash run_slinn.sh 5      mini-test, 5 uzoraka po splitu
#   bash run_slinn.sh        puni run
#
# Okolina:
#   EVAL_DIR=SLINN_OPTIM     mjerna mapa (radi i s BASELINE_OPTIM)
#   CELLS=voc,yolo           samo celije cije ime sadrzi jednu od podniski

set -u
ROOT="$HOME/Documents/code/dipl/EVAL"
BM="$ROOT/${EVAL_DIR:-SLINN_OPTIM}"
LIM="${1:-}"

if [ ! -d "$BM" ]; then
  echo "nema mjerne mape: $BM"
  echo "pokreni:  python shared/make_slinn.py  pa  OPTIM_DIR=SLINN_OPTIM python shared/export.py"
  exit 1
fi

export RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export EVAL_SPLITS="val,test"
export WRITE_CACHE=0                    # yolo: ne generiraj KD kes
if [ -n "$LIM" ]; then
  export EVAL_LIMIT="$LIM"
else
  unset EVAL_LIMIT        # naslijedjen iz ljuske bi tiho skratio "puni" run
fi

source "$HOME/dipl-venv/bin/activate"

# --- popis celija -------------------------------------------------------------
# Poredak: po velicini mape, od najmanje. Nije kozmetika — ako nesto sistemski ne valja,
# vidi se na najjeftinijoj celiji za nekoliko sekundi umjesto nakon sat i pol na yolo26l.
mapfile -t CELL_LIST < <(
  for d in "$BM"/*/; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    if [ -n "${CELLS:-}" ]; then
      keep=0
      IFS=',' read -ra PATS <<< "$CELLS"
      for p in "${PATS[@]}"; do
        case "$n" in *"$p"*) keep=1 ;; esac
      done
      [ "$keep" -eq 1 ] || continue
    fi
    echo "$(du -sk "$d" | cut -f1) $n"
  done | sort -n | cut -d' ' -f2-
)

if [ "${#CELL_LIST[@]}" -eq 0 ]; then
  echo "nema nijedne celije u $BM${CELLS:+ (filtar CELLS=$CELLS)}"
  exit 1
fi

echo "############################################################"
if [ -n "$LIM" ]; then
  echo "###  MINI-TEST — $LIM uzoraka po splitu"
else
  echo "###  PUNI RUN — svi uzorci"
fi
echo "###  mapa: $BM"
echo "###  celija: ${#CELL_LIST[@]}   EVAL_SPLITS=$EVAL_SPLITS   WRITE_CACHE=$WRITE_CACHE"
echo "###  EVAL_LIMIT=${EVAL_LIMIT:-<nije postavljen>}   RUN_STAMP=$RUN_STAMP"
echo "############################################################"
echo "### $(date '+%H:%M:%S')"
printf '###  %s\n' "${CELL_LIST[@]}"

N_OK=0
N_TOTAL=0
FAILED=""
T0=$(date +%s)

for CELL in "${CELL_LIST[@]}"; do
  # Ime skripte ide po BAZNOM modelu (ono ispred `__`), jer je celija njegova inacica.
  BASE="${CELL%%__*}"
  case "$BASE" in
    yolo*) SCRIPT="evaluate.py" ;;
    *)     SCRIPT="eval_baseline.py" ;;
  esac
  echo
  echo "==================== $CELL  ($(date '+%H:%M:%S'))"
  N_TOTAL=$((N_TOTAL + 1))
  if [ ! -f "$BM/$CELL/$SCRIPT" ]; then
    FAILED="$FAILED $CELL"
    echo "  [PAD] nema $SCRIPT — celija nije sagradjena do kraja"
    continue
  fi
  cd "$BM/$CELL" || continue
  if timeout 21600 python "$SCRIPT" > "/tmp/eval_$CELL.log" 2>&1; then
    N_OK=$((N_OK + 1))
    grep -E "^=====|mAP@50:95 = |R2   =|Accuracy  =|macro-F1  =|mIoU |AbsRel |acc |Inference \(batch=1\)|CPU: " \
         "/tmp/eval_$CELL.log" | head -14
    echo "  [OK] puni log: /tmp/eval_$CELL.log"
  else
    RC=$?                      # PRIJE svega: dodjela ispod bi $? postavila na 0
    FAILED="$FAILED $CELL"
    echo "  [PAD] izlazni kod $RC — zadnjih 8 redaka:"
    tail -8 "/tmp/eval_$CELL.log" | sed 's/^/     /'
  fi
done

echo
echo "### GOTOVO  $(date '+%H:%M:%S')   ($N_OK/$N_TOTAL proslo)"

# --- neobavezna obavijest mailom ---------------------------------------------
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
    "SLINN $KIND, splitovi $EVAL_SPLITS
proslo: $N_OK/$N_TOTAL   trajanje: ${MINS} min
palo:${FAILED:- nista}
logovi: /tmp/eval_<celija>.log"
fi
