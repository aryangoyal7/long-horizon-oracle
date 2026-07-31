#!/bin/bash
# h1 vs h4 sequence heads on every remaining labeled dataset (2026-07-31).
# LIBERO and robomimic feature files predate the proprio-sequence format, so
# train_head_seq.py now tiles the snapshot for those; the h1/h4 contrast is
# about visual history, which those files do carry.
set -u
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
AZPY=/anaconda/envs/azureml_py38/bin/python3
FE=$ROOT/artifacts/features
LA=$ROOT/artifacts/labels
HOUT=$ROOT/results/predictor/seq_allplat
LOG=$ROOT/results/robocasa/seq_allplat.log
Q=$HOUT/queue.txt
mkdir -p $HOUT
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

if [ ! -f $Q ]; then
  : > $Q
  for f in $FE/feat2_*.npz; do
    b=$(basename $f .npz); d=${b#feat2_}
    case "$d" in rc_*) continue;; esac
    [ -f $LA/final2_ol_$d.npz ] || continue
    for h in 1 4; do echo "$d $h" >> $Q; done
  done
fi
touch $Q.lock
say "=== seq all-platform heads start, $(wc -l < $Q) runs queued ==="

worker(){
  local GPU=$1 line d h out
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q") || break
    [ -z "$line" ] && break
    d=$(echo $line | cut -d' ' -f1); h=$(echo $line | cut -d' ' -f2)
    out=$HOUT/A_${d}_h${h}_s0
    [ -f $out/metrics.json ] && { say "skip $d h$h"; continue; }
    local FF=$FE/feat2_$d.npz
    [ -f "$FF" ] || FF=$FE/feat_$d.npz
    [ -f "$FF" ] || { say "NOFEAT $d"; continue; }
    CUDA_VISIBLE_DEVICES=$GPU $AZPY $ROOT/impl/predictor/train_head_seq.py \
      --features $FF --labels $LA/final2_ol_$d.npz \
      --hist $h --seed 0 --out $out --tag A_${d}_h${h}_s0 > $out.log 2>&1 \
      && say "OK A_${d}_h${h}_s0" || say "FAIL A_${d}_h${h}_s0"
  done
}
worker 0 & worker 1 & worker 7 & worker 3 &
wait
say "SEQ_ALLPLAT_DONE"
