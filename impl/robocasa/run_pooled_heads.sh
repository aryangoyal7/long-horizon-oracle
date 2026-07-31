#!/bin/bash
# Pooled 4-task stability heads (2026-07-31). One head over all four RoboCasa
# kitchen tasks, so a composed episode needs no stage signal to pick a head.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
AZPY=/anaconda/envs/azureml_py38/bin/python3
HOUT=$ROOT/results/predictor/rc_pool_v3
LOG=$ROOT/results/robocasa/pooled_heads.log
mkdir -p $HOUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
T="OpenCabinet OpenDrawer PickPlaceCounterToCabinet TurnOnSinkFaucet"
FEATS=""; LABS=""
for t in $T; do
  [ -f $LH/features/feat2_rc_$t.npz ] || cp $ROOT/artifacts/features/feat2_rc_$t.npz $LH/features/
  FEATS="$FEATS $LH/features/feat2_rc_$t.npz"
  LABS="$LABS $LH/labels/final3_env_rc_$t.npz"
done
say "=== pooled head training start ==="
train(){  # hist gpu
  local h=$1 GPU=$2 out=$HOUT/POOL_h$1_s0
  [ -f $out/metrics.json ] && { say "skip h$h"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU $AZPY $ROOT/impl/predictor/train_head_seq_pool.py \
    --features $FEATS --labels $LABS --hist $h --seed 0 \
    --out $out --tag POOL_h${h}_s0 > $out.log 2>&1 \
    && say "POOL_HEAD_OK h$h" || say "POOL_HEAD_FAIL h$h"
}
train 1 3 &
train 4 5 &
wait
say "POOLED_HEADS_DONE"
