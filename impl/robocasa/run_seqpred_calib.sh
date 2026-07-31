#!/bin/bash
# Predictor-driven switching, ONLINE-calibrated operating point (2026-07-31, v2).
# Pass 1 fired on delta_label: lambda_hat is compressed by regression so nothing
# ever fired (mean_k 16.0 everywhere) -- those runs are, usefully, pure online
# SHADOW logs of lambda_hat. Pass 2 used a threshold calibrated on the OFFLINE
# (demo) split; it transferred badly (2.1%-28.9% online fire rate vs a ~4%
# target, TOSF worst) because the policy's own state distribution differs from
# the demos. This pass takes the threshold from the pass-1 online logs at the
# quantile matching the ORACLE's measured fire rate per task, so the predictor
# gets the same intervention budget as the oracle and the ablation compares
# like with like. Thresholds: results/predictor/rc_seq_v3/online_thresholds.json
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
HOUT=$ROOT/results/predictor/rc_seq_v3
OUT=$ROOT/results/robocasa/seqpred_calib
LOG=$ROOT/results/robocasa/seqpred.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

cell(){  # task hist gpu
  local t=$1 h=$2 GPU=$3
  local d=$HOUT/S_${t}_h${h}_s0
  local name=${t}_h${h}pred_calib_16-1
  [ -f $OUT/$name.json ] && return 0
  local THR
  THR=$(python3 -c "import json;print(json.load(open('$HOUT/online_thresholds.json'))['${t}_h${h}']['thresh'])") || return 1
  local win=$((h * 16))
  say "CELL_START $name (gpu $GPU, online thr $THR)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_switch_rollout.py \
    --env-name $t --pred-head $d/head.pt \
    --bridge $ROOT/impl/eval/predictor_bridge_seq.py --frames-window $win \
    --fire-on lam --delta $THR \
    --k-stable 16 --k-unstable 1 --n-episodes 50 \
    --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    && say "CELL_DONE $name" || say "CELL_FAIL $name"
}

say "=== online-calibrated seq-predictor cells start ==="
cell OpenCabinet               1 0 &
cell OpenCabinet               4 1 &
cell OpenDrawer                1 2 &
cell OpenDrawer                4 3 &
cell PickPlaceCounterToCabinet 1 4 &
cell PickPlaceCounterToCabinet 4 5 &
cell TurnOnSinkFaucet          1 6 &
cell TurnOnSinkFaucet          4 7 &
wait
say "SEQPRED_CALIB_ALL_DONE"
