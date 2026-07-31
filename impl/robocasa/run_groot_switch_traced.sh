#!/bin/bash
# Instrumented switching reruns for regime visualization: per-replan lambda_hat
# and chosen k saved in the episode records, plus per-episode video. One cell
# per task: the two wins (OC 16-4, TOSF 16-1) and the two losses (OD 16-1,
# PP 16-4). GPUs 2/6/7 (PP queues behind OC on GPU 2).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/switch_traced
HOUT=$ROOT/results/predictor/rc_indomain
VID=$LH/rollouts/switch_traced
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT $VID
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # task ku gpu
  local TASK=$1 KU=$2 GPU=$3
  local name=${TASK}_traced_16-${KU}
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, traced)"
  mkdir -p $VID/$name
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --k-stable 16 --k-unstable $KU --n-episodes 50 \
      --video-dir $VID/$name \
      --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== groot switch traced start ==="
( run_cell OpenCabinet 4 2; run_cell PickPlaceCounterToCabinet 4 2 ) &
run_cell TurnOnSinkFaucet 1 6 &
run_cell OpenDrawer 1 7 &
wait
say "GROOT_SWITCH_TRACED_DONE"
