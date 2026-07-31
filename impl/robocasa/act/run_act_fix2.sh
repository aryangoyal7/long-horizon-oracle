#!/bin/bash
# ACT fixed grids under the action-order remap (2026-07-29). All previous
# ACT cells fed LeRobot-ordered actions to an env expecting controller
# order; act_rollout.py now remaps in Policy.chunk. Oracle cells are held
# until the v3 relabel yields fresh per-task deltas.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
OUT=$ROOT/results/robocasa/act
DATA=$LH/data/robocasa/hdf5
LOG=$ROOT/results/robocasa/act_pipeline.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

eval_fixed(){
  local TASK=$1 GPU=$2 HOR=$3
  local CKPT=$ROOT/results/training/act_rc_${TASK}/act_final.pt
  local K name
  for K in 16 8 4 1; do
    name=${TASK}_act_fixed_${K}
    [ -f $OUT/$name.json ] && { say "skip $name"; continue; }
    say "CELL_START $name (gpu $GPU, remap)"
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      $PY $ROOT/impl/robocasa/act/act_rollout.py \
      --hdf5 $DATA/rc_${TASK}.hdf5 --ckpt $CKPT --mode fixed --k $K \
      --horizon $HOR --n-episodes 50 --out $OUT/$name.json \
      > $OUT/$name.log 2>&1
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
  done
  say "ACT_FIXED_DONE $TASK"
}
say "ACT_FIX2_START (remap protocol)"
eval_fixed OpenDrawer 2 750 &
eval_fixed OpenCabinet 3 1050 &
eval_fixed PickPlaceCounterToCabinet 4 750 &
eval_fixed TurnOnSinkFaucet 7 600 &
wait
say "ACT_FIX2_ALL_DONE"
