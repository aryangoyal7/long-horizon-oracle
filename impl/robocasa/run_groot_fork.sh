#!/bin/bash
# Paired fork cells (2026-07-29): isolate the oracle's effect to the fired
# segments. PP and TOSF (the two winning tasks), GPUs 5/6, n=50, same
# deltas as the headline oracle runs.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/fork
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # name gpu extra...
  local name=$1 GPU=$2; shift 2
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, fork)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_fork_rollout.py \
      --k-stable 16 --k-unstable 1 --n-episodes 50 \
      --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name"
}

say "=== fork cells start ==="
run_cell PickPlaceCounterToCabinet_fork_16-1 5 --env-name PickPlaceCounterToCabinet --delta 0.035765 &
run_cell TurnOnSinkFaucet_fork_16-1 6 --env-name TurnOnSinkFaucet --delta 0.077113 &
wait
say "GROOT_FORK_DONE"
