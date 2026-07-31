#!/bin/bash
# Composed long-horizon task chain, fully parallel: OpenCabinet ->
# PickPlaceCounterToCabinet in one episode (doors forced closed at reset,
# instruction switched per stage). All 8 cells launch at once, one per GPU
# 0-7, co-located with the lighter jobs already there (80GB cards, existing
# jobs use 3-9GB). 50 eps, horizon 1800, stage-matched heads for predictor.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/sequence
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # name gpu extra...
  local name=$1 GPU=$2; shift 2
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, sequence)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_sequence_rollout.py \
      --n-episodes 50 --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;d=json.load(open('$OUT/$name.json'));print(d['success_rate'],'s1',d['stage1_success_rate'])" 2>/dev/null)"
}

say "=== groot sequence chain start (parallel) ==="
run_cell seq_fixed_16 0 --mode fixed --k 16 &
run_cell seq_fixed_8  1 --mode fixed --k 8 &
run_cell seq_fixed_4  2 --mode fixed --k 4 &
run_cell seq_fixed_1  3 --mode fixed --k 1 &
run_cell seq_pred_16-4 4 --mode predictor --k-stable 16 --k-unstable 4 &
run_cell seq_pred_16-1 5 --mode predictor --k-stable 16 --k-unstable 1 &
run_cell seq_pred_8-4  6 --mode predictor --k-stable 8 --k-unstable 4 &
run_cell seq_pred_8-1  7 --mode predictor --k-stable 8 --k-unstable 1 &
wait
say "GROOT_SEQUENCE_DONE"
