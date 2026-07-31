#!/bin/bash
# Composed task (OpenCabinet -> PickPlaceCounterToCabinet) re-evaluated with
# the controllers that actually matter (2026-07-31). The original cells used
# heads trained on the v2 labels, which the action-order fix invalidated, and
# had no oracle cell at all. This runs: simulator oracle, h1 and h4 heads
# trained on the corrected v3 labels at their online-calibrated thresholds,
# all at (16,1) to match the atomic-task tables. 50 episodes, horizon 1800.
# Co-located on the GPUs already holding lighter jobs (80GB cards).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/sequence
LOG=$ROOT/results/robocasa/seq_v3.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

cell(){  # name gpu extra...
  local name=$1 GPU=$2; shift 2
  [ -f $OUT/$name.json ] && { say "skip $name"; return 0; }
  say "CELL_START $name (gpu $GPU)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_sequence_rollout.py \
    --n-episodes 50 --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    && say "CELL_DONE $name" || say "CELL_FAIL $name"
}

say "=== composed task v3 chain start ==="
cell seq_oracle_16-1 2 --mode oracle    --k-stable 16 --k-unstable 1 &
cell seq_h1_16-1     4 --mode predictor --head-variant h1 --k-stable 16 --k-unstable 1 &
cell seq_h4_16-1     6 --mode predictor --head-variant h4 --k-stable 16 --k-unstable 1 &
wait
say "SEQ_V3_ALL_DONE"
