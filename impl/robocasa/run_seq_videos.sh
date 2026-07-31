#!/bin/bash
# Video-enabled sequence rollouts for the side-by-side demo:
# fixed_16 (left) and predictor (16,1) (right), 6 episodes each with
# per-episode mp4s and replan logs. GPUs 0 and 3 (co-located).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/sequence_videos
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # name gpu extra...
  local name=$1 GPU=$2; shift 2
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, seqvideo)"
  mkdir -p $OUT/vid_$name
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_sequence_rollout.py \
      --n-episodes 6 --video-dir $OUT/vid_$name \
      --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name"
}

say "=== sequence demo videos start ==="
run_cell seqvid_fixed_16 0 --mode fixed --k 16 &
run_cell seqvid_pred_16-1 3 --mode predictor --k-stable 16 --k-unstable 1 &
wait
say "SEQ_VIDEOS_DONE"
