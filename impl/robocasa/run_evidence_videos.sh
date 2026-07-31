#!/bin/bash
# Video-enabled evidence runs for the switching hypothesis, GPU 6 after the
# PP oracle (8,1) cell finishes. Episode inits are index-deterministic per
# split, so shadow (= fixed 16 with lambda-hat logged) and oracle runs of the
# same index should show the same scene: rescue evidence is side-by-side.
#   1. PP shadow_16 n=8            (fixed-16 behavior, lambda-hat logged)
#   2. PP oracle (16,1) n=8        (the .80 config)
#   3. TOSF shadow_16 n=8
#   4. TOSF oracle (16,1) n=8      (the .62-vs-.44 config)
#   5. OD pred (16,1) n=6          (fires a lot, fails anyway: honesty case)
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/evidence_videos
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while [ ! -f $ROOT/results/robocasa/oracle/PickPlaceCounterToCabinet_oracle_8-1.json ]; do sleep 180; done

run_switch() {  # name task neps extra...
  local name=$1 TASK=$2 NEPS=$3; shift 3
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu 6, evidence video)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=6 $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --n-episodes $NEPS --video-dir $OUT/vid_$name \
      --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name"
}

run_oracle() {  # name task neps delta extra...
  local name=$1 TASK=$2 NEPS=$3 DELTA=$4; shift 4
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu 6, evidence video)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=6 $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --env-name $TASK --control oracle --delta $DELTA \
      --k-stable 16 --k-unstable 1 --n-episodes $NEPS \
      --video-dir $OUT/vid_$name \
      --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name"
}

say "=== evidence video runs start ==="
run_switch PP_shadow16_vid PickPlaceCounterToCabinet 8 --control shadow
run_oracle PP_oracle161_vid PickPlaceCounterToCabinet 8 0.035765
run_switch TOSF_shadow16_vid TurnOnSinkFaucet 8 --control shadow
run_oracle TOSF_oracle161_vid TurnOnSinkFaucet 8 0.077113
run_switch OD_pred161_vid OpenDrawer 6 --k-stable 16 --k-unstable 1
say "EVIDENCE_VIDEOS_DONE"
