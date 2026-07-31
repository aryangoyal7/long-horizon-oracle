#!/bin/bash
# Protect the headline PP oracle (16,1) .80 result with the two missing cells:
#   1. rate-matched random control, (16,1) at p=0.11, n=50
#   2. replication of the oracle cell at seed 1, n=50
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
COUT=$ROOT/results/robocasa/switch_controls
OOUT=$ROOT/results/robocasa/oracle
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/groot_ksweep.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

say "=== PP headline checks start ==="
TASK=PickPlaceCounterToCabinet

( name=${TASK}_random_16-1_p11
  [ -f $COUT/$name.json ] || {
    say "CELL_START $name (gpu 2, random control)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=2 $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
        --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
        --control random --fire-rate 0.11 --k-stable 16 --k-unstable 1 \
        --seed 1 --n-episodes 50 --out $COUT/$name.json \
        > $COUT/$name.log 2>&1 \
      || say "CELL_FAIL $name"
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$COUT/$name.json'))['success_rate'])" 2>/dev/null)"
  } ) &

( name=${TASK}_oracle_16-1_s1
  [ -f $OOUT/$name.json ] || {
    say "CELL_START $name (gpu 3, oracle replication seed 1)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=3 $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
        --env-name $TASK --control oracle --delta 0.035765 \
        --k-stable 16 --k-unstable 1 --seed 1 --n-episodes 50 \
        --out $OOUT/$name.json > $OOUT/$name.log 2>&1 \
      || say "CELL_FAIL $name"
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OOUT/$name.json'))['success_rate'])" 2>/dev/null)"
  } ) &
wait
say "PP_HEADLINE_CHECKS_DONE"
