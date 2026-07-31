#!/bin/bash
# Supplemental switching cells for PickPlaceCounterToCabinet: k_stable=8
# (its best fixed k), ku in {4,1}. Waits for the main switching chain.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
KOUT=$ROOT/results/robocasa/ksweep
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/groot_ksweep.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while ! grep -q GROOT_SWITCH_DONE $LOG 2>/dev/null; do sleep 180; done
TASK=PickPlaceCounterToCabinet
for ku in 4 1; do
  name=${TASK}_predictor_full_8-${ku}
  [ -f $KOUT/$name.json ] && continue
  say "CELL_START $name (gpu 6, extra)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=6 $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --k-stable 8 --k-unstable $ku --n-episodes 50 \
      --out $KOUT/$name.json > $KOUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; continue; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
done
say "GROOT_SWITCH_EXTRA_DONE"
