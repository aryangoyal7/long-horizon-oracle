#!/bin/bash
# GR00T predictor-switching cells: 4 tasks x full head x ku in {4,1}.
# Waits for the fixed-k sweep to finish, then one task per GPU 4-7.
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

while ! grep -q GROOT_KSWEEP_DONE $LOG 2>/dev/null; do sleep 120; done

run_task() {
  local TASK=$1 GPU=$2
  local ku name
  for ku in 4 1; do
    name=${TASK}_predictor_full_16-${ku}
    [ -f $KOUT/$name.json ] && continue
    say "CELL_START $name (gpu $GPU)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
        --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
        --k-stable 16 --k-unstable $ku --n-episodes 50 \
        --out $KOUT/$name.json > $KOUT/$name.log 2>&1 \
      || { say "CELL_FAIL $name"; continue; }
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
  done
  say "SWITCH_TASK_DONE $TASK"
}

say "=== groot switching cells start ==="
run_task OpenCabinet 4 &
run_task OpenDrawer 5 &
run_task PickPlaceCounterToCabinet 6 &
run_task TurnOnSinkFaucet 7 &
wait
say "GROOT_SWITCH_DONE"
