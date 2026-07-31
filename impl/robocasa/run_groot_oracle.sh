#!/bin/bash
# Oracle-switching cells: switch on the MEASURED open-loop FTLE at each
# replan (labeler procedure at rollout time), plus a contact-oracle cell for
# PickPlace. The ceiling experiment for the hypothesis: if measured-lambda
# switching beats every fixed k, regime change is proven independent of
# predictor quality. Waits for the causal-control cells, then GPUs 2/6/7.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/oracle
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while ! grep -q GROOT_SWITCH_CONTROLS_DONE $LOG 2>/dev/null; do sleep 120; done

run_cell() {  # name gpu extra...
  local name=$1 GPU=$2; shift 2
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, oracle)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --n-episodes 50 --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== groot oracle cells start ==="
( run_cell OpenCabinet_oracle_16-4 2 --env-name OpenCabinet \
    --control oracle --delta 0.057841 --k-stable 16 --k-unstable 4 ) &
( run_cell PickPlaceCounterToCabinet_oracle_16-4 6 --env-name PickPlaceCounterToCabinet \
    --control oracle --delta 0.035765 --k-stable 16 --k-unstable 4
  run_cell PickPlaceCounterToCabinet_contact_16-4 6 --env-name PickPlaceCounterToCabinet \
    --control contact --delta 0.035765 --k-stable 16 --k-unstable 4 ) &
( run_cell OpenDrawer_oracle_16-1 7 --env-name OpenDrawer \
    --control oracle --delta 0.054712 --k-stable 16 --k-unstable 1
  run_cell TurnOnSinkFaucet_oracle_16-1 7 --env-name TurnOnSinkFaucet \
    --control oracle --delta 0.077113 --k-stable 16 --k-unstable 1 ) &
wait
say "GROOT_ORACLE_DONE"
