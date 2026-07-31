#!/bin/bash
# Oracle (16,1) cells per the standing protocol, run in parallel co-located
# on GPUs 1/4/5. OpenDrawer capped at 15 episodes (95% measured firing makes
# each episode ~1h at k_u=1).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/oracle
LOG=$ROOT/results/robocasa/groot_ksweep.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # name gpu neps extra...
  local name=$1 GPU=$2 NEPS=$3; shift 3
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, oracle 16-1)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --control oracle --k-stable 16 --k-unstable 1 --n-episodes $NEPS \
      --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== oracle 16-1 cells start ==="
run_cell OpenCabinet_oracle_16-1 1 50 --env-name OpenCabinet --delta 0.057841 &
run_cell PickPlaceCounterToCabinet_oracle_16-1 4 50 --env-name PickPlaceCounterToCabinet --delta 0.035765 &
run_cell OpenDrawer_oracle_16-1_n15 5 15 --env-name OpenDrawer --delta 0.054712 &
wait
say "GROOT_ORACLE_161_DONE"
