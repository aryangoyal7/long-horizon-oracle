#!/bin/bash
# Proxy-comparison cells (2026-07-30, ICLR reframe, v2): lambda vs the
# adaptive-chunking literature's replan signals, measured at the same states.
#   spread        - sample variance (DVAC-family confidence signal)
#   cons/cons_cos - queued-vs-fresh chunk disagreement (SGAC/BID family)
#   cmd_speed_min - predicted-chunk speed valleys (PACE)
#   speed         - instantaneous eef speed (control)
# Shares GPUs 2/3/4/7 with the tail of the ACT grid (A100-80GB fits both).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/proxy
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell(){  # task gpu delta
  local t=$1 GPU=$2 D=$3
  local name=${t}_proxyshadow_16
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, proxy v2)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_proxy_shadow.py \
    --env-name $t --delta $D --n-episodes 50 \
    --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    && say "CELL_DONE $name" || say "CELL_FAIL $name"
}

say "=== proxy shadow chain v2 start ==="
run_cell OpenDrawer 2 0.054712 &
run_cell OpenCabinet 3 0.057841 &
run_cell PickPlaceCounterToCabinet 4 0.035765 &
run_cell TurnOnSinkFaucet 7 0.077113 &
wait
say "PROXY_SHADOW_ALL_DONE"
