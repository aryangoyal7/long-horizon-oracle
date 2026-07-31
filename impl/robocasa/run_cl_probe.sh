#!/bin/bash
# Closed-loop vs open-loop stability at the same states, RoboCasa + GR00T
# (2026-07-31, user request). Fills the closed-loop axis that previously
# existed only for 4 robomimic tasks -- and with a corrected nominal branch
# (policy-closed-loop unperturbed, not demo replay), plus a sampling-floor
# control. Pure shadow: the episode always runs full 16-step chunks.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/clprobe
LOG=$ROOT/results/robocasa/clprobe.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

cell(){  # task gpu n_eps
  local t=$1 GPU=$2 N=$3
  local name=${t}_clprobe
  [ -f $OUT/$name.DONE ] && return 0
  say "CELL_START $name (gpu $GPU, n=$N)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_cl_probe.py \
    --env-name $t --n-episodes $N --n-probe-cl 4 --n-ctrl 2 \
    --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    && { touch $OUT/$name.DONE; say "CELL_DONE $name"; } || say "CELL_FAIL $name"
}

say "=== cl probe start ==="
cell TurnOnSinkFaucet          6 25 &
cell PickPlaceCounterToCabinet 4 25 &
cell OpenDrawer                2 25 &
cell OpenCabinet               7 20 &
wait
say "CL_PROBE_ALL_DONE"
