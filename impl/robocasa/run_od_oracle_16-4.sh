#!/bin/bash
# OpenDrawer oracle rerun at (16,4), 25 episodes: the (16,1) cell was killed
# because measured lambda fires on ~95% of replans, making per-step
# measurement untenable. Waits for the TurnOnSinkFaucet oracle cell to free
# GPU 7.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/oracle
LOG=$ROOT/results/robocasa/groot_ksweep.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while [ ! -f $OUT/TurnOnSinkFaucet_oracle_16-1.json ]; do sleep 120; done
name=OpenDrawer_oracle_16-4_n25
[ -f $OUT/$name.json ] && exit 0
say "CELL_START $name (gpu 7, oracle rerun)"
cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES=7 $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
    --env-name OpenDrawer --control oracle --delta 0.054712 \
    --k-stable 16 --k-unstable 4 --n-episodes 25 \
    --out $OUT/$name.json > $OUT/$name.log 2>&1 \
  || { say "CELL_FAIL $name"; exit 1; }
say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
