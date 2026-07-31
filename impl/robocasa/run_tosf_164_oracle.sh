#!/bin/bash
# TOSF oracle (16,4): replicates the content-attribution contrast (same
# measured moments, milder drop) on a second task. GPU 6 after the evidence
# video runs.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/oracle
LOG=$ROOT/results/robocasa/groot_ksweep.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while ! grep -q EVIDENCE_VIDEOS_DONE $LOG 2>/dev/null; do sleep 300; done

name=TurnOnSinkFaucet_oracle_16-4
[ -f $OUT/$name.json ] && exit 0
say "CELL_START $name (gpu 6, content contrast)"
cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=6 PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES=6 $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
    --env-name TurnOnSinkFaucet --control oracle --delta 0.077113 \
    --k-stable 16 --k-unstable 4 --n-episodes 50 \
    --out $OUT/$name.json > $OUT/$name.log 2>&1 \
  || say "CELL_FAIL $name"
say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
say "TOSF_164_DONE"
