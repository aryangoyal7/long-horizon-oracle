#!/bin/bash
# Rollout-recalibrated-delta side evaluation (user-approved idea 1).
# New deltas set so firing matches each task's label unstable fraction,
# computed from shadow (head) and oracle (measured lambda) rollout
# distributions - see results/robocasa/recalib_deltas.json.
# Head recalib is a no-op on OD/PP (already calibrated), so cells are:
#   head OC 16-4 delta .0401 (gpu 3, now)
#   head TOSF 16-1 delta .0636 (gpu 0, now)
#   oracle PP 16-4 delta .0245 (gpu 6, after PP oracle cell)
#   oracle OC 16-4 delta from OC oracle json quantile (gpu 2, after OC oracle)
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/recalib
HOUT=$ROOT/results/predictor/rc_indomain
ORC=$ROOT/results/robocasa/oracle
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_head() {  # task ku gpu delta
  local TASK=$1 KU=$2 GPU=$3 DELTA=$4
  local name=${TASK}_recalib_head_16-${KU}
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, recalib delta $DELTA)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --k-stable 16 --k-unstable $KU --delta $DELTA --n-episodes 50 \
      --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

run_oracle() {  # task ku gpu delta
  local TASK=$1 KU=$2 GPU=$3 DELTA=$4
  local name=${TASK}_recalib_oracle_16-${KU}
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, recalib oracle delta $DELTA)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --env-name $TASK --control oracle --delta $DELTA \
      --k-stable 16 --k-unstable $KU --n-episodes 50 \
      --out $OUT/$name.json > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== recalibrated-delta cells start ==="
run_head OpenCabinet 4 3 0.0401 &
run_head TurnOnSinkFaucet 1 0 0.0636 &
( while [ ! -f $ORC/PickPlaceCounterToCabinet_oracle_16-4.json ]; do sleep 120; done
  run_oracle PickPlaceCounterToCabinet 4 6 0.0245 ) &
( while [ ! -f $ORC/OpenCabinet_oracle_16-4.json ]; do sleep 120; done
  D=$(python3 -c "
import json, numpy as np
d = json.load(open('$ORC/OpenCabinet_oracle_16-4.json'))
lam = [r['lam'] for e in d['episodes'] for r in e['replans']]
print(round(float(np.quantile(lam, 1 - 0.117)), 4))")
  run_oracle OpenCabinet 4 2 $D ) &
wait
say "GROOT_RECALIB_DONE"
