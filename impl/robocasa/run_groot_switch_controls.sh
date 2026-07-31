#!/bin/bash
# Causal controls for the switching hypothesis. Waits for the traced cells.
# Proof 1 (placement matters): random switching matched on firing rate,
#   OpenCabinet 16-4 @ p=0.05 and TurnOnSinkFaucet 16-1 @ p=0.01.
# Proof 2 (long chunks at unstable moments hurt): shadow-scored fixed-16 on
#   all four tasks (head logs lambda_hat every replan, never switches), giving
#   per-state damage comparison vs the switched cells plus the rollout
#   lambda_hat distribution for calibration analysis.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/switch_controls
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

while ! grep -q GROOT_SWITCH_TRACED_DONE $LOG 2>/dev/null; do sleep 120; done

run_cell() {  # name task gpu extra...
  local name=$1 TASK=$2 GPU=$3; shift 3
  [ -f $OUT/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, control)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --n-episodes 50 --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== groot switch controls start ==="
( run_cell OpenCabinet_shadow_16 OpenCabinet 2 --control shadow --k-stable 16
  run_cell OpenCabinet_random_16-4_p05 OpenCabinet 2 \
    --control random --fire-rate 0.05 --k-stable 16 --k-unstable 4 ) &
( run_cell TurnOnSinkFaucet_shadow_16 TurnOnSinkFaucet 6 --control shadow --k-stable 16
  run_cell TurnOnSinkFaucet_random_16-1_p01 TurnOnSinkFaucet 6 \
    --control random --fire-rate 0.01 --k-stable 16 --k-unstable 1 ) &
( run_cell OpenDrawer_shadow_16 OpenDrawer 7 --control shadow --k-stable 16
  run_cell PickPlaceCounterToCabinet_shadow_16 PickPlaceCounterToCabinet 7 \
    --control shadow --k-stable 16 ) &
wait
say "GROOT_SWITCH_CONTROLS_DONE"
