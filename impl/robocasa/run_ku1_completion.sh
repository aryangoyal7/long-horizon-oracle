#!/bin/bash
# Complete the k_unstable=1 protocol across every RoboCasa cell family.
# Missing counterparts being filled:
#   GPU 4: PP oracle recalib (16,1) delta .02452086      (had 16-4 only)
#   GPU 5: OC head recalib (16,1) .04006984 -> OC oracle recalib (16,1) .0338
#          -> PP contact (16,1)                          (had 16-4 only)
#   GPU 6: PP oracle (8,1) label delta .035765           (best fixed chunk + drop to 1)
#   GPU 7: learned (8,1) at training delta: OC -> OD -> TOSF
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
ROUT=$ROOT/results/robocasa/recalib
OOUT=$ROOT/results/robocasa/oracle
SOUT=$ROOT/results/robocasa/switch_81
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $SOUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_oracle() {  # name gpu dir task ks ku delta extra...
  local name=$1 GPU=$2 DIR=$3 TASK=$4 KS=$5 KU=$6 DELTA=$7; shift 7
  [ -f $DIR/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, ku1 completion)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --env-name $TASK --delta $DELTA --k-stable $KS --k-unstable $KU \
      --n-episodes 50 --out $DIR/$name.json "$@" > $DIR/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$DIR/$name.json'))['success_rate'])" 2>/dev/null)"
}

run_pred() {  # name gpu dir task ks ku extra...
  local name=$1 GPU=$2 DIR=$3 TASK=$4 KS=$5 KU=$6; shift 6
  [ -f $DIR/$name.json ] && return 0
  say "CELL_START $name (gpu $GPU, ku1 completion)"
  cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $TASK --pred-head $HOUT/R_${TASK}_full_s0/head.pt \
      --k-stable $KS --k-unstable $KU --n-episodes 50 \
      --out $DIR/$name.json "$@" > $DIR/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$DIR/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== ku1 completion cells start ==="
run_oracle PickPlaceCounterToCabinet_recalib_oracle_16-1 4 $ROUT \
  PickPlaceCounterToCabinet 16 1 0.02452086 --control oracle &
( run_pred OpenCabinet_recalib_head_16-1 5 $ROUT OpenCabinet 16 1 --delta 0.04006984
  run_oracle OpenCabinet_recalib_oracle_16-1 5 $ROUT OpenCabinet 16 1 0.0338 --control oracle
  run_oracle PickPlaceCounterToCabinet_contact_16-1 5 $OOUT \
    PickPlaceCounterToCabinet 16 1 0.035765 --control contact ) &
run_oracle PickPlaceCounterToCabinet_oracle_8-1 6 $OOUT \
  PickPlaceCounterToCabinet 8 1 0.035765 --control oracle &
( run_pred OpenCabinet_pred_8-1 7 $SOUT OpenCabinet 8 1
  run_pred OpenDrawer_pred_8-1 7 $SOUT OpenDrawer 8 1
  run_pred TurnOnSinkFaucet_pred_8-1 7 $SOUT TurnOnSinkFaucet 8 1 ) &
wait
say "KU1_COMPLETION_DONE"
