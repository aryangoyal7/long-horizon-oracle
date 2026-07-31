#!/bin/bash
# Probe-noise robustness sweep (2026-07-30, user request): label at sigma_u
# 0.025 and 0.100 (v3 is the 0.05 point). Same seed 0 -> identical
# perturbation directions across sigma runs; per-stamp lambdas pair exactly.
# Runs the two sigma streams in parallel, each sequential over tasks, each
# load-gated so the three concurrent campaigns (v3 + 2 sweeps) self-stagger.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
RCPY=$LH/envs/rc/bin/python
LOG=$ROOT/results/robocasa/labeling_sigma.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
TASKS="rc_OpenCabinet rc_OpenDrawer rc_PickPlaceCounterToCabinet rc_TurnOnSinkFaucet"

run_sigma(){
  local SIG=$1 TAG=$2
  for t in $TASKS; do
    F=$LH/data/robocasa/hdf5/$t.hdf5
    OUT=$LH/labels/final3_env_${TAG}_$t.npz
    [ -f "$OUT" ] && { say "exists ${TAG} $t"; continue; }
    [ -f "$F" ] || { say "NO_DATA $t"; continue; }
    until [ "$(awk '{printf "%d", $1}' /proc/loadavg)" -lt 70 ]; do sleep 300; done
    say "LABELER_START ${TAG} $t"
    cd $ROOT
    MUJOCO_GL=egl $RCPY impl/robocasa/ftle_labeler_robocasa.py \
      --dataset "$F" --output "$OUT" \
      --sigma-u $SIG --sigma-u-source sweep \
      --n-demos 200 --workers 24 --k-horizon 24 \
      >> $ROOT/results/robocasa/label3_${TAG}_$t.log 2>&1 \
      && say "LABELER_DONE ${TAG} $t" || say "LABELER_FAIL ${TAG} $t"
  done
}

say "SIGMA_SWEEP_START (immediate, parallel streams)"
run_sigma 0.025 sig025 &
run_sigma 0.100 sig100 &
wait
say "SIGMA_SWEEP_DONE"
