#!/bin/bash
# v3 labeling, ALL FOUR TASKS IN PARALLEL (2026-07-30): the tasks are
# independent and v3 is the critical path for the predictor experiments, so
# it gets the whole CPU pool (4 x 24 workers). The sigma sweep (0.025/0.100)
# is chained AFTER all v3 files exist.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
RCPY=$LH/envs/rc/bin/python
LOG=$ROOT/results/robocasa/labeling_v3.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "V3_PARALLEL_START (4 tasks concurrent)"

label(){
  local t=$1
  local OUT=$LH/labels/final3_env_$t.npz
  [ -f "$OUT" ] && { say "labels exist $t"; return 0; }
  say "LABELER_START $t"
  cd $ROOT
  MUJOCO_GL=egl $RCPY impl/robocasa/ftle_labeler_robocasa.py \
    --dataset $LH/data/robocasa/hdf5/$t.hdf5 --output "$OUT" \
    --sigma-u 0.05 --sigma-u-source placeholder \
    --n-demos 200 --workers 24 --k-horizon 24 \
    >> $ROOT/results/robocasa/label3_$t.log 2>&1 \
    && say "LABELER_DONE $t" || say "LABELER_FAIL $t"
}
label rc_OpenCabinet & label rc_OpenDrawer &
label rc_PickPlaceCounterToCabinet & label rc_TurnOnSinkFaucet &
wait
say "V3_PARALLEL_DONE, launching sigma sweep"
setsid nohup bash $ROOT/impl/robocasa/run_sigma_sweep_labeling.sh \
  > $LH/sigma_sweep.out 2>&1 < /dev/null &
