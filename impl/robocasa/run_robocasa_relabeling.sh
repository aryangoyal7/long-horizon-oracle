#!/bin/bash
# RoboCasa labeling v3 (2026-07-29): rerun of the v2 chain after finding the
# LeRobot->env action-order mismatch. lerobot_to_hdf5.py stores actions as
# [base3, torso1, mode1, arm6, grip1] but the composite controller consumes
# [arm6, grip1, base3, torso1, mode1]; v2 labels (final2_ol_*) replayed
# scrambled actions. ftle_labeler_robocasa.py now remaps at load. Same
# params as v2 (n-demos 200, workers 24, k-horizon 24, sigma 0.05, seed 0)
# so v3 differs from v2 only by the fix.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
RCPY=$LH/envs/rc/bin/python
LOG=$ROOT/results/robocasa/labeling_v3.log
mkdir -p $LH/labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "RC_LABELING_V3_START (env-order remap)"

for F in $LH/data/robocasa/hdf5/rc_*.hdf5; do
  t=$(basename "$F" .hdf5)
  OUT=$LH/labels/final3_env_$t.npz
  if [ -f "$OUT" ]; then
    say "labels exist $t"; continue
  fi
  until [ "$(awk '{printf "%d", $1}' /proc/loadavg)" -lt 70 ]; do sleep 300; done
  say "LABELER_START $t"
  cd $ROOT
  MUJOCO_GL=egl $RCPY impl/robocasa/ftle_labeler_robocasa.py \
    --dataset "$F" --output "$OUT" \
    --sigma-u 0.05 --sigma-u-source placeholder \
    --n-demos 200 --workers 24 --k-horizon 24 \
    >> $ROOT/results/robocasa/label3_$t.log 2>&1 \
    && say "LABELER_DONE $t" || say "LABELER_FAIL $t"
done
say "RC_LABELING_V3_DONE"
