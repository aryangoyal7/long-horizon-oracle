#!/bin/bash
# RoboCasa labeling chain v2 (2026-07-25): consumes the converted
# robomimic-style hdf5s (lerobot_to_hdf5.py output) with the robocasa-specific
# labeler. Sequential per task with a load gate; labels land in the shared
# labels dir where the 15-min sync loop mirrors them to the share.
# Feature extraction + heads follow once the video-frame extractor exists
# (lerobot ships frames as mp4, not hdf5 images).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
RCPY=$LH/envs/rc/bin/python
LOG=$ROOT/results/robocasa/labeling.log
mkdir -p $LH/labels
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "RC_LABELING_V2_START"

until ls $LH/data/robocasa/hdf5/rc_*.hdf5 >/dev/null 2>&1; do sleep 120; done
for F in $LH/data/robocasa/hdf5/rc_*.hdf5; do
  t=$(basename "$F" .hdf5)
  OUT=$LH/labels/final2_ol_$t.npz
  if [ -f "$OUT" ] || [ -f $ROOT/artifacts/labels/final2_ol_$t.npz ]; then
    say "labels exist $t"; continue
  fi
  until [ "$(awk '{printf "%d", $1}' /proc/loadavg)" -lt 70 ]; do sleep 300; done
  say "LABELER_START $t (n-demos 200, workers 24, sigma placeholder 0.05)"
  cd $ROOT
  MUJOCO_GL=egl $RCPY impl/robocasa/ftle_labeler_robocasa.py \
    --dataset "$F" --output "$OUT" \
    --sigma-u 0.05 --sigma-u-source placeholder \
    --n-demos 200 --workers 24 --k-horizon 24 \
    >> $ROOT/results/robocasa/label_$t.log 2>&1 \
    && say "LABELER_DONE $t" || say "LABELER_FAIL $t"
done
say "RC_LABELING_V2_DONE (features/heads pending video-frame extractor)"
