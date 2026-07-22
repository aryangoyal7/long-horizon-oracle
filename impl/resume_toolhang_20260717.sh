#!/bin/bash
set -u
LH_DIR=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
SETUP_LOG=$LH_DIR/results/setup_scratch_20260717.log
PY=/mnt/scratch/lh/envs/lh/bin/python
for i in $(seq 1 200); do
  grep -q SETUP_SCRATCH_COMPLETE "$SETUP_LOG" 2>/dev/null && break
  if ! pgrep -f "bash impl/setup_scratch.sh" > /dev/null; then
    bash "$LH_DIR/impl/setup_scratch.sh" >> "$SETUP_LOG" 2>&1
  fi
  sleep 60
done
grep -q SETUP_SCRATCH_COMPLETE "$SETUP_LOG" || { echo FATAL; exit 1; }
declare -A GPU=( [tool_hang]=3 [tool_hang_s2]=6 )
for t in tool_hang tool_hang_s2; do
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU[$t]} \
    setsid nohup bash -c "yes y | $PY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py \
      --config $LH_DIR/impl/configs/dp_${t}.json --resume" \
      >> "$LH_DIR/results/training/dp_${t}.launch.log" 2>&1 < /dev/null &
  echo "launched dp_$t on GPU ${GPU[$t]} (pid $!)"
done
# restart the artifact sync loop too
setsid nohup bash "$LH_DIR/impl/sync_scratch_artifacts.sh" > /dev/null 2>&1 < /dev/null &
echo "TOOLHANG_RESUMED $(date -u +%F' '%H:%M)"
