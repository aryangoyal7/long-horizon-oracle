#!/bin/bash
# Post-restart recovery chain: wait for setup_scratch.sh to finish, then resume all
# six DP trainings (4 seed-1 via the launcher on GPUs 0-3, 2 seed-2 on GPUs 5-6).
# Run detached (setsid) so it survives Claude session teardown. Idempotent-ish:
# if setup died, reruns it inline (setup_scratch.sh is itself idempotent).
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_LOG="$LH_DIR/results/setup_scratch_20260714.log"
PY=/mnt/scratch/lh/envs/lh/bin/python

# ---- wait for bootstrap -------------------------------------------------------
for i in $(seq 1 200); do
  grep -q SETUP_SCRATCH_COMPLETE "$SETUP_LOG" 2>/dev/null && break
  if ! pgrep -f "bash impl/setup_scratch.sh" > /dev/null; then
    echo "[chain] setup process gone without marker; rerunning inline ($(date -u +%H:%M))"
    bash "$LH_DIR/impl/setup_scratch.sh" >> "$SETUP_LOG" 2>&1
  fi
  sleep 60
done
if ! grep -q SETUP_SCRATCH_COMPLETE "$SETUP_LOG"; then
  echo "[chain] FATAL: setup never completed"; exit 1
fi
echo "[chain] setup complete ($(date -u +%H:%M)); resuming trainings"

# ---- seed-1 x4 (GPUs 0-3) via the validated launcher ---------------------------
bash "$LH_DIR/impl/configs/launch_dp_training.sh" --resume

# ---- seed-2 x2 (GPUs 5-6, matching the pre-restart allocation) ------------------
declare -A GPU2=( [square_s2]=5 [tool_hang_s2]=6 )
for t in square_s2 tool_hang_s2; do
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU2[$t]} \
    setsid nohup bash -c "yes y | $PY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py \
      --config $LH_DIR/impl/configs/dp_${t}.json --resume" \
      >> "$LH_DIR/results/training/dp_${t}.launch.log" 2>&1 < /dev/null &
  echo "[chain] launched dp_$t on GPU ${GPU2[$t]} (pid $!)"
done
echo "ALL_SIX_RESUMED $(date -u +%F' '%H:%M)"
