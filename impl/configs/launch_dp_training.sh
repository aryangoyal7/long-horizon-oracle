#!/bin/bash
# Launch (or resume) Diffusion Policy training for all 4 tasks, one GPU each (0-3).
#
# setsid + no-wait: each train.py gets its OWN session/process group, so it survives
# Claude session teardown (which SIGKILLs the launching process group — plain nohup
# does NOT protect against that; learned the hard way twice).
# Completion/crash detection is external (Monitor on the launch logs).
#
# Usage: bash launch_dp_training.sh [--resume]
set -euo pipefail
PY=/mnt/scratch/lh/envs/lh/bin/python
IMPL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$IMPL/../../results/training"
mkdir -p "$OUT"
RESUME="${1:-}"

declare -A GPU=( [lift]=0 [can]=1 [square]=2 [tool_hang]=3 )
for t in lift can square tool_hang; do
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU[$t]} \
    setsid nohup bash -c "yes y | $PY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py \
      --config $IMPL/dp_${t}.json ${RESUME:+--resume}" \
      >> "$OUT/dp_${t}.launch.log" 2>&1 < /dev/null &
  echo "launched dp_$t on GPU ${GPU[$t]} (pid $!, resume='${RESUME}')"
done
echo "LAUNCHED_DETACHED (no wait — jobs own their sessions)"
