#!/bin/bash
# @reboot recovery: rebuild scratch, resume unfinished trainings, restart sync loop.
# Idempotent; safe if nothing needs recovering.
LH_DIR=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$LH_DIR/results/boot_recover_$(date +%Y%m%d_%H%M).log
exec > "$LOG" 2>&1
echo "boot_recover $(date -u)"
[ -x /mnt/scratch/lh/envs/lh/bin/python ] && { echo "scratch alive, nothing to do"; exit 0; }
bash $LH_DIR/impl/setup_scratch.sh
PY=/mnt/scratch/lh/envs/lh/bin/python
# resume any training not yet at epoch 2000
declare -A GPU=( [lift]=0 [can]=1 [square]=2 [tool_hang]=3 [square_s2]=5 [tool_hang_s2]=6 )
for t in "${!GPU[@]}"; do
  last=$(grep -aoE "Epoch [0-9]+" $LH_DIR/results/training/dp_${t}.launch.log 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  [ "${last:-0}" -ge 2000 ] && continue
  cfg=$LH_DIR/impl/configs/dp_${t}.json
  [ -f "$cfg" ] || continue
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU[$t]} setsid nohup bash -c \
    "yes y | $PY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py --config $cfg --resume" \
    >> $LH_DIR/results/training/dp_${t}.launch.log 2>&1 < /dev/null &
  echo "resumed dp_$t on GPU ${GPU[$t]}"
done
# reseed labels/rollouts and restart the artifact sync loop
mkdir -p /mnt/scratch/lh/{labels,features,rollouts}
rsync -a $LH_DIR/artifacts/labels/ /mnt/scratch/lh/labels/
rsync -a $LH_DIR/artifacts/rollouts/ /mnt/scratch/lh/rollouts/
setsid nohup bash $LH_DIR/impl/sync_scratch_artifacts.sh > /dev/null 2>&1 < /dev/null &
echo "BOOT_RECOVER_DONE"
