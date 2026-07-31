#!/bin/bash
# Resume checkpoint selection after the Jul 22 14:55 UTC machine stop.
# Missing: kitchen epoch 2000; coffee epochs 1000/1400/1700/2000.
# Runs each missing eval on its own GPU, appends the per-task DONE markers
# and the final CKPT_SELECT_ALL_DONE to the same ckpt_select_run.log that
# auto_shutdown.sh watches.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MGPY=/mnt/scratch/lh/envs/mg/bin/python
MG=/mnt/scratch/lh/data/mimicgen
OUT=$LH_DIR/results/stage2/ckpt_select
RUNLOG=$LH_DIR/results/stage2/ckpt_select_run.log

run_eval() { # task epoch gpu horizon
  local t=$1 e=$2 gpu=$3 hor=$4
  local models=$(ls -d $LH_DIR/results/training/dp_mg_${t}/*/*/models | head -1)
  local ck=$models/model_epoch_${e}.pth
  [ -f $ck ] || { echo "CKPT_SELECT_MISSING_${t}_${e}" >> $RUNLOG; return; }
  echo "EVAL $t epoch $e (gpu $gpu) $(date -u +%H:%M:%S)" >> $RUNLOG
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $MGPY $LH_DIR/impl/mimicgen/bridge_eval.py \
      --dataset $MG/${t}_image.hdf5 --checkpoint $ck \
      --n-episodes 25 --horizon $hor --seed 0 \
      --out $OUT/${t}_epoch${e}.json \
      > $OUT/${t}_epoch${e}.log 2>&1 \
    || echo "CKPT_SELECT_EVAL_FAIL_${t}_${e}" >> $RUNLOG
}

run_eval kitchen_d0 2000 6 980 &
K=$!
( run_eval coffee_preparation_d0 1000 4 1140 ) &
( run_eval coffee_preparation_d0 1400 5 1140 ) &
( run_eval coffee_preparation_d0 1700 7 1140 ) &
wait
run_eval coffee_preparation_d0 2000 4 1140
echo "CKPT_SELECT_kitchen_d0_DONE $(date -u +%H:%M:%S)" >> $RUNLOG
echo "CKPT_SELECT_coffee_preparation_d0_DONE $(date -u +%H:%M:%S)" >> $RUNLOG
echo CKPT_SELECT_ALL_DONE >> $RUNLOG
