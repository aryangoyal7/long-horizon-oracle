#!/bin/bash
# Backfill the LIBERO predictor cells that fast-failed on the 8-vs-9-dim
# proprio mismatch (fixed in libero_ksweep.py). Runs on GPUs 4/5, one
# completed task per GPU, mirroring run_libero_ksweep.sh's Stage C exactly.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
MGPY=$LH/envs/mg/bin/python
DATA=$LH/data/libero/libero_10
KOUT=$ROOT/results/stage2/ksweep_libero
HOUT=$ROOT/results/predictor/libero
LOG=$ROOT/results/stage2/libero_ksweep_chain.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$LH/repos/LIBERO
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # task ds ckpt gpu name extra...
  local task=$1 ds=$2 ckpt=$3 gpu=$4 name=$5; shift 5
  [ -f $KOUT/$name.json ] && return 0
  say "CELL_START $name (gpu $gpu, backfill)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $MGPY $ROOT/impl/libero/libero_ksweep.py \
      --dataset $ds --checkpoint "$ckpt" --horizon 600 --seed 0 \
      --out $KOUT/$name.json "$@" > $KOUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

backfill_task() {  # task epoch gpu
  local task=$1 ep=$2 GPU=$3 ds ck cond ku
  ds=$DATA/${task}_demo.hdf5
  ck=$(ls $ROOT/results/training/dp_libero_${task}/*/*/models/model_epoch_${ep}.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { say "BACKFILL_NO_CKPT $task"; return 1; }
  for cond in vj_in vis_in prop_in; do
    for ku in 4 1; do
      [ -f $HOUT/L_${cond}_s0/head.pt ] || { say "NO_HEAD $cond"; continue; }
      run_cell $task $ds "$ck" $GPU ${task}_predictor_${cond}_16-${ku} \
        --mode predictor --k-stable 16 --k-unstable $ku \
        --pred-head $HOUT/L_${cond}_s0/head.pt --n-episodes 50
    done
  done
  say "BACKFILL_TASK_DONE $task"
}

say "=== libero predictor backfill start ==="
backfill_task KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it 400 4 &
backfill_task LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket 1000 5 &
wait
say "LIBERO_BACKFILL_DONE"
