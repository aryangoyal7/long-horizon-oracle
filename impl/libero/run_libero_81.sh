#!/bin/bash
# LIBERO (8,1) switching cells for K3 and K4, per the k_unstable=1 protocol.
# Gated: each stream waits for its GPU's current RoboCasa job to finish.
#   GPU 3: after PP oracle replication -> K4 (8,1) vj/vis/prop
#   GPU 7: after TOSF pred 8-1 (end of learned 8-1 chain) -> K3 (8,1) vj/vis/prop
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
  say "CELL_START $name (gpu $gpu, 8-1 protocol)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $MGPY $ROOT/impl/libero/libero_ksweep.py \
      --dataset $ds --checkpoint "$ckpt" --horizon 600 --seed 0 \
      --out $KOUT/$name.json "$@" > $KOUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

task_81() {  # task epoch gpu
  local task=$1 ep=$2 GPU=$3 ds ck cond
  ds=$DATA/${task}_demo.hdf5
  ck=$(ls $ROOT/results/training/dp_libero_${task}/*/*/models/model_epoch_${ep}.pth 2>/dev/null | head -1)
  [ -n "$ck" ] || { say "81_NO_CKPT $task"; return 1; }
  for cond in vj_in vis_in prop_in; do
    [ -f $HOUT/L_${cond}_s0/head.pt ] || { say "NO_HEAD $cond"; continue; }
    run_cell $task $ds "$ck" $GPU ${task}_predictor_${cond}_8-1 \
      --mode predictor --k-stable 8 --k-unstable 1 \
      --pred-head $HOUT/L_${cond}_s0/head.pt --n-episodes 50
  done
  say "81_TASK_DONE $task"
}

say "=== libero 8-1 cells armed (waiting for gpu 3 and 7) ==="
( while [ ! -f $ROOT/results/robocasa/oracle/PickPlaceCounterToCabinet_oracle_16-1_s1.json ]; do sleep 180; done
  task_81 KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it 400 3 ) &
( while [ ! -f $ROOT/results/robocasa/switch_81/TurnOnSinkFaucet_pred_8-1.json ]; do sleep 180; done
  task_81 KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it 400 7 ) &
wait
say "LIBERO_81_DONE"
