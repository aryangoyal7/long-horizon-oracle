#!/bin/bash
# LIBERO-Long rollout evaluation chain (native env harness libero_ksweep.py).
# Stage A: checkpoint selection (epochs 400/600/800/1000, 20 eps, k=16)
# Stage B: fixed-k sweep k in {16,8,4,1}, 50 eps, best checkpoint
# Stage C: predictor cells, heads L_vj_in/L_vis_in/L_prop_in s0, ku in {4,1}
# Two workers: GPU 0 = tasks 0-4, GPU 1 = tasks 5-9. Idempotent per-cell jsons.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
MGPY=$LH/envs/mg/bin/python
DATA=$LH/data/libero/libero_10
KOUT=$ROOT/results/stage2/ksweep_libero
HOUT=$ROOT/results/predictor/libero
TRAIN=$ROOT/results/training
LOG=$ROOT/results/stage2/libero_ksweep_chain.log
mkdir -p $KOUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$LH/repos/LIBERO
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # task ds ckpt gpu name extra...
  local task=$1 ds=$2 ckpt=$3 gpu=$4 name=$5; shift 5
  [ -f $KOUT/$name.json ] && return 0
  say "CELL_START $name (gpu $gpu)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $MGPY $ROOT/impl/libero/libero_ksweep.py \
      --dataset $ds --checkpoint "$ckpt" --horizon 600 --seed 0 \
      --out $KOUT/$name.json "$@" > $KOUT/$name.log 2>&1 \
    || { say "CELL_FAIL $name"; return 1; }
  say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

ckpt_for_epoch() {  # taskdir epoch
  ls $1/*/*/models/model_epoch_$2.pth 2>/dev/null | head -1
}

worker() {
  local GPU=$1; shift
  local ds task tdir ep best bestrate rate ck cond ku
  for ds in "$@"; do
    task=$(basename $ds _demo.hdf5)
    tdir=$TRAIN/dp_libero_$task
    [ -d "$tdir" ] || { say "NO_TRAINING_DIR $task"; continue; }
    # Stage A: checkpoint selection
    for ep in 400 600 800 1000; do
      ck=$(ckpt_for_epoch $tdir $ep)
      [ -n "$ck" ] && run_cell $task $ds "$ck" $GPU sel_${task}_e${ep} \
        --mode fixed --k 16 --n-episodes 20
    done
    best=""; bestrate=-1
    for ep in 400 600 800 1000; do
      rate=$(python3 -c "import json;print(json.load(open('$KOUT/sel_${task}_e${ep}.json'))['success_rate'])" 2>/dev/null) || continue
      # >= so later epochs win ties
      python3 -c "exit(0 if float('$rate') >= float('$bestrate') else 1)" && { best=$ep; bestrate=$rate; }
    done
    [ -n "$best" ] || { say "SELECT_FAIL $task"; continue; }
    say "SELECTED $task epoch $best (rate $bestrate)"
    ck=$(ckpt_for_epoch $tdir $best)
    # Stage B: fixed-k sweep
    for k in 16 8 4 1; do
      run_cell $task $ds "$ck" $GPU ${task}_fixed_${k} \
        --mode fixed --k $k --n-episodes 50
    done
    # Stage C: predictor cells, (16,1) always included
    for cond in vj_in vis_in prop_in; do
      for ku in 4 1; do
        [ -f $HOUT/L_${cond}_s0/head.pt ] || { say "NO_HEAD $cond"; continue; }
        run_cell $task $ds "$ck" $GPU ${task}_predictor_${cond}_16-${ku} \
          --mode predictor --k-stable 16 --k-unstable $ku \
          --pred-head $HOUT/L_${cond}_s0/head.pt --n-episodes 50
      done
    done
    say "TASK_DONE $task"
  done
}

say "=== libero ksweep chain start ==="
ALL=($DATA/*_demo.hdf5)
worker 0 "${ALL[@]:0:5}" &
worker 1 "${ALL[@]:5}" &
wait
say "LIBERO_KSWEEP_DONE"
