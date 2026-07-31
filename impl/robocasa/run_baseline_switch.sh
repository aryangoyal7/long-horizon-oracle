#!/bin/bash
# Inference-only literature triggers as controllers on RoboCasa + GR00T
# (2026-07-31). 4 signals x 4 tasks = 16 cells, n=50, (16,1), every signal
# thresholded to the oracle's per-task firing rate so the intervention budget
# is equal. Queue is worked by 4 persistent GPU workers on the GPUs the
# closed-loop probes are not using; workers pick the next cell when free.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/baseline
LOG=$ROOT/results/robocasa/baseline.log
QUEUE=$OUT/queue.txt
mkdir -p $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

if [ ! -f $QUEUE ]; then
  : > $QUEUE
  for s in spread cons cons_cos cmd_speed_min; do
    for t in TurnOnSinkFaucet PickPlaceCounterToCabinet OpenDrawer OpenCabinet; do
      echo "$t $s" >> $QUEUE
    done
  done
fi

worker(){
  local GPU=$1 line t s name
  while true; do
    line=$(flock $QUEUE.lock -c "head -1 $QUEUE; sed -i 1d $QUEUE") || break
    [ -z "$line" ] && break
    t=$(echo $line | cut -d' ' -f1); s=$(echo $line | cut -d' ' -f2)
    name=${t}_${s}_16-1
    [ -f $OUT/$name.json ] && { say "skip $name"; continue; }
    say "CELL_START $name (gpu $GPU)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_baseline_switch.py \
      --env-name $t --signal $s --k-stable 16 --k-unstable 1 --n-episodes 50 \
      --out $OUT/$name.json > $OUT/$name.log 2>&1 \
      && say "CELL_DONE $name" || say "CELL_FAIL $name"
  done
  say "worker gpu $GPU drained"
}

touch $QUEUE.lock
say "=== baseline switching start, $(wc -l < $QUEUE) cells queued ==="
worker 0 & worker 1 & worker 3 & worker 5 &
wait
say "BASELINE_SWITCH_ALL_DONE"
