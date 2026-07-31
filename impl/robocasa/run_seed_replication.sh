#!/bin/bash
# Multi-seed replication of the RoboCasa switching cells (2026-07-31).
# Every headline cell so far is a single 50-episode run, where the gap between
# two identical runs has a standard error near 10 points. Published work in
# this area uses 3 seeds or more. This adds seeds 1 and 2 for the vanilla
# baseline, the oracle, and both learned predictors on all four tasks.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
HOUT=$ROOT/results/predictor/rc_seq_v3
OUT=$ROOT/results/robocasa/seeds
LOG=$ROOT/results/robocasa/seeds.log
Q=$OUT/queue.txt
mkdir -p $OUT
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
declare -A D=( [OpenCabinet]=0.0848 [OpenDrawer]=0.0760 \
               [PickPlaceCounterToCabinet]=0.0649 [TurnOnSinkFaucet]=0.0579 )

if [ ! -f $Q ]; then
  : > $Q
  for s in 1 2; do
    for t in TurnOnSinkFaucet PickPlaceCounterToCabinet OpenCabinet OpenDrawer; do
      echo "$t fixed16 $s" >> $Q
      echo "$t oracle $s"  >> $Q
      echo "$t h1 $s"      >> $Q
      echo "$t h4 $s"      >> $Q
    done
  done
fi
touch $Q.lock
say "=== seed replication start, $(wc -l < $Q) cells queued ==="

worker(){
  local GPU=$1 line t kind s name thr
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q") || break
    [ -z "$line" ] && break
    t=$(echo $line|cut -d' ' -f1); kind=$(echo $line|cut -d' ' -f2); s=$(echo $line|cut -d' ' -f3)
    name=${t}_${kind}_s${s}
    [ -f $OUT/$name.json ] && { say "skip $name"; continue; }
    say "CELL_START $name (gpu $GPU)"
    cd $FORK
    case $kind in
      fixed16)
        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU \
        $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py --env-name $t --control oracle \
          --delta 99 --k-stable 16 --k-unstable 16 --n-episodes 50 --seed $s \
          --out $OUT/$name.json > $OUT/$name.log 2>&1 ;;
      oracle)
        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU \
        $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py --env-name $t --control oracle \
          --delta ${D[$t]} --k-stable 16 --k-unstable 1 --n-episodes 50 --seed $s \
          --out $OUT/$name.json > $OUT/$name.log 2>&1 ;;
      h1|h4)
        thr=$(python3 -c "import json;print(json.load(open('$HOUT/online_thresholds.json'))['${t}_${kind}']['thresh'])")
        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU \
        $GPY $ROOT/impl/robocasa/groot_switch_rollout.py --env-name $t \
          --pred-head $HOUT/S_${t}_${kind}_s0/head.pt \
          --bridge $ROOT/impl/eval/predictor_bridge_seq.py \
          --frames-window $(( ${kind#h} * 16 )) --fire-on lam --delta $thr \
          --k-stable 16 --k-unstable 1 --n-episodes 50 --seed $s \
          --out $OUT/$name.json > $OUT/$name.log 2>&1 ;;
    esac
    [ -f $OUT/$name.json ] && say "CELL_DONE $name" || say "CELL_FAIL $name"
  done
}
worker 0 & worker 1 & worker 3 & worker 7 &
wait
say "SEED_REPLICATION_DONE"
