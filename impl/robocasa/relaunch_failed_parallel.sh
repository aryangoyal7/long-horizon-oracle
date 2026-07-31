#!/bin/bash
# Relaunch every cell with no result JSON, distributed across all 8 GPUs.
#
# BUG THIS FIXES (2026-07-31): the previous launcher assigned GPUs with
#     G=0; next_gpu(){ echo $G; G=$(( (G+1) % 8 )); };  g=$(next_gpu)
# Command substitution runs in a subshell, so the increment was thrown away
# every call and G stayed 0. All 53 processes were sent to GPU 0, which then
# ran out of memory and killed 33 of them. The counter is now incremented in
# the parent shell with no subshell in the path.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
AZPY=/anaconda/envs/azureml_py38/bin/python3
FORK=$LH/repos/Isaac-GR00T-rc365
HOUT=$ROOT/results/predictor/rc_seq_v3
LOG=$ROOT/results/robocasa/parallel.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
declare -A D=( [OpenCabinet]=0.0848 [OpenDrawer]=0.0760 \
               [PickPlaceCounterToCabinet]=0.0649 [TurnOnSinkFaucet]=0.0579 )
# GPU 0 already carries the head-training jobs, so start the rollouts at 1.
G=1
say "=== relaunch, fixed GPU round-robin ==="

OUT=$ROOT/results/robocasa/breadth
while read -r t kind; do
  [ -z "${t:-}" ] && continue
  name=${t}_${kind}; [ -f $OUT/$name.json ] && continue
  pgrep -f "breadth/$name.json" >/dev/null && { say "running already $name"; continue; }
  case $kind in
    shadow16) A="--control oracle  --delta 99 --k-stable 16 --k-unstable 16" ;;
    fixed8)   A="--control contact --delta 99 --k-stable 8  --k-unstable 8"  ;;
    fixed1)   A="--control contact --delta 99 --k-stable 1  --k-unstable 1"  ;;
  esac
  g=$G; G=$(( (G+1) % 8 ))
  ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
    --env-name $t $A --n-episodes 50 --seed 0 --out $OUT/$name.json \
    > $OUT/$name.log 2>&1 ) &
  say "LAUNCH breadth $name gpu $g"
done < /tmp/q_breadth.txt

OUT=$ROOT/results/robocasa/seeds
while read -r t kind s; do
  [ -z "${t:-}" ] && continue
  name=${t}_${kind}_s${s}; [ -f $OUT/$name.json ] && continue
  pgrep -f "seeds/$name.json" >/dev/null && { say "running already $name"; continue; }
  g=$G; G=$(( (G+1) % 8 ))
  case $kind in
    fixed16) ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
        CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
        --env-name $t --control contact --delta 99 --k-stable 16 --k-unstable 16 \
        --n-episodes 50 --seed $s --out $OUT/$name.json > $OUT/$name.log 2>&1 ) & ;;
    oracle)  ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
        CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
        --env-name $t --control oracle --delta ${D[$t]} --k-stable 16 --k-unstable 1 \
        --n-episodes 50 --seed $s --out $OUT/$name.json > $OUT/$name.log 2>&1 ) & ;;
    h1|h4)
      thr=$($AZPY -c "import json;print(json.load(open('$HOUT/online_thresholds.json'))['${t}_${kind}']['thresh'])")
      ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
        CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_switch_rollout.py \
        --env-name $t --pred-head $HOUT/S_${t}_${kind}_s0/head.pt \
        --bridge $ROOT/impl/eval/predictor_bridge_seq.py \
        --frames-window $(( ${kind#h} * 16 )) --fire-on lam --delta $thr \
        --k-stable 16 --k-unstable 1 --n-episodes 50 --seed $s \
        --out $OUT/$name.json > $OUT/$name.log 2>&1 ) & ;;
  esac
  say "LAUNCH seed $name gpu $g"
done < /tmp/q_seeds.txt

say "=== relaunch pass complete ==="
wait
say "RELAUNCH_DONE"
