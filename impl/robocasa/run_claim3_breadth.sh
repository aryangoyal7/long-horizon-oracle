#!/bin/bash
# Claim-3 breadth: eight NEW RoboCasa atomic tasks (2026-07-31).
# The GR00T checkpoint is multitask, so a new task needs no training, and the
# oracle needs no offline labels: a shadow pass that measures lambda at every
# replan without acting supplies both the fixed-16 baseline AND the online
# lambda distribution from which delta is set at a target firing rate. This
# takes claim 3 from 4 tasks to 12 and makes a suite average meaningful.
#
# Phase 1 per task: shadow16 (fixed-16 success + lambda log), fixed8, fixed1.
# Phase 2 per task: oracle (16,1) at delta = the 90th percentile of that
#   task's own online lambda, i.e. a 10 percent firing budget.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
OUT=$ROOT/results/robocasa/breadth
LOG=$ROOT/results/robocasa/breadth.log
Q=$OUT/queue.txt
mkdir -p $OUT
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
TASKS="CloseDrawer TurnOnStove PickPlaceCounterToSink OpenMicrowave \
       CloseDishwasher PickPlaceCounterToStove TurnOffSinkFaucet OpenFridge"

if [ ! -f $Q ]; then
  : > $Q
  for t in $TASKS; do
    echo "$t shadow16" >> $Q
    echo "$t fixed8"   >> $Q
    echo "$t fixed1"   >> $Q
  done
fi
touch $Q.lock
say "=== claim3 breadth start, $(wc -l < $Q) phase-1 cells ==="

run1(){  # gpu
  local GPU=$1 line t kind name
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q") || break
    [ -z "$line" ] && break
    t=$(echo $line|cut -d' ' -f1); kind=$(echo $line|cut -d' ' -f2)
    name=${t}_${kind}
    [ -f $OUT/$name.json ] && { say "skip $name"; continue; }
    say "CELL_START $name (gpu $GPU)"
    cd $FORK
    case $kind in
      # shadow needs lambda at every replan; the plain fixed-k cells do not,
      # and paying for it made fixed1 ~20x slower than the run it baselines.
      # contact mode computes a cheap privileged flag instead, and with
      # k_stable == k_unstable the chunk length never changes either way.
      shadow16) ARGS="--control oracle  --delta 99 --k-stable 16 --k-unstable 16" ;;
      fixed8)   ARGS="--control contact --delta 99 --k-stable 8  --k-unstable 8"  ;;
      fixed1)   ARGS="--control contact --delta 99 --k-stable 1  --k-unstable 1"  ;;
    esac
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU \
      $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py --env-name $t $ARGS \
      --n-episodes 50 --seed 0 --out $OUT/$name.json > $OUT/$name.log 2>&1 \
      && say "CELL_DONE $name" || say "CELL_FAIL $name"
  done
}
run1 2 & run1 4 & run1 5 & run1 6 &
wait
say "=== phase 1 done, computing deltas ==="

python3 - <<'PY' >> $LOG 2>&1
import json, glob, os
import numpy as np
OUT="/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/robocasa/breadth"
d={}
for f in sorted(glob.glob(f"{OUT}/*_shadow16.json")):
    r=json.load(open(f))
    lam=np.array([x["lam"] for e in r["episodes"] for x in e["replans"] if "lam" in x])
    if len(lam) < 100: continue
    d[r["env"]]={"delta": float(np.percentile(lam, 90)), "n": int(len(lam)),
                 "fixed16": r["success_rate"]}
json.dump(d, open(f"{OUT}/deltas.json","w"), indent=1)
print("deltas:", json.dumps(d, indent=1))
PY

say "=== phase 2: oracle cells ==="
: > $Q
for t in $TASKS; do echo "$t oracle" >> $Q; done
run2(){
  local GPU=$1 line t name thr
  while true; do
    line=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q") || break
    [ -z "$line" ] && break
    t=$(echo $line|cut -d' ' -f1); name=${t}_oracle_16-1
    [ -f $OUT/$name.json ] && continue
    thr=$(python3 -c "import json;print(json.load(open('$OUT/deltas.json')).get('$t',{}).get('delta',''))")
    [ -z "$thr" ] && { say "NO_DELTA $t"; continue; }
    say "CELL_START $name (gpu $GPU, delta $thr)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
      --env-name $t --control oracle --delta $thr --k-stable 16 --k-unstable 1 \
      --n-episodes 50 --seed 0 --out $OUT/$name.json > $OUT/$name.log 2>&1 \
      && say "CELL_DONE $name" || say "CELL_FAIL $name"
  done
}
run2 2 & run2 4 & run2 5 & run2 6 &
wait
say "CLAIM3_BREADTH_DONE"
