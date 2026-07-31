#!/bin/bash
# Launch every remaining cell at once, one process per cell, round-robin over
# the 8 GPUs (2026-07-31). Replaces the worker-queue scripts, which stalled
# twice: once on a lock race when the queue file was rewritten underneath a
# worker, and once because an in-flight worker had already parsed an older
# copy of its own script. Direct launch has no shared state to corrupt.
# Threads are held low per process since many run concurrently; GR00T
# inference is GPU-bound and MuJoCo stepping is single-threaded per env.
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
G=0
next_gpu(){ echo $G; G=$(( (G+1) % 8 )); }
declare -A D=( [OpenCabinet]=0.0848 [OpenDrawer]=0.0760 \
               [PickPlaceCounterToCabinet]=0.0649 [TurnOnSinkFaucet]=0.0579 )
say "=== direct parallel launch ==="

# ---- breadth phase 1 ----
OUT=$ROOT/results/robocasa/breadth
while read -r t kind; do
  [ -z "${t:-}" ] && continue
  name=${t}_${kind}
  [ -f $OUT/$name.json ] && continue
  pgrep -f "breadth/$name.json" >/dev/null && continue
  case $kind in
    shadow16) A="--control oracle  --delta 99 --k-stable 16 --k-unstable 16" ;;
    fixed8)   A="--control contact --delta 99 --k-stable 8  --k-unstable 8"  ;;
    fixed1)   A="--control contact --delta 99 --k-stable 1  --k-unstable 1"  ;;
  esac
  g=$(next_gpu)
  ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
    --env-name $t $A --n-episodes 50 --seed 0 --out $OUT/$name.json \
    > $OUT/$name.log 2>&1 ) &
  say "LAUNCH breadth $name gpu $g"
done < /tmp/q_breadth.txt

# ---- seed replication ----
OUT=$ROOT/results/robocasa/seeds
while read -r t kind s; do
  [ -z "${t:-}" ] && continue
  name=${t}_${kind}_s${s}
  [ -f $OUT/$name.json ] && continue
  pgrep -f "seeds/$name.json" >/dev/null && continue
  g=$(next_gpu)
  case $kind in
    fixed16)
      ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
        CUDA_VISIBLE_DEVICES=$g $GPY $ROOT/impl/robocasa/groot_oracle_rollout.py \
        --env-name $t --control contact --delta 99 --k-stable 16 --k-unstable 16 \
        --n-episodes 50 --seed $s --out $OUT/$name.json > $OUT/$name.log 2>&1 ) & ;;
    oracle)
      ( cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g PYOPENGL_PLATFORM=egl \
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

# ---- LIBERO head training ----
FE=$ROOT/artifacts/features; LA=$ROOT/artifacts/labels
HO=$ROOT/results/predictor/seq_allplat
while read -r d h; do
  [ -z "${d:-}" ] && continue
  out=$HO/A_${d}_h${h}_s0
  [ -f $out/metrics.json ] && continue
  FF=$FE/feat2_$d.npz; [ -f "$FF" ] || FF=$FE/feat_$d.npz
  [ -f "$FF" ] || { say "NOFEAT $d"; continue; }
  g=$(next_gpu)
  ( CUDA_VISIBLE_DEVICES=$g $AZPY $ROOT/impl/predictor/train_head_seq.py \
      --features $FF --labels $LA/final2_ol_$d.npz --hist $h --seed 0 \
      --out $out --tag A_${d}_h${h}_s0 > $out.log 2>&1 ) &
  say "LAUNCH head A_${d}_h${h}_s0 gpu $g"
done < /tmp/q_allplat.txt

say "=== all launched, $(jobs -p | wc -l) processes ==="
wait
say "ALL_PARALLEL_DONE"
