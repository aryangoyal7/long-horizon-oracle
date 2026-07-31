#!/bin/bash
# Sequence-history predictor experiment (2026-07-30, user request).
# Per task, as soon as its v3 labels land: train the 4-clip history head and
# the matched 1-clip baseline (3 seeds each, v3 lambda targets, surviving
# V-JEPA features), then run online GR00T switching cells (16,1) n=50 driven
# by each head (seed 0). Two GPU streams: GPU 0 = OpenCabinet,
# PickPlaceCounterToCabinet; GPU 1 = OpenDrawer, TurnOnSinkFaucet.
# Offline metrics land in results/predictor/rc_seq_v3/*/metrics.json;
# online cells in results/robocasa/seqpred/.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
AZPY=/anaconda/envs/azureml_py38/bin/python3
GPY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
HOUT=$ROOT/results/predictor/rc_seq_v3
COUT=$ROOT/results/robocasa/seqpred
LOG=$ROOT/results/robocasa/seqpred.log
mkdir -p $HOUT $COUT $LH/features
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_task(){
  local t=$1 GPU=$2
  local L=$LH/labels/final3_env_rc_$t.npz
  local F=$LH/features/feat2_rc_$t.npz
  until [ -f "$L" ]; do sleep 600; done
  [ -f "$F" ] || cp $ROOT/artifacts/features/feat2_rc_$t.npz "$F"
  say "TASK_START $t (gpu $GPU, v3 labels present)"
  local hist s out
  for hist in 4 1; do
    for s in 0 1 2; do
      out=$HOUT/S_${t}_h${hist}_s$s
      [ -f $out/metrics.json ] && continue
      CUDA_VISIBLE_DEVICES=$GPU $AZPY $ROOT/impl/predictor/train_head_seq.py \
        --features "$F" --labels "$L" --hist $hist --seed $s \
        --out $out --tag S_${t}_h${hist}_s$s > $out.log 2>&1
      [ -f $out/metrics.json ] && say "HEAD_OK S_${t}_h${hist}_s$s" \
                               || say "HEAD_FAIL S_${t}_h${hist}_s$s"
    done
  done
  local name head win bridge
  for hist in 4 1; do
    name=${t}_h${hist}pred_16-1
    head=$HOUT/S_${t}_h${hist}_s0/head.pt
    [ -f "$head" ] || { say "CELL_SKIP $name (no head)"; continue; }
    [ -f $COUT/$name.json ] && continue
    win=$((hist * 16))
    say "CELL_START $name (gpu $GPU)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $GPY $ROOT/impl/robocasa/groot_switch_rollout.py \
      --env-name $t --pred-head "$head" \
      --bridge $ROOT/impl/eval/predictor_bridge_seq.py --frames-window $win \
      --k-stable 16 --k-unstable 1 --n-episodes 50 \
      --out $COUT/$name.json > $COUT/$name.log 2>&1 \
      && say "CELL_DONE $name" || say "CELL_FAIL $name"
  done
  say "TASK_DONE $t"
}

say "=== seq predictor chain start ==="
( run_task OpenCabinet 0; run_task PickPlaceCounterToCabinet 0 ) &
( run_task OpenDrawer 1; run_task TurnOnSinkFaucet 1 ) &
wait
say "SEQ_PREDICTOR_ALL_DONE"
