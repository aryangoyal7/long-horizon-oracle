#!/bin/bash
# Full ACT program on RoboCasa: render training data from the labeled demos,
# train ACT per task, then evaluate fixed k in {16,8,4,1} and oracle (16,1)
# at the label threshold. Third policy class for the paper.
#   Render: GPU 4 (co-located, EGL render is light; physics is CPU)
#   Train+eval OC, PP:  GPU 3 after the LIBERO K4 (8,1) cells finish
#   Train+eval OD, TOSF: GPU 7 after the LIBERO K3 (8,1) cells finish
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
ACT=$ROOT/impl/robocasa/act
DEMOS=$LH/data/robocasa/hdf5
DATA=$LH/data/robocasa/act
TRAIN=$ROOT/results/training
OUT=$ROOT/results/robocasa/act
LOG=$ROOT/results/robocasa/act_pipeline.log
mkdir -p $DATA $OUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

declare -A HOR=( [OpenCabinet]=1050 [OpenDrawer]=750
                 [PickPlaceCounterToCabinet]=750 [TurnOnSinkFaucet]=600 )
declare -A DELTA=( [OpenCabinet]=0.057841 [OpenDrawer]=0.054712
                   [PickPlaceCounterToCabinet]=0.035765
                   [TurnOnSinkFaucet]=0.077113 )

render_task() {  # task gpu
  local TASK=$1 GPU=$2
  [ -f $DATA/${TASK}_act.hdf5 ] && return 0
  say "RENDER_START $TASK (gpu $GPU)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$GPU $PY $ACT/render_act_dataset.py \
      --hdf5 $DEMOS/rc_$TASK.hdf5 --out $DATA/${TASK}_act.hdf5.tmp \
      > $OUT/render_$TASK.log 2>&1 \
    && mv $DATA/${TASK}_act.hdf5.tmp $DATA/${TASK}_act.hdf5 \
    || { say "RENDER_FAIL $TASK"; return 1; }
  say "RENDER_DONE $TASK"
}

train_eval_task() {  # task gpu
  local TASK=$1 GPU=$2
  while [ ! -f $DATA/${TASK}_act.hdf5 ]; do sleep 300; done
  local CK=$TRAIN/act_rc_$TASK/act_final.pt
  if [ ! -f $CK ]; then
    say "TRAIN_START $TASK (gpu $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $PY $ACT/train_act.py \
      --data $DATA/${TASK}_act.hdf5 --out $TRAIN/act_rc_$TASK \
      > $OUT/train_$TASK.log 2>&1 \
      || { say "TRAIN_FAIL $TASK"; return 1; }
    say "TRAIN_DONE $TASK"
  fi
  local k name
  for k in 16 8 4 1; do
    name=${TASK}_act_fixed_$k
    [ -f $OUT/$name.json ] && continue
    say "CELL_START $name (gpu $GPU)"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $PY $ACT/act_rollout.py \
        --hdf5 $DEMOS/rc_$TASK.hdf5 --ckpt $CK --mode fixed --k $k \
        --horizon ${HOR[$TASK]} --n-episodes 50 --out $OUT/$name.json \
        > $OUT/$name.log 2>&1 || { say "CELL_FAIL $name"; continue; }
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
  done
  name=${TASK}_act_oracle_16-1
  if [ ! -f $OUT/$name.json ]; then
    say "CELL_START $name (gpu $GPU)"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $PY $ACT/act_rollout.py \
        --hdf5 $DEMOS/rc_$TASK.hdf5 --ckpt $CK --mode oracle \
        --k-stable 16 --k-unstable 1 --delta ${DELTA[$TASK]} \
        --horizon ${HOR[$TASK]} --n-episodes 50 --out $OUT/$name.json \
        > $OUT/$name.log 2>&1 || say "CELL_FAIL $name"
    say "CELL_DONE $name $(python3 -c "import json;print(json.load(open('$OUT/$name.json'))['success_rate'])" 2>/dev/null)"
  fi
  say "ACT_TASK_DONE $TASK"
}

say "=== ACT pipeline start ==="
( render_task OpenCabinet 4
  render_task PickPlaceCounterToCabinet 4
  render_task OpenDrawer 4
  render_task TurnOnSinkFaucet 4
  say "ACT_RENDER_ALL_DONE" ) &

KL=$ROOT/results/stage2/ksweep_libero
K4=KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
K3=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
( while [ ! -f $KL/${K4}_predictor_prop_in_8-1.json ]; do sleep 300; done
  train_eval_task OpenCabinet 3
  train_eval_task PickPlaceCounterToCabinet 3 ) &
( while [ ! -f $KL/${K3}_predictor_prop_in_8-1.json ]; do sleep 300; done
  train_eval_task OpenDrawer 7
  train_eval_task TurnOnSinkFaucet 7 ) &
wait
say "ACT_PIPELINE_DONE"
