#!/usr/bin/env bash
# Long-horizon labeling chain for MimicGen Kitchen_D0.
#
# Kitchen_D0 was chosen over the other three Stage 2 long-horizon sets because it is
# the only one whose policy clears the 0.9-1.0 bar used for the robomimic closed-loop
# channel (1.00 success at epochs 1000-2000, 25/25 episodes), and its demos average
# 617 steps (min 585, max 654) -- the longest well-solved horizon in the project.
# coffee_preparation_d0 is nominally longer (1140 horizon) but its policy sits at 0.04.
#
# Chain:
#   1. wait for kitchen_d0_image.hdf5 (rebuild_scratch.sh step 5 produces it on GPU 6)
#   2. sigma_u = policy validation action RMSE, lh venv, epoch-1000 checkpoint
#   3. wait for the encoder-ablation grid to release the CPUs (labeler is CPU-bound
#      MuJoCo physics; the grid runs 16 cells at ~700% CPU each)
#   4. open-loop FTLE labeler, mg venv, same constants as every other dataset
#      (--pos-only --k-horizon 24, N=8, stride 2)
#
# Run: setsid nohup bash impl/mimicgen/run_kitchen_labeling.sh > results/stage2/kitchen_label_chain.log 2>&1 &
set -u

LH=/mnt/scratch/lh
LH_DIR=/mnt/batch/tasks/shared/LS_root/mounts/clusters/garyan181/code/Users/garyan18/long-horizon
LHPY=$LH/envs/lh/bin/python
MGPY=$LH/envs/mg/bin/python

TASK=kitchen_d0
RAW=$LH/data/mimicgen/$TASK.hdf5
IMG=$LH/data/mimicgen/${TASK}_image.hdf5
CKPT=$LH_DIR/results/training/dp_mg_kitchen_d0/dp_mg_kitchen_d0_image/20260721135025/models/model_epoch_1000.pth
RMSE_OUT=$LH_DIR/results/rmse/mg_$TASK.json
LABELS=$LH/labels/final2_ol_mg_$TASK.npz
N_DEMOS=200          # same 200-of-1000 convention as stack_d0 / square_d0
WORKERS=40
LOAD_CEIL=200        # 96 cores; the ablation grid drives load past 1000

ts() { date -u +%H:%M:%S; }
say() { echo "[$(ts)] $*"; }

say "KITCHEN_CHAIN_START task=$TASK"

# ---- 1. wait for the image conversion the rebuild is producing ----------------
if [ ! -f "$RMSE_OUT" ]; then
  say "waiting for $IMG (rebuild_scratch.sh step 5, GPU 6)"
  for i in $(seq 1 720); do            # up to 12 h
    [ -f "$IMG" ] && { say "image hdf5 present after ${i} min"; break; }
    sleep 60
  done
  if [ ! -f "$IMG" ]; then
    say "KITCHEN_CHAIN_FAIL image conversion never appeared"; exit 1
  fi
  # the converter writes incrementally; wait for the file to stop growing
  prev=0
  while :; do
    cur=$(stat -c%s "$IMG" 2>/dev/null || echo 0)
    [ "$cur" = "$prev" ] && [ "$cur" != "0" ] && break
    prev=$cur; sleep 60
  done
  say "image hdf5 settled at $(stat -c%s "$IMG") bytes"

  # ---- 2. sigma_u = validation action RMSE ------------------------------------
  say "measuring sigma_u from $(basename "$CKPT") (success 1.00)"
  CUDA_VISIBLE_DEVICES=3 $LHPY $LH_DIR/impl/eval/policy_action_rmse.py \
      --ckpt "$CKPT" --dataset "$IMG" --out "$RMSE_OUT" \
      >> $LH_DIR/results/rmse/run.log 2>&1 \
    || { say "KITCHEN_CHAIN_FAIL rmse"; exit 1; }
  say "RMSE_DONE -> $RMSE_OUT"
else
  say "sigma_u already measured, reusing $RMSE_OUT"
fi

SIGMA=$($LHPY -c "import json;print('%.4f'%json.load(open('$RMSE_OUT'))['pos_action_rmse'])")
say "sigma_u = $SIGMA (source: policy_rmse, kitchen_d0 epoch 1000)"

# ---- 3. wait for the ablation grid to release the CPUs ------------------------
say "waiting for CPU (labeler is CPU-bound; ceiling load $LOAD_CEIL)"
for i in $(seq 1 720); do
  load=$(awk '{printf "%d", $1}' /proc/loadavg)
  [ "$load" -lt "$LOAD_CEIL" ] && { say "load $load, proceeding after ${i} min"; break; }
  sleep 60
done

# ---- 4. open-loop FTLE labeler ------------------------------------------------
say "LABELER_START demos=$N_DEMOS workers=$WORKERS k=24 sigma_u=$SIGMA"
cd "$LH_DIR" || exit 1
MUJOCO_GL=egl $MGPY impl/labeler/ftle_labeler.py \
    --dataset "$RAW" --output "$LABELS" \
    --sigma-u "$SIGMA" --sigma-u-source policy_rmse \
    --n-demos $N_DEMOS --workers $WORKERS --pos-only --k-horizon 24 \
  && say "LABELER_DONE -> $LABELS" \
  || { say "KITCHEN_CHAIN_FAIL labeler"; exit 1; }

$MGPY impl/labeler/analyze_labels.py --labels "$LABELS" \
    --out $LH_DIR/results/mimicgen/analyze_mg_$TASK.json 2>&1 | tail -25
say "KITCHEN_CHAIN_DONE"
