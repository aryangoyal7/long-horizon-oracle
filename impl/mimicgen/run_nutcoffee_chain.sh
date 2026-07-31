#!/bin/bash
# In-domain labeling + predictor + switching eval for nut_assembly_d0 and
# coffee_preparation_d0 (user request 2026-07-25). Per task:
#   1. wait for the raw and image hdf5 (rebuild_scratch reconverts them)
#   2. sigma_u = policy validation action RMSE (patched fallback split)
#   3. open-loop FTLE labels, same constants as every other dataset
#   4. analyze (deadband, contact fractions)
#   5. V-JEPA feature extraction for the stamps
#   6. train 9 heads: {vj+proprio, vj vision-only, proprio-only} x seeds {0,1,2}
#   7. stage2_ksweep predictor cells with the 3 in-domain seed-0 heads, 50 eps
# Fixed-k and panda-pool-predictor baselines come from run_stage2_ksweep.sh.
# Idempotent: every step skips if its artifact already exists.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LHPY=$LH/envs/lh/bin/python
MGPY=$LH/envs/mg/bin/python
VJPY=$LH/envs/vjepa/bin/python
AZPY=/anaconda/envs/azureml_py38/bin/python3
MG=$LH/data/mimicgen
KOUT=$ROOT/results/stage2/ksweep
HOUT=$ROOT/results/predictor/mg_indomain
LOG=$ROOT/results/stage2/nutcoffee_chain.log
mkdir -p $HOUT $ROOT/results/rmse $ROOT/results/mimicgen
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

readable() { # hdf5 with >=900 demos
  $MGPY - "$1" <<'PYG' 2>/dev/null
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    sys.exit(0 if len(f["data"]) >= 900 else 1)
PYG
}
settled() { # file exists and size unchanged over 60 s
  local prev cur
  prev=$(stat -c%s "$1" 2>/dev/null || echo 0); [ "$prev" = 0 ] && return 1
  sleep 60; cur=$(stat -c%s "$1" 2>/dev/null || echo 0)
  [ "$cur" = "$prev" ]
}

pipeline() { # TASK HORIZON BEST_EPOCH GPU SHORT
  local TASK=$1 HOR=$2 BEP=$3 GPU=$4 SHORT=$5
  local RAW=$MG/$TASK.hdf5 IMG=$MG/${TASK}_image.hdf5
  local LABELS=$LH/labels/final2_ol_mg_$TASK.npz
  local FEATS=$LH/features/feat2_mg_$TASK.npz
  local RMSE=$ROOT/results/rmse/mg_$TASK.json

  say "CHAIN_START $TASK (gpu $GPU)"
  # ---- 1. datasets -----------------------------------------------------------
  until [ -f "$RAW" ] && readable "$RAW"; do sleep 300; done
  say "$TASK raw ready"
  until [ -f "$IMG" ] && settled "$IMG" && readable "$IMG"; do sleep 300; done
  say "$TASK image ready"
  local CK
  CK=$(ls -d $ROOT/results/training/dp_mg_$TASK/*/*/models | head -1)/model_epoch_${BEP}.pth
  [ -f "$CK" ] || { say "CHAIN_FAIL $TASK no checkpoint $CK"; return 1; }

  # ---- 2. sigma_u ------------------------------------------------------------
  if [ ! -f "$RMSE" ]; then
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU $LHPY $ROOT/impl/eval/policy_action_rmse.py \
      --ckpt "$CK" --dataset "$IMG" --out "$RMSE" >> $ROOT/results/rmse/run.log 2>&1 \
      || { say "CHAIN_FAIL $TASK rmse"; return 1; }
  fi
  local SIG
  SIG=$($LHPY -c "import json;print('%.4f'%json.load(open('$RMSE'))['pos_action_rmse'])")
  say "$TASK sigma_u=$SIG"

  # ---- 3. FTLE labels --------------------------------------------------------
  if [ ! -f "$LABELS" ] && [ ! -f "$ROOT/artifacts/labels/$(basename $LABELS)" ]; then
    until [ "$(awk '{printf "%d", $1}' /proc/loadavg)" -lt 200 ]; do sleep 60; done
    say "$TASK LABELER_START demos=200 workers=36 sigma_u=$SIG"
    cd $ROOT
    MUJOCO_GL=egl $MGPY impl/labeler/ftle_labeler.py \
      --dataset "$RAW" --output "$LABELS" \
      --sigma-u "$SIG" --sigma-u-source policy_rmse \
      --n-demos 200 --workers 36 --pos-only --k-horizon 24 \
      >> $ROOT/results/mimicgen/label_$TASK.log 2>&1 \
      || { say "CHAIN_FAIL $TASK labeler"; return 1; }
    say "$TASK LABELER_DONE"
  else
    [ -f "$LABELS" ] || cp $ROOT/artifacts/labels/$(basename $LABELS) "$LABELS"
    say "$TASK labels already exist"
  fi
  $MGPY $ROOT/impl/labeler/analyze_labels.py --labels "$LABELS" \
    --out $ROOT/results/mimicgen/analyze_mg_$TASK.json \
    >> $ROOT/results/mimicgen/label_$TASK.log 2>&1 || say "$TASK analyze failed (nonfatal)"

  # ---- 4. V-JEPA features ----------------------------------------------------
  if [ ! -f "$FEATS" ] && [ ! -f "$ROOT/artifacts/features/$(basename $FEATS)" ]; then
    until $VJPY -c "import torch" 2>/dev/null; do sleep 300; done
    say "$TASK EXTRACT_START"
    CUDA_VISIBLE_DEVICES=$GPU $VJPY $ROOT/impl/predictor/vjepa_extract.py \
      --task mg_$TASK --labels "$LABELS" --dataset "$IMG" --out "$FEATS" \
      --cam-key agentview_image --batch 32 \
      >> $ROOT/results/predictor/extract_mg_$TASK.log 2>&1 \
      || { say "CHAIN_FAIL $TASK extract"; return 1; }
    say "$TASK EXTRACT_DONE"
  else
    [ -f "$FEATS" ] || cp $ROOT/artifacts/features/$(basename $FEATS) "$FEATS"
    say "$TASK features already exist"
  fi

  # ---- 5. heads: 3 conditions x 3 seeds, per-task auto-delta -----------------
  local cond flags s name
  for cond in full vis prop; do
    case $cond in
      full) flags="";;
      vis)  flags="--no-proprio";;
      prop) flags="--proprio-only";;
    esac
    for s in 0 1 2; do
      name=M_${SHORT}_${cond}_s$s
      [ -f $HOUT/$name/metrics.json ] && continue
      CUDA_VISIBLE_DEVICES=$GPU $AZPY $ROOT/impl/predictor/train_head.py \
        --features "$FEATS" --out $HOUT/$name --tag $name --epochs 40 --seed $s $flags \
        > $HOUT/$name.log 2>&1
      [ -f $HOUT/$name/metrics.json ] && say "HEAD_OK $name" || say "HEAD_FAIL $name"
    done
  done

  # ---- 6. in-domain predictor switching cells, 50 episodes -------------------
  # (16,4) and (16,1): the single-step-when-unstable cell is standing protocol
  # (user directive 2026-07-25) alongside the chunked-recovery cell
  local ku
  for cond in full vis prop; do
    for ku in 4 1; do
      name=${TASK}_predictor_indom_${cond}_16-$ku
      [ -f $KOUT/$name.json ] && continue
      [ -f $HOUT/M_${SHORT}_${cond}_s0/head.pt ] || { say "KSWEEP_SKIP $name (no head)"; continue; }
      say "KSWEEP_EVAL $name (gpu $GPU)"
      MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU \
        $MGPY $ROOT/impl/mimicgen/stage2_ksweep.py \
          --dataset "$IMG" --checkpoint "$CK" --mode predictor \
          --k-stable 16 --k-unstable $ku \
          --pred-head $HOUT/M_${SHORT}_${cond}_s0/head.pt \
          --n-episodes 50 --horizon $HOR --seed 0 \
          --out $KOUT/$name.json > $KOUT/$name.log 2>&1 \
        || say "KSWEEP_EVAL_FAIL $name"
    done
  done
  say "CHAIN_DONE $TASK"
}

pipeline nut_assembly_d0        650  2000 2 nut    &
pipeline coffee_preparation_d0  1140 1000 3 coffee &
wait
say "NUTCOFFEE_ALL_DONE"
