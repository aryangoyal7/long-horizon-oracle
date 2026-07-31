#!/bin/bash
# RoboCasa features + heads: waits for each task's labels, extracts V-JEPA
# features from the lerobot videos (GPU 4), trains {full, vis, prop} x 3 seeds.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
VJPY=$LH/envs/vjepa/bin/python
AZPY=/anaconda/envs/azureml_py38/bin/python3
HOUT=$ROOT/results/predictor/rc_indomain
LOG=$ROOT/results/robocasa/labeling.log
mkdir -p $HOUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
TASKS="OpenCabinet OpenDrawer PickPlaceCounterToCabinet TurnOnSinkFaucet"
for t in $TASKS; do
  L=$LH/labels/final2_ol_rc_$t.npz
  F=$LH/features/feat2_rc_$t.npz
  until [ -f "$L" ]; do sleep 300; done
  if [ ! -f "$F" ] && [ ! -f $ROOT/artifacts/features/feat2_rc_$t.npz ]; then
    D=$(ls -d $LH/repos/robocasa/datasets/v1.0/*/atomic/$t/*/lerobot 2>/dev/null | head -1)
    [ -z "$D" ] && { say "RC_EXTRACT_FAIL $t (no lerobot dir)"; continue; }
    say "RC_EXTRACT_START $t"
    CUDA_VISIBLE_DEVICES=4 $VJPY $ROOT/impl/robocasa/extract_rc_features.py \
      --lerobot-dir "$D" --labels "$L" --out "$F" --batch 32 \
      >> $ROOT/results/robocasa/extract_$t.log 2>&1 \
      && say "RC_EXTRACT_DONE $t" || { say "RC_EXTRACT_FAIL $t"; continue; }
  fi
  [ -f "$F" ] || cp $ROOT/artifacts/features/feat2_rc_$t.npz "$F"
  for cond in full vis prop; do
    case $cond in full) flags="";; vis) flags="--no-proprio";; prop) flags="--proprio-only";; esac
    for s in 0 1 2; do
      name=R_${t}_${cond}_s$s
      [ -f $HOUT/$name/metrics.json ] && continue
      CUDA_VISIBLE_DEVICES=4 $AZPY $ROOT/impl/predictor/train_head.py \
        --features "$F" --out $HOUT/$name --tag $name --epochs 40 --seed $s $flags \
        > $HOUT/$name.log 2>&1
      [ -f $HOUT/$name/metrics.json ] && say "RC_HEAD_OK $name" || say "RC_HEAD_FAIL $name"
    done
  done
done
say "RC_FEATURES_HEADS_DONE"
