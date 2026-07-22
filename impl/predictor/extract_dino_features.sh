#!/bin/bash
# DINOv2 single-frame ablation: extract features for the 8 run1-pool datasets
# on GPUs 4-6, then train the head on the pinned run1 validation split.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VPY=/mnt/scratch/lh/envs/vjepa/bin/python
F=/mnt/scratch/lh/features
RM=/mnt/scratch/lh/data/robomimic
MG=/mnt/scratch/lh/data/mimicgen
RO=/mnt/scratch/lh/rollouts
LOG=$LH_DIR/results/predictor/dino_extract.log

ex() { # gpu stamps dataset cam prefix out
  local gpu=$1 stamps=$2 ds=$3 cam=$4 prefix=$5 out=$6
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "EXTRACT $(basename $out) (gpu $gpu) $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpu $VPY $LH_DIR/impl/predictor/dino_extract.py \
    --stamps "$stamps" --dataset "$ds" --cam-key "$cam" \
    --demo-prefix "$prefix" --out "$out"
}

( ex 4 $F/feat2_lift.npz $RM/lift/ph/image_v15.hdf5 agentview_image demo $F/dino_lift.npz
  ex 4 $F/feat2_can.npz $RM/can/ph/image_v15.hdf5 agentview_image demo $F/dino_can.npz
  ex 4 $F/feat2_square.npz $RM/square/ph/image_v15.hdf5 agentview_image demo $F/dino_square.npz ) &
( ex 5 $F/feat2_tool_hang.npz $RM/tool_hang/ph/image_v15.hdf5 sideview_image demo $F/dino_tool_hang.npz
  ex 5 $F/feat2_rollouts_lift.npz $RO/rollouts_lift_image.hdf5 agentview_image demo $F/dino_rollouts_lift.npz ) &
( ex 6 $F/feat2_rollouts_can.npz $RO/rollouts_can_image.hdf5 agentview_image demo $F/dino_rollouts_can.npz
  ex 6 $F/feat_mg_stack_d0.npz $MG/stack_d0_image.hdf5 agentview_image demo $F/dino_mg_stack_d0.npz
  ex 6 $F/feat_mg_square_d0.npz $MG/square_d0_image.hdf5 agentview_image demo $F/dino_mg_square_d0.npz ) &
wait
echo DINO_FEATURES_DONE

# Head training: same file ORDER as run1 so global demo ids line up with the
# pinned validation split.
CUDA_VISIBLE_DEVICES=4 $VPY $LH_DIR/impl/predictor/train_head.py \
  --features $F/dino_lift.npz $F/dino_can.npz $F/dino_square.npz \
             $F/dino_tool_hang.npz $F/dino_rollouts_lift.npz \
             $F/dino_rollouts_can.npz $F/dino_mg_stack_d0.npz \
             $F/dino_mg_square_d0.npz \
  --val-demos-json $LH_DIR/results/predictor/run1_val_demos.json \
  --out $LH_DIR/results/predictor/run4_dino_single_frame --epochs 40
echo DINO_ABLATION_DONE
