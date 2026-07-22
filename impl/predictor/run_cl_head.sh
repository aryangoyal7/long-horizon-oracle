#!/bin/bash
# CL head: extract V-JEPA features at the closed-loop label stamps (all four
# Panda tasks), then train a head on lambda_cl. GPU 4.
# The extractor stores lambda_cl_task under the npz key lambda_task, so
# train_head.py runs unchanged; interpret run5 outputs as CL quantities.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VPY=/mnt/scratch/lh/envs/vjepa/bin/python
F=/mnt/scratch/lh/features
L=/mnt/scratch/lh/labels

for t in lift can square tool_hang; do
  [ -f $F/featcl_$t.npz ] && { echo "SKIP featcl_$t"; continue; }
  echo "EXTRACT featcl_$t $(date -u +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=4 $VPY $LH_DIR/impl/predictor/vjepa_extract.py \
    --task $t --labels $L/labels_cl_$t.npz --lambda-key lambda_cl_task \
    --out $F/featcl_$t.npz
done
echo CL_FEATURES_DONE

CUDA_VISIBLE_DEVICES=4 $VPY $LH_DIR/impl/predictor/train_head.py \
  --features $F/featcl_lift.npz $F/featcl_can.npz $F/featcl_square.npz \
             $F/featcl_tool_hang.npz \
  --out $LH_DIR/results/predictor/run5_cl_head --epochs 40
echo CL_HEAD_DONE
