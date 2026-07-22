#!/bin/bash
# Stage 1a next phase: ftle_probe cells (seed 0) + predictor seed repeats (1,2).
# GPUs 0-3. GPU 7 is the CL labeler; GPUs 4-6 run the DINOv2 extraction.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/mnt/scratch/lh/envs/lh/bin/python
CK_SQ=$LH_DIR/results/training/dp_square/dp_square_image/20260712190622/models/model_epoch_950_image_v15_success_0.9.pth
CK_TH=$LH_DIR/results/training/dp_tool_hang/dp_tool_hang_image/20260712190622/models/model_epoch_1200_image_v15_success_0.9.pth
HEAD=$LH_DIR/results/predictor/run1_panda_pool/head.pt

run() { # task mode ks ku seed gpu extra...
  local task=$1 mode=$2 ks=$3 ku=$4 seed=$5 gpu=$6; shift 6
  local out=$LH_DIR/results/ksweep/$task
  local name="${mode}_k0_ks${ks}_ku${ku}_seed${seed}"
  [ -f "$out/$name.json" ] && { echo "SKIP $task/$name"; return; }
  local ck=$CK_SQ hz=400
  [ "$task" = tool_hang ] && { ck=$CK_TH; hz=700; }
  echo "RUN $task/$name (gpu $gpu) $(date -u +%H:%M:%S)"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY $LH_DIR/impl/eval/eval_k_sweep.py \
    --ckpt "$ck" --mode $mode --k-stable $ks --k-unstable $ku \
    --n-episodes 50 --horizon $hz --seed $seed --out "$out/$name.json" "$@"
}

( run square ftle_probe 16 4 0 0 ; echo NEXT_GPU0_DONE ) &
( run tool_hang ftle_probe 16 4 0 1 ; echo NEXT_GPU1_DONE ) &
( run square predictor 16 4 1 2 --pred-head "$HEAD"
  run square predictor 16 4 2 2 --pred-head "$HEAD"
  echo NEXT_GPU2_DONE ) &
( run tool_hang predictor 16 4 1 3 --pred-head "$HEAD" --pred-cam sideview_image
  run tool_hang predictor 16 4 2 3 --pred-head "$HEAD" --pred-cam sideview_image
  echo NEXT_GPU3_DONE ) &
wait
echo STAGE1A_NEXT_DONE
