#!/bin/bash
# Stage 1a on square seed-1: fixed-k row + (k_stable, k_unstable) oracle grid.
# 4 GPU queues; 50 eps each, horizon 400, seed 0. Cells named by mode+params.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/mnt/scratch/lh/envs/lh/bin/python
CK=$LH_DIR/results/training/dp_tool_hang/dp_tool_hang_image/20260712190622/models/model_epoch_1200_image_v15_success_0.9.pth
OUT=$LH_DIR/results/ksweep/tool_hang
run() { # mode k ks ku gpu
  local mode=$1 k=$2 ks=$3 ku=$4 gpu=$5
  local name="${mode}_k${k}_ks${ks}_ku${ku}"
  [ -f "$OUT/$name.json" ] && { echo "SKIP $name"; return; }
  echo "RUN $name (gpu $gpu)"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY $LH_DIR/impl/eval/eval_k_sweep.py \
    --ckpt "$CK" --mode $mode --k $k --k-stable $ks --k-unstable $ku \
    --n-episodes 50 --horizon 700 --seed 0 --out "$OUT/$name.json"
}
queue() { # gpu cells...
  local gpu=$1; shift
  for c in "$@"; do run $c $gpu; done
  echo "QUEUE_GPU${gpu}_DONE"
}
# 13 cells split across 4 queues (fixed uses k; oracle uses ks/ku)
queue 0 "fixed 1 0 0" "oracle_seg 0 16 1" "oracle_seg 0 1 4" &
queue 1 "fixed 4 0 0" "fixed 2 0 0" "oracle_seg 0 8 1" &
queue 3 "fixed 8 0 0" "oracle_seg 0 16 4" "oracle_seg 0 1 16" &
queue 5 "fixed 16 0 0" "oracle_seg 0 8 4" "oracle_seg 0 4 1" &
wait
echo STAGE1A_TOOLHANG_DONE
