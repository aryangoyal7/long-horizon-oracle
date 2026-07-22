#!/bin/bash
# Stage 1a seed repeats: all 12 cells x seeds {1,2} for square and tool_hang.
# GPU 0 starts immediately; GPUs 1-5 wait for LIBERO feature extraction to end.
# GPU 6 (training) and GPU 7 (CL labeler) are left alone.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/mnt/scratch/lh/envs/lh/bin/python
CK_SQ=$LH_DIR/results/training/dp_square/dp_square_image/20260712190622/models/model_epoch_950_image_v15_success_0.9.pth
CK_TH=$LH_DIR/results/training/dp_tool_hang/dp_tool_hang_image/20260712190622/models/model_epoch_1200_image_v15_success_0.9.pth
EXLOG=$LH_DIR/results/predictor/extract_libero.log

run() { # task mode k ks ku seed gpu
  local task=$1 mode=$2 k=$3 ks=$4 ku=$5 seed=$6 gpu=$7
  local out=$LH_DIR/results/ksweep/$task
  local name="${mode}_k${k}_ks${ks}_ku${ku}_seed${seed}"
  [ -f "$out/$name.json" ] && { echo "SKIP $task/$name"; return; }
  local ck=$CK_SQ hz=400
  [ "$task" = tool_hang ] && { ck=$CK_TH; hz=700; }
  echo "RUN $task/$name (gpu $gpu) $(date -u +%H:%M:%S)"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY $LH_DIR/impl/eval/eval_k_sweep.py \
    --ckpt "$ck" --mode $mode --k $k --k-stable $ks --k-unstable $ku \
    --n-episodes 50 --horizon $hz --seed $seed --out "$out/$name.json"
}

CELLS=("fixed 1 0 0" "fixed 2 0 0" "fixed 4 0 0" "fixed 8 0 0" "fixed 16 0 0" \
       "oracle_seg 0 1 4" "oracle_seg 0 1 16" "oracle_seg 0 4 1" "oracle_seg 0 8 1" \
       "oracle_seg 0 8 4" "oracle_seg 0 16 1" "oracle_seg 0 16 4")

# Flat work list: 48 jobs, square first (faster cells drain queues sooner)
JOBS=()
for s in 1 2; do
  for c in "${CELLS[@]}"; do JOBS+=("square $c $s"); done
done
for s in 1 2; do
  for c in "${CELLS[@]}"; do JOBS+=("tool_hang $c $s"); done
done

queue() { # gpu wait_flag job_indices...
  local gpu=$1 wait_flag=$2; shift 2
  if [ "$wait_flag" = wait ]; then
    until grep -q LIBERO_FEATURES_DONE "$EXLOG" 2>/dev/null || \
          ! pgrep -f extract_libero_features.sh >/dev/null; do sleep 120; done
  fi
  for i in "$@"; do run ${JOBS[$i]} $gpu; done
  echo "SEEDS_QUEUE_GPU${gpu}_DONE"
}

n=${#JOBS[@]}
gpus=(0 1 2 3 4 5)
pids=()
for gi in "${!gpus[@]}"; do
  idxs=()
  for ((j=gi; j<n; j+=6)); do idxs+=($j); done
  wf=wait; [ "${gpus[$gi]}" = 0 ] && wf=now
  queue "${gpus[$gi]}" $wf "${idxs[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo STAGE1A_SEEDS_DONE
