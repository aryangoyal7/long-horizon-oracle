#!/bin/bash
# Extract V-JEPA 2 features for all 10 labeled LIBERO-10 tasks.
# Distributes tasks across GPUs 1-5 (2 sequential tasks per GPU).
# Label files are truncated to 50 chars; match each to its hdf5 by prefix.
set -u
PY=/mnt/scratch/lh/envs/vjepa/bin/python
EXTRACT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/impl/predictor/vjepa_extract.py
LABELS=/mnt/scratch/lh/labels
DATA=/mnt/scratch/lh/data/libero/libero_10
FEAT=/mnt/scratch/lh/features
LOG=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/predictor
mkdir -p "$LOG"

# Build task list: label_npz|hdf5|out_npz
TASKS=()
for lbl in "$LABELS"/final2_ol_libero10_*.npz; do
  stem=$(basename "$lbl" .npz)
  short=${stem#final2_ol_libero10_}
  hdf5=""
  for f in "$DATA"/*.hdf5; do
    base=$(basename "$f" _demo.hdf5)
    case "$base" in "$short"*) hdf5="$f"; break;; esac
  done
  if [ -z "$hdf5" ]; then echo "NO MATCH for $short" >&2; continue; fi
  TASKS+=("$lbl|$hdf5|$FEAT/feat_libero10_$short.npz")
done
echo "matched ${#TASKS[@]} tasks"

run_queue() {
  local gpu=$1; shift
  for spec in "$@"; do
    IFS='|' read -r lbl hdf5 out <<<"$spec"
    name=$(basename "$out" .npz)
    if [ -f "$out" ]; then echo "skip $name (exists)"; continue; fi
    echo "[gpu$gpu] $name start $(date -u +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$EXTRACT" --task libero \
      --labels "$lbl" --out "$out" --dataset "$hdf5" \
      --cam-key agentview_rgb --proprio-keys ee_pos ee_ori gripper_states \
      --batch 32 >>"$LOG/${name}.log" 2>&1
    echo "[gpu$gpu] $name exit=$? $(date -u +%H:%M:%S)"
  done
}

pids=()
n=${#TASKS[@]}
gpus=(1 2 3 4 5)
for i in "${!gpus[@]}"; do
  q=()
  for ((j=i; j<n; j+=5)); do q+=("${TASKS[$j]}"); done
  [ ${#q[@]} -eq 0 ] && continue
  run_queue "${gpus[$i]}" "${q[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
ok=$(ls "$FEAT"/feat_libero10_*.npz 2>/dev/null | wc -l)
echo "LIBERO_FEATURES_DONE $ok/10"
