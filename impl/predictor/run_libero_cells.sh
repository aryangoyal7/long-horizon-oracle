#!/bin/bash
# LIBERO-specific predictor cells. The existing run3/run3b pool LIBERO with the
# panda tasks and validate on panda demos, so neither gives a LIBERO in-domain
# number. These do: in-domain (three input conditions) and both transfer
# directions. V-JEPA only; no DINOv2 features were ever extracted for LIBERO.
set -uo pipefail
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
FEAT=/mnt/scratch/lh/features
OUT=$ROOT/results/predictor/libero
LOG=$ROOT/results/predictor/libero_run.log
mkdir -p "$OUT"
PY=""
for _p in /anaconda/envs/azureml_py38/bin/python3 /anaconda/bin/python3 python3; do
  if command -v "$_p" >/dev/null 2>&1 && "$_p" -c "import torch" 2>/dev/null; then
    PY=$(command -v "$_p"); break
  fi
done
[ -n "$PY" ] || { echo "FATAL: no torch interpreter"; exit 1; }
# torch sizes its intra-op pool to all 96 cores, so N concurrent cells spawn ~96N
# threads and thrash. On 2026-07-24 that meant 3082 threads on 96 cores and zero
# cells finished in 43 minutes. The per-batch gather is memory bandwidth bound,
# so extra threads buy nothing anyway.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

LIB=""
for f in $FEAT/featq_libero10_*.npz; do LIB="$LIB $f"; done
PANDA="$FEAT/feat2_lift.npz $FEAT/feat2_can.npz $FEAT/feat2_square.npz $FEAT/feat2_tool_hang.npz $FEAT/feat2_rollouts_lift.npz $FEAT/feat2_rollouts_can.npz $FEAT/feat_mg_stack_d0.npz $FEAT/feat_mg_square_d0.npz"

say() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }
CELLS=()
add() { CELLS+=("$1|$2|$3|$4"); }
for S in 0 1 2; do
  add "L_vj_in_s$S"   "--seed $S"                "$LIB" ""
  add "L_prop_in_s$S" "--seed $S --proprio-only" "$LIB" ""
  add "L_vis_in_s$S"  "--seed $S --no-proprio"   "$LIB" ""
  add "L_p2l_s$S"     "--seed $S"                "$PANDA" "$LIB"
  add "L_l2p_s$S"     "--seed $S"                "$LIB" "$FEAT/feat2_tool_hang.npz"
done
say "LIBERO_START ${#CELLS[@]} cells"

run_cell() {
  local gpu=$1 spec=$2 name flags feats hold extra
  name=$(cut -d'|' -f1 <<<"$spec"); flags=$(cut -d'|' -f2 <<<"$spec")
  feats=$(cut -d'|' -f3 <<<"$spec"); hold=$(cut -d'|' -f4 <<<"$spec")
  [ -f "$OUT/$name/metrics.json" ] && { say "SKIP $name"; return; }
  extra=""; [ -n "$hold" ] && extra="--holdout-features $hold"
  say "RUN $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$ROOT/impl/predictor/train_head.py" \
      --features $feats $extra --out "$OUT/$name" --tag "$name" \
      --epochs 40 $flags > "$OUT/$name.log" 2>&1
  [ -f "$OUT/$name/metrics.json" ] && say "OK $name" || say "FAIL $name"
}
# 3 workers on GPUs 0,3,6 (the ablation spreads over all 8 but each cell is a
# small head; this keeps peak concurrency bounded)
worker() { local w=$1 gpu=$2; for ((i=w; i<${#CELLS[@]}; i+=3)); do run_cell "$gpu" "${CELLS[$i]}"; done; }
worker 0 0 & worker 1 3 & worker 2 6 &
wait
say "LIBERO_ALL_DONE $(ls -d $OUT/*/ 2>/dev/null | wc -l)/${#CELLS[@]}"
