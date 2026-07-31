#!/usr/bin/env bash
# Encoder ablation and transfer grid for the stability predictor head.
#
# Grid A (conclusiveness): does the DINOv2 / V-JEPA tie survive seed noise, and
#   how much of the signal is vision at all?  Conditions: full, vision-only,
#   proprio-only, for each encoder, 3 seeds.  Pool, split and delta pinned to
#   run 1 so every number is mutually comparable.
# Grid B (transfer): square-only -> tool_hang (mirrors run 2) and the
#   deployment-relevant leave-one-task-out (pool minus tool_hang -> tool_hang),
#   for each encoder plus a proprio-only control, 3 seeds.
set -u
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
FEAT=/mnt/scratch/lh/features
OUT=$ROOT/results/predictor/ablation
LOG=$ROOT/results/predictor/ablation_run.log
# Resolved explicitly: under the @reboot PATH, python3 is /usr/bin/python3, which
# has no torch, so a boot-time resume would fail every cell silently.
PY=""
# Order matters: the conda python (torch 2.9.1) trained the first 14 cells, so it
# must stay first or later cells would differ from earlier ones by torch version.
for _p in /anaconda/envs/azureml_py38/bin/python3 /anaconda/bin/python3 python3 \
          /mnt/scratch/lh/envs/lh/bin/python /mnt/scratch/lh/envs/vjepa/bin/python; do
  if command -v "$_p" >/dev/null 2>&1 && "$_p" -c "import torch" 2>/dev/null; then
    PY=$(command -v "$_p"); break
  fi
done
[ -n "$PY" ] || { echo "FATAL: no interpreter with torch found"; exit 1; }
# torch sizes its intra-op pool to all 96 cores, so N concurrent cells spawn ~96N
# threads and thrash. On 2026-07-24 that meant 3082 threads on 96 cores and zero
# cells finished in 43 minutes. The per-batch gather is memory bandwidth bound,
# so extra threads buy nothing anyway.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
DELTA=0.03545861691236496          # run 1 pooled delta, pinned across the whole grid
VALJSON=$ROOT/results/predictor/run1_val_demos.json
mkdir -p "$OUT"

VJ_POOL="$FEAT/feat2_lift.npz $FEAT/feat2_can.npz $FEAT/feat2_square.npz $FEAT/feat2_tool_hang.npz $FEAT/feat2_rollouts_lift.npz $FEAT/feat2_rollouts_can.npz $FEAT/feat_mg_stack_d0.npz $FEAT/feat_mg_square_d0.npz"
DI_POOL="$FEAT/dino_lift.npz $FEAT/dino_can.npz $FEAT/dino_square.npz $FEAT/dino_tool_hang.npz $FEAT/dino_rollouts_lift.npz $FEAT/dino_rollouts_can.npz $FEAT/dino_mg_stack_d0.npz $FEAT/dino_mg_square_d0.npz"
# leave-one-task-out pools: same order, tool_hang removed
VJ_LOTO="$FEAT/feat2_lift.npz $FEAT/feat2_can.npz $FEAT/feat2_square.npz $FEAT/feat2_rollouts_lift.npz $FEAT/feat2_rollouts_can.npz $FEAT/feat_mg_stack_d0.npz $FEAT/feat_mg_square_d0.npz"
DI_LOTO="$FEAT/dino_lift.npz $FEAT/dino_can.npz $FEAT/dino_square.npz $FEAT/dino_rollouts_lift.npz $FEAT/dino_rollouts_can.npz $FEAT/dino_mg_stack_d0.npz $FEAT/dino_mg_square_d0.npz"

say() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

# name | gpu-slot | extra flags | feature list | holdout list | pinned-val?
CELLS=()
add() { CELLS+=("$1|$2|$3|$4|$5"); }

for S in 0 1 2; do
  # ---- Grid A: pinned pool, pinned val split, pinned delta ----
  add "A_vj_full_s$S"  "--seed $S"                  "$VJ_POOL" "" "pin"
  add "A_di_full_s$S"  "--seed $S"                  "$DI_POOL" "" "pin"
  add "A_vj_vis_s$S"   "--seed $S --no-proprio"     "$VJ_POOL" "" "pin"
  add "A_di_vis_s$S"   "--seed $S --no-proprio"     "$DI_POOL" "" "pin"
  add "A_prop_s$S"     "--seed $S --proprio-only"   "$VJ_POOL" "" "pin"
  # longer training, to check the tie is not an artifact of stopping at 40 epochs
  add "A_vj_e100_s$S"  "--seed $S --epochs 100"      "$VJ_POOL" "" "pin"
  add "A_di_e100_s$S"  "--seed $S --epochs 100"      "$DI_POOL" "" "pin"
  # ---- Grid B: transfer ----
  add "B_sq2th_vj_s$S"   "--seed $S"                "$FEAT/feat2_square.npz" "$FEAT/feat2_tool_hang.npz" ""
  add "B_sq2th_di_s$S"   "--seed $S"                "$FEAT/dino_square.npz"  "$FEAT/dino_tool_hang.npz"  ""
  add "B_sq2th_prop_s$S" "--seed $S --proprio-only" "$FEAT/feat2_square.npz" "$FEAT/feat2_tool_hang.npz" ""
  add "B_loto_vj_s$S"    "--seed $S"                "$VJ_LOTO" "$FEAT/feat2_tool_hang.npz" ""
  add "B_loto_di_s$S"    "--seed $S"                "$DI_LOTO" "$FEAT/dino_tool_hang.npz"  ""
  add "B_loto_prop_s$S"  "--seed $S --proprio-only" "$VJ_LOTO" "$FEAT/feat2_tool_hang.npz" ""
done

say "ABLATION_START ${#CELLS[@]} cells"

run_cell() {
  local gpu=$1 spec=$2
  local name flags feats hold pin
  name=$(cut -d'|' -f1 <<<"$spec")
  flags=$(cut -d'|' -f2 <<<"$spec")
  feats=$(cut -d'|' -f3 <<<"$spec")
  hold=$(cut -d'|' -f4 <<<"$spec")
  pin=$(cut -d'|' -f5 <<<"$spec")
  if [ -f "$OUT/$name/metrics.json" ]; then say "SKIP $name (exists)"; return; fi
  local extra=""
  [ -n "$hold" ] && extra="--holdout-features $hold"
  [ "$pin" = "pin" ] && extra="$extra --val-demos-json $VALJSON"
  say "RUN $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$ROOT/impl/predictor/train_head.py" \
      --features $feats $extra --out "$OUT/$name" \
      --delta $DELTA --tag "$name" --epochs 40 $flags \
      > "$OUT/$name.log" 2>&1
  if [ -f "$OUT/$name/metrics.json" ]; then say "OK $name"; else say "FAIL $name"; fi
}

worker() {
  local gpu=$1
  for ((i=gpu; i<${#CELLS[@]}; i+=8)); do run_cell "$gpu" "${CELLS[$i]}"; done
}

for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait

n=$(ls -d "$OUT"/*/ 2>/dev/null | wc -l)
say "ABLATION_ALL_DONE $n/${#CELLS[@]} cells produced metrics"
