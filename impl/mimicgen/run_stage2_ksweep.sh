#!/bin/bash
# Stage 2 experiment: fixed k in {1,4,8,16} plus the predictor (16,4) switch
# on the four MimicGen tasks, 50 episodes each at seed 0, on the selected
# checkpoints. Sequencing: smoke-test stage2_ksweep.py as soon as the first
# reconverted dataset lands (the old smoke died with the scratch wipe, exit
# 137), then wait for checkpoint selection to finish, then run all 20 cells
# round-robin over the 8 GPUs. Appends markers to ksweep_run.log; the
# shutdown watchdog fires only on STAGE2_KSWEEP_ALL_DONE, which is written
# only if every cell produced its result file.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MGPY=/mnt/scratch/lh/envs/mg/bin/python
MG=/mnt/scratch/lh/data/mimicgen
OUT=$LH_DIR/results/stage2/ksweep
SEL=$LH_DIR/results/stage2/ckpt_select
RUNLOG=$LH_DIR/results/stage2/ksweep_run.log
mkdir -p $OUT
log() { echo "$*" >> $RUNLOG; }

TASKS=(three_piece_assembly_d0 nut_assembly_d0 kitchen_d0 coffee_preparation_d0)
declare -A HOR=( [three_piece_assembly_d0]=560 [nut_assembly_d0]=650
                 [kitchen_d0]=980 [coffee_preparation_d0]=1140 )

models_dir() { ls -d $LH_DIR/results/training/dp_mg_$1/*/*/models | head -1; }

# ---- smoke: 2 episodes per mode on three_piece as soon as its data lands ----
until [ -f $MG/three_piece_assembly_d0_image.hdf5 ] \
      && grep -q "RECONVERT_three_piece_assembly_d0_DONE" \
           $LH_DIR/results/stage2/rebuild_20260723.log; do sleep 120; done
ck=$(models_dir three_piece_assembly_d0)/model_epoch_1700.pth
smoke() { # mode extra...
  local mode=$1; shift
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    $MGPY $LH_DIR/impl/mimicgen/stage2_ksweep.py \
      --dataset $MG/three_piece_assembly_d0_image.hdf5 --checkpoint $ck \
      --mode $mode "$@" --n-episodes 2 --horizon 560 --seed 99 \
      --out $OUT/smoke_$mode.json > $OUT/smoke_$mode.log 2>&1
}
if smoke fixed --k 8 && smoke predictor; then
  log "KSWEEP_SMOKE_OK $(date -u +%H:%M:%S)"
else
  log "KSWEEP_SMOKE_FAIL"; exit 1
fi

# ---- wait for checkpoint selection, then pick best epochs ----
until grep -q "CKPT_SELECT_ALL_DONE" \
      $LH_DIR/results/stage2/ckpt_select_run.log 2>/dev/null; do sleep 300; done
declare -A BEST
for t in "${TASKS[@]}"; do
  BEST[$t]=$(python3 - "$SEL" "$t" <<'PY'
import glob, json, re, sys
sel, task = sys.argv[1], sys.argv[2]
cells = []
for p in glob.glob(f"{sel}/{task}_epoch*.json"):
    e = int(re.search(r"epoch(\d+)", p).group(1))
    cells.append((json.load(open(p))["success_rate"], e))
best = max(cells, key=lambda c: (c[0], c[1]))
print(best[1])
PY
)
  log "KSWEEP_BEST_${t}_epoch${BEST[$t]}"
done

# ---- 20 cells, round-robin over 8 GPUs ----
CELLS=()
for t in "${TASKS[@]}"; do
  for k in 1 4 8 16; do CELLS+=("$t|fixed|$k"); done
  CELLS+=("$t|predictor|16-4")
done

run_cell() { # spec gpu
  local spec=$1 gpu=$2
  local t=${spec%%|*} rest=${spec#*|}
  local mode=${rest%%|*} k=${rest#*|}
  local tag=${t}_${mode}_${k}
  [ -f $OUT/$tag.json ] && { log "KSWEEP_SKIP_$tag"; return; }
  local ck=$(models_dir $t)/model_epoch_${BEST[$t]}.pth
  local extra
  if [ $mode = fixed ]; then extra="--k $k"; else extra="--k-stable 16 --k-unstable 4"; fi
  log "KSWEEP_EVAL $tag (gpu $gpu) $(date -u +%H:%M:%S)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $MGPY $LH_DIR/impl/mimicgen/stage2_ksweep.py \
      --dataset $MG/${t}_image.hdf5 --checkpoint $ck \
      --mode $mode $extra --n-episodes 50 --horizon ${HOR[$t]} --seed 0 \
      --out $OUT/$tag.json > $OUT/$tag.log 2>&1 \
    || log "KSWEEP_EVAL_FAIL_$tag"
}

worker() { # gpu
  local gpu=$1 i
  for ((i=gpu; i<${#CELLS[@]}; i+=8)); do run_cell "${CELLS[$i]}" $gpu; done
}
for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait

# ---- completion: every cell must have a result file ----
missing=0
for spec in "${CELLS[@]}"; do
  t=${spec%%|*}; rest=${spec#*|}; mode=${rest%%|*}; k=${rest#*|}
  [ -f $OUT/${t}_${mode}_${k}.json ] || { log "KSWEEP_MISSING_${t}_${mode}_${k}"; missing=1; }
done
if [ $missing -eq 0 ]; then
  log "STAGE2_KSWEEP_ALL_DONE"
else
  log "KSWEEP_INCOMPLETE"
fi
