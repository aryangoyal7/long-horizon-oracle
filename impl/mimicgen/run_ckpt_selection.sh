#!/bin/bash
# Stage 2 checkpoint selection over the surviving checkpoints (trainings died
# at epochs 1596-1721 in the July 21 disk event; saved grids end at 1550-1700,
# all past the epoch ~950-1200 plateau Stage 1 selection favored).
# Per task: wait for that task's image reconversion marker (run_stage2.sh,
# GPUs 0-3), then evaluate its checkpoint list through the cross-venv bridge
# on GPUs 4-7. 25 episodes per checkpoint (SE ~0.1): coarse but enough to pick
# a plateau checkpoint; the chosen one gets full 50-episode cells later.
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MGPY=/mnt/scratch/lh/envs/mg/bin/python
MG=/mnt/scratch/lh/data/mimicgen
OUT=$LH_DIR/results/stage2/ckpt_select
CONVLOG=$LH_DIR/results/training/stage2_run.log
mkdir -p $OUT

TASKS=(three_piece_assembly_d0 nut_assembly_d0 kitchen_d0 coffee_preparation_d0)
GPUS=(4 5 6 7)
declare -A HOR=( [three_piece_assembly_d0]=560 [nut_assembly_d0]=650
                 [kitchen_d0]=980 [coffee_preparation_d0]=1140 )
declare -A EP=( [three_piece_assembly_d0]="600 1000 1400 1700 2000"
                [nut_assembly_d0]="600 1000 1400 1700 2000"
                [kitchen_d0]="600 1000 1400 1700 2000"
                [coffee_preparation_d0]="600 1000 1400 1700 2000" )

# immediately back up each checkpoint grid once its training finishes, so an
# accidental overwrite can never destroy the only copy again (Jul 21 lesson)
backup_ckpts() { # task
  local t=$1
  local models=$(ls -d $LH_DIR/results/training/dp_mg_${t}/*/*/models 2>/dev/null | head -1)
  [ -n "$models" ] && cp -rn "$models" "/home/azureuser/ckpt_backup_dp_mg_${t}" \
    && echo "CKPT_BACKUP_${t}_DONE"
}

select_task() { # task gpu
  local t=$1 gpu=$2
  until grep -q "finished run successfully" \
        $LH_DIR/results/training/dp_mg_${t}.launch.log 2>/dev/null; do sleep 600; done
  backup_ckpts $t
  local models=$(ls -d $LH_DIR/results/training/dp_mg_${t}/*/*/models | head -1)
  for e in ${EP[$t]}; do
    local ck=$models/model_epoch_${e}.pth
    [ -f $ck ] || { echo "CKPT_SELECT_MISSING_${t}_${e}"; continue; }
    echo "EVAL $t epoch $e (gpu $gpu) $(date -u +%H:%M:%S)"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu CUDA_VISIBLE_DEVICES=$gpu \
      $MGPY $LH_DIR/impl/mimicgen/bridge_eval.py \
        --dataset $MG/${t}_image.hdf5 --checkpoint $ck \
        --n-episodes 25 --horizon ${HOR[$t]} --seed 0 \
        --out $OUT/${t}_epoch${e}.json \
        > $OUT/${t}_epoch${e}.log 2>&1 \
      || echo "CKPT_SELECT_EVAL_FAIL_${t}_${e}"
  done
  echo "CKPT_SELECT_${t}_DONE $(date -u +%H:%M:%S)"
}

for i in 0 1 2 3; do
  select_task "${TASKS[$i]}" "${GPUS[$i]}" &
done
wait
echo CKPT_SELECT_ALL_DONE
