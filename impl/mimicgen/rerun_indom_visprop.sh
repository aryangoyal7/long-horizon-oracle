#!/bin/bash
# Rerun the 8 predictor_indom vis/prop cells that failed on the predictor_bridge
# head-architecture bug (bridge now passes proprio_only/no_proprio from the ckpt).
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
MGPY=$LH/envs/mg/bin/python
MG=$LH/data/mimicgen
KOUT=$ROOT/results/stage2/ksweep
HOUT=$ROOT/results/predictor/mg_indomain
LOG=$ROOT/results/stage2/nutcoffee_chain.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_task() {
  local TASK=$1 HOR=$2 BEP=$3 GPU=$4 SHORT=$5
  local IMG=$MG/${TASK}_image.hdf5
  local CK
  CK=$(ls -d $ROOT/results/training/dp_mg_$TASK/*/*/models | head -1)/model_epoch_${BEP}.pth
  local cond ku name
  for cond in vis prop; do
    for ku in 4 1; do
      name=${TASK}_predictor_indom_${cond}_16-$ku
      [ -f $KOUT/$name.json ] && continue
      say "KSWEEP_EVAL $name (gpu $GPU, rerun)"
      MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU \
        $MGPY $ROOT/impl/mimicgen/stage2_ksweep.py \
          --dataset "$IMG" --checkpoint "$CK" --mode predictor \
          --k-stable 16 --k-unstable $ku \
          --pred-head $HOUT/M_${SHORT}_${cond}_s0/head.pt \
          --n-episodes 50 --horizon $HOR --seed 0 \
          --out $KOUT/$name.json > $KOUT/$name.log 2>&1 \
        || say "KSWEEP_EVAL_FAIL $name (rerun)"
    done
  done
  say "RERUN_DONE $TASK"
}

run_task nut_assembly_d0       650  2000 2 nut    &
run_task coffee_preparation_d0 1140 1000 3 coffee &
wait
say "RERUN_INDOM_VISPROP_ALL_DONE"
