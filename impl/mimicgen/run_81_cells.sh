#!/bin/bash
# MimicGen (8,1) switching cells with the full in-domain heads, per the
# k_unstable=1 protocol. Sequential on GPU 2. Coffee first: its best fixed
# chunk is 8, so (8,1) is the meaningful cell there.
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
MGPY=$LH/envs/mg/bin/python
MG=$LH/data/mimicgen
KOUT=$ROOT/results/stage2/ksweep
HOUT=$ROOT/results/predictor/mg_indomain
LOG=$ROOT/results/stage2/nutcoffee_chain.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_cell() {  # task hor bep short
  local TASK=$1 HOR=$2 BEP=$3 SHORT=$4
  local IMG=$MG/${TASK}_image.hdf5
  local CK
  CK=$(ls -d $ROOT/results/training/dp_mg_$TASK/*/*/models | head -1)/model_epoch_${BEP}.pth
  local name=${TASK}_predictor_indom_full_8-1
  [ -f $KOUT/$name.json ] && return 0
  say "KSWEEP_EVAL $name (gpu 2, 8-1 protocol)"
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 \
    $MGPY $ROOT/impl/mimicgen/stage2_ksweep.py \
      --dataset "$IMG" --checkpoint "$CK" --mode predictor \
      --k-stable 8 --k-unstable 1 \
      --pred-head $HOUT/M_${SHORT}_full_s0/head.pt \
      --n-episodes 50 --horizon $HOR --seed 0 \
      --out $KOUT/$name.json > $KOUT/$name.log 2>&1 \
    || say "KSWEEP_EVAL_FAIL $name"
  say "KSWEEP_EVAL_DONE $name $(python3 -c "import json;print(json.load(open('$KOUT/$name.json'))['success_rate'])" 2>/dev/null)"
}

say "=== mimicgen 8-1 cells start ==="
run_cell coffee_preparation_d0 1140 1000 coffee
run_cell nut_assembly_d0       650  2000 nut
say "MIMICGEN_81_DONE"
