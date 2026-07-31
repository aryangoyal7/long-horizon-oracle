#!/bin/bash
# Sequence-history vs single-clip heads on every CLEAN dataset with V-JEPA
# features (2026-07-30, claim-2 evidence sweep). RoboCasa runs separately on
# v3 labels (run_seq_predictor.sh); these datasets' final2 labels are clean
# (the action-order bug was RoboCasa-only). GPU 0, shared with the oracle
# cell (head training is minutes per run).
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
AZPY=/anaconda/envs/azureml_py38/bin/python3
FEATS=$ROOT/artifacts/features
LABS=$ROOT/artifacts/labels
HOUT=$ROOT/results/predictor/seq_multids
LOG=$ROOT/results/robocasa/seqpred.log
mkdir -p $HOUT
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "=== multi-dataset seq heads start ==="

DS="can lift square tool_hang mg_coffee_preparation_d0 mg_nut_assembly_d0 rollouts_can rollouts_lift"
for d in $DS; do
  F=$FEATS/feat2_$d.npz
  L=$LABS/final2_ol_$d.npz
  [ -f "$F" ] && [ -f "$L" ] || { say "MDS_SKIP $d (missing files)"; continue; }
  $AZPY -c "import numpy as np; z=np.load('$F'); exit(0 if 'proprio_seq' in z.files else 1)" \
    || { say "MDS_SKIP $d (no proprio_seq)"; continue; }
  for hist in 4 1; do
    for s in 0 1 2; do
      out=$HOUT/M_${d}_h${hist}_s$s
      [ -f $out/metrics.json ] && continue
      CUDA_VISIBLE_DEVICES=0 $AZPY $ROOT/impl/predictor/train_head_seq.py \
        --features "$F" --labels "$L" --hist $hist --seed $s \
        --out $out --tag M_${d}_h${hist}_s$s > $out.log 2>&1
      [ -f $out/metrics.json ] && say "MDS_HEAD_OK M_${d}_h${hist}_s$s" \
                               || say "MDS_HEAD_FAIL M_${d}_h${hist}_s$s"
    done
  done
done
say "MDS_ALL_DONE"
