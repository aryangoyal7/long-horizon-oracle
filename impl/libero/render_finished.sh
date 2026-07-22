#!/bin/bash
set -u
cd /home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
export PYTHONPATH=/mnt/scratch/lh/repos/LIBERO
for L in /mnt/scratch/lh/labels/final2_ol_libero10_*.npz; do
  short=$(basename $L .npz | sed 's/final2_ol_libero10_//')
  out=results/label_videos/libero10_${short}
  [ -f $out/README.json ] && continue
  /mnt/scratch/lh/envs/mg/bin/python impl/labeler/analyze_labels.py --labels $L --out results/labels2/libero10_${short} || continue
  D=$(/mnt/scratch/lh/envs/mg/bin/python -c "import json;raw=open('results/labels2/libero10_${short}/label_stats.json').read().replace('NaN','null');print(round(json.loads(raw)['suggested_deadband_delta_p95_free'],4))")
  F=$(ls /mnt/scratch/lh/data/libero/libero_10/${short}*.hdf5 2>/dev/null | head -1)
  [ -n "$F" ] || continue
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 /mnt/scratch/lh/envs/mg/bin/python impl/libero/render_label_videos_libero.py     --dataset "$F" --labels $L --out-dir $out --delta $D --n-demos 5
done
echo LIBERO_VIDEOS_PASS_DONE
