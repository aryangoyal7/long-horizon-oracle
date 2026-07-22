#!/bin/bash
# True-sigma_u relabel chain + sanity outputs. Detached, restart-tolerant-ish.
# 1. tool_hang validation RMSE (GPU 0)
# 2. OL relabel all 4 robomimic ph tasks with per-task sigma_u -> final2_ol_<task>.npz
# 3. analyze_labels per task -> results/labels2/<task>
# 4. label sanity VIDEOS (5 demos/task) -> results/label_videos/<task>/
# 5. divergence CURVES (3 stable + 3 unstable stamps/task) -> .../divergence_curves/
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/mnt/scratch/lh/envs/lh/bin/python
DATA=/mnt/scratch/lh/data/robomimic
LAB=/mnt/scratch/lh/labels
cd "$LH_DIR"

# ---- 1. tool_hang RMSE ----------------------------------------------------------
if [ ! -f results/rmse/tool_hang.json ]; then
  CK=$(ls results/training/dp_tool_hang/dp_tool_hang_image/*/models/*success*.pth \
       | awk -F'success_' '{print $2+0, $0}' | sort -k1,1g -k2,2V | tail -1 | cut -d' ' -f2-)
  echo "[chain2] tool_hang RMSE on $CK"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY impl/eval/policy_action_rmse.py \
    --ckpt "$CK" --dataset $DATA/tool_hang/ph/image_v15.hdf5 \
    --out results/rmse/tool_hang.json --max-demos 20 --stride 2
fi

sig() { $PY -c "import json;print(round(json.load(open('results/rmse/$1.json'))['pos_action_rmse'],4))"; }

# ---- 2+3. relabel + analyze -------------------------------------------------------
for t in lift can square tool_hang; do
  S=$(sig $t)
  echo "[chain2] relabel $t with sigma_u=$S"
  if [ ! -f $LAB/final2_ol_$t.npz ]; then
    $PY impl/labeler/ftle_labeler.py --dataset $DATA/$t/ph/low_dim_v15.hdf5 \
      --output $LAB/final2_ol_$t.npz --sigma-u $S --sigma-u-source policy_rmse \
      --workers 40 --pos-only --k-horizon 24
  fi
  mkdir -p results/labels2
  $PY impl/labeler/analyze_labels.py --labels $LAB/final2_ol_$t.npz \
    --out results/labels2/$t
done
echo "RELABEL_ALL_DONE"

delta_of() { $PY -c "
import json
raw=open('results/labels2/$1/label_stats.json').read().replace('NaN','null')
print(round(json.loads(raw)['suggested_deadband_delta_p95_free'],4))"; }

# ---- 4+5. videos + curves (GPU 0) --------------------------------------------------
for t in lift can square tool_hang; do
  S=$(sig $t); D=$(delta_of $t)
  CAM=agentview; [ $t = tool_hang ] && CAM=sideview
  echo "[chain2] videos+curves $t (delta=$D cam=$CAM)"
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY impl/labeler/render_label_videos.py \
    --dataset $DATA/$t/ph/low_dim_v15.hdf5 --labels $LAB/final2_ol_$t.npz \
    --out-dir results/label_videos/$t --delta $D --camera $CAM --n-demos 5
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY impl/labeler/plot_divergence_curves.py \
    --dataset $DATA/$t/ph/low_dim_v15.hdf5 --labels $LAB/final2_ol_$t.npz \
    --out-dir results/label_videos/$t/divergence_curves --sigma-u $S --delta $D
done
echo "CHAIN2_ALL_DONE $(date -u +%F' '%H:%M)"
