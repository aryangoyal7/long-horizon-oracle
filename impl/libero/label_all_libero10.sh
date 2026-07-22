#!/bin/bash
set -u
LH=/mnt/scratch/lh
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=$LH/envs/mg/bin/python
export PYTHONPATH=$LH/repos/LIBERO
# 1) fetch all libero_10 files from HF
$PY - <<'PYEOF'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
files = [f for f in api.list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
         if f.startswith("libero_10/") and f.endswith(".hdf5")]
print(len(files), "files in suite")
for fn in files:
    hf_hub_download("yifengzhu-hf/LIBERO-datasets", fn, repo_type="dataset",
                    local_dir="/mnt/scratch/lh/data/libero")
    print("have", fn, flush=True)
PYEOF
# 2) label each
for F in $LH/data/libero/libero_10/*.hdf5; do
  base=$(basename "$F" .hdf5); short=$(echo "$base" | cut -c1-40)
  out=$LH/labels/final2_ol_libero10_${short}.npz
  [ -f "$out" ] && { echo "SKIP $short"; continue; }
  echo "[libero-label] $short"
  MUJOCO_GL=egl $PY "$LH_DIR/impl/libero/ftle_labeler_libero.py" \
    --dataset "$F" --output "$out" --sigma-u 0.203 --workers 10
done
echo LIBERO10_LABELS_DONE
