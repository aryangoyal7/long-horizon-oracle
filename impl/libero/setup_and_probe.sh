#!/bin/bash
# LIBERO (LIBERO-Long = libero_10) setup + compatibility probe.
# Reuses the mg venv (robosuite 1.4.1 + mujoco 2.3.2 — LIBERO's own pins are 1.4-era).
set -uo pipefail
LH=/mnt/scratch/lh
PY=$LH/envs/mg/bin/python
PIP=$LH/envs/mg/bin/pip
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PIP_CACHE_DIR=$LH/pipcache

[ -d $LH/repos/LIBERO ] || \
  git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git $LH/repos/LIBERO
# LIBERO ships no top-level libero/__init__.py, so the editable install alone does
# not make it importable. It resolves as a namespace package off the repo root, and
# every consumer must therefore have the repo root on PYTHONPATH.
export PYTHONPATH="$LH/repos/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
$PY -c "import libero" 2>/dev/null || $PIP install -q --no-deps -e $LH/repos/LIBERO
$PIP install -q bddl easydict "hydra-core>=1.1" gym cloudpickle future \
  huggingface_hub 2>/dev/null || true
echo "LIBERO_IMPORT: $($PY -c 'import libero; print("ok")' 2>&1 | tail -1)"

# ---- dataset: LIBERO-10 (long-horizon suite) --------------------------------------
mkdir -p $LH/data/libero
if ! ls $LH/data/libero/**/*.hdf5 >/dev/null 2>&1; then
  echo "[libero] trying official downloader..."
  (cd $LH/repos/LIBERO && timeout 3600 $PY benchmark_scripts/download_libero_datasets.py \
     --datasets libero_10 --save-dir $LH/data/libero 2>&1 | tail -4) || true
fi
# LIBERO-10 is exactly 10 tasks; anything less means an interrupted download
NHDF5=$(ls $LH/data/libero/libero_10/*.hdf5 2>/dev/null | wc -l)
if [ "$NHDF5" -lt 10 ]; then
  echo "[libero] have $NHDF5/10, trying HuggingFace mirrors..."
  $PY - <<'PYEOF'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
for repo in ["yifengzhu-hf/LIBERO-datasets", "openvla/LIBERO-datasets",
             "physical-intelligence/libero"]:
    try:
        files = [f for f in api.list_repo_files(repo, repo_type="dataset")
                 if "libero_10" in f and f.endswith(".hdf5")]
        print(repo, "->", len(files), "libero_10 files")
        if files:
            for fn in sorted(files):
                p = hf_hub_download(repo, fn, repo_type="dataset",
                                    local_dir="/mnt/scratch/lh/data/libero")
                print("GOT", p, flush=True)
            break
    except Exception as e:
        print("no:", repo, type(e).__name__, str(e)[:80])
PYEOF
fi

F=$(ls $LH/data/libero/**/*.hdf5 2>/dev/null | head -1)
if [ -n "$F" ]; then
  echo "[libero] probing $F"
  MUJOCO_GL=egl $PY "$LH_DIR/impl/libero/probe_libero.py" --dataset "$F"
else
  echo "LIBERO_DATASET_MISSING (both download paths failed)"
fi
echo "LIBERO_PROBE_DONE"
