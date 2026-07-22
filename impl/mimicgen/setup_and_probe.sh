#!/bin/bash
# MimicGen setup + compatibility probe, take 2.
# Findings from take 1 (results/mimicgen/probe_20260717.log):
#   - mimicgen envs need robosuite<=1.4 (robosuite.environments.manipulation.
#     single_arm_env was removed in 1.5) -> DEDICATED venv, as planned fallback.
#   - download_datasets.py needs gdown -> skip it, pull straight from HuggingFace.
# Pins per mimicgen docs: robosuite v1.4.1, mujoco 2.3.2, robomimic v0.3.0.
set -uo pipefail
LH=/mnt/scratch/lh
MG=$LH/envs/mg
PY=$MG/bin/python
PIP=$MG/bin/pip
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PIP_CACHE_DIR=$LH/pipcache

# ---- dedicated venv -------------------------------------------------------------
if [ ! -x $PY ]; then python3 -m venv $MG; $PIP install -q --upgrade pip; fi
$PY -c "import torch" 2>/dev/null || \
  $PIP install -q torch --index-url https://download.pytorch.org/whl/cu128
$PY -c "import mujoco; assert mujoco.__version__=='2.3.2'" 2>/dev/null || \
  $PIP install -q "mujoco==2.3.2"
$PY -c "import robosuite; assert robosuite.__version__.startswith('1.4')" 2>/dev/null || \
  $PIP install -q "robosuite==1.4.1"
PATCH_V03() { $MG/bin/python - <<PYEOF
p = "/mnt/scratch/lh/repos/robomimic_v03/robomimic/envs/env_robosuite.py"
s = open(p).read()
if "except ImportError" not in s:
    s = s.replace("import mujoco_py\n", "try:\n    import mujoco_py\nexcept ImportError:\n    mujoco_py = None\n", 1)
    open(p, "w").write(s)
PYEOF
}
if [ ! -d $LH/repos/robomimic_v03 ]; then
  git clone -q --branch v0.3.0 https://github.com/ARISE-Initiative/robomimic.git \
    $LH/repos/robomimic_v03
fi
$PY -c "import robomimic" 2>/dev/null || $PIP install -q -e $LH/repos/robomimic_v03
if [ ! -d $LH/repos/mimicgen ]; then
  git clone -q https://github.com/NVlabs/mimicgen.git $LH/repos/mimicgen
fi
$PY -c "import mimicgen" 2>/dev/null || $PIP install -q --no-deps -e $LH/repos/mimicgen
if [ ! -d $LH/repos/robosuite-task-zoo ]; then
  git clone -q https://github.com/ARISE-Initiative/robosuite-task-zoo.git \
    $LH/repos/robosuite-task-zoo
fi
$PY -c "import robosuite_task_zoo" 2>/dev/null || \
  $PIP install -q --no-deps -e $LH/repos/robosuite-task-zoo
$PIP install -q h5py numpy tqdm imageio 2>/dev/null
echo "VENV_READY: $($PY -c 'import robosuite, mujoco, robomimic, mimicgen; \
print(robosuite.__version__, mujoco.__version__, robomimic.__version__)')"

# ---- datasets straight from HuggingFace ------------------------------------------
HF=https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core
mkdir -p $LH/data/mimicgen
for ds in stack_d0 square_d0; do
  f=$LH/data/mimicgen/$ds.hdf5
  [ -f "$f" ] || wget -q -O "$f" "$HF/$ds.hdf5" || { echo "DOWNLOAD_FAIL $ds"; rm -f "$f"; }
done
ls -la $LH/data/mimicgen/

# ---- probe ------------------------------------------------------------------------
for ds in stack_d0 square_d0; do
  f=$LH/data/mimicgen/$ds.hdf5
  [ -f "$f" ] || { echo "PROBE_FAIL(download): $ds missing"; continue; }
  echo "=== probing $ds ==="
  MUJOCO_GL=egl $PY "$LH_DIR/impl/mimicgen/probe_replay.py" --dataset "$f" --n-demos 3
done
echo "MIMICGEN_PROBE_DONE"
