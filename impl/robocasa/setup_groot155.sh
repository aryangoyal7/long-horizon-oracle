#!/bin/bash
# Env for the robocasa-benchmark Isaac-GR00T fork (GR00T N1.5, matches the
# robocasa365 checkpoint-120000). py3.10, torch 2.5.1, transformers 4.51.3.
# tensorflow (base extra) is skipped: it pins numpy<2 but robocasa365 asserts
# numpy==2.2.5 and nothing on the eval path imports tf.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
ENV=$LH/envs/groot155
FORK=$LH/repos/Isaac-GR00T-rc365
RC365=$LH/repos/Isaac-GR00T/external_dependencies/robocasa365
LOG=$ROOT/results/robocasa/setup_groot155.log
MARK=$ROOT/results/robocasa
export PIP_CACHE_DIR=$LH/pipcache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

say "=== groot155 setup start ==="
if [ ! -x $ENV/bin/python ]; then
  /anaconda/bin/conda create -y -p $ENV -c conda-forge --override-channels python=3.10 >> $LOG 2>&1 \
    || { say "CONDA_CREATE_FAIL"; exit 1; }
fi
P=$ENV/bin/pip
$P install --upgrade pip setuptools wheel psutil ninja packaging >> $LOG 2>&1

$P install torch==2.5.1 torchvision==0.20.1 >> $LOG 2>&1 || { say "TORCH_FAIL"; exit 1; }
say "torch OK"

$P install -e $FORK >> $LOG 2>&1 || { say "GR00T_FORK_FAIL"; exit 1; }
$P install diffusers==0.30.2 pyzmq >> $LOG 2>&1
say "gr00t fork OK"

$P install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" >> $LOG 2>&1 \
  || { say "FLASH_WHEEL_FAIL (will try compile-free fallback later)"; }
say "flash-attn step done"

$P install "git+https://github.com/ARISE-Initiative/robosuite.git@85abee228d1c43ab1939bce33028099945d453b4" >> $LOG 2>&1 || { say "ROBOSUITE_FAIL"; exit 1; }
$P install -e $RC365 --no-deps --config-settings editable_mode=compat >> $LOG 2>&1 || { say "RC365_FAIL"; exit 1; }
$P install numpy==2.2.5 numba scipy mujoco==3.3.1 pygame Pillow opencv-python-headless pyyaml pynput tqdm termcolor imageio h5py lxml hidapi tianshou gymnasium==1.0.0 loguru tenacity >> $LOG 2>&1
say "robocasa365 deps OK"

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $ENV/bin/python - >> $LOG 2>&1 <<'PY'
import numpy, torch, flash_attn, transformers
print("versions:", numpy.__version__, torch.__version__, flash_attn.__version__, transformers.__version__)
import robocasa, robosuite
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
print("IMPORTS_OK", list(TASK_SET_REGISTRY.keys())[:5])
PY
if [ $? -eq 0 ]; then say "GROOT155_OK"; touch $MARK/GROOT155_OK; else say "GROOT155_IMPORT_FAIL"; fi
say "=== groot155 setup end ==="
