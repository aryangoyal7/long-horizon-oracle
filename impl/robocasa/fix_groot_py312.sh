#!/bin/bash
# Round-4 groot fix: gr00t requires Python >=3.12,<3.13 (the 3.10 venv failed
# the version gate every round). Build a conda 3.12 env, install GR00T's own
# pins, compile flash-attn against /usr/local/cuda.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$ROOT/results/robocasa/setup.log
export PIP_CACHE_DIR=$LH/pipcache CUDA_HOME=/usr/local/cuda MAX_JOBS=32
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "GROOT_FIX_PY312_START"

if [ ! -x $LH/envs/groot312/bin/python ]; then
  /anaconda/bin/conda create -y -p $LH/envs/groot312 -c conda-forge --override-channels python=3.12 >> $ROOT/results/robocasa/groot_pip.log 2>&1 \
    || { say "GROOT_CONDA_CREATE_FAIL"; exit 1; }
fi
GPY=$LH/envs/groot312/bin/python
GPIP="$GPY -m pip"
$GPIP install -q --upgrade pip setuptools wheel psutil ninja packaging >> $ROOT/results/robocasa/groot_pip.log 2>&1
$GPY -c "import torch" 2>/dev/null || \
  $GPIP install torch==2.9.0 torchvision==0.24.0 >> $ROOT/results/robocasa/groot_pip.log 2>&1
(cd $LH/repos/Isaac-GR00T && $GPIP install -e ".[base]" >> $ROOT/results/robocasa/groot_pip.log 2>&1) \
  || say "GROOT_PIP_BASE_FAIL_ROUND4"
$GPY -c "import flash_attn" 2>/dev/null || \
  $GPIP install --no-build-isolation flash-attn==2.7.1.post4 >> $ROOT/results/robocasa/groot_pip.log 2>&1 \
  || say "GROOT_FLASHATTN_FAIL_ROUND4"
if $GPY -c "import gr00t; print('gr00t import ok')" >> $LOG 2>&1; then
  say "GROOT_VENV_OK (envs/groot312, python 3.12)"
else
  say "GROOT_VENV_FAIL_ROUND4"
fi
say "GROOT_FIX_PY312_DONE"
