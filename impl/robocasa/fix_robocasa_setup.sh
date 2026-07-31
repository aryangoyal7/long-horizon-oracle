#!/bin/bash
# Round-2 fixes for the robocasa setup (2026-07-25 10:10):
#   groot venv: preinstall torch (a [base] dep imports it at build time)
#   datasets: correct downloader flags (--tasks ... --source human)
#   probe: real env name from the 365 registry
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$ROOT/results/robocasa/setup.log
RCPY=$LH/envs/rc/bin/python
GPY=$LH/envs/groot/bin/python
GPIP=$LH/envs/groot/bin/pip
export PIP_CACHE_DIR=$LH/pipcache HF_HOME=$LH/hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "SETUP_FIX_ROUND2_START"

# ---- datasets, correct flags ---------------------------------------------------
TASKS="PickPlaceCounterToCabinet OpenDrawer OpenCabinet TurnOnStove TurnOnSinkFaucet"
if ! grep -q RC_DATA_OK $LOG; then
  yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_datasets.py \
    --tasks $TASKS --source human >> $ROOT/results/robocasa/data_download.log 2>&1
  N=$(find $LH/repos/robocasa $LH/data/robocasa -name "*.hdf5" 2>/dev/null | wc -l)
  say "dataset hdf5 count after download: $N"
  if [ "$N" -ge 5 ]; then say "RC_DATA_OK"; else say "RC_DATA_FAIL_ROUND2"; fi
fi

# ---- groot venv with torch first ----------------------------------------------
if ! $GPY -c "import gr00t" 2>/dev/null; then
  $GPIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> $LOG 2>&1
  (cd $LH/repos/Isaac-GR00T && $GPIP install -e ".[base]" >> $ROOT/results/robocasa/groot_pip.log 2>&1) \
    || say "GROOT_PIP_BASE_FAIL_ROUND2 (see groot_pip.log)"
  $GPIP install --no-build-isolation flash-attn==2.7.1.post4 >> $ROOT/results/robocasa/groot_pip.log 2>&1 \
    || say "GROOT_FLASHATTN_FAIL_ROUND2 (eager attention fallback)"
fi
$GPY -c "import gr00t" 2>/dev/null && say "GROOT_VENV_OK" || say "GROOT_VENV_FAIL_ROUND2"

# ---- probe with a real env name ------------------------------------------------
$RCPY - >> $ROOT/results/robocasa/probe2.log 2>&1 <<'PYP'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from robocasa.utils.env_utils import create_env
env = create_env(env_name="TurnOnStove", render_onscreen=False)
env.reset()
print("[probe] lang:", env.get_ep_meta().get("lang"))
s0 = env.sim.get_state().flatten()
env.step(np.zeros(env.action_dim))
env.sim.set_state_from_flattened(s0); env.sim.forward()
drift = float(abs(s0 - env.sim.get_state().flatten()).max())
print("[probe] state save/restore drift:", drift)
assert drift < 1e-8
print("[probe] PROBE_ENV_OK")
PYP
grep -q "PROBE_ENV_OK" $ROOT/results/robocasa/probe2.log && say "RC_PROBE_OK" || say "RC_PROBE_FAIL_ROUND2"
say "SETUP_FIX_ROUND2_DONE"
