#!/bin/bash
# Stage A setup for the generalist-policy switching experiment (approved
# 2026-07-25): RoboCasa Kitchen (Franka) + GR00T N1.5 robocasa365 checkpoint.
# Staged with markers; each stage skips if already done, fails soft so later
# stages still run. The pipeline loop reads setup.log and iterates on failures.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$ROOT/results/robocasa/setup.log
mkdir -p $ROOT/results/robocasa $LH/repos $LH/data/robocasa $LH/hf
export PIP_CACHE_DIR=$LH/pipcache HF_HOME=$LH/hf
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "ROBOCASA_SETUP_START"

BASEPY=""
for _p in /anaconda/envs/azureml_py310_sdkv2/bin/python /anaconda/envs/azureml_py38/bin/python3 /anaconda/bin/python3 python3; do
  command -v "$_p" >/dev/null 2>&1 && "$_p" -c "import ensurepip" 2>/dev/null && { BASEPY=$(command -v "$_p"); break; }
done
say "basepy=$BASEPY"

# ---- 1. rc venv: robosuite + robocasa -----------------------------------------
RCPY=$LH/envs/rc/bin/python
RCPIP=$LH/envs/rc/bin/pip
if ! $RCPY -c "import robocasa" 2>/dev/null; then
  [ -x $RCPY ] || "$BASEPY" -m venv $LH/envs/rc
  $RCPIP install -q --upgrade pip setuptools wheel
  [ -d $LH/repos/robosuite_rc ] || git clone -q https://github.com/ARISE-Initiative/robosuite $LH/repos/robosuite_rc
  [ -d $LH/repos/robocasa ] || git clone -q https://github.com/robocasa/robocasa $LH/repos/robocasa
  $RCPIP install -q -e $LH/repos/robosuite_rc >> $LOG 2>&1
  $RCPIP install -q -e $LH/repos/robocasa >> $LOG 2>&1
  $RCPIP install -q h5py imageio imageio-ffmpeg huggingface_hub >> $LOG 2>&1
fi
$RCPY -c "import robocasa" 2>/dev/null && say "RC_VENV_OK" || say "RC_VENV_FAIL"

# ---- 2. kitchen assets (~5 GB) -------------------------------------------------
if [ ! -f $ROOT/results/robocasa/ASSETS_OK ]; then
  yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_kitchen_assets.py >> $LOG 2>&1 \
    && { touch $ROOT/results/robocasa/ASSETS_OK; say "RC_ASSETS_OK"; } \
    || say "RC_ASSETS_FAIL"
  $RCPY $LH/repos/robocasa/robocasa/scripts/setup_macros.py >> $LOG 2>&1 || true
fi

# ---- 3. groot venv: Isaac-GR00T ------------------------------------------------
GPY=$LH/envs/groot/bin/python
GPIP=$LH/envs/groot/bin/pip
if ! $GPY -c "import gr00t" 2>/dev/null; then
  [ -x $GPY ] || "$BASEPY" -m venv $LH/envs/groot
  $GPIP install -q --upgrade pip setuptools wheel
  [ -d $LH/repos/Isaac-GR00T ] || git clone -q https://github.com/NVIDIA/Isaac-GR00T $LH/repos/Isaac-GR00T
  (cd $LH/repos/Isaac-GR00T && $GPIP install -e ".[base]" >> $LOG 2>&1) || say "GROOT_PIP_BASE_FAIL"
  $GPIP install --no-build-isolation flash-attn==2.7.1.post4 >> $LOG 2>&1 || say "GROOT_FLASHATTN_FAIL (may still run with eager attention)"
fi
$GPY -c "import gr00t" 2>/dev/null && say "GROOT_VENV_OK" || say "GROOT_VENV_FAIL"

# ---- 4. checkpoint: GR00T N1.5 robocasa365 multitask ---------------------------
if [ ! -f $ROOT/results/robocasa/CKPT_OK ]; then
  $RCPY - >> $LOG 2>&1 <<'PYC'
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
found = None
for repo in ["robocasa/robocasa365_checkpoints", "robocasa/robocasa365-checkpoints",
             "robocasa/checkpoints", "nvidia/GR00T-N1.5-robocasa"]:
    try:
        files = api.list_repo_files(repo)
        hits = [f for f in files if "n1" in f.lower() and "multitask" in f.lower()]
        print(f"[ckpt] {repo}: {len(files)} files, {len(hits)} n1.5-multitask hits", flush=True)
        if hits:
            found = repo
            print("[ckpt] sample:", hits[:5], flush=True)
            pref = sorted({h.split("checkpoint-")[0] for h in hits if "checkpoint-120000" in h})
            pats = [p + "checkpoint-120000/*" for p in pref] or None
            snapshot_download(repo, allow_patterns=pats,
                              local_dir="/mnt/scratch/lh/data/robocasa/ckpt")
            print("[ckpt] downloaded", pats, flush=True)
            break
    except Exception as e:
        print(f"[ckpt] no: {repo} {type(e).__name__} {str(e)[:100]}", flush=True)
if not found:
    raise SystemExit(1)
PYC
  [ $? -eq 0 ] && { touch $ROOT/results/robocasa/CKPT_OK; say "RC_CKPT_OK"; } || say "RC_CKPT_FAIL (repo id needs manual resolution)"
fi

# ---- 5. starter demo datasets (states+actions, for FTLE labeling) --------------
if [ ! -f $ROOT/results/robocasa/DATA_OK ]; then
  DL=$LH/repos/robocasa/robocasa/scripts/download_datasets.py
  $RCPY $DL --help >> $LOG 2>&1
  yes y 2>/dev/null | $RCPY $DL --ds_types human_raw >> $LOG 2>&1 \
    && { touch $ROOT/results/robocasa/DATA_OK; say "RC_DATA_OK"; } \
    || say "RC_DATA_FAIL (see --help output in log)"
fi

# ---- 6. probes -----------------------------------------------------------------
say "PROBE_ENV_START"
timeout 600 bash -c "MUJOCO_GL=egl $RCPY -m robocasa.demos.demo_random_action --help" >> $LOG 2>&1 \
  && say "RC_DEMO_HELP_OK" || say "RC_DEMO_HELP_FAIL"
$RCPY - >> $LOG 2>&1 <<'PYP'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite, robocasa
print("[probe] robosuite", robosuite.__version__)
from robocasa.utils.env_utils import create_env
env = create_env(env_name="PnPCounterToCab", render_onscreen=False)
env.reset()
s0 = env.sim.get_state().flatten()
import numpy as np
a = np.zeros(env.action_dim)
env.step(a)
env.sim.set_state_from_flattened(s0); env.sim.forward()
s1 = env.sim.get_state().flatten()
print("[probe] state save/restore drift:", float(abs(s0 - s1).max()))
print("[probe] PROBE_ENV_OK")
PYP
grep -q "PROBE_ENV_OK" $LOG && say "RC_PROBE_OK" || say "RC_PROBE_FAIL"

say "ROBOCASA_SETUP_DONE (check markers above for stage failures)"
