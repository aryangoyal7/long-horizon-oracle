#!/bin/bash
# Rebuild /mnt/scratch after the July 21 wipe (repos, mg/vjepa/lh venvs,
# MimicGen datasets, image conversions), then relaunch checkpoint selection.
# Survivors: labels/ and lift/can rollouts (rescued to persistent rescue/),
# the running dp_mg trainings (deleted-inode code+data, checkpoints safe).
set -uo pipefail
LH=/mnt/scratch/lh
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PIP_CACHE_DIR=$LH/pipcache
mkdir -p $LH/repos $LH/data/mimicgen $LH/features

# ---- 1. mg venv + repos (idempotent, reuses the original setup script) ----
bash $LH_DIR/impl/mimicgen/setup_and_probe.sh && echo REBUILD_MG_VENV_DONE \
  || { echo REBUILD_MG_VENV_FAIL; exit 1; }

# ---- 2. lh venv (robomimic main + diffusion policy deps) ----
LHV=$LH/envs/lh
python3 -m venv --clear $LHV
$LHV/bin/pip install -q --upgrade pip
$LHV/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
if [ ! -d $LH/repos/robomimic ]; then
  git clone -q https://github.com/ARISE-Initiative/robomimic.git $LH/repos/robomimic
  git -C $LH/repos/robomimic checkout -q e10526b || true
fi
$LHV/bin/pip install -q "robosuite==1.5.1" "mujoco==3.2.6" diffusers h5py numpy tqdm imageio
$LHV/bin/pip install -q -e $LH/repos/robomimic
$LHV/bin/python -c "import robomimic, robosuite, diffusers, torch; print('lh ok')" \
  && echo REBUILD_LH_VENV_DONE || { echo REBUILD_LH_VENV_FAIL; exit 1; }

# ---- 3. vjepa venv (encoder + head inference) ----
VJ=$LH/envs/vjepa
python3 -m venv --clear $VJ
$VJ/bin/pip install -q --upgrade pip
$VJ/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
$VJ/bin/pip install -q transformers h5py numpy
$VJ/bin/python -c "import transformers, torch; print('vjepa ok')" \
  && echo REBUILD_VJEPA_VENV_DONE || { echo REBUILD_VJEPA_VENV_FAIL; exit 1; }

# ---- 4. re-download the four Stage 2 source datasets ----
HF=https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core
TASKS=(three_piece_assembly_d0 nut_assembly_d0 kitchen_d0 coffee_preparation_d0)
for ds in "${TASKS[@]}"; do
  f=$LH/data/mimicgen/$ds.hdf5
  [ -f "$f" ] || wget -q -O "$f" "$HF/$ds.hdf5" \
    || { echo "REBUILD_DOWNLOAD_FAIL_$ds"; rm -f "$f"; exit 1; }
done
echo REBUILD_DATASETS_DONE

# ---- 5. reconvert to image obs, GPUs 4-7 (trainings own 0-3) ----
MGPY=$LH/envs/mg/bin/python
V03=$LH/repos/robomimic_v03/robomimic/scripts/dataset_states_to_obs.py
convert() { # task gpu
  local t=$1 gpu=$2
  [ -f $LH/data/mimicgen/${t}_image.hdf5 ] && { echo "RECONVERT_${t}_SKIP"; return; }
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu $MGPY -c "
import mimicgen, runpy, sys
sys.argv = ['dataset_states_to_obs', '--dataset', '$LH/data/mimicgen/${t}.hdf5',
            '--output_name', '${t}_image.hdf5',
            '--camera_names', 'agentview', 'robot0_eye_in_hand',
            '--camera_height', '84', '--camera_width', '84',
            '--done_mode', '2', '--exclude-next-obs']
runpy.run_path('$V03', run_name='__main__')" \
    > $LH/data/mimicgen/${t}_img_regen.log 2>&1 \
    && echo "RECONVERT_${t}_DONE" || echo "RECONVERT_${t}_FAIL"
}
convert three_piece_assembly_d0 4 &
convert nut_assembly_d0 5 &
convert kitchen_d0 6 &
convert coffee_preparation_d0 7 &
wait
echo REBUILD_ALL_DONE

# ---- 6. relaunch checkpoint selection (waits for trainings itself) ----
setsid nohup bash $LH_DIR/impl/mimicgen/run_ckpt_selection.sh \
  > $LH_DIR/results/stage2/ckpt_select_run.log 2>&1 < /dev/null &
echo CKPT_SELECT_RELAUNCHED

# NOTE (Jul 21): after cloning robomimic_v03, ALWAYS re-apply the mujoco_py
# import guard in robomimic/envs/env_robosuite.py (robosuite 1.4 uses the DM
# bindings; the unconditional import breaks env construction):
#   try: import mujoco_py / except ImportError: mujoco_py = None
