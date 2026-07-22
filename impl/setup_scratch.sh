#!/bin/bash
# Idempotent bootstrap of the ephemeral /mnt/scratch workspace after an instance restart.
# Everything here is re-derivable; persistent artifacts (code, results, checkpoints)
# live on the CIFS share. Safe to re-run at any time.
#
# Validated pins (2026-07-12):
#   robomimic  master @ e10526b  (v0.4.x, has diffusion_policy, HF datasets)
#   robosuite  v1.5.1 (from source, per robomimic docs)
#   mujoco     3.2.6  (robosuite 1.5.1 predates the mj_fullM signature change in 3.10)
#   torch      cu128 wheel
set -euo pipefail

SCRATCH=/mnt/scratch
LH=$SCRATCH/lh
PY=$LH/envs/lh/bin/python
PIP=$LH/envs/lh/bin/pip
ROBOMIMIC_COMMIT=e10526b

if [ ! -d $SCRATCH ]; then
  sudo mkdir -p $SCRATCH && sudo chown azureuser:azureuser $SCRATCH
fi
mkdir -p $LH/{data,ckpt,envs,pipcache,tmp,repos}
export PIP_CACHE_DIR=$LH/pipcache

# ---- venv -------------------------------------------------------------------
if [ ! -x $PY ]; then
  python3 -m venv $LH/envs/lh
  $PIP install -q --upgrade pip
fi
$PY -c "import torch" 2>/dev/null || \
  $PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
$PY -c "import scipy, matplotlib, h5py, einops, diffusers" 2>/dev/null || \
  $PIP install -q numpy scipy matplotlib h5py tqdm imageio imageio-ffmpeg einops tensorboard psutil

# ---- robomimic (pinned master) ------------------------------------------------
if [ ! -d $LH/repos/robomimic ]; then
  git clone -q https://github.com/ARISE-Initiative/robomimic.git $LH/repos/robomimic
fi
git -C $LH/repos/robomimic checkout -q $ROBOMIMIC_COMMIT
$PY -c "import robomimic" 2>/dev/null || $PIP install -q -e $LH/repos/robomimic

# ---- robosuite v1.5.1 + mujoco pin -------------------------------------------
$PY -c "import robosuite; assert robosuite.__version__=='1.5.1'" 2>/dev/null || \
  $PIP install -q "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
$PY -c "import mujoco; assert mujoco.__version__=='3.2.6'" 2>/dev/null || \
  $PIP install -q "mujoco==3.2.6"

# ---- vjepa venv (separate: needs transformers>=4.52, robomimic pins 4.41) ------
VPY=$LH/envs/vjepa/bin/python
if [ ! -x $VPY ]; then
  python3 -m venv $LH/envs/vjepa
  $LH/envs/vjepa/bin/pip install -q --upgrade pip
fi
$VPY -c "import torch" 2>/dev/null || \
  $LH/envs/vjepa/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
$VPY -c "import transformers, h5py" 2>/dev/null || \
  $LH/envs/vjepa/bin/pip install -q "transformers>=4.52" h5py numpy tqdm pillow

# ---- datasets -----------------------------------------------------------------
NEED_DL=0
for t in lift can square tool_hang; do
  [ -f $LH/data/robomimic/$t/ph/low_dim_v15.hdf5 ] || NEED_DL=1
done
if [ $NEED_DL = 1 ]; then
  $PY $LH/repos/robomimic/robomimic/scripts/download_datasets.py \
    --download_dir $LH/data/robomimic \
    --tasks lift can square tool_hang --dataset_types ph --hdf5_types low_dim
fi

# ---- image observation regeneration (parallel) ---------------------------------
regen () {  # task cam1 size
  local t=$1 cam=$2 sz=$3
  if [ ! -f $LH/data/robomimic/$t/ph/image_v15.hdf5 ]; then
    MUJOCO_GL=egl $PY $LH/repos/robomimic/robomimic/scripts/dataset_states_to_obs.py \
      --dataset $LH/data/robomimic/$t/ph/low_dim_v15.hdf5 --output_name image_v15.hdf5 \
      --camera_names $cam robot0_eye_in_hand --camera_height $sz --camera_width $sz \
      --done_mode 2 --exclude-next-obs > $LH/data/robomimic/$t/ph/img_gen.log 2>&1
  fi
}
regen lift agentview 84 &
regen can agentview 84 &
regen square agentview 84 &
regen tool_hang sideview 240 &
wait

# ---- sanity -------------------------------------------------------------------
$PY - <<'EOF'
import torch, h5py
import robomimic, robosuite, mujoco
from robomimic.config import config_factory
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
config_factory(algo_name="diffusion_policy")
for t in ["lift", "can", "square", "tool_hang"]:
    for kind in ["low_dim", "image"]:
        with h5py.File(f"/mnt/scratch/lh/data/robomimic/{t}/ph/{kind}_v15.hdf5", "r") as f:
            n = len(f["data"])
            assert n == 200, (t, kind, n)
print("BOOTSTRAP SANITY: 8 GPUs, diffusion_policy config, 4x200 demos low_dim+image OK")
EOF
echo "SETUP_SCRATCH_COMPLETE"
