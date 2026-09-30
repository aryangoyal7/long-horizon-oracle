#!/bin/bash
# Rebuild the RoboCasa demo-replay stack on /mnt/scratch for the label videos.
#
# Same revisions as the rc_relabel_20260831 demo-label campaign
# (impl/crn/bootstrap_rc.sh): robosuite 5ce6643f, robocasa a07e365, target-split
# LeRobot demos converted by impl/robocasa/lerobot_to_hdf5.py (first 200
# episodes, the demo_id mapping the labels use). No GR00T, no checkpoint.
# Logs go to /mnt/scratch only; nothing in the research repo is written.
# Idempotent: every stage checks the real artifact.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$LH/site_logs/setup_robocasa.log
BASEPY=/anaconda/envs/azureml_py310_sdkv2/bin/python
RCPY=$LH/envs/rc/bin/python
RCPIP=$LH/envs/rc/bin/pip
TASKS="OpenCabinet OpenDrawer PickPlaceCounterToCabinet TurnOnSinkFaucet"
export PIP_CACHE_DIR=$LH/pipcache
mkdir -p $LH/repos $LH/envs $LH/data/robocasa/hdf5 $(dirname $LOG)
say(){ echo "[$(date -u +%F' '%T)] $*" | tee -a $LOG; }

clone_at(){ # url dir ref
  [ -e "$2/.git" ] && { say "$(basename $2) @ $(git -C $2 rev-parse --short HEAD) (existing)"; return 0; }
  git clone -q "$1" "$2" && git -C "$2" checkout -q "$3" || { say "CLONE_FAIL $2"; return 1; }
  say "$(basename $2) @ $(git -C $2 rev-parse --short HEAD)"
}
clone_at https://github.com/ARISE-Initiative/robosuite $LH/repos/robosuite_rc 5ce6643f
clone_at https://github.com/robocasa/robocasa $LH/repos/robocasa a07e365

if ! $RCPY -c "import robocasa, pandas" 2>/dev/null; then
  [ -x $RCPY ] || "$BASEPY" -m venv $LH/envs/rc
  $RCPIP install -q --upgrade pip setuptools wheel >> $LOG 2>&1
  $RCPIP install -q -e $LH/repos/robosuite_rc >> $LOG 2>&1
  $RCPIP install -q -e $LH/repos/robocasa >> $LOG 2>&1
  $RCPIP install -q h5py imageio imageio-ffmpeg pandas pyarrow pillow termcolor >> $LOG 2>&1
fi
$RCPY -c "import robosuite, robocasa; print(robosuite.__version__, robocasa.__version__)" >> $LOG 2>&1 \
  && say "RC_VENV_OK" || { say "RC_VENV_FAIL"; exit 1; }
[ -f $LH/repos/robocasa/robocasa/macros_private.py ] || \
  $RCPY $LH/repos/robocasa/robocasa/scripts/setup_macros.py >> $LOG 2>&1 || true

A=$LH/repos/robocasa/robocasa/models/assets
if [ "$(ls $A/objects/objaverse 2>/dev/null | wc -l)" -lt 10 ]; then
  # the default download at a07e365 already fetches all six packs (textures,
  # generative textures, lightwheel fixtures/objects, objaverse, aigen objects)
  say "downloading kitchen assets"
  yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_kitchen_assets.py >> $LOG 2>&1 \
    && say "BASE_ASSETS_OK" || say "BASE_ASSETS_FAIL"
fi
say "assets: $(du -sh $A 2>/dev/null | cut -f1)"

echo y | $RCPY $LH/repos/robocasa/robocasa/scripts/download_datasets.py \
  --tasks $TASKS --split target --source human >> $LOG 2>&1 && say "DATASETS_OK" || say "DATASETS_FAIL"

declare -A SRC=(
 [OpenCabinet]=$LH/repos/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot
 [OpenDrawer]=$LH/repos/robocasa/datasets/v1.0/target/atomic/OpenDrawer/20250816/lerobot
 [PickPlaceCounterToCabinet]=$LH/repos/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
 [TurnOnSinkFaucet]=$LH/repos/robocasa/datasets/v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot
)
for T in $TASKS; do
  DS=$LH/data/robocasa/hdf5/rc_$T.hdf5
  [ -f $DS ] && { say "hdf5 $T present"; continue; }
  $RCPY $ROOT/impl/robocasa/lerobot_to_hdf5.py --lerobot-dir "${SRC[$T]}" --out $DS.part \
    --n-demos 200 >> $LOG 2>&1 && mv $DS.part $DS && say "CONVERT_OK $T" || say "CONVERT_FAIL $T"
done
say "SETUP_ROBOCASA_DONE"
