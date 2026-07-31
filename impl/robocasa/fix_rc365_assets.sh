#!/bin/bash
# Post-rebuild fix (2026-07-30): the Jul 30 scratch rebuild downloaded only
# the base kitchen assets; robocasa365 scenes also need the lightwheel /
# objaverse / aigen packs. Original layout (setup_rc365_uv.log): packs merge
# into the classic repo's assets dir (the shared cache) and robocasa365's
# assets dir symlinks each cache subdir. Recreate that, then relaunch the
# GPU chains that died on the missing assets.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
RCPY=$LH/envs/rc/bin/python
CACHE=$LH/repos/robocasa/robocasa/models/assets
RC365A=$LH/repos/Isaac-GR00T/external_dependencies/robocasa365/robocasa/models/assets
LOG=$ROOT/results/robocasa/rebuild_robocasa.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "RC365_ASSET_FIX_START"

yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_kitchen_assets.py \
  --type tex tex_generative fixtures_lw objs_lw objs_objaverse objs_aigen \
  >> $ROOT/results/robocasa/asset_packs.log 2>&1 \
  && say "ASSET_PACKS_OK" || { say "ASSET_PACKS_FAIL"; exit 1; }

mkdir -p $RC365A
for d in $CACHE/*/; do
  name=$(basename "$d")
  [ -e $RC365A/$name ] || ln -s "$d" $RC365A/$name
done
say "RC365_SYMLINKS_OK ($(ls $RC365A | wc -l) entries)"

# sanity: the two files the crashed chains asked for
ok=1
[ -e $RC365A/fixtures/windows/Window050/model.xml ] || { say "STILL_MISSING Window050"; ok=0; }
[ -e $RC365A/objects/lightwheel/utensil_rack/UtensilRack007/model.xml ] || { say "STILL_MISSING UtensilRack007"; ok=0; }

if [ $ok -eq 1 ]; then
  setsid nohup bash $ROOT/impl/robocasa/act/run_act_fix2.sh > $LH/relaunch_act.out 2>&1 < /dev/null &
  setsid nohup bash $ROOT/impl/robocasa/run_groot_fork.sh > $LH/relaunch_fork.out 2>&1 < /dev/null &
  say "GPU_CHAINS_RELAUNCHED"
else
  say "GPU_CHAINS_HELD (assets still incomplete)"
fi
say "RC365_ASSET_FIX_DONE"
