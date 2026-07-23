#!/bin/bash
# Recovery for the Jul 23 torchvision fault: the rebuild's reconversion step
# failed on all four tasks (mismatched PyPI torchvision against cu128 torch),
# which made resume_ckpt_selection.sh fail vacuously. Reruns steps 5-6 of
# rebuild_scratch.sh now that the venv is fixed. Appends to the rebuild log.
set -u
LH=/mnt/scratch/lh
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
    && echo "RECONVERT_${t}_DONE" || { echo "RECONVERT_${t}_FAIL"; exit 1; }
}
convert three_piece_assembly_d0 4 &
convert nut_assembly_d0 5 &
convert kitchen_d0 6 &
convert coffee_preparation_d0 7 &
wait
ok=1
for t in three_piece_assembly_d0 nut_assembly_d0 kitchen_d0 coffee_preparation_d0; do
  [ -f $LH/data/mimicgen/${t}_image.hdf5 ] || ok=0
done
[ $ok -eq 1 ] || { echo RECONVERT_RETRY_FAILED; exit 1; }
echo REBUILD_ALL_DONE_RETRY

setsid nohup bash $LH_DIR/impl/mimicgen/resume_ckpt_selection.sh \
  > $LH_DIR/results/stage2/resume_select.log 2>&1 < /dev/null &
echo CKPT_SELECT_RESUMED
