#!/bin/bash
# Stage 2: MimicGen long-horizon policy training, one GPU chain per task.
# Chain per GPU: image conversion (mg venv, robosuite 1.4 + mimicgen envs via
# runpy wrapper) -> DP training (lh venv, rollouts disabled, own setsid).
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LHPY=/mnt/scratch/lh/envs/lh/bin/python
MGPY=/mnt/scratch/lh/envs/mg/bin/python
V03=/mnt/scratch/lh/repos/robomimic_v03/robomimic/scripts/dataset_states_to_obs.py
MG=/mnt/scratch/lh/data/mimicgen
OUT=$LH_DIR/results/training

chain() { # task gpu
  local t=$1 gpu=$2
  if [ ! -f $MG/${t}_image.hdf5 ]; then
    echo "CONVERT $t (gpu $gpu) $(date -u +%H:%M:%S)"
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu $MGPY -c "
import mimicgen, runpy, sys
sys.argv = ['dataset_states_to_obs', '--dataset', '$MG/${t}.hdf5',
            '--output_name', '${t}_image.hdf5',
            '--camera_names', 'agentview', 'robot0_eye_in_hand',
            '--camera_height', '84', '--camera_width', '84',
            '--done_mode', '2', '--exclude-next-obs']
runpy.run_path('$V03', run_name='__main__')" \
      > $MG/${t}_img_gen.log 2>&1 || { echo "STAGE2_CONVERT_${t}_FAIL"; return; }
  fi
  echo "STAGE2_CONVERT_${t}_DONE $(date -u +%H:%M:%S)"
  # NEVER auto-overwrite an existing experiment dir: robomimic's "y" answer
  # DELETES it, checkpoints included (this destroyed the surviving grids on
  # Jul 21). If the dir exists, refuse and let a human decide.
  if [ -d "$OUT/dp_mg_${t}/dp_mg_${t}_image" ]; then
    echo "STAGE2_TRAIN_${t}_REFUSED_DIR_EXISTS"
    return
  fi
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu \
    setsid nohup $LHPY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py \
      --config $LH_DIR/impl/configs/dp_mg_${t}.json \
      >> "$OUT/dp_mg_${t}.launch.log" 2>&1 < /dev/null &
  echo "STAGE2_TRAIN_${t}_LAUNCHED (gpu $gpu, pid $!)"
}

chain three_piece_assembly_d0 0 &
chain nut_assembly_d0 1 &
chain kitchen_d0 2 &
chain coffee_preparation_d0 3 &
wait
echo STAGE2_ALL_LAUNCHED
