#!/bin/bash
# Select the six stamps per task and render the 58 label clips.
#
#   bash tools/render_videos.sh [robomimic|mimicgen|robocasa ...]
#
# Needs the research repo's simulator venvs on /mnt/scratch/lh/envs:
#   lh  robomimic e10526b + robosuite 1.5.1 + mujoco 3.2.6  (impl/crn/bootstrap_dp.sh)
#   mg  robomimic 0.3 + robosuite 1.4.1 + mujoco 2.3.2 + mimicgen (same script)
#   rc  robosuite 5ce6643f + robocasa a07e365 + assets + rc_*.hdf5 (tools/setup_robocasa_stack.sh)
# and the demo datasets under /mnt/scratch/lh/data (see README). Each clip
# replays its demo from the start, so a late stamp takes a few minutes; the
# platforms run in parallel on CPU with EGL rendering.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ENVS=/mnt/scratch/lh/envs
PLATFORMS=${*:-robomimic mimicgen robocasa}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$HERE/tools/render_log"
python3 "$HERE/tools/select_stamps.py" > "$HERE/tools/stamps.json"
declare -A VENV=([robomimic]=lh [mimicgen]=mg [robocasa]=rc)
pids=()
for p in $PLATFORMS; do
  "$ENVS/${VENV[$p]}/bin/python" "$HERE/tools/render_label_videos.py" --platform "$p" \
    > "$HERE/tools/render_log/$p.out" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
grep -h -E "mp4:|FAIL" "$HERE"/tools/render_log/*.out || true
python3 "$HERE/tools/build_video_manifest.py"
