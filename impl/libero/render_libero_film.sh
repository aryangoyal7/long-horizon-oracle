#!/bin/bash
# Rebuild the LIBERO stack if scratch was wiped, then render the stitched
# label film. Kept separate from render_finished.sh (which makes 5-demo clips).
set -uo pipefail
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LH=/mnt/scratch/lh
PY=$LH/envs/mg/bin/python
# LIBERO resolves as a namespace package off the repo root (no top-level
# __init__.py at commit 8f1084e), so the editable install alone is not enough.
export PYTHONPATH="$LH/repos/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
LOG=$ROOT/results/label_videos/films/libero_render.log
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

# 1. the mg venv is built by mimicgen/setup_and_probe.sh via rebuild_scratch.sh
until [ -x "$PY" ] && $PY -c "import robosuite" 2>/dev/null; do sleep 60; done
say "mg venv ready"

# 2. LIBERO repo + libero_10 datasets (idempotent, downloads if absent)
bash $ROOT/impl/libero/setup_and_probe.sh >> "$LOG" 2>&1
$PY -c "import libero" 2>/dev/null || { say "FATAL libero import"; exit 1; }
say "libero ready"

TASK=${TASK:-KITCHEN_SCENE3_turn_on_the_stove_and_put}
L=$ROOT/artifacts/labels/final2_ol_libero10_${TASK}.npz
STATS=$ROOT/results/labels2/libero10_${TASK}/label_stats.json
D=$($PY -c "import json;raw=open('$STATS').read().replace('NaN','null');print(round(json.loads(raw)['suggested_deadband_delta_p95_free'],4))")
F=$(ls $LH/data/libero/libero_10/${TASK}*.hdf5 2>/dev/null | head -1)
[ -n "$F" ] || { say "FATAL no dataset for $TASK"; exit 1; }
say "task=$TASK delta=$D dataset=$(basename $F)"

OUT=$ROOT/results/label_videos/films/libero10_${TASK}_labeled_film.mp4
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 $PY $ROOT/impl/libero/render_label_films_libero.py \
  --task "libero10 ${TASK}" --dataset "$F" --labels "$L" --delta "$D" \
  --out "$OUT" ${FILM_ARGS:-} >> "$LOG" 2>&1
say "film exit=$? out=$OUT"
