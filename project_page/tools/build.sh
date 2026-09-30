#!/bin/bash
# Build the site's figures, tables and video manifest from the paper draft.
#
#   bash tools/build.sh [PAPER_SRC]
#
# PAPER_SRC: the draft .zip or directory (default: the latest writeup zip in
# results/report_stability_20260902). Videos are rendered separately with
# tools/render_videos.sh (needs the simulator stacks); this script only
# rebuilds static/js/label_videos.js from what has been rendered.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PAPER_SRC=${1:-$ROOT/results/report_stability_20260902/Measuring_the_Stability_Assumption_Behind_Action_Chunking__1_.zip}

bash "$HERE/tools/build_figures.sh" "$PAPER_SRC"
python3 "$HERE/tools/build_tables.py" "$PAPER_SRC" "$HERE/index.html"
python3 "$HERE/tools/build_video_manifest.py"
touch "$HERE/.nojekyll"
echo "BUILD_OK  preview: cd $HERE && python3 -m http.server 8000"
