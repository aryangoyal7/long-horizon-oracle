#!/bin/bash
# Render the paper's TikZ figures to SVG for the site (vector, no rasterising).
#
#   bash tools/build_figures.sh [PAPER_SRC]
#
# PAPER_SRC is the draft directory (the one holding iclr2027_conference.tex,
# method.tex and figs/) or the draft .zip. Default: the latest writeup zip in
# results/report_stability_20260902. Needs latex and dvisvgm (>= 2.13).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PAPER_SRC=${1:-$ROOT/results/report_stability_20260902/Measuring_the_Stability_Assumption_Behind_Action_Chunking__1_.zip}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [ -f "$PAPER_SRC" ]; then
  unzip -q "$PAPER_SRC" -d "$WORK/paper"
  PAPER_SRC=$WORK/paper
fi
python3 "$HERE/tools/derive_figures.py" "$PAPER_SRC" "$WORK/tex"

OUT=$HERE/static/images
mkdir -p "$OUT"
cd "$WORK/tex"
for f in fig_method fig_hist fig_stack fig_lamK8; do
  latex -interaction=nonstopmode -halt-on-error "$f.tex" > "$f.build.log" 2>&1 \
    || { tail -30 "$f.build.log"; echo "LATEX_FAIL $f"; exit 1; }
  ZOOM=1
  [ "$f" = fig_method ] && ZOOM=2   # hero image: 2x
  # --no-fonts: glyphs become paths, so the SVG renders the same in every
  # browser without the (unsupported) SVG-font elements.
  dvisvgm --no-fonts --exact-bbox --zoom=$ZOOM --output="$OUT/$f.svg" "$f.dvi" \
    > "$f.svg.log" 2>&1 || { cat "$f.svg.log"; echo "DVISVGM_FAIL $f"; exit 1; }
  echo "wrote static/images/$f.svg ($(du -h "$OUT/$f.svg" | cut -f1))"
done
