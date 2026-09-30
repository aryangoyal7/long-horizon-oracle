# Project page: Measuring the Stability Assumption Behind Action Chunking

Static site for GitHub Pages, built on the [Nerfies](https://github.com/nerfies/nerfies.github.io)
template with the layout of [action-chunking.github.io](https://action-chunking.github.io/).
It covers the **open-loop results only**. The closed-loop section is a commented-out
placeholder in `index.html` (heading only), to be filled in once the closed-loop labels are
re-labeled.

Every number, claim and caption comes from the latest draft
(`results/report_stability_20260902/Measuring_the_Stability_Assumption_Behind_Action_Chunking__1_.zip`,
`iclr2027_conference.tex`). Table bodies are parsed out of the draft's LaTeX, not retyped.

## Build

```bash
cd project_page
bash tools/build.sh            # figures (SVG) + tables + video manifest, from the draft zip
python3 -m http.server 8000    # preview at http://localhost:8000
```

`tools/build.sh [PAPER_SRC]` takes the draft `.zip` or an unpacked draft directory. It needs
`latex` and `dvisvgm` (>= 2.13; `sudo apt-get install dvisvgm`).

The label videos are rendered separately, because they need the simulator stacks:

```bash
bash tools/render_videos.sh    # all three platforms; or: bash tools/render_videos.sh robomimic
```

To publish on the personal site instead (aryangoyal7.github.io, as an incomplete-draft
publication at `/publication/2026-stability-action-chunking`):

```bash
python3 tools/export_personal_site.py /home/azureuser/cloudfiles/code/aryangoyal7.github.io \
    --drive-url 'https://drive.google.com/...'
```

This writes `_publications/2026-09-30-stability-action-chunking.html` (the site's
`layout: none` project-page convention, CSS/JS inlined, Bulma and Font Awesome from a CDN) and
copies the figures and videos to `assets/figures/stability-action-chunking/`. Re-run it after
`tools/build.sh` to refresh the copy.

To deploy standalone, push the contents of `project_page/` (including `.nojekyll`) to the root of the
Pages repository. The `tools/` directory is only needed to rebuild. Before going live, see
**Placeholders to fill** below.

## Layout

```
index.html                 the page
static/css/index.css       styles (reference-site classes; paper's class colours)
static/css/bulma.min.css   Nerfies
static/js/index.js         section rail, KaTeX, label-video grid (reference-site pattern)
static/js/label_videos.js  generated: clip list + button tooltips
static/js/fontawesome.all.min.js   Nerfies
static/images/fig_*.svg    generated: the paper's TikZ figures, vector
static/videos/*.mp4        generated: 58 label clips, <platform>_<task>_<class>_<idx>.mp4
static/videos/manifest.json  generated: provenance per clip (stamp, lambda, verification)
tools/                     build scripts (below)
```

| script | does |
|---|---|
| `tools/build_figures.sh` | unpacks the draft, derives the site variants (`derive_figures.py`), `latex` + `dvisvgm --no-fonts` to `static/images/` (hero at `--zoom=2`) |
| `tools/derive_figures.py` | copies the paper's `\definecolor` lines into each wrapper; removes the closed-loop parts (see below); asserts the text it cuts, so a changed draft fails instead of leaking a closed-loop panel |
| `tools/build_tables.py` | parses `tab:ol-shares-window` and `tab:pred-ol-app` (asserted equal to `tab:pred-ol`) into `index.html` |
| `tools/select_stamps.py` | picks the six stamps per task, writes `tools/stamps.json` |
| `tools/render_label_videos.py` | renders one platform's clips (run in that platform's venv) |
| `tools/build_video_manifest.py` | writes `label_videos.js` + `manifest.json`; fails on a missing clip, a clip over 3 MB, or an unchecked one |
| `tools/setup_robocasa_stack.sh` | rebuilds the RoboCasa replay stack on `/mnt/scratch` at the campaign's pins |

## Figures

Rendered from the draft's TikZ with the draft preamble's colours (`unst` 204,102,92;
`stab` 88,152,120; `dead` 135,135,140), no rasterising. Three are derived to drop closed-loop
content; the cut is done by `tools/derive_figures.py` on a copy, never in the paper sources.

| site file | source | change |
|---|---|---|
| `fig_method.svg` (hero, 2x) | `method.tex` (the draft zip has no `figs/fig_method.tex`; `method.tex` is the method schematic) | panel (b), the closed-loop probe with the lambda_pert - lambda_ctrl estimator, removed; panel (a) centred over (c), (d) |
| `fig_hist.svg` | `figs/fig_hist.tex` | closed-loop row removed, legend moved up |
| `fig_stack.svg` | `figs/fig_stack.tex` | closed-loop row removed, legend moved up |
| `fig_lamK8.svg` | `figs/fig_lamK8.tex` | none |

## Label videos

One panel per task (robomimic lift, can, square, tool_hang; MimicGen coffee, nut_assembly;
RoboCasa OpenCabinet, OpenDrawer, PickPlaceCounterToCabinet, TurnOnSinkFaucet), six buttons
each, picked from the **demonstration-stamp labels at m = 2.0, t = 2.365, K = 24** (the
labels behind the paper's tables: `results/labels_bothK_20260902/bothK_ol_<task>.npz`, `k24_*`):

* 1, 2 unstable: highest lambda_bar clearing both gates
* 3, 4 deadband: lambda_bar nearest zero
* 5, 6 stable: most negative lambda_bar clearing both gates

The two picks of a class come from different demonstrations. Consecutive stamps of one demo
are two steps apart and render as near-identical clips; on OpenDrawer the plain top-2 unstable
and top-2 stable picks are such pairs. `python3 tools/select_stamps.py --allow-same-demo`
restores the plain top-2.

**The branches shown are the labeled branches, reproduced exactly.** The labelers reuse one
env across a demo's stamps and robosuite carries state across `reset_to` (gripper command
ramp, observable timers), so a branch rolled out from a cold reset is not the branch that was
labeled. Each clip therefore replays its labeler's call sequence from the demo's reset to the
target stamp (the labeler's own `make_env`, the same `reset_to` calls, the same RNG draws),
records the target stamp's branches and recomputes lambda_bar from them. No GL context exists
during that rollout (on RoboCasa, attaching one moved lambda_bar at the 1e-5 level); frames are
rendered afterwards by restoring each recorded state. The recomputed value is stored in
`manifest.json` next to the label's (`abs_diff`).

On RoboCasa the history is per labeling *attempt*: the demo pass resumed demos across rounds
and relaunches, so a stamp's row may come from a fresh attempt (replay gate, then stamps from
t = 0) or a resumed one (no replay gate, first pending stamp onward). `rc_attempt()` reads the
campaign's per-stamp rows (`results/rc_relabel_20260831/demo_stamps/`), finds the row the
consolidated label kept, and replays that attempt; the manifest records which (`history`).
RoboCasa labels are stored to 6 decimals, so there `abs_diff` is at the 1e-7 level; elsewhere
it is at machine precision.

Each clip: agentview camera (RoboCasa: `robot0_agentview_left`), 640x480, 20 fps, K = 24 frames
(the state after each replayed action) plus 1 s hold, H.264 (`yuv420p`, `+faststart`). The
image is the nominal branch blended 50/50 with the mean of the 8 perturbed branches, so the
scene ghosts where branches separate; the nominal end-effector path is white, the perturbed
paths are in the class colour, and the top-left shows task, lambda_bar (3 decimals), class and
K.

Names: `coffee` is MimicGen `coffee_preparation_d0` and `nut_assembly` is `nut_assembly_d0`,
as in the paper's tables.

## Placeholders to fill

* **Authors.** The draft's `\author` block is still the template placeholder (Jane E. Doe,
  UC Berkeley EECS), so the header says "Anonymous authors" and the BibTeX says `Anonymous`.
  Both spots are marked in `index.html`.
* **Paper PDF.** The Paper button points to `static/paper.pdf`, which is not included: the
  only PDF available is the draft, which contains the closed-loop results.
* **arXiv, Code.** Buttons present, links empty (`is-disabled-coming-soon`).

## Text taken from the draft, and what was cut

* Abstract: closed-loop content removed. "under two execution regimes: open-loop, where
  ..., and closed-loop, where the policy replans after the perturbation" became "under
  open-loop execution, where ..."; the "In contrast, allowing the policy to replan ..."
  sentence was removed; "whereas closed-loop stability is not reliably recoverable from
  observation alone" and "and provide a measurable state-dependent characterization of when
  commitment versus replanning changes error propagation" were cut.
* Finding 1: the "Three sources of open-loop states" paragraph, verbatim. The draft notes
  after it (task medians, "G2 keep is 90-99%") are unfinished and were not used.
* Finding 2: the window-dependence paragraph and the front-loaded paragraph, verbatim.
* Finding 3: the first two "Predicting open-loop stability" paragraphs, verbatim. The table is
  the draft's open-loop predictor table with the appendix caption (`tab:pred-ol-app`); the
  main-text caption of the same table describes the dual-axis table (12/14 vs 4/14) and was
  not used. Variant names are the table's: P and A1.
* Captions: "four stamp types" -> "stamp types"; closed-loop rows and the lambda_excess
  sentence removed from the histogram caption. The histogram caption's clause "these are
  shown as a common visual reference and do not replace the window-specific threshold used
  for labeling" was also dropped: the stored labels behind the tables use ln(m)/16 at every
  window, which contradicts it (see the gate note in the delivery report).
* Method: the open-loop parts of "Measuring stability per state", "Open-loop stability",
  "State and action sources", the stamp-type table without its closed-loop row, and
  "Deciding a label" ("applied identically to both regimes" -> "applied").
