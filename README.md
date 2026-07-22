# Long-Horizon Oracle: Segment-Dependent Chunk Length via a Video Stability Predictor

Chunked imitation policies (Diffusion Policy, ACT) commit to a fixed action chunk length k. A large k gives smooth, temporally consistent motion but reacts slowly; a small k reacts fast but jitters. This project tests a simple idea: the right k is not a constant of the task, it is a property of the current segment of the task. Free-space transport tolerates long chunks; contact-rich insertion needs short ones.

We build a per-timestep stability signal, label demonstrations with it, train a video-based predictor of that signal, and use the prediction at rollout time to switch the chunk length online. The end metric is task success rate against the best fixed-k baselines.

## Method in one paragraph

For each demonstration timestep we measure two things by perturbation rollouts in simulation: open-loop stability (does the motion tolerate executing a long chunk blindly from here) and closed-loop stability (does replanning from a perturbed state recover). Both are summarized by a scalar log amplification factor lambda: negative means perturbations shrink (stable), positive means they grow (unstable), and a deadband around zero means neither dominates. A frozen V-JEPA video encoder with a small attentive head is trained to predict lambda from the last 16 frames. At rollout time the policy replans with a long chunk (k = 16) while the predicted lambda is below a threshold delta and drops to a short chunk (k = 4) when it rises above, which is the "oracle switching" the repo name refers to.

## Repository layout

| Path | What it is |
|---|---|
| `adaptive_chunking_hypothesis_v0.4.tex/.pdf` | The hypothesis document: claims, metrics, falsification criteria |
| `implementation_plan_v0.1.md`, `plan_latex/` | The staged implementation plan and its LaTeX version with status updates |
| `stability_label_generation.md`, `label_report_latex/` | How the perturbation labeler works and the full label report, including per-demonstration label timelines |
| `predictor_doc_latex/` | The V-JEPA predictor: architecture, training runs, AUROC results, zero-shot transfer check |
| `stage1a_report_latex/` | Stage 1a report: fixed-k sweeps and the oracle-switching evaluation on robomimic tasks |
| `stage2_doc_latex/` | Stage 2 companion doc: MimicGen long-horizon tasks, training setup, the July 21 disk incident and restart |
| `AAAI/` | Paper draft (AAAI 2027 template) |
| `impl/` | All code: labeler, predictor training, evaluation bridges, stage scripts |
| `impl/labeler/` | Perturbation-rollout labeling of demonstrations (open-loop and closed-loop lambda) |
| `impl/predictor/` | V-JEPA feature extraction and attentive-head training |
| `impl/eval/` | k-sweep evaluation and the predictor bridge used at rollout time |
| `impl/mimicgen/` | Stage 2: dataset conversion, Diffusion Policy training, cross-venv policy bridge, checkpoint selection, k-sweep runner, auto-shutdown |
| `impl/configs/` | robomimic training configs for the four Stage 2 tasks |
| `results/` | Small result files: JSONs, figures, label npz files (heavy outputs are gitignored, see below) |
| `PROGRESS.md` | Chronological log of every phase, incident, and decision |
| `JOURNAL.md` | Working notes |

## Pipeline stages

1. **Stage 0, signal sanity.** Verify the perturbation labeler recovers known amplification factors on synthetic dynamics. Result: recovery to 4 decimal places.
2. **Stage 1, labels and predictor.** Label robomimic demonstrations (lift, can, square, tool_hang) with open-loop and closed-loop lambda. Train the attentive head on frozen V-JEPA features. Evaluate oracle switching (ground-truth labels driving k) against the best fixed k.
3. **Stage 2, long-horizon tasks.** Train Diffusion Policy on four MimicGen D0 tasks (three_piece_assembly, nut_assembly, kitchen, coffee_preparation, 1000 demos each, mean episode lengths 337 to 689 steps). Run fixed k in {1, 4, 8, 16} against predictor-switched (16, 4) at 50 episodes per cell. The predictor is applied zero-shot (trained on robomimic frames, evaluated on MimicGen frames) after a transfer sanity check passed.

## Key results so far

- Labels are phase-structured, not salt-and-pepper: 82 to 94 percent of stamp mass sits in runs of 3 or more consecutive stamps, median open-loop run 4 to 8 steps. Per-demonstration color-coded timelines are in the label report.
- Open-loop occupancy is deadband-dominant (58 to 75 percent), with 20 to 40 percent unstable and almost no confidently stable steps. This matches the physics of the tasks and the threshold construction; it is a property of the signal, not a labeling bug.
- Closed-loop stability head reaches AUROC 0.627 on held-out demonstrations (run 5).
- Oracle switching fires in 100 percent of oracle episodes, with stable-mode occupancy about 44 percent on square and 16 percent on tool_hang, so the online experiment has room to show a difference.
- Zero-shot transfer check on MimicGen frames passed: predicted lambda distributions land in a plausible range with a plausible task ordering.

## What is not in this repo

`results/training/` (about 515 GB of checkpoints and logs), `results/predictor/` features, rollout videos, `artifacts/` (88 GB V-JEPA feature mirror), and `rescue/` (recovered rollout hdf5 files) are gitignored for size. Datasets come from the robomimic and MimicGen public releases (`https://huggingface.co/datasets/amandlek/mimicgen_datasets`).

## Reproducing

The pipeline needs two incompatible simulator stacks, so it runs from three virtualenvs on scratch: `mg` (robosuite 1.4.1, robomimic v0.3, mimicgen) for MimicGen environments and dataset conversion, `lh` (robosuite 1.5.1, robomimic main, diffusers) for Diffusion Policy training, and `vjepa` (torch, transformers) for the predictor. `impl/mimicgen/rebuild_scratch.sh` rebuilds all of it from scratch, including dataset download and image conversion. Policy rollouts across the venv boundary go through a line-protocol bridge (`impl/mimicgen/policy_bridge.py` serving, `impl/mimicgen/bridge_eval.py` and `impl/mimicgen/stage2_ksweep.py` as clients).

## Status (July 22, 2026)

Stage 0 and Stage 1 are complete and written up. Stage 2 Diffusion Policy trainings for the four MimicGen tasks are at roughly epoch 1400 to 1650 of 2000 and finish today; checkpoint selection and the final k-sweep experiment run next. `PROGRESS.md` has the full history, including the July 21 disk incident and recovery.
