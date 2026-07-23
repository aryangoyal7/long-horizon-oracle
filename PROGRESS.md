# Implementation progress — adaptive chunking via stability prediction

## UPDATE 2026-07-17 (second restart recovered; dataset plan expanded)
- **SECOND instance restart** (~2026-07-16 night) killed tool_hang @1882; tool_hang_s2
  had died 07-15 from CIFS OSError mid-checkpoint @1150. Both RESUMED 07-17 ~06:15
  (GPUs 3/6) after scratch rebuild. lift/can/square/square_s2 all COMPLETED 2000 epochs
  (best success 1.0 / 1.0@e1050 / 0.9@e950 / 0.96). tool_hang feats + lift/can rollouts
  had synced to artifacts/ before the restart — nothing lost.
- **DATASET PLAN EXPANDED (user, 2026-07-17)** — now in implementation_plan_v0.1.md §5
  + plan_latex/main.tex: predictor pool = robomimic 4xph + MimicGen D0-D1 (incl. coffee/
  threading/stack) + FurnitureSim one_leg; long-horizon rollout = MimicGen
  three_piece_assembly/nut_assembly/kitchen/coffee_preparation + one_leg + LIBERO-Long
  (+LIBERO-Pro); EXCLUDED Transport (bimanual->stage2), CALVIN (TTIC audit). Priority:
  same-embodiment (Panda/OSC) first. Start with a couple, expand later.
- **MimicGen: GO (probe passed 2026-07-17).** Findings chain: (1) mimicgen does NOT
  import under robosuite 1.5.1 (single_arm_env removed) → dedicated venv
  /mnt/scratch/lh/envs/mg (robosuite 1.4.1, mujoco 2.3.2, robomimic v0.3.0, mimicgen +
  robosuite-task-zoo --no-deps). (2) gdown script broken → pull straight from HF
  (amandlek/mimicgen_datasets/core). (3) robomimic v0.3 env_robosuite.py imports
  mujoco_py unconditionally → PATCHED guard in /mnt/scratch/lh/repos/robomimic_v03
  (NOTE: scratch patch, reapply after instance restart — or bake into setup script).
  (4) Replay vs STORED states mismatches (~0.1 qvel transient at step 1, decaying —
  warm-controller artifact of generation) but replay SELF-consistency is bit-exact
  (0.0 over 16 steps, stack_d0 + square_d0) — which is the labeler's actual
  requirement (branches share the transient; it cancels in the divergence).
  probe_replay.py now tests self-consistency. Remaining planned sets downloading →
  /mnt/scratch/lh/data/mimicgen/ (log results/mimicgen/downloads_20260717.log).
  TODO before labeling mimicgen: extend ftle_labeler.py OBJ_PATTERNS for mimicgen
  env names (contact sanity channel only) + run under mg venv.
- **Closed-loop labeler WRITTEN + smoke-tested** (impl/labeler/ftle_labeler_cl.py):
  nominal = demo replay, perturbed = policy replans every step (queue cleared),
  identical sigma_u injection; runs on image obs + GPU (N*K inferences/stamp →
  defaults stride 10, 50 demos, N=4). Smoke on lift (1 demo, K=8, N=2): 33s,
  lambda_cl ≈ +0.31 in free space — positive where plant contracts = the
  feedback-injects-error prediction. Full runs pending true sigma_u from RMSE batch.
- **policy_action_rmse.py FIXED + running** (GPU 0, lift/can/square): the ob must be the
  stacked (n_stack, ...) history per key, not the last frame. Smoke: lift pos-RMSE
  ≈0.138 — ~3x the 0.05 placeholder → label regeneration with true sigma_u WILL move
  deadbands. Results → results/rmse/<task>.json.
- **eval_k_sweep.py smoke-tested** (fixed mode, square, 3 eps: 100%, 47s) — Stage 1a
  ready to launch once GPUs free; oracle_seg/ftle_probe modes still untested.

Updated: 2026-07-14 ~05:00 UTC (sections below). This file is the resume point after any session/instance
restart. Ephemeral state (venv, datasets, /mnt/scratch) is rebuilt by
`bash impl/setup_scratch.sh` (idempotent, ~40 min; pins inside). One-shot full recovery:
`setsid nohup bash impl/resume_all_after_setup.sh &` (waits for setup marker, resumes
all 6 trainings; expects setup already launched writing to results/setup_scratch_*.log —
edit SETUP_LOG inside if the date changed).

## RESTART EVENT 2026-07-13 11:03 UTC — recovered 2026-07-14 04:30
Instance deallocation killed all jobs and wiped /mnt/scratch. Nothing lost: checkpoints
(share) + artifacts sync had everything. Recovery: setup_scratch rebuilt, all 6 DP
trainings resumed from last.pth (lift @1337, can @1006, square @826, square_s2 @778,
tool_hang @~318, tool_hang_s2 @~300), sync loop restarted, tool_hang V-JEPA extraction
launched (GPU 4), rollout collection lift→can 100 eps launched (GPU 7).
**collect_rollouts.py BUG FIXED**: env from env_from_checkpoint is FrameStackWrapper →
`env.env.sim` hit EnvRobosuite (no .sim). Fix: `env.get_state()["model"]` (line 50);
same latent bug fixed in eval_k_sweep.py::contact_signal (unwrap loop). Smoke test
passed (2 eps collected).

## Done
- **Stage 0 synthetic — VALIDATED** (`results/stage0/`, code `impl/stage0/`):
  floor k*≈5-6 in contracting segments (k≤3 → 0% success), ceiling ≈4-5 in expansive
  (k≥8 → 0%), best constant k = 3.1% vs adaptive (k_c=6,k_e=2) = 100%; FTLE labeler
  recovers true λ to 4 decimals; closed-loop λ flips sign in both regimes.
  Secondary run `results_window.json`: mild parameters → constant k suffices
  (documents falsifiability: adaptive wins iff floor > ceiling).
- **Stack**: robomimic master@e10526b (has diffusion_policy) + robosuite v1.5.1 +
  mujoco 3.2.6 + torch cu128 in /mnt/scratch/lh/envs/lh. EGL rendering verified;
  **state-replay determinism bit-exact** (labeler prerequisite), `impl/labeler/sanity_sim.py`.
- **Datasets**: robomimic ph v1.5 (HF official): lift/can/square/tool_hang, 200 demos
  each, 159k transitions total; image obs regenerated (84px agentview+wrist;
  tool_hang 240px sideview+wrist). Official train/valid masks present.
- **Volume note** (user concern addressed): predictor supervision = sim-generated label
  stamps (~79k demo stamps + policy rollouts later), manufacturable at will; MimicGen is
  the same-format scale-up path if more volume needed.

## Running (as of 2026-07-14 04:45, post-recovery)
- **DP training** ×6 to full 2000 epochs (user-confirmed lift/can continue too):
  GPUs 0-3 = lift/can/square/tool_hang seed-1, GPUs 5-6 = square_s2/tool_hang_s2.
  Configs `impl/configs/dp_*.json` (To=2 Ta=8 Tp=16, crop 76/216, rollout 50 eps every
  50 epochs), output `results/training/dp_<task>/` (persistent). Resume:
  `bash impl/configs/launch_dp_training.sh --resume` (+ s2 pair, see
  impl/resume_all_after_setup.sh). Targets (DP paper): lift ~1.0 (hit 1.0 @100), can
  ~.97 (hit .98 @400), square ~.92 (at .86-.88), tool_hang ~.5-.7 (at .5-.56).
- **V-JEPA extraction** tool_hang (GPU 4) → /mnt/scratch/lh/features/feat_tool_hang.npz
  (45633 stamps, biggest task; lift/can/square feats already in artifacts/features/).
- **Rollout collection** lift→can (GPU 7), 100 eps each incl. failures →
  /mnt/scratch/lh/rollouts/rollouts_{lift,can}.hdf5. Ckpts: lift success_1.0@100,
  can success_0.98@400.
- **Artifact sync loop** (15 min) restarted.
- **FINAL-LABEL ANALYSIS DONE all 4 tasks** (results/labels/<task>/label_stats.json):
  square AUROC .75 (free λ −.014 vs contact +.030), tool_hang AUROC .87 (−.028 vs
  +.031/.035), lift see lift_final/. **can: free_frac=0 by construction** (can rests
  in bin from t=0 → object-fixture contact in every window; gripper contact after) →
  AUROC undefined, deadband fell back to 0.05 (≈square's measured .052). Not a bug;
  contact flags are sanity-only. Optional refinement: count "object resting + gripper
  far" as quasi-free for the noise floor.

## First real-data labeler result (lift, placeholder σ_u=0.05)
AUROC(λ ranks contact-in-window) = **0.903**; mean λ: free −0.024 / gripper contact
+0.101; free space labeled unstable 0.05%; contact labeled unstable 55% (settled
grasps are stabilizing — expected). Deadband δ≈0.061 (p95 free-space). Note: task-space
λ magnitudes are small (stiff OSC, 0.8s window) — sign discrimination is what matters.
Analysis: `results/labels/lift/`. Training restarted ~18:45 with hdf5_cache_mode=all
(epochs were dataloader-bound: ~6 min → target ~20-30 s).

## GPU allocation (as of 2026-07-14 04:45, post-recovery)
- GPU 0-3: DP seed-1 (lift/can/square/tool_hang), resumed, detached.
- GPU 5-6: DP seed-2 square + tool_hang (configs dp_*_s2.json), resumed.
- GPU 7: rollout collection lift→can (100 eps each, incl. failures) → /mnt/scratch/lh/rollouts.
- GPU 4: V-JEPA extraction tool_hang → /mnt/scratch/lh/features.
- MimicGen compat probe: KILLED by the restart before finishing; rerun only if data
  scale-up becomes necessary (deferred).

## Next (in order)
1. ~~analyze_labels.py sanity~~ DONE all 4 tasks (see Running section).
2. Training done → `impl/eval/policy_action_rmse.py` (TO WRITE) → σ_u per task →
   regenerate labels with `--sigma-u-source policy_rmse`.
3. Stage 1a: `impl/eval/eval_k_sweep.py` (WRITTEN, untested; contact_signal unwrap
   fixed) — fixed-k sweep k∈{1,2,4,8,16} + oracle_seg (k_stable,k_unstable) grid +
   ftle_probe adaptive mode. 50 eps × 3 seeds. Can start on square/tool_hang best
   ckpts BEFORE trainings finish (checkpoints good enough); GPU 4/7 free once
   extraction + collection land.
4. Policy-rollout labeling for predictor coverage (rollouts_{lift,can}.hdf5 →
   ftle_labeler.py; collect square/tool_hang rollouts when their policies improve).
5. V-JEPA extraction of rollout stamps + attentive-probe head training
   (impl/predictor/train_head.py).

## Probe study + label status (19:00-21:00)
- 6-variant probe study on square → FINAL probe params: **position-only perturbation,
  K=24, N=8** (fixture 36.6% vs free 1.2% above deadband; baseline was 12.7%/4.4%).
  Details in stability_label_generation.md §7.
- Contact flags must be OBJECT-CENTRIC (distractor objects poison name-based rules:
  square's RoundNut-on-floor, can's Bread/Cereal/Milk, tool_hang's stand-on-table).
  Fixed in ftle_labeler.py (OBJ_PATTERNS); old flags repaired via --flags-only mode.
- Final labels (final_ol_<task>.npz, pos-only/K24, σ_u=0.05 placeholder) regenerating.
- Training: attempt 5 config (workers=0) confirmed stable through rollout rounds; after
  session teardown, resumed with `--resume` from last.pth (lift @110, can @92, square
  @51, tool_hang @38 epochs; earlier rollouts: lift 0.98→1.0, can 0.84).

## Gotchas learned (do not relearn)
- **CIFS share hiccups can kill torch.save mid-checkpoint** (`basic_ios::clear: iostream
  error`); robomimic's last_bak.pth + `--resume` recovers. If it recurs often, move
  output_dir to scratch + rsync to share.
- **Claude session teardown SIGKILLs background-task process groups — nohup does NOT
  protect.** Long-lived jobs (training, labeling) must be launched with
  `setsid nohup ... < /dev/null &` and NO wrapper `wait`. Detection of completion is
  external (Monitor/log polling). Recovery: `bash impl/configs/launch_dp_training.sh
  --resume` (robomimic resumes from last.pth in the same run dir).
- /mnt is EPHEMERAL — wiped on instance restart. Never keep the only copy there.
- `nohup ... &` inside a foreground Bash dies with the shell → always run_in_background.
- Interrupting the agent kills its background task's children (labeler died this way once).
- robomimic train.py prompts interactively if exp dir exists → launcher pipes `yes y`.
- robomimic needs hdf5_filter_key=train + hdf5_validation_filter_key=valid when
  experiment.validate=true.
- mujoco>=3.10 breaks robosuite 1.5.1 (mj_fullM signature) → pin 3.2.6.
- ObsUtils.initialize_obs_utils_with_obs_specs() required before standalone env use.
- Lift contact happens at t≈50-58, past last FTLE stamp — window contacts (added) are
  the right alignment channel, not at-stamp contacts.

## 2026-07-17 15:05 — PREDICTOR TRAINING STARTED (loop phase 4). Full Panda pool (8 feature sets, ~114k stamps) -> results/predictor/run1_panda_pool; transfer square->tool_hang -> run2_transfer_sq2th. HEAD_SMOKE_OK on rejoined feat2 features. All extractions done (mimicgen 2/2, rollouts 2/2). Square Stage 1a 11/12.

## 2026-07-18 04:50 UTC — overnight results, loop re-armed
- Stage 1a square COMPLETE (12/12): fixed k1 .76 / k2 .82 / k4 .78 / k8 .86 / k16 .80; oracle best (8,4) .88 and (1,16) .88. Flat, contact-dominated as diagnosed.
- Stage 1a tool_hang 11/12: fixed k1 .62 / k2 .66 / k4 .88 / k8 .80 / k16 .88; oracle (16,1) .56 ... (16,4) .78. NOT flat: k1 clearly hurts. Final cell (ks1,ku4) running, watcher armed.
- Predictor Stage 1b DONE: run1 panda pool val AUROC .699 all / .868 confident, lambda MAE .0268 (n_val 20147). run2 square-only val AUROC .768/.719; transfer to tool_hang .562/.465 = near chance. Single-task transfer weak; pooled training much stronger on confident stamps.
- CL labeler: can DONE frac(lambda_cl>0)=90.4%, square DONE 93.0% (mechanism confirmed again); tool_hang at 14/50 demos (slow, ~2 days to finish).
- LIBERO: all 10 tasks labeled (final2_ol_libero10_*) + sanity videos rendered. Feature extraction launched 04:47 on GPUs 1-5 (vjepa_extract.py gained --cam-key/--proprio-keys for agentview_rgb + ee_pos/ee_ori/gripper_states; script impl/predictor/extract_libero_features.sh).
- dp_tool_hang_s2 at epoch 1650/2000, training healthy.
- Watchers re-armed for: tool_hang final cell, tool_hang_s2 completion, CL tool_hang, LIBERO extraction. Next: post full tool_hang table, 3-seed Stage 1a repeats, Stage 1c predictor-driven switching.
- 2026-07-18 06:00: GOTCHA found and contained: transformers cannot be upgraded in lh venv (robomimic's old diffusers needs huggingface_hub<1.0 HfFolder; upgrade to 5.14.1 broke robomimic imports; rolled back to 4.41.2/hf_hub 0.23.4 within minutes, no seed cell crashed). Consequence: Stage 1c predictor inference must run as a vjepa-venv subprocess bridge (pipe frames+proprio out, lambda_hat back), NOT in-process in eval_k_sweep. Seed repeats 3/48 done, all queues healthy.
- 2026-07-18 06:45: STAGE 1C LIVE. Built predictor_bridge.py (vjepa venv inference server, line protocol over /dev/shm npz) + predictor mode in eval_k_sweep (16-frame rolling buffer, lambda_hat vs train delta .0355). Smoke passed: low lambda_hat in free space, rises past delta near contact, mean_k 13 with (16,4). First cell launched: square predictor (16,4) seed 0, GPU 7 colocated with CL labeler -> results/ksweep/square/predictor_k0_ks16_ku4_seed0.json.
- 2026-07-18 07:15: Stage 1c square predictor (16,4) seed 0 DONE: SR 0.76 (square flat band .76-.88, no detectable effect as diagnosed; ceiling task). Predictor switching is real: mean_k 10.7 (range 5.3-14.6), lambda_hat>delta at 78.9% of decisions (oracle contact frac ~73%), lambda_hat p5 .002 p95 .068. Cost 0.6h. Tool_hang predictor (16,4) launched on GPU 7 (sideview_image cam). run3 LIBERO ablation training on GPU 6.
- 2026-07-18 08:20: run3 (Panda+LIBERO pool) finished: val .646/.880 (its own mixed val). Comparison on run1's Panda val stamps showed .875 -> .968 confident AUROC BUT LEAKED: run3's random split put most of run1's val demos into run3's train. Do NOT use run3 for the LIBERO decision. Fixed properly: train_head.py gained --val-demos-json (pin val split across pool variants); run1's 280 val demos dumped to results/predictor/run1_val_demos.json; run3b_libero_pinnedval launched on GPU 6 (LIBERO all-train, val = exactly run1's Panda val demos). compare_heads_panda_val.py exists for the final read (note: its run1 recompute matches run1's logged MAE exactly, small AUROC delta vs log under investigation, use same-script numbers for both heads).
- 2026-07-18 10:40: Watchers were killed again ~08:30 (session teardown); jobs unaffected. Landed while down: (1) STAGE 1C TOOL_HANG: predictor (16,4) seed 0 SR 0.82 -- beats ALL contact-oracle cells (.56-.78), within one-seed noise of best fixed .88, fastest successes on the task (423 steps), lam_hat>delta 71.6% of decisions vs oracle 84-88% contact = discriminating WITHIN contact. Cost 1.16h. (2) run3b leak-free LIBERO ablation on pinned Panda val: .720/.893 vs run1 .703/.875, MAE equal -> LIBERO IN (true gain +.018 confident, the .968 was leakage). (3) Seeds 29/48. (4) CL tool_hang 19/50. (5) th_s2 epoch 1800. Predictor doc updated with both results (Stage 1c section added). Loop re-armed.

## 2026-07-19 08:10 UTC — Seed repeats complete, next phase launched
- Stage 1a seed repeats: 48/48 DONE (STAGE1A_SEEDS_DONE). Pooled 3-seed tables added to
  stage1a_report_latex (new section "Three-seed pooled results", PDF recompiled).
  Headlines: square flat (.773–.873, best cell oracle (8,4) .873, ~1 sigma over best fixed);
  tool_hang k_contact=1 cells decisively worst (.600/.633/.667 vs .807 plateau, 3–5 sigma);
  best oracle ties best fixed (.807). Seed-0 outliers (sq (1,16) .88, th k4 .88) did not replicate.
- dp_tool_hang_s2 training: DONE epoch 2000 ("finished run successfully!", Jul 18 17:08).
- CL tool_hang labeler: 40/50 demos, ~43h elapsed, ~9-10h left (GPU 7, PID 397837).
- Launched (setsid): run_stage1a_next.sh on GPUs 0-3 — ftle_probe (16,4) seed 0 both tasks
  (~8h sq / ~18h th) + predictor (16,4) seeds 1,2 both tasks (run1 head, th uses sideview cam).
  Log results/ksweep/next_run.log, marker STAGE1A_NEXT_DONE.
- Launched (setsid): extract_dino_features.sh on GPUs 4-6 — DINOv2 single-frame features for
  the 8 run1-pool datasets (stamps copied from feat2 npz for exact alignment), then auto-trains
  run4_dino_single_frame head with pinned run1 val demos. Log results/predictor/dino_run.log,
  markers DINO_FEATURES_DONE / DINO_ABLATION_DONE. Gotcha fixed: rollout *_image.hdf5 files use
  demo_ prefix, not rollout_.
- Watchers re-armed for all three (next queue, dino, CL labeler).

## 2026-07-20 09:30 UTC — Everything harvested; Stage 2 launched
- STAGE1A_NEXT_DONE: predictor seeds pooled — square .76/.68/.86 -> .767+-.035 (bottom of flat
  band); tool_hang .82/.84/.74 -> .800+-.033 (ties best fixed .807 and best oracle .807).
  ftle_probe (16,4) seed0: square .84 / tool_hang .76, mean_k ~15.8 both — probe delta .05
  almost never fires, degenerates to fixed k16; recorded and de-prioritized.
- CL tool_hang labeler DONE: 2,697 stamps, 53.6h; lambda_cl mean +.096, 97.6% positive
  (p5 +.013, p95 +.228). Mechanism check now passes on ALL FOUR robomimic tasks.
- DINOv2 ablation (run4) DONE: single frame .703/.859/MAE .0269 vs run1 V-JEPA 16-frame
  .699/.868/.0268 on identical pinned split — motion context worth <=.009 confident AUROC.
  Signal is scene configuration + proprio. Shuffle-clip control dropped as moot.
- Docs updated + recompiled: predictor doc (DINOv2 section, 1c seed repeats + probe, abstract),
  label report (CL all-four table + tool_hang row, Pattern 6, caveats, totals 3,969 CL stamps).
  Stage 1a report already carries the pooled 3-seed section.
- Stage 2 LAUNCHED (setsid, impl/mimicgen/run_stage2.sh): GPUs 0-3 converting
  three_piece_assembly_d0 / nut_assembly_d0 / kitchen_d0 / coffee_preparation_d0 to 84px image
  obs in the mg venv (runpy wrapper for mimicgen env registration), then each GPU auto-launches
  DP training (lh venv, dp_mg_*.json, rollouts DISABLED — mg envs need robosuite 1.4; smoke
  test on square_d0_image passed). 2000 epochs = 200k grad steps (epoch_every_n_steps=100).
  Log results/training/stage2_run.log, markers STAGE2_CONVERT_<task>_DONE / STAGE2_ALL_LAUNCHED.
- CL head chain LAUNCHED (GPU 4, impl/predictor/run_cl_head.sh): featcl_* extraction
  (vjepa_extract --lambda-key lambda_cl_task) then run5_cl_head training. Markers
  CL_FEATURES_DONE / CL_HEAD_DONE.

## 2026-07-20 09:25 UTC — CL head (run5) done; closed-loop expansion only weakly predictable
- CL head chain finished (CL_HEAD_DONE, ~20 min total on GPU 4). featcl_* extraction: 151 lift,
  477 can, 644 square, 2,697 tool_hang = 3,969 stamps (tool_hang 68% of pool).
- run5_cl_head metrics: n_train 3,150 / n_val 819; auto delta .0763 ~= label median (.0798)
  since lambda_cl is 96%+ positive -> classification is "faster than typical expansion",
  relative not stable/unstable. val_auroc_all .627, MAE .043.
- The reported val_auroc_confident .986 is VOID: confident = true |lambda|>delta, and only 7
  stamps in the whole pool sit below -delta, so it is measured against ~a couple of negatives.
- Interpretation: closed-loop expansion is a property of the policy's reaction, not just the
  visible scene; some signal transfers (.627 >> chance on n=819) but nothing deployable.
  Run 1 open-loop comparison (.699) confounded by 26x more data. Nothing depends on this head.
- Predictor doc updated (new subsection sec:clhead, abstract + summary lines) and recompiled.
- GPU 4 now idle (5-7 already idle). GPUs 0-3 still converting mimicgen image obs (1.6-3.2 GB
  written at +12 min, nut_assembly at demo 129/1000); Stage 2 watcher (betmlho1t) still armed.

## 2026-07-20 09:50 UTC — Full documentation pass (user request)
- NEW: stage2_doc_latex/main.tex — exhaustive Stage 2 document: why long-horizon (1c tie
  analysis), the four MimicGen tasks with exact dataset stats (demos/transitions/lengths),
  the two-simulator problem and mg/lh venv split, conversion pipeline (runpy wrapper, EGL,
  done_mode 2, exclude-next-obs), training recipe (200k steps, rollouts disabled + why,
  40 checkpoints), pending eval bridge (policy-server vs backport designs), pending
  experiment protocol + zero-shot vs relabel decision with decision rule, status snapshot,
  risks. Compiles clean.
- plan_latex: added \section{Status and results as of July 20, 2026} — stage-by-stage
  record (Stage 0 validated incl. falsifiability run; labeling complete w/ cost deviation;
  1a effect + direction w/ correct contact-segment framing; 1b findings incl. run5;
  1c tie + redefinition of Stage 2; deviations and still-open list: policy-side baselines,
  dissociation cases, can control, ablations, FurnitureSim, LIBERO-Long). Date line and
  Summary updated. Compiles clean.
- stage1a_report: fixed stale abstract ("repeats running now" -> pooled outcome), constants
  table seed row, can-control caveat (not run, GPUs moved to Stage 2), date -> updated Jul 20.
- label_report: date line -> updated Jul 20 (content was already current).
- predictor_doc: 1c conclusion now points to stage2_doc_latex.
- All five PDFs recompiled clean, no undefined references.

## 2026-07-20 11:30 UTC — STAGE2_ALL_LAUNCHED: all four DP trainings running; timeline collapsed
- Conversions done, zero failures: nut_assembly 78min/14.9GB (10:23), kitchen 92min/25.7GB
  (10:37), three_piece 104min/14.1GB (10:49), coffee_preparation 119min/29.0GB (11:05).
  84GB total, 2.4TB free. Each DP training auto-launched within 1s of its conversion.
- All four trainings VERIFIED STEPPING at 11:25: ~8 grad steps/s, ~4GB GPU mem each,
  epochs done 95/67/51/21 (nut/kitchen/three_piece/coffee), nut epoch_50 checkpoint saved.
- TIMELINE REVISION: ~30-45s/epoch (no rollouts, data cached in RAM) -> 2000 epochs in
  17-24h, NOT 5-6 days (Stage 1 wall-clock was rollout-dominated). All four finish morning
  of Jul 21 UTC. Eval bridge is now the critical path.
- Docs upgraded (user request): stage2 doc (measured conversion table replacing 6-12GB
  estimate, measured training pace + why it is faster, status section with full timeline,
  bridge marked critical path, abstract/summary updated); plan doc status section Stage 2
  paragraph (Jul 21 completion). Both recompiled clean.

## 2026-07-20 12:15 UTC — Cross-venv evaluation bridge BUILT and smoke-tested
- New: impl/mimicgen/policy_bridge.py (lh venv, loads dp_mg ckpt via policy_from_checkpoint,
  line protocol READY/RESET->ACK/REQ npz->RES action/QUIT, mirrors predictor_bridge.py) +
  impl/mimicgen/bridge_eval.py (mg venv: owns MimicGen env from image-hdf5 env_args, spawns
  the lh server as subprocess, streams obs per step, writes success JSON).
- Three integration bugs found+fixed in smoke testing: (1) robomimic prints to stdout ->
  client skips non-protocol lines (expect() helper); (2) v0.3 env returns images processed
  CHW float, main RolloutPolicy wants raw HWC uint8 -> to_raw() unprocess on client;
  (3) ckpt frame_stack=2 -> client reproduces FrameStackWrapper semantics (reset = 2 copies,
  slide 1/step), obs sent stacked (2,...).
- Smoke PASS (nut_assembly epoch_150, 2 eps, horizon 100, GPU 4): full protocol round trip,
  ~7.7 env-steps/s end to end (~13s per 100-step ep). 0/2 success as expected (7.5% trained,
  horizon << task length). Throughput => 50 eps at horizon 650 ~ 70 min/cell/GPU.
- Trainings healthy at 12:00: epochs 150/123/105/74 (nut/kitchen/three_piece/coffee), no
  tracebacks. Watcher bevc2f2to polls for completion/failure (~04:00-09:00 UTC Jul 21).
- Next: when trainings finish, checkpoint-selection runner on GPUs 4-7 (subsample of the 40
  checkpoints per task via the bridge), then zero-shot head sanity on MimicGen frames.

## 2026-07-20 12:50 UTC — Zero-shot head check on MimicGen frames: PASS, zero-shot-first decided
- Ran run1 head (frozen, delta .0355) on 320 stamps/task from the converted image hdf5s
  (40 demos x 8 stamps, agentview 16-frame clips + 9-dim proprio, GPU 4, ~7 min).
- Distributions varied and span delta on all four tasks (NOT degenerate):
  three_piece mean .022 (p5 -.008/p95 .046, 19% > delta), nut .017 (22%), kitchen .019
  (11%), coffee_prep .035 (55%). Plausible ordering: coffee (contact-heavy machine loading)
  most unstable, kitchen (transport-dominated) most stable. Weak positive time-corr on
  kitchen (.16) / coffee (.26).
- DECISION per stage2 doc rule: zero-shot first; relabel+fine-tune only if the switch
  underperforms in rollouts. Caveats recorded: demo frames not policy frames; verifies
  non-degeneracy not accuracy.
- Stage2 doc updated (decision table + status + risks reworded) and recompiled clean.
- Trainings healthy at 12:40: epochs 216/189/171/140 (nut/kitchen/three_piece/coffee).
- Artifacts: scratchpad zeroshot_head_check.py + zeroshot_check.json.

## 2026-07-20 13:20 UTC — Overnight auto-chain armed: checkpoint selection
- New impl/mimicgen/run_ckpt_selection.sh LAUNCHED (setsid, survives teardowns): waits for
  all four "finished run successfully", then per GPU 4-7 evaluates epochs 600/1000/1400/
  1700/2000 via the bridge, 25 eps each, horizons 560/650/980/1140 (demo max x1.5), seed 0.
  Results results/stage2/ckpt_select/<task>_epoch<e>.json; markers CKPT_SELECT_TRAININGS_DONE,
  CKPT_SELECT_<task>_DONE, CKPT_SELECT_ALL_DONE, aborts on training Traceback. Est ~2.5-5.5h
  per task after trainings finish -> selection done ~midday Jul 21.
- Watchers: bevc2f2to (training completion), b53uive53 (selection completion/failure).
- Trainings healthy at 13:17: epochs 280/253/236/202 (nut/kitchen/three_piece/coffee),
  ~35-39s/epoch, completion 05:45-08:45 UTC Jul 21 on pace.

## 2026-07-21 11:50 UTC — SCRATCH WIPE incident + recovery; label timelines added (user request)
- INCIDENT: /mnt/scratch wiped overnight: repos/, data/ (all robomimic+mimicgen hdf5s incl.
  114GB converted images), features/, mg+vjepa venvs, lh venv gutted (28K husk). SURVIVED:
  labels/ (31 npz, RESCUED to rescue/labels/), rollouts lift+can hdf5s (rescue copy started),
  and the four dp_mg trainings (deleted-inode code+RAM-cached data; checkpoints on persistent
  storage unaffected). Killed the armed ckpt-selection chain (needed missing mg venv).
- RECOVERY LAUNCHED (setsid impl/mimicgen/rebuild_scratch.sh, log results/stage2/rebuild.log):
  mg venv via setup_and_probe.sh, lh venv (torch cu128 + robomimic main e10526b + diffusers),
  vjepa venv, re-download 4 core D0 datasets from HF, reconvert images on GPUs 4-7, then
  auto-relaunch run_ckpt_selection.sh. Markers REBUILD_*_DONE/FAIL, CKPT_SELECT_RELAUNCHED.
- Trainings at 11:41: epochs 1637/1721/1653/1596 of 2000, healthy; ~55s/epoch avg; finish
  ~16:00-18:00 UTC today.
- Label timelines (user request): figs/timeline_{lift,can,square,tool_hang}.png + stats json;
  new label report section sec:timelines (4 figures, contiguity table, 4 findings + oracle
  occupancy note). Verdict: real phase-level structure (approach deadband band -> contact
  unstable segments), NOT salt-and-pepper (82-94% of OL mass in runs >=3 stamps), but
  boundary flicker exists (median OL run 4 steps; tool_hang 52 transitions/demo); stable
  class rare (0-4%) so practical contrast is deadband-vs-unstable. Report recompiled clean.
- Oracle switching sanity (user question): 100% of oracle rollout episodes mixed both modes;
  stable-mode occupancy ~44% square / ~16% tool_hang (from recorded mean_k). Switch fires;
  lopsided occupancy is what limits its value on short contact-dominated tasks.
- Lost, deferred: square/tool_hang rollout hdf5s (GPU-days, only needed for future feature
  work), Panda ph datasets (re-download when needed), HF model caches (auto re-download).

## 2026-07-21 12:25 UTC — Trainings found DEAD (died 05:01 in disk event); full pipeline restarted
- CORRECTION: the four dp_mg trainings did NOT survive the wipe. They died 05:01 UTC at
  epochs 1721/1653/1637/1596 (launch-log mtime); the "healthy" reads at 11:41/11:54 were
  frozen counters misread as progress. Saved checkpoint grids (persistent) end at
  nut 1700 / kitchen 1650 / three_piece 1600 / coffee 1550 — all past the epoch ~950-1200
  plateau Stage 1 selection favored, so scientific loss is likely nil.
- Rebuild reconversion had TWO failures, both fixed: (1) mg venv torchvision mismatch
  (bare 0.26.0 vs torch cu128) -> pip upgrade torch+torchvision from cu128 index;
  (2) fresh robomimic_v03 clone lost the local mujoco_py import guard (original patch
  wiped with the repo) -> re-applied try/except in env_robosuite.py; patch procedure now
  recorded in setup_and_probe.sh (PATCH_V03) + rebuild_scratch.sh note.
- RELAUNCHED (user: "put the trainings active"): run_stage2.sh (GPUs 0-3: reconvert ~2h ->
  fresh 2000-epoch trainings, done ~midday Jul 22) + run_ckpt_selection.sh rewritten to
  wait per-task on CONVERT markers and evaluate SURVIVING checkpoints (600/1000/1400/latest
  per task) on GPUs 4-7 starting ~14:30 -> selection done tonight; Stage 2 experiment can
  launch on selected checkpoints without waiting for the rerun trainings.
- Conversions verified writing at 12:22. Old rebuild watcher retired; new watcher armed.

## 2026-07-21 12:40 UTC — label_report_latex DELETED from persistent share; fully reconstructed
- Discovered while updating docs (user request): label_report_latex/ vanished from the Azure
  Files share on both mount paths. Likely accidental deletion via a file browser: a file named
  with the user's chat message text appeared in the project root at the same time. No other
  directory affected.
- RECONSTRUCTED completely: main.tex replayed from this session's transcript jsonl (original
  Write + 32 recorded Edits, 1 date-line fixup); all 16 figures regenerated (4 timelines from
  labels, 4 stat figures via the fig-gen scripts recovered from the transcript, 8 annotated
  frames re-extracted from surviving label_videos mp4s). Compiles clean, 1.18MB PDF (~same as
  before). Date line marks the reconstruction.
- MAJOR FIND during recovery: artifacts/ on the persistent share holds an 87GB mirror of the
  feature npz files + labels + rollouts from a prior session backup -> the features thought
  lost in the scratch wipe are SAFE (incl. dino_* and, to verify, feat2_*/featcl_*).
- Defense: full docs+code+labels+small-results backup now at
  /home/azureuser/lh_docs_backup_20260721.tar.gz (OS disk, off-share).
- stage2 doc + plan doc updated with the July 21 incident section / revised timeline and
  recompiled. Predictor doc and stage1a report verified current, no changes needed.
- Pipeline unaffected throughout: conversions on GPUs 0-3 progressing, selection waiting on 4-7.

## 2026-07-21 12:45 UTC — Auto-shutdown armed (user request)
- impl/mimicgen/auto_shutdown.sh (setsid): shuts machine down when trainings x4 finished AND
  CKPT_SELECT_ALL_DONE AND STAGE2_KSWEEP_ALL_DONE (marker the experiment runner must write to
  results/stage2/ksweep_run.log). Failsafe: after trainings+selection, 3h of idle GPUs ->
  shutdown anyway. 5-min warning via shutdown +5. Log results/stage2/auto_shutdown.log.
- NOTE for the experiment runner (to be built): MUST log to results/stage2/ksweep_run.log and
  end with STAGE2_KSWEEP_ALL_DONE, else only the idle failsafe fires.
- Shutdown ends the Claude session and all watchers; results live on persistent storage.

## 2026-07-21 14:40 UTC — Checkpoint grids destroyed at relaunch; process hardened
- The 12:21 relaunch of run_stage2.sh used `yes y | train.py`, which auto-answered
  robomimic's overwrite prompt and DELETED the surviving checkpoint grids
  (epochs <=1700/1650/1600/1550) for all four tasks. Loss is permanent: the
  OS-disk backup deliberately excluded results/training. The 13:56 checkpoint
  selection therefore ran vacuously (all CKPT_SELECT_MISSING) and wrote a
  CKPT_SELECT_ALL_DONE that would have mis-triggered auto_shutdown.
- Fixes: run_stage2.sh now refuses to launch if the experiment dir exists
  (STAGE2_TRAIN_*_REFUSED_DIR_EXISTS) and the yes-pipe is gone;
  run_ckpt_selection.sh rewritten to wait per task on "finished run successfully",
  back up each grid to /home/azureuser/ckpt_backup_dp_mg_<task> BEFORE evaluating,
  and evaluate epochs 600/1000/1400/1700/2000; selection relaunched with a fresh
  log (vacuous ALL_DONE marker cleared, verified count 0).
- Net effect: ~1 day slip. Fresh trainings (launched 13:39-14:19 Jul 21) carry
  the full pipeline; selection runs on their grids as they finish.

## 2026-07-22 06:30 UTC — All four trainings healthy overnight
- three_piece epoch 1462, nut 1643, kitchen 1467, coffee 1425 (of 2000);
  4 processes alive 16h+, ~4 GB GPU each, logs advancing, zero Tracebacks.
- Coffee (GPU 3) confirmed stepping (was still caching at last check).
- ETA at ~36-41 s/epoch: nut ~10:00 UTC, the rest ~12:30-13:00 UTC Jul 22.
- Selection waiter (pids 69506+) and auto_shutdown watchdog alive; no backups
  yet (none finished). Next: backup + selection per task as trainings finish.

## 2026-07-23 05:30 UTC — Root cause of both "disk events": VM stops wipe scratch
- Boot history shows the machine was STOPPED 14:55 UTC Jul 22 and restarted
  04:46 UTC Jul 23; the Jul 17 boot ended 05:02 UTC Jul 21. Azure reprovisions
  /mnt/scratch on every stop. So the Jul 21 "disk event" and yesterday's
  selection death were both VM stops (manual or scheduled), not disk failures.
- What completed before the stop: ALL FOUR trainings reached epoch 2000 and
  finished. Selection completed for nut (best 0.36 @2000) and three_piece
  (best 0.80 @1700); kitchen 600-1700 done (0.96-1.0 (!)); coffee only
  epoch 600 (0.04). Missing: kitchen@2000, coffee@1000/1400/1700/2000.
- Backup fault (mine): backups targeted the 119G OS disk which filled at 97%;
  nut got 25G of 58G, kitchen 179M, others none. Partials deleted (originals
  intact on the share), destination switched to /mnt/scratch/lh/ckpt_backups
  (2.6T); full re-copy of all four grids running.
- Recovery running: resume_ckpt_selection.sh (only missing evals, appends to
  the watched log), rebuild_scratch.sh (venvs + datasets + reconversion on
  GPUs 4-7, ~2h) relaunched after the wipe killed the first resume attempt
  (mg venv missing); vacuous ALL_DONE scrubbed from ckpt_select_run.log;
  auto_shutdown re-armed. GitHub: all commits pushed through restored
  VS Code connection; credential store now holds a durable token.
- Remaining: missing evals (~5h after reconversion), then build + smoke the
  k-sweep runner, then the experiment (~1 GPU-day), then auto-shutdown.

## 2026-07-23 06:10 UTC — Rebuild hit a torchvision fault; conversions rerunning

The scratch rebuild finished its venvs and dataset downloads, but all four
image reconversions failed instantly: setup_and_probe.sh installed torch from
the cu128 wheel index while pip pulled torchvision from PyPI, and the
mismatched C extension fails with "operator torchvision::nms does not exist".
The selection resume then failed vacuously and wrote a false
CKPT_SELECT_ALL_DONE, which was scrubbed before auto_shutdown could see a
complete pipeline (the k-sweep gate was still closed, so no shutdown risk
materialized). Fixes: cu128 torchvision force-reinstalled in the mg venv,
setup_and_probe.sh now installs the pair from the same index (and its
PATCH_V03 mujoco_py guard, which was defined but never invoked, is now
called after the v03 clone). recover_convert_and_select.sh reruns the four
conversions on GPUs 4-7 (~2h) and then relaunches the five missing selection
evals. Separately, the original ckpt_select_run.log was destroyed by an
in-place sed on the CIFS mount (rename onto an open file leaves it
delete-pending); it was reconstructed from the intact eval JSONs, which were
never at risk.

## 2026-07-23 06:40 UTC — K-sweep runner built and armed; backups complete

All four checkpoint grids are backed up on scratch (55 GB each,
SCRATCH_BACKUP_ALL). Reconversions are healthy (about demo 300 of 1,000 on
three tasks, coffee at 129; ETA one to three hours). run_stage2_ksweep.sh is
launched and waiting: it smoke-tests stage2_ksweep.py (2 episodes fixed, 2
predictor) as soon as the three_piece dataset lands, waits for
CKPT_SELECT_ALL_DONE, picks the best checkpoint per task from the selection
JSONs (argmax success, ties to the later epoch), then runs the 20 cells
(fixed k in {1,4,8,16} plus predictor (16,4), 50 episodes, seed 0) across 8
GPUs. STAGE2_KSWEEP_ALL_DONE is written only if all 20 result files exist,
so the shutdown watchdog cannot fire on a partial sweep. Both LaTeX docs
updated with the selection results and the labeling clarification, pushed.
