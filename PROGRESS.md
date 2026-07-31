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

## 2026-07-23 12:15 UTC - Reboot #3 recovered (root cause fixed); encoder ablation launched

Machine restarted ~11:21 UTC, wiping /mnt/scratch for the third time and killing
the three running coffee checkpoint-selection evals. Root cause of the *silent*
recovery failures found and fixed: `@reboot` runs with a minimal PATH, so
`python3` resolved to /usr/bin/python3, which has no `ensurepip` (python3.10-venv
is not installed on this image). Both of today's automatic recoveries had failed
at venv creation for this reason while manual recovery always worked, because
interactively `python3` is the conda interpreter. All five venv creation sites in
impl/setup_scratch.sh, impl/mimicgen/rebuild_scratch.sh and
impl/mimicgen/setup_and_probe.sh now resolve the interpreter explicitly.
Rebuild relaunched and is past its old failure point (lh, vjepa, mg venvs and
datasets all rebuilt).

New experiment: 39-cell encoder ablation and transfer grid
(impl/predictor/run_encoder_ablation.sh -> results/predictor/ablation/), all
pinned to run 1's pool, validation demos and delta. Grid A answers whether the
V-JEPA/DINOv2 tie survives seed noise and decomposes the signal into vision-only,
proprio-only and both, plus 100-epoch cells. Grid B is transfer: square ->
tool_hang for both encoders plus a proprio control, and the previously untested
leave-one-task-out (pool minus tool_hang -> tool_hang).

train_head.py gained --no-proprio (vision-only), Spearman rank correlation, and
holdout scoring against a delta refit on the holdout set, so a ranking failure
can be told apart from a calibration failure.

First results: transfer failure is a REPRESENTATION failure, not calibration.
Square -> tool_hang Spearman drops 0.574 -> 0.053, and refitting delta on
tool_hang changes nothing (its own delta is 0.0351 vs the training 0.0355).
Recalibration cannot recover transfer.
DINOv2 vision-only (no proprio at all) reaches 0.844 confident AUROC against
0.863 with proprio, so the visual features carry most of the signal.

## 2026-07-23 12:45 UTC - Proprioception beats both visual encoders; transfer fails even leave-one-task-out

Ablation grid at 10/39 cells. Two results change the picture.

1. PROPRIO-ONLY IS THE BEST CONDITION. All same code path, pinned pool, pinned
   validation demos, pinned delta, seed 0, AUROC on confident stamps:
     proprio only (9 numbers, 21,954-param head)   0.736 all / 0.895 confident
     V-JEPA 2 16-frame + proprio (4M-param head)   0.726 / 0.883
     V-JEPA 2 16-frame, vision only                0.712 / 0.871
     DINOv2 single frame + proprio                 0.699 / 0.863
     DINOv2 single frame, vision only              0.688 / 0.844
   Every addition of visual information makes it slightly worse. The proprio-only
   head was verified vision-free: its state dict has only prop.* and mlp.* keys,
   no query/attn/norm. Seed spread on a repeated condition (A_vj_vis s0 vs s1,
   0.871 vs 0.878) is about 0.007, so the ordering is near single-seed
   resolution; the robust claim is that vision adds nothing, not that proprio
   strictly wins. Remaining seeds pending.

2. TRANSFER FAILS EVEN LEAVE-ONE-TASK-OUT. Confident AUROC on tool_hang:
     square-only head, V-JEPA          0.493
     square-only head, DINOv2          0.358
     square-only head, proprio only    0.353
     pool-minus-tool_hang, DINOv2      0.351   <- 7 training families, still chance
   The hope that a pooled head would generalize where a single-task head could
   not is not supported. DINOv2 transfers worse than V-JEPA, as expected for an
   appearance-heavy encoder.

3. SHORTCUT CONTROLS PASSED (impl/predictor/shortcut_controls.py). The head is
   not a contact detector: within gripper-contact stamps only, where the contact
   cue is constant and episode progress is at chance (0.514), it still scores
   0.827. It is not a clock either: AUROC stratified within episode-progress
   deciles is 0.831 against 0.863 unstratified. Trivial baselines are far below:
   progress 0.652, fixture flag 0.640, gripper flag 0.599. Per-task the head
   ranges 0.600 (can) to 0.884 (tool_hang), so the pooled 0.86 is partly a
   between-task effect and should be quoted as a range.

Stage 2: rebuild past venvs and datasets, 4 image conversions still running.
The k-sweep runner had died in the reboot and was never re-armed; relaunched.
Its dataset gate was also wrong (it accepted an existing _image.hdf5 plus a
RECONVERT marker in a STALE rebuild log, so it opened mid-write and the smoke
died on h5py's exclusive lock). Gate now opens the file and checks the demo
count. Two spurious KSWEEP_SMOKE_FAIL lines in ksweep_run.log are annotated.

Blog post drafted at aryangoyal7.github.io/_pages/blog/vision-or-proprioception.md
(not pushed, awaiting confirmation).

## 2026-07-24 08:40 UTC, restart #4 and two structural recovery gaps closed

Fourth Azure reprovision of /mnt/scratch (08:05 UTC). The BASEPY fix from
yesterday held: recovery got past venv creation for the first time and was
downloading robomimic data on its own. Two further gaps were found, both of
which had been silently papered over by me relaunching things by hand.

Gap 1: boot_recover.sh rsynced labels/ and rollouts/ back to scratch but not
features/. It created an empty features/ directory. The encoder ablation reads
its 38 GB of .npz features from /mnt/scratch/lh/features, so after every reboot
it had nothing to train on. This is what stranded the grid at 14/39 cells from
12:56 UTC yesterday until now. Fixed by adding the features rsync.

Gap 2: nothing launched rebuild_scratch.sh at boot at all. /mnt/scratch/mg did
not exist and no Stage 2 process was running. Every reboot killed Stage 2 dead
until relaunched manually. Fixed by launching it from boot_recover.sh, and by
arming the k-sweep runner there too. To make that safe, rebuild_scratch.sh
steps 2 and 3 are now idempotent: they previously ran venv --clear on the lh
venv, which would have deleted it out from under the setup_scratch.sh run that
boot_recover.sh had just used to resume trainings.

Also pinned the ablation interpreter. run_encoder_ablation.sh used PY=python3,
which under the @reboot PATH is /usr/bin/python3 and has no torch, so the
boot-time resume would have failed every cell silently. It now resolves the
conda python first, which is what trained the first 14 cells (torch 2.9.1);
preferring a venv would have changed torch version mid-grid.

Regression check resolved without needing the repro run. A_vj_full_s0 and run 1
have identical splits (n_train 82212, n_val 20147, frac_unstable 0.351616, same
delta) and identical weight init. The only difference is that run 1 consumed
rng.choice for the random split before the training loop's rng.shuffle, so the
minibatch order differs. 0.699 -> 0.726 all, 0.868 -> 0.883 confident, is
shuffle-order noise. The train_head.py rewrite is neutral.

That also revises the noise estimate. Same cell, same data, only shuffle order
changed, moves confident AUROC by 0.015. The proprio-over-VJEPA margin is 0.012.
The ordering in the encoder table is not resolved by one seed. What survives is
that vision adds nothing, not that proprioception wins. The blog post quotes
0.007 and needs updating once the remaining seeds land.

Transfer results from the 7 B-cells that finished are worse than previously
recorded and cleaner. Refitting delta on the holdout does not rescue any of
them (0.351 -> 0.352), so this is a ranking failure and not a calibration
failure. Spearman between predicted and true lambda on the unseen task runs
from -0.119 to +0.053, i.e. no rank correlation at all.

  cell              xfer_conf  xfer_own_delta  spearman
  B_loto_di_s0          0.351       0.352      -0.004
  B_loto_prop_s0        0.309       0.308       0.025
  B_loto_vj_s0          0.664       0.665      -0.007
  B_sq2th_di_s0         0.358       0.356       0.001
  B_sq2th_prop_s0       0.353       0.352      -0.086
  B_sq2th_vj_s0         0.493       0.492       0.053
  B_sq2th_vj_s1         0.332       0.331      -0.119

B_loto_vj_s0 at 0.664 is the one cell above chance, but B_sq2th_vj moves 0.493
to 0.332 across two seeds, so seed spread on transfer is large and 0.664 should
not be read as a result until its other seeds land.

Stage 2 state at the restart: checkpoint selection DONE for
three_piece_assembly_d0 and nut_assembly_d0, and kitchen_d0 plus
coffee_preparation_d0 were mid-eval at 07:36. Those two need redoing. The chain
is rebuild -> image conversions -> resume_ckpt_selection -> k-sweep.

## 2026-07-24 09:30 UTC, thread oversubscription found; LIBERO promoted to a training target

Two hours of GPU time were lost to a scheduling bug, not a science one. The
encoder ablation and the new LIBERO predictor cells ran for 43 minutes and
completed exactly one cell between them. GPU utilisation sat at 3-26% while
load average hit 1069 on 96 cores. Cause: torch sizes its intra-op thread pool
to all 96 cores, OMP_NUM_THREADS was never set, and 19 concurrent cells were
running 3082 threads. The heavy CPU work in train_head.py is the per-batch
gather and dtype conversion over a ~30 GB feature array, which is exactly what
OMP parallelises, so every cell fought every other one. Capped OMP_NUM_THREADS
and MKL_NUM_THREADS at 8 in both runners: 3082 threads -> 144, load 1069 -> 37.
No effect on results, the arithmetic is unchanged. Worth remembering that the
8-worker grid yesterday was already 768 threads and only looked fine.

Restart #4 recovery worked end to end this time. boot_recover.sh now also
stages features, resumes the ablation, launches rebuild_scratch.sh, arms the
k-sweep runner and launches the LIBERO stage, and all of it fired unattended.

LIBERO promoted from predictor training data to a training target (user
decision). What is in place: all 10 LIBERO-10 datasets downloaded and verified
(50 demos each, 138,090 transitions), mask/train+valid 45/5 splits written,
10 diffusion policy configs generated, and a 2-epoch smoke test passing at
25.4 GB on an A100 80GB. Batch 128 rather than 100: LIBERO frames are 128x128
from two cameras, so this is already the largest effective batch in the
project, and on 13k transitions per task a much larger batch would only cut the
number of distinct gradient steps. Trainings are gated behind the ablation and
the predictor cells so they do not contend.

Two LIBERO blockers found and fixed. LIBERO does not import at commit 8f1084e
because it ships no top-level libero/__init__.py; it only resolves as a
namespace package off the repo root, so every consumer now exports PYTHONPATH.
And the HuggingFace fallback in setup_and_probe.sh downloaded files[0] then
broke out of the loop, while the gate only tested "any hdf5 present", so we had
1 of 10 datasets and it looked complete.

Measured the premise behind promoting LIBERO, since it is checkable. Mean
open-loop regime alternations per episode, from our own labels:

  tool_hang        455 steps   15.8 alternations
  LIBERO-10        252 steps    6.4
  square           126 steps    4.2
  mg_square_d0     128 steps    4.2
  mg_stack_d0       83 steps    3.2
  can               92 steps    2.0
  lift              24 steps    1.0

Every suite alternates about once per 30 steps, so alternation count tracks
episode length and LIBERO's advantage over square is length, not a different
character of task. LIBERO is a reasonable main dataset and clearly better than
square, can, lift or the MimicGen sets we have labels for, but tool_hang is
2.5x richer on the axis the hypothesis actually depends on, and we already have
Stage 1a, oracle and Stage 1c results there.

LIBERO labeling is only calibrated on 4 of 10 tasks. delta is the 95th
percentile of |lambda| over free-space stamps, where free means neither
gripper-object nor object-fixture contact. The six LIVING_ROOM and STUDY tasks
are two-object goals, so while one object is carried the other always touches a
fixture, free space is empty, and analyze_labels.py:59 falls back to a
hardcoded delta of 0.05. Those six report frac_free_space = 0.000 and delta =
0.0500 exactly. The 4 KITCHEN tasks have real deltas (0.0174-0.0199) and show
mean lambda under fixture contact at 2.19x free-space lambda. This does not
affect the predictor cells, which take delta from the lambda distribution.

No pretrained LIBERO policy was ever found, contrary to a reasonable
assumption; the only pretrained models here are the V-JEPA 2 and DINOv2
encoders. Public LIBERO policies do not fit the framework either: chunk length
k is enforced by loading a robomimic DiffusionPolicyUNet with action_horizon
forced to prediction_horizon and popping k off policy.policy.action_queue, so a
usable policy must predict >=16 actions per forward pass and expose that queue.
OpenVLA emits one action per pass, so k is undefined for it.

Still owed: the one-page LIBERO LaTeX document (waiting on the 15 predictor
cells), and the LIBERO rollout harness. The harness is smaller than first
estimated: eval_k_sweep.py already implements fixed-k, oracle FTLE-probe and
predictor switching, and only needs its env factory swapped from robomimic's
env_from_checkpoint to LIBERO's native OffScreenRenderEnv, plus manual frame
stacking and sim state save/restore for the probe.

## 2026-07-24 09:55 UTC, PRIOR WORK FOUND: adaptive execution horizon is published

While sourcing external datasets we found the core idea of this project already
published, and it should change what we claim and how we evaluate.

  Che-Sang Park, Junsu Ha, Jianlong Fu, Frank C. Park.
  "Spatial Attention: Adapting Execution Horizons for Diffusion Policies via
  Observation Sensitivity." IEEE RA-L vol 11 no 6, June 2026. arXiv 2607.04739.

Their motivation is our motivation almost verbatim: "A fixed execution horizon
forces a single compromise between responsiveness and efficiency for the entire
task, even though different phases of a task demand different levels of
responsiveness."

What differs is the signal. Theirs is policy-internal: Spatial Attention, the
expected squared norm of the gradient of the action log-likelihood with respect
to the observation, i.e. how sensitive the policy's action distribution is to
the observation. A transformer forecasts the SA sequence alongside the action
chunk, and the execution horizon is the smallest k whose cumulative
SA^(1/(2gamma+1)) exceeds a threshold. Ours is plant-physical: FTLE of the
system under perturbation. Their Fig 2 reports SA rising during grasping and
inserting on Tool Hang, which is the same regime structure our lambda labels
pick up, so the two signals probably correlate.

They evaluate on robomimic Lift/Can/Square/ToolHang, exactly our Stage 1a tasks,
and crucially they control for the SAME AVERAGE EXECUTION HORIZON (T_avg 16
simple, 32 complex) rather than comparing against a best fixed k. Evaluation is
5 best checkpoints x 4 seeds = 20 checkpoints x 100 episodes.

Tool Hang, baseline -> +SA:
  DDPM      state 0.47 -> 0.57   vision 0.61 -> 0.75
  DDIM      state 0.47 -> 0.57   vision 0.63 -> 0.78
  3-step CP state 0.42 -> 0.53   vision 0.49 -> 0.62
  1-step CP state 0.27 -> 0.46   vision 0.36 -> 0.48
Square DDPM: state 0.89 -> 0.94, vision 0.75 -> 0.80. Lift and Can sit at 0.99
for every method, which matches our own finding that they are ceiling tasks.

Why they get +10 to +19 points where Stage 1c got a tie (tool_hang predictor
0.800 vs best fixed 0.807):
  1. Evaluation budget. They use 2000 episodes per cell; we used 3 seeds x 50.
     Our 0.007 gap is far inside our own noise, so our "tie" was never able to
     resolve an effect of the size they report.
  2. Ceiling. Their tool_hang vision baseline is 0.61; our best fixed was 0.88.
     We were measuring near saturation where there is little room to win.
  3. Control. Matched average horizon is a different and cleaner comparison
     than "beat the best fixed k".

Actions this implies, in priority order:
  1. Adopt matched-average-execution-horizon as the primary control. We already
     log mean_k (square 10.7), so this is a reporting and design change, not new
     machinery.
  2. Raise the evaluation budget by roughly an order of magnitude before any
     claim of tie or win.
  3. Implement Spatial Attention as a baseline. The concrete contribution then
     becomes a direct comparison of a plant-stability signal against a
     policy-sensitivity signal for choosing chunk length, which is a real
     question and one we are unusually well equipped to answer because we
     already have the FTLE labeling machinery.
  4. Reconsider task choice on ceiling grounds, not just alternation count.

Dataset sourcing assessment, same session. RoboCasa is the best structural fit
(robosuite, so our labeler, contact signal and eval harness transfer) and its
checkpoint repo robocasa/robocasa365_checkpoints does contain a diffusion
policy directory, but it is a diffusion_policy-codebase checkpoint
(train_diffusion_transformer_hybrid) rather than robomimic, and the RoboCasa365
leaderboard puts Diffusion Policy between 6.1% and 23.9%, which is a floor
problem rather than the 40-75% band we want. The leaderboard's best model
overall is 57.4%. Given the paper above, the accepted evaluation for this
question is robomimic Lift/Can/Square/ToolHang, which we already have, so more
datasets is not the binding constraint. Evaluation protocol is.

## 2026-07-24 10:15 UTC, LIBERO reverses the encoder result

First LIBERO in-domain predictor cells (seed 0, delta 0.0241 auto-fit on the
LIBERO pool, 50082 train / 13084 val stamps):

  vision only (V-JEPA)        all 0.592   confident 0.905
  V-JEPA + proprioception     all 0.574   confident 0.878
  proprioception only         all 0.484   confident 0.859

The panda ordering is the opposite: proprio-only 0.895 > vj_vis 0.878 >
di_vis 0.848. On LIBERO proprioception alone is at 0.484 on all stamps, which
is chance.

This makes the earlier mechanism a testable claim instead of a rationalisation.
Proprioception suffices exactly when the scene is fixed. On square and
tool_hang the fixture never moves and the object is rigidly held, so
end-effector pose is a sufficient statistic for contact geometry. LIBERO has
ten scenes with varied layouts and two-object goals, so pose stops being
sufficient and vision carries real signal. The blog post's "vision adds
nothing" is a property of the panda suite, not of the problem.

Unexplained: adding proprioception to vision HURTS on LIBERO, 0.905 -> 0.878.
Seed 0 only so far.

Transfer still fails in this direction too. LIBERO -> tool_hang: 0.443
confident, 0.441 at refit delta, Spearman -0.074.

Panda grid A with 3 seeds where available:
  A_prop     0.895 0.897 0.893   mean 0.895  spread 0.004
  A_vj_vis   0.871 0.878 0.886   mean 0.878  spread 0.015
  A_di_vis   0.844 0.852         mean 0.848
  A_di_full  0.863 0.872         mean 0.867
  A_di_e100  0.844 0.814         mean 0.829  (100 epochs is worse than 40)
A_prop is remarkably tight at 0.004 spread across 3 seeds, so proprio-only
beating both vision-only conditions on the panda suite is solid.

INFRASTRUCTURE NOTE: impl/predictor/train_head.py was modified at 09:50 UTC by
someone other than this session. It went 240 -> 280 lines and gained
proprio-sequence handling plus a --proprio-last-only flag, with load_features
now defaulting to seq=True. The grid is NOT confounded: our feature .npz files
store proprio as 2-D (N, 9), so the sequence path is inert and both post-edit
cells record proprio_seq=False, proprio_window=1, identical to the 27 pre-edit
cells. One cell (L_vj_in_s1) failed with UnboundLocalError on P, almost
certainly because it started while the file was mid-rewrite; the same code path
succeeded in B_loto_vj_s2 and B_sq2th_vj_s2. Queued for re-run.

## 2026-07-24 10:50 UTC, LIBERO reversal confirmed across seeds

Two seeds per condition now. The seed-0 result holds and the all-stamps figure
is the decisive one.

  cell         all (mean)   confident (mean)   seeds
  L_vis_in       0.579          0.900            2
  L_vj_in        0.583          0.885            2
  L_prop_in      0.502          0.870            2

Proprioception alone is at 0.502 on all stamps on LIBERO, i.e. chance, against
0.736 for the same condition on the panda suite (A_prop_s0). The 0.077 gap is
well outside the observed seed spreads of 0.009 to 0.022. On confident stamps
the gap is 0.030 and closer to the noise floor, same direction.

This is the cleanest statement of the mechanism so far. Proprioception is a
sufficient statistic for contact geometry only when the scene is fixed. On
square and tool_hang the fixture never moves and the object is rigidly held, so
end-effector pose determines the contact configuration. LIBERO has ten scenes
with varied layouts and two-object goals, and there pose alone carries nothing.

Transfer fails in every direction tested, two of three below chance:
  LIBERO -> tool_hang  s0  0.443 conf, 0.441 refit-delta, spearman -0.074
  LIBERO -> tool_hang  s1  0.429 conf, 0.429 refit-delta, spearman -0.093
  panda  -> LIBERO     s1  0.400 conf, 0.406 refit-delta, spearman +0.031
                           (mean over all 10 LIBERO holdout tasks)

Consequence for the blog post "Do You Need Vision to Predict Stability
Regimes?": its headline claim, that vision adds nothing and nine proprioceptive
numbers beat a video transformer, is a property of the fixed-scene panda suite
and does not generalise. The post needs revising once the third seeds land. The
revised claim is stronger and more interesting than the original.

Pipeline: ablation 31/39, LIBERO cells 9/15, Stage 2 nut_assembly_d0 conversion
DONE and the other three at demo 639-809 of 1000, k-sweep runner armed, 5 films
on disk. Load 16.5 since the OMP fix.

## 2026-07-25 08:30 UTC, restart #5 and final 3-seed grids

The VM was stopped at 13:56 yesterday and came back at 07:44 today, so
/mnt/scratch was reprovisioned for the fifth time. Nothing of value was lost:
the predictor cells write their metrics straight to the share, the 15-minute
sync loop had mirrored labels and features, and all 10 LIBERO diffusion policy
trainings checkpoint to the share (last.pth at roughly epoch 124 of 1000 at
shutdown, about 2h20m of training each).

Before the shutdown the grids finished: ABLATION_ALL_DONE at 11:33, 
LIBERO_ALL_DONE at 11:40, and the LIBERO stage launched all 8 first-wave
trainings at 11:34. Final 3-seed LIBERO predictor table (val AUROC):

  condition                 all    conf   spread(all)
  L_vis_in  (vision only)  0.578  0.896   0.027
  L_vj_in   (vis+proprio)  0.585  0.894   0.018
  L_prop_in (proprio only) 0.499  0.861   0.035

Proprio-only on all stamps is chance across all 3 seeds on LIBERO, against
0.895 confident / tight-spread wins for A_prop on the panda grid. The
scene-dependence claim is now 3-seed solid on both suites.

Transfer, 3 seeds, still fails everywhere:
  LIBERO -> tool_hang: all 0.44-0.48, conf 0.43-0.44 (below chance)
  panda -> LIBERO: all 0.54-0.62, conf 0.40-0.50 (conf below chance;
  the all-stamps number is propped up by the label imbalance, not signal)

Panda ablation final (confident AUROC, 3 seeds unless noted):
  A_prop 0.895 (spread 0.003), A_vj_full 0.883, A_vj_vis 0.879,
  A_di_full 0.864, A_di_vis 0.851, A_vj_e100 0.849, A_di_e100 0.829 (n=2)
  B_loto: prop 0.834 (n=2) > vj 0.804 > di 0.800
  B_sq2th: prop 0.774 > vj 0.755 > di 0.739

Two cells (A_di_e100_s2, B_loto_prop_s1) died in yesterday's 09:50
train_head.py mid-rewrite window and were never retried; the runner's ALL_DONE
line counts cell dirs, not metrics files, so it reported 39/39 anyway. A
detached retry chain is running them now on GPU 7.

Recovery actions today: boot_recover fired on its own (fifth restart, second
fully automatic recovery). Two gaps found and fixed in the LIBERO resume path:
(1) run_libero_stage.sh trained without --resume, which would have thrown away
the 124 epochs; train_one now passes --resume whenever a last.pth exists on
the share. (2) Nothing re-downloads the LIBERO datasets after a wipe; a
detached waiter now runs setup_and_probe.sh once the mg venv exists.
boot_recover.sh itself still needs the same waiter added for restart #6; it
was mid-execution and editing a running bash script is unsafe, so that patch
waits for BOOT_RECOVER_DONE.

Timeline from here: LIBERO datasets ~1-2 h after the mg venv is up, then the
stage resumes 8 trainings from epoch ~124 (about 15 h to 1000) and starts the
2 second-wave tasks from scratch (about 17 h). Rollout harness for success
rates is still the next thing to write.

## 2026-07-25 09:50 UTC, generalist-policy experiment approved and staged

User decisions this morning: (1) every switching eval must include a (16,1)
cell alongside (16,4), now baked into run_nutcoffee_chain.sh and
run_stage2_ksweep.sh (24 baseline cells now). (2) The next experiment platform
is GR00T N1.5 (flow-matching DiT head, 16-step action chunks) on RoboCasa
Kitchen with the Franka arm, using NVIDIA's released robocasa365 multitask
checkpoint (43.0% atomic-seen mean, so real headroom, and no policy training
on our side). Chosen over pi0+SIMPLER (SAPIEN backend would need a labeler
port, no demos shipped), GR1 tabletop (bimanual), LIBERO VLA checkpoints
(ceiling). impl/robocasa/setup_robocasa.sh is running: rc venv (robosuite +
robocasa), kitchen assets, groot venv (Isaac-GR00T + flash-attn), checkpoint
and starter datasets from HF, then an env state save/restore probe. Markers in
results/robocasa/setup.log. boot_recover.sh now also re-arms the LIBERO
dataset download, the nut/coffee chain, and the robocasa setup after a
restart. Integration order stays: nut/coffee + LIBERO results first, RoboCasa
harness work gated on that outcome.

## 2026-07-25 12:05 UTC, RoboCasa integration: data solved, labeler ported

The robocasa365 release ships everything in LeRobot format. Two findings that
shaped the port: (1) the extras/ dir of every episode carries states.npz
(full per-step MuJoCo states), model.xml.gz (the episode's compiled kitchen)
and ep_meta.json, so no replay reconstruction is needed at all; (2) every
demo has its OWN kitchen scene, so labeling must reset scene per demo, and
the robot is PandaOmron with a 12-dim HYBRID_MOBILE_BASE action where the
arm position dims must be resolved from the composite controller's action
split rather than assumed to be the first three.

Built today: impl/robocasa/lerobot_to_hdf5.py (pure repackaging; 4 tasks
converted, 500-514 demos each, 113k-184k steps, zero skips) and
impl/robocasa/ftle_labeler_robocasa.py (uses robocasa's own EnvRobocasa
wrapper; stock robomimic 0.5 crashes querying action specs before robosuite
1.5 initializes controllers, and the wrapper needs robocasa's vendored
obs_utils initialized, not robomimic's). Contact semantics for the deadband:
free space = gripper touching nothing; movable vs static contact recorded.
Dev run over 2 demos in flight; labeling chain v2 (sequential per task,
load-gated at 70, 200 demos each) takes over once it passes.

GR00T venv saga: rounds 1-3 failed for stacked reasons (a [base] dep imports
torch at build time; flash-attn needs psutil preinstalled and CUDA_HOME; and
the actual root cause, gr00t requires Python >=3.12 while the venv was 3.10).
Round 4 builds a conda-forge 3.12 env with GR00T's exact pins. Checkpoint
(16 GB) and 23 GB of kitchen assets are fully in; the state save/restore
probe passed with zero drift, so the labeling method is valid on RoboCasa.

Elsewhere: LIBERO 8 trainings resumed from checkpoint (epoch ~292 by 11:35),
nut labeling started 11:13 with policy sigma_u 0.1030, coffee image ready
11:17, the 24 Stage 2 baseline cells (incl. the new 16-1 protocol cells)
running since 11:01, panda ablation a true 39/39 after the 2-cell retry.

## 2026-07-25 12:50 UTC, nut assembly labeled; RoboCasa feature path complete

nut_assembly_d0 open-loop labels landed in 69 minutes: 33,251 stamps from 200
demos, sigma_u 0.1030 (policy RMSE, ~3x the panda value, consistent with a
0.36-success policy), auto-deadband 0.0620, 12.4% of stamps unstable, 31%
free-space. The chain's V-JEPA extract step then crashed on a missing
MAIN_CAM entry for the new task; fixed with an explicit
--cam-key agentview_image and the chain relaunched (nut resumes at extract,
coffee skips its finished sigma_u and goes straight to labeling).

First two 50-episode baseline cells: three_piece fixed-16 0.64, fixed-8 0.72
(the 25-episode ckpt-select figure of 0.80 was optimistic, as expected at
that budget). The three_piece predictor(16,1) cell runs mean_k ~15.5, i.e.
the panda-pool predictor almost never fires the short chunk there.

RoboCasa: labeling chain v2 is running (OpenCabinet first). Built and
launched the remaining piece of the predictor path:
impl/robocasa/extract_rc_features.py (frames decoded from the lerobot mp4s,
proprio from parquet observation.state, output schema identical to
vjepa_extract.py) and impl/robocasa/run_rc_features_heads.sh (waits per task
for labels, extracts on GPU 4, trains {full, vis, prop} x 3 seeds). With
that, every stage from raw robocasa365 release to predictor AUROC is wired;
only the GR00T rollout harness remains (flash-attn still compiling).

## 2026-07-26 14:55 UTC — GR00T env ready; RoboCasa predictor path fully landed; nut/coffee chain done
- GR00T N1.5 environment complete: groot312 (py3.12) with torch 2.9.0+cu128, official flash-attn 2.8.3 wheel (the 3h self-compiled 2.7.1.post4 did not match gr00t's ==2.8.3 pin; replaced with NVIDIA's prebuilt wheel), gr00t 0.1.0 editable install, import test GROOT_IMPORT_OK. Checkpoint-120000 (16 GB) on scratch; sim probe passed earlier (state save/restore drift 0.0). Repo ships an official robocasa365 eval harness (gr00t/eval/sim/robocasa365/gymnasium_groot.py + scripts/eval/check_sim_eval_ready.py) — first rollout validation will use it.
- RoboCasa labeling/predictor path 100% done: 4 tasks labeled (OpenCabinet 34.1k stamps 64.6% lambda>0; OpenDrawer 24.3k 68.1%; PickPlaceCounterToCabinet 23.7k 84.5%; TurnOnSinkFaucet 20.2k 47.4%), features extracted, 36 heads trained. 3-seed val AUROC (all/confident): TurnOnSinkFaucet .89/.95, OpenCabinet .85/.98, OpenDrawer .79/.86, PickPlace .78/.87. Sigma_u was placeholder 0.05; revisit once GR00T action RMSE is measurable.
- nut/coffee chain NUTCOFFEE_ALL_DONE (10:24). Nut: fixed 1/4/8/16 = .52/.42/.42/.34 (single-step BEST, 18pt headroom over k=16); contact-oracle predictor(16-4/16-1) .40/.42; indom full head .40/.32. Coffee: fixed 1/4/8/16 = .04/.10/.24/.18 (k=8 best, single-step WORST); indom full .16/.10. In-domain heads (nut auroc_all .70, coffee .76) do not yet beat fixed-k.
- 8 indom vis/prop eval cells had failed: predictor_bridge built the full head architecture regardless of ckpt condition flags. Patched (passes proprio_only/no_proprio from ckpt) and relaunched the 8 cells on GPUs 2/3 (rerun_indom_visprop.sh).
- LIBERO: 2 second-wave trainings still running on GPUs 0/1 (STUDY_SCENE1 book, LIVING_ROOM_SCENE6 mug+pudding).

## 2026-07-26 15:40 UTC — first GR00T rollout blocked by repo/checkpoint mismatch, correct fork identified and building
- First rollout attempt failed cleanly: the upstream Isaac-GR00T repo is now the N1.7 release and no longer registers the gr00t_n1_5 architecture, so AutoModel cannot load checkpoint-120000. No commit of upstream has both the robocasa365 harness and the N1.5 class.
- The robocasa365 benchmark docs point to the paired fork github.com/robocasa-benchmark/Isaac-GR00T (N1.5 training + scripts/run_eval.py, in-process ZMQ server/client eval over the robocasa365 task registry). Cloned to repos/Isaac-GR00T-rc365.
- Building envs/groot155 (py3.10, torch 2.5.1, transformers 4.51.3 = checkpoint's own version, official flash-attn 2.7.1.post4 wheel for torch2.5/cp310, pinned robosuite 85abee2, robocasa365 pinned clone with our kitchen assets symlinked). tensorflow from the base extra deliberately skipped (pins numpy<2, robocasa365 asserts numpy==2.2.5, not needed on the eval path). setup_groot155.sh, marker GROOT155_OK, added to boot_recover.
- The official robocasa365_uv sim venv (NVIDIA setup script) is built and passed its env sanity check; kept for the N1.7-era harness, but the N1.5 eval path uses the fork.
- 8 indom vis/prop cells rerunning cleanly after the second predictor_bridge fix (flags live in ck["args"], not top level; all three head architectures verified to load). LIBERO second wave at epochs ~976/956.

## 2026-07-26 16:00 UTC — GR00T N1.5 PLATFORM VALIDATED: first rollout 2/2 successes
- groot155 env (fork + torch 2.5.1 + transformers 4.51.3 + official flash-attn 2.7.1.post4 wheel, no compile) built in ~15 min and passed all imports.
- First rollout validation: TurnOnSinkFaucet, pretrain scenes, checkpoint-120000, 16-step chunks: 2/2 SUCCESS, 138 s for 2 episodes, videos in /mnt/scratch/lh/rollouts/groot_val/. The full stack (Gr00tPolicy ZMQ server -> robocasa365 sim -> success predicate -> video) works.
- Launched the fixed-k baseline sweep (run_groot_ksweep.sh): 4 labeled tasks x k in {16,8,4,1}, 50 episodes, 5 parallel envs, one GPU per task (4-7), distinct ZMQ ports; added to boot_recover. This gives the per-task success table and the headroom picture for switching.
- GR00T action RMSE (sigma_u) measurement running on GPU 5 (measure_groot_rmse.py, first-action pos-dim RMSE vs the same target-split demos the labels used); relabel decision when it lands.

## 2026-07-26 16:05 UTC — LIBERO training stage COMPLETE; sigma_u investigation opened
- All 10 LIBERO-Long DP trainings reached epoch 1000 (36-41 checkpoints each). GPUs 0/1 free. Next: the LIBERO rollout harness (native OffScreenRenderEnv, frame stacking, sim save/restore) for checkpoint selection, fixed-k sweep, and oracle cells incl. (16,1).
- GR00T sigma_u: measured first-action pos RMSE on target-split demos = 0.30-0.56 (pooled 0.467) vs placeholder 0.05. Context: demo arm-action std is ~0.25, so the placeholder is ~20% of action std (consistent with the robomimic convention in relative terms) while the naive measurement is 1-2x action std, inflated by one-sample flow-policy stochasticity and the target/pretrain split shift. Two checks running before any full relabel: (a) RMSE on the freshly downloaded pretrain-split demos (in-distribution), (b) sensitivity relabel of TurnOnSinkFaucet at sigma 0.30 (60 demos) to measure label agreement vs the 0.05 labels.
- GR00T fixed-k sweep: all four fixed_16 cells running (GPUs 4-7).

## 2026-07-26 16:40 UTC — LIBERO harness validated (0.80 first cell); GR00T k-response matches label profiles
- LIBERO chain: after the init-states torch.load fix, KITCHEN_SCENE3 e400 selection cell scored 0.80 (20 eps) — harness, obs mapping, and policy bridge all confirmed working. LIVING_ROOM_SCENE2 e400 = 0.0 (checking later epochs; large per-task spread is expected for DP on LIBERO-Long).
- GR00T fixed-k (50 eps/cell): OpenCabinet .52/.52, OpenDrawer .64/.56, PickPlaceCounterToCabinet .58/.78, TurnOnSinkFaucet .42/.36 for k=16/k=8. NOTABLE: the one task with 50% unstable stamps (PickPlace, carry-phase instability) gains 20 points from halving the chunk; the three mostly-stable tasks do not improve. First evidence on the generalist platform that the open-loop label profile predicts the fixed-k response. k=4/k=1 cells running.
- Sensitivity relabel had died on an argument mismatch (--output, no --env-name); relaunched correctly (60 demos, sigma 0.30).
- 8 indom vis/prop reruns: nut vis 16-4 completed cleanly (~65 min/cell), no failures since the ck["args"] fix.

## 2026-07-26 17:15 UTC — GR00T fixed-k near-complete; switching harness built and queued
- Fixed-k so far (50 eps): OpenCabinet .52/.52/.44, OpenDrawer .64/.56/.64, PickPlaceCounterToCabinet .58/.78/.70, TurnOnSinkFaucet .42/.36/.10 for k=16/8/4. Task-dependence is strong and matches the label profiles: the carry-unstable task peaks at k=8 (+20 pts over k=16), the mostly-stable precision task collapses at k=4 (.42 -> .10). k=1 cells running.
- Built groot_switch_rollout.py: in-process Gr00tPolicy + fork env with MultiStepWrapper(n_action_steps=1) so the loop controls executed chunk length; predictor_bridge (vjepa venv) serves lambda_hat from the RoboCasa heads with training-faithful inputs (16-frame robot0_agentview_left window + 16-step 16-dim state window in modality.json order). run_groot_switch.sh queued behind the fixed sweep: 4 tasks x full head x (16,4) and (16,1). Added to boot_recover.
- LIBERO selection: KITCHEN_SCENE3 e400 .80 > e600 .60 > e800 .55 (earlier checkpoints better); LIVING_ROOM_SCENE2 0.0 at e400/e600.

## 2026-07-26 18:15 UTC — GR00T fixed-k COMPLETE; oracle switching cells live
- Final fixed-k table (50 eps/cell, k=16/8/4/1): OpenCabinet .52/.52/.44/.48; OpenDrawer .64/.56/.64/.40; PickPlaceCounterToCabinet .58/.78/.70/.52; TurnOnSinkFaucet .42/.36/.10/.04.
- Switching (16,4) cells rolling on GPUs 4-7 (first-cell partials, not final): TurnOnSinkFaucet 15/26 with 29% unstable calls (vs .42 best fixed); OpenCabinet 15/20; PickPlace 8/16 with the oracle firing on 69-91% of replans mid-carry exactly as the labels predicted; OpenDrawer 14/23 with the head almost never firing. (16,1) cells follow, then the PickPlace (8,4)/(8,1) supplement.
- Sanity videos: one successful captioned episode per task delivered and saved to results/robocasa/sanity_videos/ (prompts are episode-randomized; in-process rollout path validated end to end, including the 0-dim language obs fix that would have broken the switching cells).
- LIBERO: KITCHEN_SCENE3 fixed_16 = 0.72 (vs 0.80 sel at 20 eps); LIVING_ROOM_SCENE2 is 0.0 at every epoch, a genuinely failed DP task.

## 2026-07-26 18:50 UTC — sigma_u robustness CONFIRMED, no relabel; first switching finals
- TurnOnSinkFaucet relabeled at sigma 0.30 (6x the placeholder, 2x the defensible measured value of ~0.15): on 6,149 common stamps, Spearman of lambda vs the 0.05 labels is 0.834 and unstable-flag agreement at the same delta is 95.4% (unstable fraction 8.6% -> 10.6%). The absolute lambda level shifts up with sigma (mean 0.004 -> 0.024) but the ranking and the top tail, which are all the pipeline consumes (auto-delta adapts per distribution), are robust. DECISION: keep the existing labels and heads; record sigma sensitivity in the writeup.
- Switching (16,4) finals: OpenCabinet 0.62 vs best fixed 0.52 (oracle fired on 5% of replans; rare targeted intervention wins +10); TurnOnSinkFaucet 0.44 vs 0.42 (3% firing); OpenDrawer 0.52 vs 0.64 fixed_16 (16% firing hurt). PickPlace (16,4) still running (fires 70-90%), then all (16,1) cells and the PickPlace k_stable=8 supplement.

## 2026-07-27 04:50 UTC - GR00T RoboCasa eval COMPLETE; LIBERO predictor bug fixed and backfilling
- GR00T switching finished (GROOT_SWITCH_DONE 20:29, GROOT_SWITCH_EXTRA_DONE 23:14). Full table (50 eps/cell):
  OpenCabinet fixed .52/.52/.44/.48 (k=16/8/4/1), switch (16,4)=.62 (5% firing), (16,1)=.58. Best: switch (16,4), +10 over best fixed.
  OpenDrawer fixed .64/.56/.64/.40, switch (16,4)=.52, (16,1)=.54. Switching slightly hurts (16-21% firing).
  PickPlaceCounterToCabinet fixed .58/.78/.70/.52, switch (16,4)=.50, (16,1)=.62, (8,4)=.68, (8,1)=.66. Heavy firing (70-90%); best remains fixed_8=.78.
  TurnOnSinkFaucet fixed .42/.36/.10/.04, switch (16,4)=.44, (16,1)=.50. Best: switch (16,1), +8 over fixed_16 with 1% firing.
  Headline: switching wins where the head is selective (OC, TOSF), loses where it fires constantly (PP) or interrupts needlessly (OD).
- robocasa_label_report_latex/main.tex updated with measured rates (new Section "Measured success rates", Table tab:rates); recompiled, now 7 pages.
- LIBERO chain predictor cells ALL fast-failed (12 cells, K3 + LR2-soup): heads train on harmonized 9-dim proprio (pos3+quat4+grip2) but harness sent 8-dim (axis-angle). Fixed libero_ksweep.py (scipy from_rotvec -> quat, mirrors harmonize_libero_proprio.py). Future chain cells use fixed code.
- Launched impl/libero/backfill_libero_predictor.sh (GPUs 4/5, K3 + LR2-soup x 6 cells each, marker LIBERO_BACKFILL_DONE). First cell healthy: K3 vj_in 16-4 ep0 SUCCESS, mean_k 12.2 (head fires on LIBERO, unlike mimicgen).
- LIBERO chain progress: K3 fixed row .72/.44/.22/.08 (k=16/8/4/1); K4 sel e400=.75, fixed_16=.76, fixed_8 running; LR2-soup all 0.0 (dead task); LR2-cream sel running.
- nut/coffee reruns 7/8: coffee prop 16-4=.02 (mean_k 10.6!), coffee vis 16-1=.12; last cell (coffee prop 16-1) at ep 33, mean_k ~14-15.

## 2026-07-27 05:40 UTC - Causal controls + composed sequence task launched
- User's proof plan: (1) show short-at-OL-unstable placement matters -> random switching matched on firing rate (OC 16-4 p=.05, TOSF 16-1 p=.01); (2) show long chunks at OL-unstable moments hurt -> shadow-scored fixed-16 on all 4 tasks (head logs lambda_hat, never acts). groot_switch_rollout.py gained --control {predictor,random,shadow} + --fire-rate + --seed; run_groot_switch_controls.sh gated on GROOT_SWITCH_TRACED_DONE, GPUs 2/6/7, marker GROOT_SWITCH_CONTROLS_DONE. Results go into a new causal-controls section of the RoboCasa LaTeX report.
- Composed long-horizon task built: impl/robocasa/groot_sequence_rollout.py - one PickPlaceCounterToCabinet env, cabinet doors forced closed after reset (cab.close_door + sim.forward + refreshed obs), stage 1 language override "Open the cabinet door(s)." with cab.is_open success (atomic predicate), stage 2 env's own instruction + success. Horizon 1050+750=1800. Predictor mode uses stage-matched heads (OC head stage 1, PP head stage 2) at their own deltas; logs per-replan stage/lam/k.
- run_groot_sequence.sh launched (marker GROOT_SEQUENCE_DONE): 8 cells x 50 eps - fixed 16/8/4/1 + predictor (16,4) (16,1) (8,4) (8,1). GPU 3 after coffee cell, GPU 4 after K3 backfill half, GPU 5 after full LIBERO backfill. Sequence results also go into the LaTeX report.

## 2026-07-27 07:40 UTC - Traced cells COMPLETE, controls running, regime figures built
- All 4 traced cells done (GROOT_SWITCH_TRACED_DONE 06:59): OC (16,4)=.56, OD (16,1)=.56, TOSF (16,1)=.60, PP (16,4)=.66. Second-seed effect vs first runs (.62/.54/.50/.50): per-config spread ~.06-.16 at 50 eps, worth error bars later.
- Regime-timeline figures built (impl/robocasa/make_traced_figs.py -> results/robocasa/switch_traced/figs/): per-task lambda-hat timelines success vs failure + firing-by-progress panel. Sent to user.
- KEY finding from traces: OpenDrawer failures fire 2.3x more than successes (0.32 vs 0.14) - consistent with firing-causes-harm OR head-detects-doom; shadow-16 control will disambiguate. TOSF failures fire LESS than successes (0.01 vs 0.04) - its rare firing is well placed. OC firing identical across outcomes (0.04). PP fires ~0.8 regardless.
- Controls chain started 07:01 (shadow_16 on OC/OD/TOSF, GPUs 2/7/6; then PP shadow + 2 random cells). Sequence chain still waiting on GPUs 3/4/5.
- Coffee prop 16-1 at ep 48/50; K3 vj (16,1) and LR2-soup vj (16,1) backfill cells running; K4 fixed_1 running; LR2-cream fixed row 0.0 (fixed_16 done).

## 2026-07-27 08:30 UTC - Nut/coffee table final; oracle-switching harness built; novelty verified
- Nut/coffee reruns COMPLETE (8/8). nut: fixed 16/8/4/1 = .34/.42/.42/.52 (k=1 best - mostly-unstable task), pooled-vjepa switch .40/.42, indom full .40/.32, prop .44/.38, vis .34/.46 (mean_k ~16 under-firing). coffee: fixed .18/.24/.10/.04 (k=8 best), all switched cells <= .16 (firing poorly placed). MimicGen heads do not transfer their offline AUROC to rollout switching - calibration/shift finding for the writeup.
- First control cell: OpenCabinet shadow_16 = .58 (fixed-16 replicate; .52/.58 spread confirms seed noise scale).
- Novelty check done (web): 2026 adaptive-horizon papers exist (DEHP arXiv 2606.11408 RL-trained horizon branch; AutoHorizon 2602.21445 attention weights; PACE 2606.00537 speed-profile transitions; HiPolicy entropy; MDPI visual-context) but NONE use measured perturbation-divergence stability or an OL/CL taxonomy - the mechanism is unclaimed; the coarse "phases exist so adapt the horizon" intuition IS published (PACE) so the paper must lead with the stability mechanism.
- Built impl/robocasa/groot_oracle_rollout.py: measured-lambda switching (labeler FTLE at each replan: state save, chunk replay x8 perturbed branches sigma_u .05, restore; K=chunk_len=16 vs labeler 24 - noted) + contact-oracle mode (gripper-on-movable). run_groot_oracle.sh launched, gated on GROOT_SWITCH_CONTROLS_DONE: OC oracle 16-4 (gpu2), PP oracle+contact 16-4 (gpu6), OD oracle 16-1 + TOSF oracle 16-1 (gpu7), deltas = exact head-train deltas. Marker GROOT_ORACLE_DONE.
- Publishable proof ladder (robust to imperfect learned head): labels prove regimes exist -> regime-specific fixed-k failure decomposition (shadow + fixed cells) -> ORACLE switching proves causality -> random/shadow controls rule out confounds -> learned head = deployable approximation; 2x2 taxonomy predicts win/loss pattern.

## 2026-07-27 09:10 UTC - Controls COMPLETE; pooled CI analysis changes the atomic-task story
- GROOT_SWITCH_CONTROLS_DONE 08:52. OpenCabinet random (16,4) p=.05 = .56. Oracle chain auto-started 08:52 and the harness works (OC oracle ep0 SUCCESS, ~144s/ep = ~2.6x cost, fired 0.00 so far).
- POOLED analysis (replicates merged, 95% binomial CIs):
  OC: fixed_16 55/100=.55, pred 59/100=.59, random 28/50=.56 -> predictor +4 over fixed, random matches predictor. NOT separable.
  TOSF: fixed_16 44/100=.44, pred 55/100=.55, random 30/50=.60 -> occasional replanning helps regardless of placement at 1% firing.
  OD: fixed_16 .60, pred .55 -> mild negative, within CI.
  PP: fixed_8 .78 (best, CI [.67,.89]), fixed_16 pooled .67, pred 16-4 .58.
- HONEST CONCLUSION: on atomic RoboCasa tasks the learned-switching wins reported earlier were substantially run noise; at n=100-150 predictor vs random vs fixed are within CIs. The effects that ARE solid: fixed-k regime correspondence (PP 8 vs 16 gap, TOSF collapse at short k, nut k=1 best) and the SEQUENCE task where switching (16,1) leads best fixed by ~.15 at n~45 (10/18 vs 18/45) - effect size scales with the switchable fraction of the episode, which the composed task maximizes by construction. Oracle cells now decisive for the atomic causal claim.
- Theory note for paper: expected switching gain is bounded by firing rate; at 5%/1% firing the max effect was always small. The composed task was designed to have a large unstable fraction inside a stable frame, exactly where the bound is loose, and there the gap is large. This becomes a feature of the story (taxonomy + effect-size bound), not a failure.
- Report/paper causal-controls + sequence sections: writing ONCE when oracle + sequence chains land (2-3h) so conclusions include the ceiling experiment.

## 2026-07-27 09:55 UTC - Oracle firing spread across tasks; OD oracle rescoped
- Oracle interim: OpenCabinet 7/10, measured lambda fires 0% (degenerates to fixed-16); PickPlace 5/12, fires ~2-12%; OpenDrawer ep0 FAILED with fires 95% -> at (16,1) that is one full FTLE measurement per step, ~1h/episode, untenable. Killed OD (16,1) oracle (CELL_FAIL 09:54), runner advanced to TurnOnSinkFaucet oracle (16,1); queued OpenDrawer oracle (16,4) n=25 rerun gated on TOSF finishing (run_od_oracle_16-4.sh).
- The measured-vs-predicted firing gap is now a 3-way spread: measured lambda at rollout fires 0% (OC), 2-12% (PP), 95% (OD) where the heads fired 5%, 78%, 16-21%. Rollout-state stability differs sharply from demo-state labels per task and from head calibration. Candidate causes: state distribution shift (policy visits different states than demos), K=16 vs K=24 slope bias, per-task contact noise floors. This is now a first-class finding for the paper: neither the labels delta nor the head transfers cleanly to rollout-measured stability; delta needs rollout-side recalibration (e.g. percentile of measured lambda on shadow runs).
- seq_fixed_4 final 0.22 (s1 .48). Sequence: 3/8 cells final (fixed 16/8/4 = .38/.44/.22), 5 running.

## 2026-07-27 11:55 UTC - Sequence grid effectively final: no switching win at n=50
- seq_pred_16-1 FINAL 0.38 (s1 .72) - fully regressed from the .75-at-n=12 interim. Final stitched-task grid: fixed 16/8/4 = .38/.44/.22 (fixed_1 at 4/46, heading ~.08); switched (16,4)=.38, (16,1)=.38, (8,4)=.40, (8,1)=.26. VERDICT: best fixed (k=8, .44) >= all switched cells at learned-delta. The composed-task "centerpiece" does not survive full sampling; stage-1 rates favor switching ((8,4) .76 vs fixed .66-.68) but stage 2 gives it back.
- Oracle interim: TOSF 20/32=.63 (fired ~14%) still the one live oracle-over-fixed signal (pooled fixed_16 .44); PP 25/39=.64 ~ fixed_16 .67; OC 11/24=.46.
- Remaining before write-up: seq_fixed_1 (~15 min), TOSF/PP oracle (~30 min), OC oracle (~1h), OD (16,4) n=25 after TOSF.

## 2026-07-27 12:30 UTC - Stitched grid 100% final; report table refreshed
- seq_fixed_1 FINAL 0.10 (s1 .62). Grid complete: fixed .38/.44/.22/.10; switched (16,4) .38, (16,1) .38, (8,4) .40, (8,1) .26. Report Table tab:sequence refreshed with final numbers and honest paragraph (sample-size lesson, stage-1 advantage that stage 2 gives back, pointer to threshold recalibration). Recompiled, 9 pages.
- Oracle: TOSF 26/40 (.65), PP 30/46 (.65), OC 13/29 (.45); OD (16,4) n=25 queued. Oracle comparison + AAAI paper marker replacement next iteration when cells land.

## 2026-07-27 13:05 UTC - Recalibrated-delta side evaluation launched (user-approved idea 1)
- Recalibrated deltas from rollout distributions (results/robocasa/recalib_deltas.json), targeting firing = label unstable fraction:
  head: OC .0578->.0401 (fired 2.3% -> target 11.7%), TOSF .0771->.0636 (1.9% -> 8.9%); OD and PP heads ALREADY calibrated (16.8% vs 15.2%, 51.6% vs 50.3%) - recalib is a no-op there, so their switching losses are NOT a threshold problem.
  oracle: PP .0358->.0245 (13.2% -> 50%); TOSF oracle would go LESS sensitive (.0771->.1041) - left untouched since it is winning at old delta; OC oracle delta computed at launch from its json.
- run_groot_recalib.sh launched (marker GROOT_RECALIB_DONE): head OC 16-4 (gpu 3) + head TOSF 16-1 (gpu 0) now; oracle PP 16-4 (gpu 6) and oracle OC 16-4 (gpu 2) gated on their oracle cells finishing.

## 2026-07-28 ~09:50 UTC - Oracle + recalib families FINAL; headline result; docs updated
- All oracle cells final: PP oracle (16,1) **.80** [.69,.91] fired .11 (HEADLINE: beats
  fixed_16 .67 and fixed_8 .78); PP (16,4) .64 same firing moments -> the k=1 drop is
  what rescues; TOSF (16,1) .62 vs fixed .44; OC (16,1)/(16,4) .60/.52 fired .04-.06
  (never fires -> fixed level); OD (16,4) n25 .60 fired .15; OD (16,1) n15 .40 fired ~.95
  in failures -> degenerates to fixed_1's exact rate.
- Recalib cells final: PP oracle delta .0245 -> .74 (up from .64, still < fixed_8 .78);
  OC head .56 (fire 2%->29%, unchanged); TOSF head .56 (fire 2%->10%, unchanged);
  OC oracle-recalib .58. Verdict: threshold matters for the measured signal, heads'
  gap is ranking quality at rollout states.
- Launched run_pp_headline_checks.sh (GPU 2/3): PP random (16,1) p=.11 n50 + PP oracle
  (16,1) seed-1 replication n50. Marker PP_HEADLINE_CHECKS_DONE.
- robocasa_label_report_latex updated (10 pp): full 8-row oracle table, recalib results.
- AAAI paper filled: abstract/intro/results/conclusion/method pendings replaced with
  final numbers; new oracle subsection + table; honest learned-switching-within-noise
  subsection; sequence grid; taxonomy rewritten on rollout states. 5 pp, 2 intentional
  pendings (PP random control).
- LIBERO: backfill DONE. K3 predictor cells: vis (16,4) .56 / (16,1) .42, prop .30/.10,
  vj .36/.40 vs fixed_16 .72 -> switching hurts. K4: vj .54/.62, vis .64/.58, prop
  .58/.38 vs fixed_8 .84 -> hurts. LR2 both tasks 0.0 everywhere; K6 dead (0.0 at
  selection + fixed 16/8/4), chain on K6 fixed_1.

## 2026-07-28 ~10:15 UTC - Non-RoboCasa report written
- New doc libero_mimicgen_report_latex/main.pdf (2 pp): LIBERO K3/K4/K6/LR2 grids
  (fixed 16/8/4/1 + vjepa/vis/prop heads at (16,4),(16,1)) and MimicGen nut/coffee
  grids. Verdict: learned switching never beats best fixed k off-RoboCasa (K3 .56 vs
  .72, K4 .64 vs .84, nut .46 vs .52, coffee .16 vs .24); K3 prop (16,1) .10 ~= fixed_1
  .08 shows false-positive cost in closed-loop unstable regime; K6 + both LR2 dead.
  Sent to user.

## 2026-07-28 ~10:45 UTC - PP random control FINAL: attribution confirmed
- PickPlaceCounterToCabinet_random_16-1_p11 FINAL .62 (31/50). Oracle .80 vs random .62
  vs fixed_16 .67: the PP gain is timing-specific (interim .71 at n=28 regressed, again).
- Report updated (11 pp) + AAAI paper now has ZERO \pending markers (5 pp).
- Still running: PP oracle replication s1 (6/11), PP oracle recalib 16-1 (2/3, slow),
  PP oracle 8-1 (0/3), OC recalib head 16-1 (20/34), OC pred 8-1 (27/39); queued: OC
  recalib oracle 16-1, PP contact 16-1, OD pred 8-1, TOSF pred 8-1. LIBERO chain on
  LIVING_ROOM_SCENE5 selection.

## 2026-07-28 ~10:55 UTC - Missing k-combination cells armed on all platforms
- MimicGen (8,1) full-head cells running on GPU 2 (coffee first, then nut);
  (8,4) contrast cells armed behind them (marker MIMICGEN_81_DONE -> _84_DONE).
- LIBERO (8,1) cells armed: K4 vj/vis/prop grabs GPU 3 after PP replication json;
  K3 vj/vis/prop grabs GPU 7 after TOSF_pred_8-1 json (marker LIBERO_81_DONE).
- With these plus the RoboCasa ku1 family, every platform has (16,1) and (8,1)
  coverage; MimicGen additionally gets the (8,4)-vs-(8,1) drop-size contrast.

## 2026-07-28 ~11:05 UTC - Evidence videos armed; AAAI methodology/results tables added
- groot_oracle_rollout.py patched with --video-dir (mirrors switch script; final
  reset finalizes last mp4). run_evidence_videos.sh armed on GPU 6 behind PP oracle
  8-1: PP shadow16 n8 + PP oracle 16-1 n8 + TOSF shadow16 n8 + TOSF oracle 16-1 n8
  + OD pred 16-1 n6, all with video (marker EVIDENCE_VIDEOS_DONE). Composer to be
  written when footage lands: rescue side-by-sides (same episode index = same scene,
  verify before claiming), left-alone case, OD fires-but-fails honesty case.
- AAAI paper: tab:headacc (head validation AUROC/MAE/base rates) in experiments,
  tab:tiny (oracle vs fixed 16 vs best fixed with steps-at-k1 and call overhead) +
  tiny-interventions paragraph in results. 5 pp, sent.

## 2026-07-28 ~11:20 UTC - ACT third-policy-class program launched
- New impl/robocasa/act/: render_act_dataset.py (state-replay -> JPEG 224 agentview
  + 9-dim proprio + 12-dim actions per step), act_model.py (faithful compact ACT:
  CVAE z=32, d512/8h/ff3200, enc4/dec7, resnet18, chunk 16), train_act.py (60k steps,
  batch 48, lr 1e-5, KL 10, norm stats in ckpt), act_rollout.py (fixed k and oracle
  16-1 with measure_lambda ported from groot_oracle_rollout; episode inits seeded by
  index so fixed and oracle share scenes).
- run_act_pipeline.sh RUNNING (log results/robocasa/act_pipeline.log): renders 4 tasks
  co-located on GPU 4; train+eval OC,PP on GPU 3 after LIBERO K4 8-1 cells; OD,TOSF on
  GPU 7 after K3 cells. Eval per task: fixed 16/8/4/1 + oracle (16,1) at label delta,
  n=50. Markers RENDER_DONE / TRAIN_DONE / CELL_DONE / ACT_PIPELINE_DONE.
- Rationale: same dataset + same labels + same oracle machinery, different policy
  class -> tests policy-independence of the stability mechanism per user request.

## 2026-07-29 ~09:40 UTC — overnight finals: replication holds (.74), ku1 family complete, ACT fixed and relaunched
- PP oracle (16,1) seed 1 FINAL: .74 (fired .15). Pooled with seed 0 (.80): 77/100 = .77 over n=100. Headline promoted from one-run to pooled status: oracle .77 vs fixed_16 .67, best fixed .78, random .62, (16,4) .64.
- New ku1 finals (all n=50): PP oracle (8,1) .50 fired .12 (vs fixed_8 .78 — switching from base 8 hurts; oracle value is rescuing LONG chunks); PP contact (16,1) .64 fired .87; PP recalib oracle (16,1) .70 fired .68; OC recalib head (16,1) .62 fired .48; OC recalib oracle (16,1) .58 fired .13. KU1_COMPLETION_DONE.
- TOSF oracle (16,4) FINAL .52: monotone content gradient on second task — fixed .44 < (16,4) .52 < (16,1) .62 at same measured moments. TOSF_164_DONE.
- Learned (8,1) finals: OC .60, OD .54, TOSF .38. Learned switching still within noise of its base fixed chunk.
- MimicGen finals: coffee (8,1) .06, (8,4) .24 (ties best fixed); nut (8,1) .42 (= fixed_8, below fixed_1 .52), (8,4) .34. MIMICGEN_84_DONE.
- LIBERO (8,1) finals: K4 vj .82 (ties best fixed .84 — best switching cell yet off-RoboCasa), vis .72, prop .58; K3 vj .20, vis .24, prop .10 (K3 closed-loop unstable, false positives expensive). LIBERO_81_DONE.
- Evidence video cells DONE (PP/TOSF shadow16 + oracle161, OD pred161, per-episode mp4s + replan logs). Composer is the next work item.
- ACT: renders OC/PP/TOSF done (500 demos each ~2.5h); OD render died on robocasa scene-init flake (Ran _load_model() 50 times) — renderer retry-patched (3 attempts, env rebuild, skip on persistent failure), rerunning on GPU 2. Trains OC/PP done (60k steps, L1 .086/.095). ALL 10 eval cells crashed instantly: arm_pos_index() before first env.reset() (composite_controller None) — fixed with lazy p0 after first reset; relaunched via impl/robocasa/act/run_act_fix.sh: GPU 3 OC evals -> PP evals, GPU 7 TOSF train -> evals, GPU 2 OD render -> train -> evals.
- LIBERO main chain still on LIVING_ROOM_SCENE5 (all cells 0.0 so far, GPUs 0/1).

## 2026-07-29 ~10:15 UTC — evidence videos composed and sent; all docs updated with overnight finals
- Evidence videos: same-scene pairing assumption FALSE (verified on first frames — episode inits differ across runs); side-by-sides annotated as separate episodes. 7 composed videos sent (impl/robocasa/compose_evidence_videos.py -> results/robocasa/evidence_videos/composed/): PP_rescue_pair_ep1 (1 fire t=160 rescue), PP_rescue_pair_ep4 (1 fire t=128), TOSF_rescue_pair_ep1 (5 fires at lever), PP_leftalone_ep3 + TOSF_leftalone_ep7 (0 fires, success), OD_highfire_fail_ep0 (78/120 fired, fail), PP_highfire_fail_ep7 (57/101 fired, fail).
- Docs folded and compiling clean: robocasa report (11pp) — pooled PP oracle .77 n=100 headline, 13-row tab:oracle, TOSF dose response .44/.52/.62, PP (8,1) .50 boundary, contact (16,1), recalib (16,1) family, learned (8,1) column, tiny-table pooled stats (1.5% steps, 1.25x); AAAI (5pp) — abstract/intro softened to "matches or beats every fixed k" with pooled .77, results tables + text updated; LIBERO/MimicGen report (2pp) — (8,1)/(8,4) columns added, K4 vjepa (8,1) .82 ties best fixed, Reading section rewritten.
- ACT relaunch healthy: fix verified (ep 0 ran full horizon, ~66 s/episode); OC evals on GPU 3, TOSF training on GPU 7, OD render retry on GPU 2.

## 2026-07-29 ~18:50 UTC - ACT zero-rate root cause: LeRobot->env action-order mismatch (affects offline labels too)

The ACT recovery went through three diagnosis layers today. (1) The
composite-controller crash was fixed with a lazy arm-index. (2) OpenCabinet
fixed_16 = 0.0 was blamed on dim 11 as a discrete mode flag and a sign-snap
patch was added; that diagnosis was wrong in an instructive way. (3) The
OpenDrawer grid then came back 0.0 on every fixed k including k=1 with
healthy training loss, and frame grids showed the base still wandering out
of the kitchen. The controller's action split settles it: the env consumes
[arm6, grip1, base3, torso1, mode1], but lerobot_to_hdf5.py packs the
LeRobot parquet action column as [base3, torso1, mode1, arm6, grip1]. Every
consumer of rc_*.hdf5 actions fed scrambled vectors: arm commands landed on
the base dims, and the demo gripper bit acted as the base-mode switch.
Proof: raw demo replay ends ~1 m off target and fails; remapped replay
SUCCEEDS with 4-7 cm final eef error (OpenCabinet demos 1-2).

Scope. Poisoned: all v2 offline labels (final2_ol_*), the per-task auto
deltas derived from them, demo-label analyses (regime structure, contact
alignment, failure localization), the learned head's training targets, and
the demo-to-rollout shift finding (the "shift" may largely be this
artifact; old unstable ~= gripper-closed segments, since the gripper bit
gated the base). Survives: every online oracle result (GR00T rollouts
replay the policy's own env-ordered actions; their .5-.8 fixed rates prove
the ordering), all LIBERO/MimicGen results, GR00T fixed cells, ACT
checkpoints (training is self-consistent in demo order; remap applied at
the env boundary).

Actions taken: killed the OD oracle cell (scrambled protocol, pure waste);
old ACT stream ended cleanly; deleted the four invalid OpenDrawer jsons;
patched act_rollout.py (snap gripper+mode, remap in chunk),
ftle_labeler_robocasa.py (remap at load), diag_act.py (remap part A);
launched labeling v3 (final3_env_*, same params as v2, CPU with load gate);
staged run_act_fix2.sh (four parallel ACT fixed grids on GPUs 2/3/4/7,
oracle cells held until v3 deltas exist); OC+OD diag reruns and a
four-task remapped-replay validation are running as launch gates.

## 2026-07-29 ~19:15 UTC - remap gates passed, corrected ACT grids running

Gate results: remapped demo replay 12/12 SUCCESS across all four tasks;
diag reruns show ACT now succeeds from demo inits (OpenCabinet 4/5,
OpenDrawer 2/5) with the base staying put (verified in frame grids).
run_act_fix2.sh launched 18:48 on GPUs 2/3/4/7 (fixed 16/8/4/1, n=50 each;
oracle held for v3 deltas). Labeling v3 on OpenCabinet since 18:44; v2
timings imply OC labels ~23:35, full set ~09:30 tomorrow. LIBERO GPUs 0/1
identified: KITCHEN_SCENE8 moka pots e1000 selection cell and
LIVING_ROOM_SCENE6 (K6) prop cell, both healthy.

## 2026-07-30 ~11:25 UTC — VM reboot wiped /mnt/scratch; RoboCasa stack rebuild launched
- The VM restarted ~10:08 UTC (this, not the harness, is what killed last night's three
  background chains). /mnt/scratch is the Azure ephemeral disk and was wiped
  (EPHEMERAL_DISK_DATALOSS_WARNING.txt). The @reboot recovery started
  impl/mimicgen/rebuild_scratch.sh (LIBERO/MimicGen half, in progress),
  sync_scratch_artifacts.sh, and the STALE v2 robocasa labeling script (no
  action-order remap) — killed the v2 labeler.
- Lost: rc + groot155 envs, Isaac-GR00T(-rc365) + robocasa365 repos, kitchen assets,
  GR00T checkpoint-120000, lerobot data, converted rc_*.hdf5, partial v3 OC labels.
  Survived on the share: all results/, artifacts/labels (final2), ACT checkpoints
  (results/training/act_rc_*), all scripts.
- Fork cells died at PP ep 8/50, TOSF ep 7/50 (logs only, no JSON). ACT grids died with
  fixed_16+fixed_8 complete (all 4 tasks) + TOSF fixed_4; missing fixed_4 x3, fixed_1 x4.
- New impl/robocasa/rebuild_scratch_robocasa.sh (idempotent, detached via setsid):
  rc venv -> assets -> Isaac-GR00T + robocasa365@be22d659 + robocasa-benchmark fork ->
  ckpt-120000 (HF) -> lerobot data x4 -> setup_groot155.sh -> lerobot_to_hdf5 x4 ->
  relaunches run_act_fix2.sh + run_groot_fork.sh + run_robocasa_relabeling.sh.
  Stale ASSETS_OK/CKPT_OK/GROOT155_OK markers removed. Log: results/robocasa/rebuild_robocasa.log.
- Episode-level stratification recomputed from surviving JSONs for the user's
  open-loop-unstable/closed-loop-stable question (flag = any replan lambda>delta; oracle
  16->1 vs shadow_16): TOSF flagged .73 (19/26) vs .45 (5/11), unflagged .50 vs .46 —
  entire win inside flagged episodes. OC flagged .50 vs .53 (null). PP not localizable
  (shadow flags 50/50 episodes). OD oracle run only n=15.

## 2026-07-30 ~11:55 UTC — probe-noise (sigma) robustness sweep queued (user request)
- New impl/robocasa/run_sigma_sweep_labeling.sh (detached, waiting): after the four v3
  (sigma 0.05) label files land, rerun the labeler at sigma_u 0.025 and 0.100 on all
  four tasks (n-demos 200, k 24, seed 0 -> identical perturbation directions across
  sigma, magnitudes pair per stamp). Outputs labels/final3_env_sig{025,100}_rc_*.npz.
- Purpose: show the lambda ranking / regime split is sigma-invariant (linear regime),
  strengthening the "segments differ in open-loop stability" claim. Planned analysis:
  per-stamp Spearman rank corr across sigma pairs + flag agreement at per-sigma
  recalibrated auto-delta. Old final2_ol_rc_TurnOnSinkFaucet_sig030.npz is v2-era
  (action-order bug) and must not be used.
- Rebuild status at queue time: REPOS_OK 11:45, RC_CKPT_OK 11:46, RC_DATA_OK 11:49;
  groot155 env build + hdf5 conversion remaining, chains auto-relaunch after.

## 2026-07-30 ~12:40 UTC — sequence-history predictor experiment launched; GPU chains recovered
- Asset gap fixed: robocasa365 needed the lightwheel/objaverse/aigen packs; downloaded
  into the shared cache (classic repo assets dir) and recursively symlink-merged
  (cp -rsn) into robocasa365 (git-tracked real dirs defeated top-level symlinks twice:
  Window050, then CabinetDoorPanel024). Fork cells (GPUs 5/6) and ACT grid remainder
  (2/3/4/7) relaunched 12:33 and confirmed alive.
- NEW seq-predictor experiment (user request): train the stability head on a HISTORY
  of 4 causal V-JEPA clips (t, t-16, t-32, t-48 = ~3.2 s) vs matched 1-clip baseline,
  both on v3 labels (features feat2_rc_* survived and are frame-only, bug-free; v2
  lambda inside them ignored; joined to final3 labels on (demo_id,t)).
  - impl/predictor/train_head_seq.py (SeqHead: per-clip pos emb + attentive probe +
    GRU proprio; metrics incl. tail recall/precision = open-loop-unstable stratum)
  - impl/eval/predictor_bridge_seq.py (slices a 64-frame buffer into 4 clips)
  - groot_switch_rollout.py: +--bridge, +--frames-window (scratchpad-edit, cat back)
  - impl/robocasa/run_seq_predictor.sh: per task waits for v3 labels -> trains
    h4+h1 x seeds 0-2 -> online (16,1) n=50 cells per head on GPUs 0/1.
    Outputs: results/predictor/rc_seq_v3/, results/robocasa/seqpred/.
- Labeling: three campaigns running in parallel (v3 sigma .05 + sweep .025/.100).

## 2026-07-30 ~12:50 UTC — full-parallel restructure (user: "why not parallel?")
- Labeling: killed the 3-way sequential campaigns; run_v3_parallel.sh now labels ALL
  FOUR tasks concurrently (4x24 workers, whole CPU pool). Sigma sweep auto-launches
  after the last v3 file. All four v3 labels ETA ~17:30-18:00 (was ~next morning).
- New oracle success cells on the idle GPUs 0/1 (user priority: oracle task-success
  over waiting for predictor): OpenDrawer oracle (16,1) n=50 full (replaces the n=15
  partial) on GPU 0; TurnOnSinkFaucet oracle (16,1) SEED 1 on GPU 1 (replication of
  the headline win against the noise critique). ACT grids NOT paused (0/1 were free).
- All 8 GPUs busy: 0 OD-oracle, 1 TOSF-oracle-s1, 2/3/4/7 ACT grid, 5/6 fork cells.
  Seq-predictor chain armed; will share/queue on 0/1 when v3 labels land.

## 2026-07-30 ~14:00 UTC — ICLR additions: proxy comparison + failure-onset analysis
- User reframe accepted: (a) measure what competing replan-trigger proxies actually
  track vs lambda at the SAME states; (b) failure-vs-divergence correlation without
  intervention. Method names user cited (DVAC/PACE/SGAC/DEHP) not verified against
  memory; proxies implemented as described (sample spread / speed minima / chunk
  consistency); learned-horizon-schedule needs that method's code — flagged.
- NEW impl/robocasa/groot_proxy_shadow.py: shadow rollout measuring per replan from
  the same state: lam (oracle), spread (3 extra chunk samples, pairwise L2, first 8
  steps), speed (eef), cons (mid-chunk fresh query vs executing chunk tail).
  run_proxy_shadow.sh: 4 cells n=50, each claims its GPU (2 OD, 3 OC, 4 PP, 7 TOSF)
  after that GPU's ACT stream logs ACT_FIXED_DONE. Launched detached.
- Failure-onset analysis (existing shadow cells, complete data): predicted pattern
  holds ONLY on OpenDrawer (failed eps 2.1x unstable exposure; P(fail|unstable
  replan) .83 vs .67). OC/PP/TOSF: absent or REVERSED (successful eps hit MORE
  unstable replans; no late-lambda concentration anywhere, rank ~.56). Reading:
  lambda-firing conditions on task ENGAGEMENT (must reach contact to be unstable;
  engagement correlates with success) — naive episode-level correlation confounded;
  fork (conditions on the fired moment, counterfactual forward) is the correct
  instrument. PP uninterpretable under v2 delta (fires 52%).

## 2026-07-30 ~15:15 UTC — ACT fixed-chunk grid COMPLETE (16/16, post action-order fix)
- ACT n=50 grid: OC .58/.64/.52/.32, OD .30/.32/.04/.10, PP .12/.12/.04/.10,
  TOSF .50/.40/.34/.36 (k=16/8/4/1). Notes: PP+OD collapse at k=4 (.04) then
  partially recover at k=1 (.10) — nonmonotonic; TOSF fixed_1 .36 vs GR00T's .04
  (ACT tolerates closed-loop where GR00T does not — policy-class contrast for the
  heterogeneity story). GPUs 2/3/4/7 now hold only the proxy cells.
- Still running: fork PP ep14/TOSF ep16; oracle OD ep9 (slow), TOSF-s1 ep22;
  proxy cells ep~5-8; v3 labels not yet landed (4-parallel since 12:47);
  MDS mg_coffee h4_s0 2h16 in, CPU-active.

## 2026-07-31 ~09:00 UTC — OVERNIGHT RESULTS: fork NULL, sigma sweep strong, proxies do not track lambda
### 1. Paired fork (the causal test) — NULL on both tasks
- PP: 40/50 episodes fired; oracle-acts .625 vs oracle-ignored .625; pairs n11=21 n10=4
  n01=4 n00=11; McNemar exact p=1.000. TOSF: 22/50 fired; .455 vs .455; n10=n01=0 (zero
  discordant pairs); p=1.000. Combined 4 vs 4 discordant across 62 forked episodes.
- Mechanism verified sound: in both-succeed episodes the branches diverge strongly in
  steps-after-fork (e.g. 115 vs 21, 265 vs 46); identical step counts occur ONLY in
  both-fail episodes that exhaust the budget. Paired sign test on steps-to-success also
  null (TOSF 6/4 p=.75; PP 11/7 p=.48), with oracle slightly SLOWER on average.
- Conclusion: acting on perfect stability detection does not causally rescue episodes at
  the fired moments. Supersedes the episode-level stratification (engagement confound).
### 2. Oracle cells — the headline win does not replicate
- TOSF oracle (16,1): seed0 .62, seed1 .50 (mean .56 vs fixed16 .42 / shadow16 .46).
- OpenDrawer oracle (16,1) full n=50 = .62 (the old n=15 partial said .40 — that partial
  was misleading; fixed16 .64, so still a null, not a loss).
### 3. Sigma robustness sweep — STRONG (claim 1)
- Spearman across probe magnitudes (0.025/0.05/0.10): .95-.98 adjacent, .82-.90 extreme
  (4x range); flag agreement .97-.99 on all four tasks. Regime structure is a property of
  the dynamics, not of the probe size.
### 4. v3 vs v2 labels + delta acceptance
- Spearman(v2,v3) only +.14/+.14/-.03/+.07 -> v2 labels were near-uncorrelated with truth
  (PP flag agreement .50 = coin flip). Redo was necessary.
- v3 auto-delta: OC .0848, OD .0760, PP .0649, TOSF .0579. Acceptance vs ONLINE lambda
  distribution: OC p97.1 PASS, PP p96.7 PASS, TOSF p95.6 PASS, OD p89.1 marginal.
### 5. Proxy comparison (ICLR addition) — proxies do NOT track lambda
- Per-replan, same states, n=1370-2339 per task. DVAC-style sample spread: rho -.07..+.10,
  AUC .61-.76 (weak tail signal, no graded tracking). SGAC-style chunk cosine: rho -.17..0,
  AUC .40-.62 (none). Chunk L2 disagreement: AUC .56-.72. PACE-style speed valley:
  rho(oriented) -.15..-.21 i.e. ANTI-correlated — commanded speed correlates POSITIVELY
  with lambda (rho ~+.2), so PACE's low-speed trigger fires at the more STABLE moments;
  AUC .31-.69, inconsistent. eef speed: rho(speed,lam) ~ +.28..+.50.
### 6. Sequence-history predictor
- Offline on v3 labels (3 seeds, held-out demos): AUC(confident) h1 .885/.863/.895/.902,
  h4 .858/.838/.887/.895 (OC/OD/PP/TOSF). History does NOT help; h1 >= h4 everywhere on
  RoboCasa (MimicGen coffee marginally favours h4 .953 vs .939, nut favours h1).
  Claim 2 now holds on RoboCasa with clean labels.
- ONLINE cells from the first pass are INVALID: fired 0.00, mean_k 16.0 on all 8 — the
  head fires on (lambda_hat > delta_label) but regression compression caps lambda_hat
  (e.g. TOSF max .064 vs delta .058, only .12% of val over delta). Ranking fine, operating
  point broken. This is the predicted under-firing failure mode, now measured.
- FIX: impl/predictor/calibrate_seq_head.py computes a quantile-matched threshold on the
  held-out split; groot_switch_rollout.py gains --fire-on {lam,prob}; bridge prints p at
  8dp. Calibrated operating points (recall/precision at matched fire rate): OC h1 .33/.33
  (13.2x lift), OD h1 .18/.18, PP h1 .25/.25, TOSF h1 .20/.20; h4 uniformly worse.
  8 recalibrated cells relaunched on 8 GPUs -> results/robocasa/seqpred_calib/.

## 2026-07-31 ~09:30 UTC — predictor operating point: offline threshold does NOT transfer online
- Pass 2 (offline/demo-calibrated quantile threshold) launched then killed: online fire
  rates came out 2.1%-28.9% against a ~4% target (TOSF worst: 28.9% in the shadow log,
  59% in live ep 0 once switching shifted the state distribution further). Over-firing is
  the harmful mode (TOSF pure k=1 = .04), so those cells would have measured a calibration
  artifact, not the signal.
- Pass 1's never-firing cells turn out to be exactly what was needed: pure ONLINE shadow
  logs of lambda_hat (mean_k 16.0, 1223-2324 replans per cell). Online lambda_hat sits far
  below the offline scale (TOSF p90 .038 vs offline delta .058) and differs per task.
- Pass 3 (running, 8 GPUs): threshold = quantile of the PASS-1 ONLINE lambda_hat matching
  the ORACLE's measured online fire rate per task (OC .06, OD .14, PP .11, TOSF .12), so
  the predictor gets the same intervention budget as the oracle. Thresholds saved to
  results/predictor/rc_seq_v3/online_thresholds.json; verified to reproduce the target
  rate on the shadow logs to 0.1%.
- Finding in its own right for the paper: a stability head calibrated on demonstrations
  mis-fires badly when deployed on the policy's own state distribution. Threshold transfer
  is a distinct failure from ranking quality (AUC .86-.90 throughout).

## 2026-07-31 ~10:15 UTC — TRAINED-PREDICTOR switching on GR00T+RoboCasa (7/8 cells)
Online-calibrated thresholds (quantile of pass-1 online lambda_hat matched to the
oracle's per-task fire rate). n=50 per cell, 16->1.
  task        vanilla16  fixed1  oracle(mean)  pred_h1 (fire)  pred_h4 (fire)
  OpenCabinet      .52     .48       .60        .54 (.16)       running
  OpenDrawer       .64     .40       .62        .60 (.34)       .62 (.26)
  PickPlace        .58     .52       .77        .60 (.38)       .66 (.14)
  TurnOnSinkFaucet .42     .04       .56        .54 (.15)       .60 (.14)
  mean             .54               .64        +.030 vs vanilla   +.080 (3 tasks)
- h4 BEATS h1 online on PP (+.06) and TOSF (+.06), reversing the offline ranking where
  h1 won every task. Firing rates explain it: h1 fired .15-.38, h4 .14-.26; the h1 cells
  with the highest fire rates (PP .38, OD .34) are the ones that gain least/lose. Online
  advantage looks like calibration, not detection quality.
- All predictor gains are within/near the +-.10 noise floor except TOSF h4 (+.18).
  Consistent with the fork null: no cell approaches the oracle's PP .77.
### Closed-loop probe (groot_cl_probe.py) RUNNING, 4 tasks, GPUs 2/4/6/7
- Corrected design: nominal = policy closed-loop UNPERTURBED (not demo replay), plus a
  sampling-floor control (2 extra unperturbed CL rollouts).
- First data confirms the control was necessary: lam_ctrl frequently EXCEEDS lam_cl
  (e.g. .1427 vs .1264, .1195 vs .0648, .1282 vs .0411) — GR00T's diffusion sampling
  alone diverges as fast as an injected perturbation, so lam_cl is only interpretable
  as lam_cl - lam_ctrl. The old robomimic labels_cl_* have no such control AND use a
  demo-replay nominal; the 3x3 built from them was retracted.
- Pace ~25s/replan, ~5.5 min/episode on PP; full set overnight.

## 2026-07-31 ~11:10 UTC — ICLR paper restructured to the three-claim spine
- sections/{abstract,introduction,related_work,method,experiments,results,conclusion}.tex
  all rewritten. New spine: Claim 1 both regimes inside one task (strong evidence),
  Claim 2 regime is labelable and predictable (strong), Claim 3 predicted regime sets
  chunk length at inference (incomplete, so far negative). Premise section added:
  compounding error in imitation learning (Ross and Bagnell) + finite-time maximal
  Lyapunov exponent (Haller; Krishna et al.) as the measured quantity.
- Removed all oracle-only tables. Oracle now appears only as a column beside vanilla,
  always-k=1, best-fixed-k (with the k value), predictor h1 and h4.
- New tables: main controller comparison, paired fork (McNemar), predictor detection
  h1 vs h4, competing-proxy audit (sec:proxies), v3 labels with both-regime columns.
- refs.bib: all 5 Anonymous placeholders replaced with verified author lists fetched
  from arXiv/Semantic Scholar (AutoHorizon = Wang et al. 2602.21445; HiPolicy = Zhang
  et al. 2604.06067; DVAC = Feng et al. 2606.03847; MDPI = Wen et al. Biomimetics
  11(5):316 via S2 DOI API; DEHP/PACE/SGAC/BID from the user's verified bib). Added
  ross2010efficient, ross2011dagger, haller2015lcs, so2025sgac, feng2026dvac.
- Compiles clean: 11 pages, no errors, no undefined refs or citations, no overfull
  boxes, zero em dashes. Stale v2-label table (tab:headacc) deleted from experiments.

## 2026-07-31 ~13:50 UTC — composed-task heads were single-task AND stage-privileged
- User caught it: the composed task (OpenCabinet -> PickPlace) used STAGE-MATCHED
  single-task heads (S_OpenCabinet_* in stage 1, S_PickPlace_* in stage 2). Two problems:
  (a) each head saw only its own task's data; (b) the stage switch is driven by the env's
  own kenv.cab.is_open predicate, so selecting the head by stage feeds the controller
  privileged ground truth that only the predictor cells benefit from. The oracle cells
  had the milder version of the same issue (per-stage delta).
- NOTE: the LANGUAGE instruction switch also uses cab.is_open, but that defines the task
  and applies identically to fixed/oracle/predictor cells, so it does not bias the
  comparison. Head selection does. Keep the distinction in the paper.
- FIX: impl/predictor/train_head_seq_pool.py trains ONE head over all four kitchen tasks.
  Pooling detail: per-task auto-delta spans .058-.085 so raw lambda is not comparable;
  the target is lambda/delta_task, putting every task's boundary at 1.0 and giving the
  pooled head a single threshold. Per-task AUROC on the pooled val split is reported.
  run_pooled_heads.sh training POOL_h1_s0 (gpu 3) and POOL_h4_s0 (gpu 5).
- Composed-task cells now running with checkpointing (oracle ~20 min/episode on the
  1800-step horizon; h1/h4 ~2 min/episode). These will be re-run with the pooled head
  once it trains, and the stage-matched cells kept only as an ablation.

## 2026-07-31 ~14:10 UTC — full 24-dataset label table + regime timelines added to paper
- New Table 2 (tab:alllabels): every labeled dataset, grouped by platform, with stamps,
  auto-delta, deadband share, unstable share, % episodes containing both regimes, median
  crossings per episode, median unstable run length. 24 datasets.
- IMPORTANT and now stated in the caption: the STABLE share is ~2.5% on every dataset by
  construction, because auto_delta = P95(|lambda| : lambda < median) puts 2.5% of all
  stamps below -delta regardless of the data. That column carries no information and is
  omitted; the informative contrast is deadband vs unstable. Do not report a three-way
  split from this rule without saying so.
- New Figure (fig:timelines): per-episode regime strings for 8 demonstrations across 4
  datasets, showing the two shapes: one or two short unstable islands (RoboCasa kitchen)
  vs alternation throughout (LIBERO K8, tool_hang) which is the shorter-fixed-chunk case.
- Closed-loop row in the table marked \pending "still pending".
- Contact alignment on v3 corrected: our contact flag is gripper-vs-MOVABLE-body, so it
  reads 0% on OpenCabinet/OpenDrawer/TurnOnSinkFaucet (articulated fixtures, not free
  bodies). Only PickPlace is testable and gives a 2.4x lift. The broad "instability
  concentrates at contact" claim needs a fixture-aware contact definition first.

## 2026-07-31 ~15:10 UTC — 3x3 open-loop x closed-loop table, 3 of 4 tasks complete
Design: OL axis = lam_ol; CL axis = lam_cl MINUS lam_ctrl (growth beyond the policy's own
sampling floor); SAME per-task v3 delta on both axes so the two ask the same question on
the same scale. n=669-683 paired probes per task, 25 episodes each.
  POOLED (3 tasks, 2029 probes)   CL stable  CL deadband  CL unstable   row
    OL stable                          2.1%        8.8%        0.2%   11.0%
    OL deadband                        8.3%       41.9%        2.3%   52.5%
    OL unstable                        9.1%       26.3%        1.1%   36.5%
    col                               19.4%       77.0%        3.6%
- KEY CELL (open-loop unstable AND closed-loop stable) = 9.1% of stamps, i.e. 25% of all
  open-loop-unstable stamps. Per task: TOSF 14.9%, PP 8.4%, OD 3.8%.
- Closed-loop UNSTABLE is rare everywhere (1.1% pooled, 0.7-1.6% per task): once the
  policy replans every step, the injected perturbation almost never outgrows sampling.
- The sampling-floor control was essential: at 65-67% of stamps on EVERY task the
  perturbed closed-loop branch grows SLOWER than two unperturbed rollouts do. Without
  that control the closed-loop axis is uninterpretable, which is what invalidated the old
  robomimic labels_cl_* (demo-replay reference, no control).
- OL-unstable share is far higher online (24-49%) than in the offline demo labels
  (2-5%), consistent with the demo-to-rollout shift.
- IMPLICATION for claim 3: even a PERFECT open-loop detector spends ~75% of its
  interventions on moments where closing the loop cannot help, because our predictors are
  trained on the open-loop axis alone. That is a quantitative explanation of the fork null
  that does not require the stability signal to be wrong, only incomplete. Motivates a
  head trained on the CONJUNCTION of both axes.
- OpenCabinet at 17/20, pooled table to be recomputed at 4/4.

## 2026-07-31 ~15:40 UTC — breadth fixed-k cells were 20x slower than needed (fixed)
- run_claim3_breadth.sh ran the plain fixed-k baselines through groot_oracle_rollout with
  --control oracle --delta 99. That disables switching but still calls measure_lambda at
  EVERY replan. At k=1 on a 450-step horizon that is 450 replans x 144 env steps of
  probing per episode: CloseDrawer_fixed1 reached only 3/50 episodes in 67 minutes,
  extrapolating to ~18 hours for a cell that baselines a ~1 hour run.
- FIX: fixed8/fixed1 now use --control contact (cheap gripper-contact flag, no lambda).
  With k_stable == k_unstable the chunk length is constant either way, so behaviour is
  identical and only the wasted probing is removed. shadow16 keeps --control oracle
  because it exists to produce the online lambda distribution.
- Killed and requeued the 4 affected cells. NOTE for future work: the original ksweep
  used groot_first_rollout.py with --n-envs 5 (5 parallel envs), which is faster still;
  worth switching to if more fixed-k baselines are needed at scale.

## 2026-07-31 ~19:20 UTC — four major result sets landed
### 3x3 COMPLETE, all four tasks (2925 paired probes)
  POOLED            CL stable  CL deadband  CL unstable   row
  OL stable              1.7%        8.7%        0.2%   10.6%
  OL deadband            7.8%       47.4%        2.1%   57.3%
  OL unstable            7.0%       24.1%        1.0%   32.1%
  col                   16.4%       80.2%        3.3%
- KEY cell 7.0% of stamps = 22% of open-loop-unstable stamps. Per task it tracks the
  task's own instability: OC 11%, OD 16%, PP 23%, TOSF 31% of OL-unstable are CL-stable.
- CL-unstable stays rare (1.0-5.6%). Sampling floor exceeds the perturbed rate at ~66%.
### Literature baselines COMPLETE (16 cells, matched firing budget)
  means: vanilla .540 | oracle .637 | h4 .620 | SGAC-cos .615 | DVAC-spread .585 |
         BID-cons .585 | h1 .570 | PACE-speed .555
- Everything except the oracle clusters within ~6 points, i.e. inside the noise band.
- CAVEAT that must go in the paper: the consistency signals systematically UNDER-fire
  (cons .00-.09, cons_cos .02-.08 against targets .06-.14). OpenCabinet_cons fired 0.00,
  so that cell is literally fixed-16. Their means are achieved with less intervention
  than ours, so the comparison is not yet budget-matched in practice despite being
  budget-matched by construction. Recalibrate or report achieved rates alongside.
### Composed task: the FIRST clear win over the best fixed chunk
  fixed 16/8/4/1 = .38/.44/.22/.10 ; h1 .48 ; h4 .48 ; oracle .54 (28/50, partial)
  stage-1 rates: fixed8 .68 ; h4 .78 ; oracle .79
- h1 and h4 both beat the best constant (.48 vs .44) and the oracle is higher still.
  This is the only setting so far where switching clearly wins, consistent with the
  hypothesis that a task with a built-in regime change is where it should.
- CAVEAT: these use STAGE-MATCHED heads, i.e. the privileged stage signal. The
  pooled-head cells are the honest version and are queued.
### Pooling costs per-task accuracy
  POOL_h1 AUROC(conf) .887 overall but per-task .69/.79/.74/.80 vs single-task
  .885/.863/.895/.902. POOL_h4 similar. Removing the stage privilege is not free.
### Seed replication (8 of 32) confirms the noise problem is severe
  PP fixed16 .58->.62, PP h1 .60->.70, PP oracle .80/.74->.60
  TOSF fixed16 .42->.54, TOSF h1 .54->.66, TOSF oracle .62/.50->.48
- The vanilla baseline itself moves 12 points between seeds on TOSF, and the PP oracle
  moves 20. No single-seed cell in Table 5 should be quoted without this spread.

## 2026-07-31 ~19:40 UTC — all four result sets written into the ICLR paper
- Claim 2 gains Table 6 (tab:threebythree): the 3x3, replacing the pending closed-loop
  subsection. Text states the three readings (key cell 7.0% = 22% of OL-unstable; CL
  instability rare; the sampling-floor control was necessary) and the ceiling argument:
  an open-loop-only detector wastes ~78% of interventions, which predicts the weak
  intervention results without the signal being wrong.
- Claim 2 gains Table 7 (tab:baselines): the reimplemented literature signals as
  controllers. Reports ACHIEVED firing rates beside targets and flags in the caption that
  the two consistency signals under-fired (SGAC .02-.08, BID .00-.09 vs .06-.14 targets;
  BID on OpenCabinet fired 0.00 = fixed-16 in disguise), so those rows are not yet
  like-for-like. \pending to recalibrate against live rates and rerun.
- Claim 3 gains Table 9 (tab:composed): the composed task, the ONLY setting where
  switching beats the best constant chunk (h1/h4 .48, oracle .54 vs best fixed .44;
  stage-1 .78/.79 vs .68). Caveats in caption: oracle partial at 28/50, and the learned
  cells use the privileged stage predicate; pooled-head replacement pending.
- Claim 3 gains Table 10 (tab:seeds): seed spread. Vanilla moves 12 points on TOSF, the
  PP oracle 20. Text now says explicitly that no difference in Table 5 is established.
- tab:alllabels closed-loop row updated from "still pending" to "measured on the four
  RoboCasa tasks, pending on the other 20".
- Compiles clean: 19 pages, no errors, no undefined refs/cites, no overfull boxes.

## 2026-07-31 ~19:55 UTC — the "history does not help" claim was WRONG at scale; corrected
- With only 6 datasets (4 RoboCasa + 2 MimicGen) h1 beat h4 nearly everywhere and the
  paper said temporal history does not help. With 14 datasets the sign flips: h1 wins 6,
  h4 wins 8, mean AUROC .848 vs .858.
- The split is systematic. Spearman(crossings per 100 stamps, h4 advantage) = +0.64.
  Slow-alternating datasets (<10 crossings/100, n=7): h4 loses .013, wins 2/7.
  Fast-alternating (>=10, n=7): h4 gains .034, wins 6/7. Biggest h4 wins are can (+.123),
  tool_hang (+.042), square (+.029), rollouts_can (+.027).
- Interpretation now in the paper: a single frame suffices when instability is isolated
  islands (regime = current configuration, consistent with the single-frame DINOv2 and
  proprio-only results); recent motion adds information only when regimes alternate
  several times per 100 steps. RoboCasa sits at 2-5 crossings/100, which is why h4 loses
  there and why the original 6-dataset conclusion was an artifact of platform choice.
- LESSON: the earlier claim was drawn from one platform. Do not generalise a
  head-architecture conclusion from RoboCasa alone.
