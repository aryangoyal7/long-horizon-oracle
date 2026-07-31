#!/bin/bash
# @reboot recovery: rebuild scratch, resume unfinished trainings, restart sync loop.
# Idempotent; safe if nothing needs recovering.
LH_DIR=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$LH_DIR/results/boot_recover_$(date +%Y%m%d_%H%M).log
exec > "$LOG" 2>&1
echo "boot_recover $(date -u)"
[ -x /mnt/scratch/lh/envs/lh/bin/python ] && { echo "scratch alive, nothing to do"; exit 0; }
bash $LH_DIR/impl/setup_scratch.sh
PY=/mnt/scratch/lh/envs/lh/bin/python
# resume any training not yet at epoch 2000
declare -A GPU=( [lift]=0 [can]=1 [square]=2 [tool_hang]=3 [square_s2]=5 [tool_hang_s2]=6 )
for t in "${!GPU[@]}"; do
  last=$(grep -aoE "Epoch [0-9]+" $LH_DIR/results/training/dp_${t}.launch.log 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  [ "${last:-0}" -ge 2000 ] && continue
  cfg=$LH_DIR/impl/configs/dp_${t}.json
  [ -f "$cfg" ] || continue
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU[$t]} setsid nohup bash -c \
    "yes y | $PY /mnt/scratch/lh/repos/robomimic/robomimic/scripts/train.py --config $cfg --resume" \
    >> $LH_DIR/results/training/dp_${t}.launch.log 2>&1 < /dev/null &
  echo "resumed dp_$t on GPU ${GPU[$t]}"
done
# reseed labels/rollouts and restart the artifact sync loop
mkdir -p /mnt/scratch/lh/{labels,features,rollouts}
rsync -a $LH_DIR/artifacts/labels/ /mnt/scratch/lh/labels/
rsync -a $LH_DIR/artifacts/rollouts/ /mnt/scratch/lh/rollouts/
# Features are 38 GB and live only on the share. The encoder ablation reads them
# from scratch, so without this it silently has nothing to train on after a reboot
# (this is what stranded the grid at 14/39 on Jul 24).
rsync -a $LH_DIR/artifacts/features/ /mnt/scratch/lh/features/
echo "features staged: $(ls /mnt/scratch/lh/features/*.npz 2>/dev/null | wc -l) files"
setsid nohup bash $LH_DIR/impl/sync_scratch_artifacts.sh > /dev/null 2>&1 < /dev/null &
# resume the encoder ablation if it has not finished; run_cell skips completed cells
if ! grep -q ABLATION_ALL_DONE $LH_DIR/results/predictor/ablation_run.log 2>/dev/null; then
  setsid nohup bash $LH_DIR/impl/predictor/run_encoder_ablation.sh > /dev/null 2>&1 < /dev/null &
  echo "resumed encoder ablation"
fi
# Stage 2 lives in its own tree (mg venv, mimicgen datasets, image conversions)
# and nothing used to restart it, so every reboot stranded Stage 2 until it was
# relaunched by hand. rebuild_scratch.sh is idempotent and ends by resuming
# checkpoint selection, which in turn releases the k-sweep runner.
if ! pgrep -f "mimicgen/rebuild_scratch.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/mimicgen/rebuild_scratch.sh \
    >> $LH_DIR/results/stage2/rebuild_boot.log 2>&1 < /dev/null &
  echo "launched rebuild_scratch (stage 2)"
fi
# the k-sweep runner waits on its own gate for the converted datasets
if ! grep -q KSWEEP_ALL_DONE $LH_DIR/results/stage2/ksweep_run.log 2>/dev/null \
   && ! pgrep -f "run_stage2_ksweep.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/mimicgen/run_stage2_ksweep.sh \
    >> $LH_DIR/results/stage2/ksweep_runner.log 2>&1 < /dev/null &
  echo "armed k-sweep runner"
fi
# LIBERO stage: datasets, masks, configs, trainings, checkpoint selection. Fully
# idempotent (skips tasks already at the target epoch) and self-sequencing: it
# waits for the ablation and predictor cells before touching the GPUs.
if ! grep -q LIBERO_STAGE_DONE $LH_DIR/results/libero/stage.log 2>/dev/null \
   && ! pgrep -f "run_libero_stage.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/libero/run_libero_stage.sh \
    >> $LH_DIR/results/libero/stage_boot.log 2>&1 < /dev/null &
  echo "launched LIBERO stage"
fi
# LIBERO datasets live only on scratch and the stage runner just waits for
# them; re-download once the mg venv from rebuild_scratch imports robosuite.
if [ "$(ls /mnt/scratch/lh/data/libero/libero_10/*.hdf5 2>/dev/null | wc -l)" -lt 10 ]; then
  setsid nohup bash -c 'until /mnt/scratch/lh/envs/mg/bin/python -c "import robosuite" 2>/dev/null; do sleep 120; done; bash '"$LH_DIR"'/impl/libero/setup_and_probe.sh' \
    >> $LH_DIR/results/libero/setup_boot.log 2>&1 < /dev/null &
  echo "armed LIBERO dataset download"
fi
# nut/coffee in-domain labeling + predictor + switching chain (2026-07-25)
if ! grep -q NUTCOFFEE_ALL_DONE $LH_DIR/results/stage2/nutcoffee_chain.log 2>/dev/null \
   && ! pgrep -f "run_nutcoffee_chain.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/mimicgen/run_nutcoffee_chain.sh > /dev/null 2>&1 < /dev/null &
  echo "launched nut/coffee chain"
fi
# RoboCasa + GR00T N1.5 setup (2026-07-25); idempotent stages
if ! grep -q ROBOCASA_SETUP_DONE $LH_DIR/results/robocasa/setup.log 2>/dev/null \
   && ! pgrep -f "setup_robocasa.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/robocasa/setup_robocasa.sh > /dev/null 2>&1 < /dev/null &
  echo "launched robocasa setup"
fi
# RoboCasa labeling + predictor chain (independent of the GR00T eval side)
if ! grep -q RC_LABELING_CHAIN_DONE $LH_DIR/results/robocasa/labeling.log 2>/dev/null \
   && ! pgrep -f "run_robocasa_labeling.sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/robocasa/run_robocasa_labeling.sh > /dev/null 2>&1 < /dev/null &
  echo "launched robocasa labeling chain"
fi
echo "BOOT_RECOVER_DONE"
# GR00T N1.5 fork env (robocasa-benchmark Isaac-GR00T) for checkpoint-120000 evals
if [ ! -f $LH_DIR/results/robocasa/GROOT155_OK ] \
   && ! pgrep -f "setup_groot155[.]sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/robocasa/setup_groot155.sh > /dev/null 2>&1 < /dev/null &
  echo "launched groot155 setup"
fi
# GR00T fixed-k baseline sweep (RoboCasa, 4 tasks x k in 16/8/4/1)
if ! grep -q GROOT_KSWEEP_DONE $LH_DIR/results/robocasa/groot_ksweep.log 2>/dev/null \
   && ! pgrep -f "run_groot_ksweep[.]sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/robocasa/run_groot_ksweep.sh > /dev/null 2>&1 < /dev/null &
  echo "launched groot fixed-k sweep"
fi
# LIBERO rollout evaluation chain (ckpt select -> fixed-k -> predictor cells)
if ! grep -q LIBERO_KSWEEP_DONE $LH_DIR/results/stage2/libero_ksweep_chain.log 2>/dev/null \
   && ! pgrep -f "run_libero_ksweep[.]sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/libero/run_libero_ksweep.sh > /dev/null 2>&1 < /dev/null &
  echo "launched libero ksweep chain"
fi
# GR00T predictor-switching cells (after fixed-k sweep)
if ! grep -q GROOT_SWITCH_DONE $LH_DIR/results/robocasa/groot_ksweep.log 2>/dev/null \
   && ! pgrep -f "run_groot_switch[.]sh" > /dev/null; then
  setsid nohup bash $LH_DIR/impl/robocasa/run_groot_switch.sh > /dev/null 2>&1 < /dev/null &
  echo "launched groot switching cells"
fi
