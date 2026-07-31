#!/bin/bash
# LIBERO training stage, fully sequenced and detached.
#   wait for all 10 datasets -> masks + configs -> smoke test -> train 10 tasks
#   across 8 GPUs -> checkpoint selection by validation loss
# Deliberately queued BEHIND the encoder ablation and the LIBERO predictor cells
# so it does not contend with work that is nearly finished.
set -uo pipefail
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LH=/mnt/scratch/lh
PY=$LH/envs/lh/bin/python          # robomimic main + diffusion policy
DATA=$LH/data/libero/libero_10
CFG=$ROOT/impl/configs/libero
OUTD=$ROOT/results/training
LOG=$ROOT/results/libero/stage.log
mkdir -p "$(dirname "$LOG")" "$CFG"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$LOG"; }

# ---- 1. all 10 LIBERO-10 datasets ------------------------------------------------
say "waiting for 10 datasets"
until [ "$(ls $DATA/*.hdf5 2>/dev/null | wc -l)" -ge 10 ] \
      && ! pgrep -f "setup_and_probe.sh" >/dev/null; do sleep 60; done
say "datasets ready: $(ls $DATA/*.hdf5 | wc -l)"

# ---- 2. masks + configs ----------------------------------------------------------
$PY $ROOT/impl/libero/prepare_libero_training.py \
    --data-dir "$DATA" --template $ROOT/impl/configs/dp_square.json \
    --config-dir "$CFG" --output-dir "$OUTD" \
    --epochs "${EPOCHS:-1000}" --batch "${BATCH:-128}" >> "$LOG" 2>&1
grep -q PREP_DONE "$LOG" || { say "FATAL prep failed"; exit 1; }
say "configs written: $(ls $CFG/*.json | wc -l)"

# ---- 3. smoke test: 2 epochs on one task, so a config error costs minutes ---------
SMOKE=$(ls $CFG/*.json | head -1)
SM=/tmp/libero_smoke.json
$PY - "$SMOKE" "$SM" <<'PYS' >> "$LOG" 2>&1
import json, sys
c = json.load(open(sys.argv[1]))
c["train"]["num_epochs"] = 2
c["experiment"]["epoch_every_n_steps"] = 5
c["experiment"]["validation_epoch_every_n_steps"] = 2
c["experiment"]["save"]["enabled"] = False
c["train"]["output_dir"] = "/mnt/scratch/lh/libero_smoke"
json.dump(c, open(sys.argv[2], "w"), indent=1)
PYS
say "smoke: $(basename $SMOKE)"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 $PY $LH/repos/robomimic/robomimic/scripts/train.py \
    --config $SM >> $ROOT/results/libero/smoke.log 2>&1
if [ $? -ne 0 ]; then say "SMOKE_FAIL (see results/libero/smoke.log)"; exit 1; fi
say "SMOKE_OK"

# ---- 4. wait for the GPUs to clear -----------------------------------------------
say "waiting for ablation + predictor cells"
until grep -q ABLATION_ALL_DONE $ROOT/results/predictor/ablation_run.log 2>/dev/null; do sleep 120; done
until grep -q LIBERO_ALL_DONE $ROOT/results/predictor/libero_run.log 2>/dev/null; do sleep 120; done
say "GPUs clear, launching trainings"

# ---- 5. train, one task per GPU, 8 at a time -------------------------------------
mapfile -t CFGS < <(ls $CFG/*.json)
say "TRAIN_START ${#CFGS[@]} tasks"
train_one() {
  local gpu=$1 cfg=$2 name resume=""
  name=$(basename "$cfg" .json)
  if grep -aoE "Epoch [0-9]+" $OUTD/$name.launch.log 2>/dev/null | tail -1 \
     | grep -oE "[0-9]+" | awk -v e="${EPOCHS:-1000}" '{exit !($1>=e)}'; then
    say "SKIP $name (already at target epoch)"; return
  fi
  # a prior run's last.pth on the share means we resume instead of restarting
  # from epoch 0 (checkpoints survive the scratch wipe, the run dir does not)
  ls $OUTD/$name/$name/*/last.pth >/dev/null 2>&1 && resume="--resume"
  say "TRAIN $name (gpu $gpu)${resume:+ resuming}"
  yes y 2>/dev/null | MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu $PY \
    $LH/repos/robomimic/robomimic/scripts/train.py --config "$cfg" $resume \
    >> $OUTD/$name.launch.log 2>&1
  say "TRAIN_DONE $name exit=$?"
}
worker() { local w=$1; for ((i=w; i<${#CFGS[@]}; i+=8)); do train_one "$w" "${CFGS[$i]}"; done; }
for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
say "TRAIN_ALL_DONE"

# ---- 6. checkpoint selection by validation loss (rollouts need the native env) ----
$PY $ROOT/impl/libero/select_libero_ckpts.py --training-dir "$OUTD" \
    --out $ROOT/results/libero/ckpt_selection.json >> "$LOG" 2>&1
say "CKPT_SELECT_DONE"
say "LIBERO_STAGE_DONE"
