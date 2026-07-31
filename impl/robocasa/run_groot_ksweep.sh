#!/bin/bash
# GR00T N1.5 fixed-k baseline sweep on the 4 labeled RoboCasa tasks.
# One GPU per task, k in {16,8,4,1} sequential per GPU, 50 episodes per cell.
# k is the executed chunk length (n_action_steps); the policy always predicts 16.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
PY=$LH/envs/groot155/bin/python
FORK=$LH/repos/Isaac-GR00T-rc365
KOUT=$ROOT/results/robocasa/ksweep
LOG=$ROOT/results/robocasa/groot_ksweep.log
mkdir -p $KOUT $LH/rollouts/groot_ksweep
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }

run_task() {
  local TASK=$1 GPU=$2
  local PORT=$((5600 + GPU))
  # GPU 5 runs the RMSE measurement first; wait for it to free up
  while pgrep -f "measure_groot_rmse[.]py" > /dev/null && [ "$GPU" = "5" ]; do sleep 60; done
  local k name
  for k in 16 8 4 1; do
    name=${TASK}_fixed_${k}
    [ -f $KOUT/$name.json ] && continue
    say "CELL_START $name (gpu $GPU)"
    cd $FORK && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PYOPENGL_PLATFORM=egl \
      CUDA_VISIBLE_DEVICES=$GPU $PY $ROOT/impl/robocasa/groot_first_rollout.py \
        --env-name $TASK --n-episodes 50 --n-envs 5 --n-action-steps $k \
        --port $PORT --out $LH/rollouts/groot_ksweep/$name \
        > $KOUT/$name.log 2>&1
    if grep -q ROLLOUT_VALIDATION_RESULT $KOUT/$name.log; then
      grep -o 'ROLLOUT_VALIDATION_RESULT.*' $KOUT/$name.log | tail -1 \
        | sed 's/ROLLOUT_VALIDATION_RESULT //' > $KOUT/$name.json
      say "CELL_DONE $name $(python3 -c "import json;d=json.load(open('$KOUT/$name.json'));print(d['success_rate'])" 2>/dev/null)"
    else
      say "CELL_FAIL $name"
    fi
  done
  say "TASK_DONE $TASK"
}

say "=== groot fixed-k sweep start ==="
run_task OpenCabinet 4 &
run_task OpenDrawer 5 &
run_task PickPlaceCounterToCabinet 6 &
run_task TurnOnSinkFaucet 7 &
wait
say "GROOT_KSWEEP_DONE"
