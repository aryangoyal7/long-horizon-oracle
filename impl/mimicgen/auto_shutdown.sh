#!/bin/bash
# Auto-shutdown when the full Stage 2 pipeline is done (user request, Jul 21).
# Shuts the machine down when ALL of:
#   1. all four fresh dp_mg trainings print "finished run successfully"
#   2. checkpoint selection prints CKPT_SELECT_ALL_DONE
#   3. the Stage 2 experiment prints STAGE2_KSWEEP_ALL_DONE
#      (results/stage2/ksweep_run.log, written by the experiment runner)
# Failsafe: once (1) and (2) hold, if the GPUs stay idle for 3 consecutive
# hours (pipeline stalled or finished without the marker), shut down anyway.
# Poll every 10 minutes. Log: results/stage2/auto_shutdown.log
set -u
LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T=$LH_DIR/results/training
KS=$LH_DIR/results/stage2/ksweep_run.log
IDLE_LIMIT=18   # 18 polls x 10 min = 3 h
idle=0

trainings_done() {
  local n=0
  for t in three_piece_assembly_d0 nut_assembly_d0 kitchen_d0 coffee_preparation_d0; do
    grep -q "finished run successfully" $T/dp_mg_${t}.launch.log 2>/dev/null && n=$((n+1))
  done
  [ $n -eq 4 ]
}

gpus_idle() {
  local busy
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '$1 > 1000 {c++} END {print c+0}')
  [ "$busy" -eq 0 ]
}

while true; do
  if trainings_done \
     && grep -q "CKPT_SELECT_ALL_DONE" $LH_DIR/results/stage2/ckpt_select_run.log 2>/dev/null; then
    if grep -q "STAGE2_KSWEEP_ALL_DONE" $KS 2>/dev/null; then
      echo "$(date -u) all pipeline markers present -> shutdown"
      sudo shutdown -h +5 "long-horizon pipeline complete, auto-shutdown in 5 min"
      exit 0
    fi
    if gpus_idle; then
      idle=$((idle+1))
      echo "$(date -u) trainings+selection done, GPUs idle ($idle/$IDLE_LIMIT)"
      if [ $idle -ge $IDLE_LIMIT ]; then
        echo "$(date -u) idle failsafe -> shutdown"
        sudo shutdown -h +5 "long-horizon pipeline idle 3h, auto-shutdown in 5 min"
        exit 0
      fi
    else
      idle=0
    fi
  fi
  sleep 600
done
