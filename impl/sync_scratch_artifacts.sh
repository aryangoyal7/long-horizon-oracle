#!/bin/bash
# Continuously mirror non-rederivable scratch artifacts to the persistent share.
# (Scheduled shutdown = deallocation = /mnt/scratch wiped. Checkpoints already write
# to the share; labels/features/rollouts are produced on scratch for speed and synced
# here.) Run detached: setsid nohup bash impl/sync_scratch_artifacts.sh &
set -u
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/artifacts"
mkdir -p "$DEST"/{labels,features,rollouts}
while true; do
  rsync -a --exclude 'probe_study' /mnt/scratch/lh/labels/ "$DEST/labels/" 2>/dev/null
  rsync -a /mnt/scratch/lh/features/ "$DEST/features/" 2>/dev/null
  rsync -a /mnt/scratch/lh/rollouts/ "$DEST/rollouts/" 2>/dev/null
  sleep 900
done
