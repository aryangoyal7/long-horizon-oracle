#!/bin/bash
# Rebuild the RoboCasa/GR00T half of /mnt/scratch after the Jul 30 reboot
# wipe (rebuild_scratch.sh covers only the LIBERO/MimicGen half). Steps are
# idempotent; failures are soft so later stages still run. On success it
# relaunches the three interrupted chains (ACT grids, fork cells, v3 labels).
# Survivors on the share: all results/, artifacts/labels (final2), ACT ckpts.
set -u
LH=/mnt/scratch/lh
ROOT=/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon
LOG=$ROOT/results/robocasa/rebuild_robocasa.log
export PIP_CACHE_DIR=$LH/pipcache HF_HOME=$LH/hf
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p $LH/repos $LH/data/robocasa/hdf5 $LH/hf $LH/labels
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "REBUILD_RC_START"

BASEPY=""
for _p in /anaconda/envs/azureml_py310_sdkv2/bin/python /anaconda/envs/azureml_py38/bin/python3 /anaconda/bin/python3 python3; do
  command -v "$_p" >/dev/null 2>&1 && "$_p" -c "import ensurepip" 2>/dev/null && { BASEPY=$(command -v "$_p"); break; }
done
say "basepy=$BASEPY"

# stale markers point at wiped content
rm -f $ROOT/results/robocasa/ASSETS_OK $ROOT/results/robocasa/CKPT_OK \
      $ROOT/results/robocasa/GROOT155_OK $ROOT/results/robocasa/DATA_OK

# ---- 1. rc venv (labeler env) --------------------------------------------------
RCPY=$LH/envs/rc/bin/python
RCPIP=$LH/envs/rc/bin/pip
if ! $RCPY -c "import robocasa" 2>/dev/null; then
  [ -x $RCPY ] || "$BASEPY" -m venv $LH/envs/rc
  $RCPIP install -q --upgrade pip setuptools wheel >> $LOG 2>&1
  [ -d $LH/repos/robosuite_rc ] || git clone -q https://github.com/ARISE-Initiative/robosuite $LH/repos/robosuite_rc
  if [ ! -d $LH/repos/robocasa ]; then
    git clone -q https://github.com/robocasa/robocasa $LH/repos/robocasa
    git -C $LH/repos/robocasa checkout -q b4684e6 2>/dev/null || true
  fi
  $RCPIP install -q -e $LH/repos/robosuite_rc >> $LOG 2>&1
  $RCPIP install -q -e $LH/repos/robocasa >> $LOG 2>&1
  $RCPIP install -q h5py imageio imageio-ffmpeg huggingface_hub pandas pyarrow >> $LOG 2>&1
fi
$RCPY -c "import robocasa" 2>/dev/null && say "RC_VENV_OK" || say "RC_VENV_FAIL"

# ---- 2. kitchen assets (~5 GB) -------------------------------------------------
if [ ! -f $ROOT/results/robocasa/ASSETS_OK ]; then
  yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_kitchen_assets.py >> $LOG 2>&1 \
    && { touch $ROOT/results/robocasa/ASSETS_OK; say "RC_ASSETS_OK"; } || say "RC_ASSETS_FAIL"
  $RCPY $LH/repos/robocasa/robocasa/scripts/setup_macros.py >> $LOG 2>&1 || true
fi

# ---- 3. repos: Isaac-GR00T (for external robocasa365) + benchmark fork ---------
[ -d $LH/repos/Isaac-GR00T ] || git clone -q https://github.com/NVIDIA/Isaac-GR00T $LH/repos/Isaac-GR00T
RC365=$LH/repos/Isaac-GR00T/external_dependencies/robocasa365
if [ ! -e $RC365/.git ]; then
  git clone -q https://github.com/robocasa/robocasa.git $RC365
  git -C $RC365 fetch -q origin be22d659b02db8f6d7f3a3c3edc742934fdcbaae
  git -C $RC365 checkout -q be22d659b02db8f6d7f3a3c3edc742934fdcbaae
fi
[ -d $LH/repos/Isaac-GR00T-rc365 ] || git clone -q https://github.com/robocasa-benchmark/Isaac-GR00T $LH/repos/Isaac-GR00T-rc365
[ -d $LH/repos/Isaac-GR00T-rc365 ] && say "REPOS_OK" || say "REPOS_FAIL"

# ---- 4. GR00T checkpoint-120000 ------------------------------------------------
if [ ! -f $ROOT/results/robocasa/CKPT_OK ]; then
  $RCPY - >> $LOG 2>&1 <<'PYC'
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
found = None
for repo in ["robocasa/robocasa365_checkpoints", "robocasa/robocasa365-checkpoints",
             "robocasa/checkpoints", "nvidia/GR00T-N1.5-robocasa"]:
    try:
        files = api.list_repo_files(repo)
        hits = [f for f in files if "n1" in f.lower() and "multitask" in f.lower()]
        print(f"[ckpt] {repo}: {len(files)} files, {len(hits)} hits", flush=True)
        if hits:
            found = repo
            pref = sorted({h.split("checkpoint-")[0] for h in hits if "checkpoint-120000" in h})
            pats = [p + "checkpoint-120000/*" for p in pref] or None
            snapshot_download(repo, allow_patterns=pats,
                              local_dir="/mnt/scratch/lh/data/robocasa/ckpt")
            print("[ckpt] downloaded", pats, flush=True)
            break
    except Exception as e:
        print(f"[ckpt] no: {repo} {type(e).__name__} {str(e)[:100]}", flush=True)
if not found:
    raise SystemExit(1)
PYC
  [ $? -eq 0 ] && { touch $ROOT/results/robocasa/CKPT_OK; say "RC_CKPT_OK"; } || say "RC_CKPT_FAIL"
fi

# ---- 5. lerobot demo datasets (4 tasks, pretrain split) ------------------------
TASKS="PickPlaceCounterToCabinet OpenDrawer OpenCabinet TurnOnSinkFaucet"
NEED=0
for t in $TASKS; do
  ls -d $LH/repos/robocasa/datasets/v1.0/*/atomic/$t/*/lerobot >/dev/null 2>&1 || NEED=1
done
if [ $NEED -eq 1 ]; then
  yes y 2>/dev/null | $RCPY $LH/repos/robocasa/robocasa/scripts/download_datasets.py \
    --tasks $TASKS --source human >> $ROOT/results/robocasa/data_download_rebuild.log 2>&1
fi
NEED=0
for t in $TASKS; do
  ls -d $LH/repos/robocasa/datasets/v1.0/*/atomic/$t/*/lerobot >/dev/null 2>&1 || { NEED=1; say "DATA_MISSING $t"; }
done
[ $NEED -eq 0 ] && say "RC_DATA_OK" || say "RC_DATA_FAIL"

# ---- 6. groot155 env (eval env: GR00T fork + robocasa365) ----------------------
bash $ROOT/impl/robocasa/setup_groot155.sh
[ -f $ROOT/results/robocasa/GROOT155_OK ] && say "GROOT155_REBUILT_OK" || say "GROOT155_REBUILT_FAIL"

# ---- 7. convert lerobot -> rc_*.hdf5 -------------------------------------------
for t in $TASKS; do
  OUT=$LH/data/robocasa/hdf5/rc_$t.hdf5
  [ -f $OUT ] && continue
  D=$(ls -d $LH/repos/robocasa/datasets/v1.0/*/atomic/$t/*/lerobot 2>/dev/null | head -1)
  [ -z "$D" ] && { say "CONVERT_SKIP $t (no lerobot dir)"; continue; }
  $RCPY $ROOT/impl/robocasa/lerobot_to_hdf5.py --lerobot-dir "$D" --out $OUT \
    >> $ROOT/results/robocasa/convert_$t.log 2>&1 \
    && say "CONVERT_DONE $t" || { say "CONVERT_FAIL $t"; rm -f $OUT; }
done

# ---- 8. relaunch the interrupted chains (all idempotent) -----------------------
OKS=1
$LH/envs/groot155/bin/python -c "import gr00t" 2>/dev/null || OKS=0
[ -f $LH/data/robocasa/hdf5/rc_OpenCabinet.hdf5 ] || OKS=0
[ -d $LH/data/robocasa/ckpt ] || OKS=0
if [ $OKS -eq 1 ]; then
  setsid nohup bash $ROOT/impl/robocasa/act/run_act_fix2.sh > /mnt/scratch/lh/relaunch_act.out 2>&1 < /dev/null &
  setsid nohup bash $ROOT/impl/robocasa/run_groot_fork.sh > /mnt/scratch/lh/relaunch_fork.out 2>&1 < /dev/null &
  setsid nohup bash $ROOT/impl/robocasa/run_robocasa_relabeling.sh > /mnt/scratch/lh/relaunch_label.out 2>&1 < /dev/null &
  say "CHAINS_RELAUNCHED"
else
  say "CHAINS_NOT_RELAUNCHED (missing env/data/ckpt, see markers above)"
fi
say "REBUILD_RC_DONE"
