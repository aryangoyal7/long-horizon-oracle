"""Diffusion Policy inference server for Stage 2 evaluation.

Runs in the lh venv (robomimic main, has diffusion_policy); the mg-venv
clients (bridge_eval.py, stage2_ksweep.py) talk to it over the same line
protocol predictor_bridge.py uses:

  bridge stdout:  READY                    after the checkpoint is loaded
  eval  -> stdin: RESET                    start_episode + clear action queue
  bridge stdout:  ACK
  eval  -> stdin: CLEARQ                   clear the action queue (forces
  bridge stdout:  ACK                      re-inference at the next REQ)
  eval  -> stdin: REQ <npz_path>           npz: one array per observation key
  bridge stdout:  RES <qlen> <a0> ... <a6> one action + queue length after pop
  eval  -> stdin: QUIT

The npz arrays are raw env observations (images HWC uint8, frame-stacked with
a leading (fs, ...) dim); RolloutPolicy does its own processing, exactly as
robomimic's run_trained_agent.py relies on.

--full-chunk applies the Stage 1c trick from eval_k_sweep.py: the checkpoint
config is patched so action_horizon = prediction_horizon, making the internal
action queue hold the full predicted chunk. The client then controls the
executed chunk length by sending CLEARQ at replan points, mirroring
eval_k_sweep.run_episode exactly.

Run: CUDA_VISIBLE_DEVICES=<gpu> python policy_bridge.py --checkpoint <ckpt.pth>
"""

import argparse
import json
import sys

import numpy as np


def load_policy(ckpt_path, full_chunk):
    import robomimic.utils.file_utils as FileUtils
    if not full_chunk:
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device="cuda", verbose=False)
        return policy
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    cfg = json.loads(ckpt_dict["config"])
    cfg["algo"]["horizon"]["action_horizon"] = \
        cfg["algo"]["horizon"]["prediction_horizon"]
    ckpt_dict["config"] = json.dumps(cfg)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device="cuda", verbose=False)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--full-chunk", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.checkpoint, args.full_chunk)

    def queue():
        return policy.policy.action_queue

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break
        if line == "RESET":
            policy.start_episode()
            queue().clear()
            print("ACK", flush=True)
            continue
        if line == "CLEARQ":
            queue().clear()
            print("ACK", flush=True)
            continue
        if not line.startswith("REQ "):
            continue
        z = np.load(line[4:])
        obs = {k: z[k] for k in z.files}
        act = policy(ob=obs)
        print(f"RES {len(queue())} "
              + " ".join(f"{a:.6f}" for a in act), flush=True)


if __name__ == "__main__":
    main()
