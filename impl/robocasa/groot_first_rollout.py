"""First GR00T N1.5 rollout validation: one task, few episodes, video on.

Uses the robocasa-benchmark fork's own server/client eval path (run_eval.py
else-branch) restricted to a single env. Run in the groot155 env:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=4 python groot_first_rollout.py \
      --env-name TurnOnSinkFaucet --n-episodes 2
"""
import argparse
import json
import os
import threading
import time

import numpy as np
from robocasa.utils.dataset_registry_utils import get_task_horizon
from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig, SimulationConfig, SimulationInferenceClient, VideoConfig)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"


def run_server(port):
    dc = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(
        model_path=CKPT, modality_config=dc.modality_config(),
        modality_transform=dc.transform(), embodiment_tag="new_embodiment",
        denoising_steps=4)
    RobotInferenceServer(policy, port=port).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="TurnOnSinkFaucet")
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--out", default="/mnt/scratch/lh/rollouts/groot_val")
    args = ap.parse_args()

    threading.Thread(target=run_server, args=(args.port,), daemon=True).start()
    time.sleep(1)
    client = SimulationInferenceClient(host="localhost", port=args.port)
    print("modality configs:", list(client.get_modality_config().keys()), flush=True)

    horizon = get_task_horizon(args.env_name)
    video_dir = os.path.join(args.out, args.env_name)
    cfg = SimulationConfig(
        env_name=f"robocasa/{args.env_name}", split=args.split,
        n_episodes=args.n_episodes, n_envs=args.n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps, max_episode_steps=horizon))
    t0 = time.time()
    env_name, successes = client.run_simulation(cfg)
    out = {"env": env_name, "horizon": horizon,
           "n_episodes": len(successes),
           "successes": [bool(s) for s in successes],
           "success_rate": float(np.mean(successes)),
           "elapsed_s": round(time.time() - t0, 1)}
    print("ROLLOUT_VALIDATION_RESULT", json.dumps(out), flush=True)
    with open(os.path.join(video_dir, "first_rollout_stats.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
