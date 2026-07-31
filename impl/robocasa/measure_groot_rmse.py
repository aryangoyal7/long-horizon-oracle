"""Measure GR00T N1.5 first-action position RMSE against the demo actions.

This is the sigma_u for RoboCasa FTLE labels (same convention as
policy_action_rmse.py on robomimic/mimicgen: first predicted action vs recorded
action, position dims only, sampled across validation demos). Uses the same
target-split demos the labels were generated from.

Run in groot155:
  CUDA_VISIBLE_DEVICES=5 python measure_groot_rmse.py --out rmse_groot.json
"""
import argparse
import glob
import json

import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"
DATA_GLOB = "/mnt/scratch/lh/repos/robocasa/datasets/v1.0/target/atomic/{}/*/lerobot"
TASKS = ["OpenCabinet", "OpenDrawer", "PickPlaceCounterToCabinet", "TurnOnSinkFaucet"]
POS_KEY = "action.end_effector_position"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", type=int, default=10, help="demos per task (from the end)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dc = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(
        model_path=CKPT, modality_config=dc.modality_config(),
        modality_transform=dc.transform(), embodiment_tag="new_embodiment",
        denoising_steps=4)
    modality = policy.get_modality_config()

    results = {}
    sq_all, n_all = 0.0, 0
    for task in TASKS:
        path = sorted(glob.glob(DATA_GLOB.format(task)))[-1]
        ds = LeRobotSingleDataset(
            dataset_path=path, modality_configs=modality,
            video_backend="torchvision_av", video_backend_kwargs=None,
            transforms=None, embodiment_tag="new_embodiment")
        lengths = ds.trajectory_lengths
        traj_ids = list(range(len(lengths)))[-args.trajs:]   # last demos = held-out-ish
        sq, n = 0.0, 0
        for tid in traj_ids:
            for t in range(0, int(lengths[tid]), args.stride):
                step = ds.get_step_data(tid, t)
                pred = np.asarray(policy.get_action(step)[POS_KEY])
                gt = np.asarray(step[POS_KEY])
                p0 = pred[0] if pred.ndim > 1 else pred      # first action of chunk
                g0 = gt[0] if gt.ndim > 1 else gt
                sq += float(np.sum((p0 - g0) ** 2)); n += p0.size
        rmse = float(np.sqrt(sq / n))
        results[task] = {"pos_action_rmse": rmse, "n_points": n,
                         "trajs": len(traj_ids), "dataset": path}
        print(f"{task}: pos RMSE {rmse:.4f} ({n} pts)", flush=True)
        sq_all += sq; n_all += n
    results["pooled_pos_action_rmse"] = float(np.sqrt(sq_all / n_all))
    results["sigma_u_placeholder_used_in_labels"] = 0.05
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("POOLED pos RMSE:", results["pooled_pos_action_rmse"], flush=True)


if __name__ == "__main__":
    main()
