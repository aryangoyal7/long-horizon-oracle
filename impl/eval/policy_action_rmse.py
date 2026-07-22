"""Compute the trained policy's validation action RMSE — this is sigma_u for the
final FTLE labels (stability_label_generation.md, Step 3).

Definition: RMSE between the policy's predicted FIRST action and the demo action,
over all timesteps of the validation demos (mask 'valid'), arm dims only (first 6).
DP is stochastic; we draw one sample per state (the error the policy actually
injects at execution). Observations are built exactly as in training (frame_stack
obs history from the dataset).

Usage:
  python policy_action_rmse.py --ckpt <model.pth> --dataset <image_v15.hdf5> --out out.json
"""

import argparse
import json

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-demos", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    import robomimic.utils.file_utils as FileUtils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.ckpt, device=device, verbose=False)
    cfg = json.loads(ckpt_dict["config"])
    obs_keys = (cfg["observation"]["modalities"]["obs"]["low_dim"]
                + cfg["observation"]["modalities"]["obs"]["rgb"])
    n_stack = cfg["train"]["frame_stack"]

    sq_err, n_pts = 0.0, 0
    sq_err_pos, n_pts_pos = 0.0, 0
    with h5py.File(args.dataset, "r") as f:
        valid = [d.decode() if isinstance(d, bytes) else d
                 for d in f["mask/valid"][()]][: args.max_demos]
        for dk in valid:
            g = f[f"data/{dk}"]
            acts = g["actions"][()]
            obs_all = {k: g[f"obs/{k}"][()] for k in obs_keys}
            T = acts.shape[0]
            policy.start_episode()
            for t in range(0, T, args.stride):
                ob = {}
                for k in obs_keys:
                    idx = [max(0, t - i) for i in range(n_stack - 1, -1, -1)]
                    # feed the stacked history exactly as FrameStackWrapper would:
                    # shape (n_stack, ...) per key
                    ob[k] = obs_all[k][idx]
                a = policy(ob=ob)
                # policy returns queued chunk actions; only the FIRST post-inference
                # action corresponds to this state
                sq_err += float(np.sum((np.asarray(a)[:6] - acts[t][:6]) ** 2))
                n_pts += 6
                # position dims only — the final labeler recipe perturbs pos only,
                # so this is the sigma_u actually fed to ftle_labeler.py
                sq_err_pos += float(np.sum((np.asarray(a)[:3] - acts[t][:3]) ** 2))
                n_pts_pos += 3
                # flush queue so every evaluated state triggers fresh inference
                if hasattr(policy.policy, "action_queue"):
                    policy.policy.action_queue.clear()
    rmse = float(np.sqrt(sq_err / n_pts))
    rmse_pos = float(np.sqrt(sq_err_pos / n_pts_pos))
    out = {"ckpt": args.ckpt, "dataset": args.dataset, "n_points": n_pts,
           "arm_action_rmse": rmse, "pos_action_rmse": rmse_pos}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
