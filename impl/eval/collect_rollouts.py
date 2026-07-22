"""Collect policy rollouts (successes AND failures) in demo-compatible hdf5 format.

Purpose: the predictor coverage rule (stability_label_generation.md §5) — labels must be
generated on the state distribution the policy actually visits, not just the demo tube.
Output hdf5 mirrors robomimic layout (data/rollout_i/{states,actions}, model_file attr,
data.attrs env_args), so ftle_labeler.py runs on it unchanged.

Usage:
  python collect_rollouts.py --ckpt <best.pth> --source-dataset <low_dim_v15.hdf5> \
      --out rollouts_lift.hdf5 --n 100
"""

import argparse
import json

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source-dataset", required=True,
                    help="demo hdf5 whose env_args to copy verbatim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_k_sweep import load_policy, make_env_from_ckpt, queue_of

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = load_policy(args.ckpt, device)
    env = make_env_from_ckpt(ckpt_dict)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    with h5py.File(args.source_dataset, "r") as f:
        env_args = f["data"].attrs["env_args"]

    n_succ = 0
    with h5py.File(args.out, "w") as f:
        grp = f.create_group("data")
        grp.attrs["env_args"] = env_args
        for ep in range(args.n):
            obs = env.reset()
            # env is FrameStackWrapper(EnvRobosuite) — use the EnvBase API, which the
            # wrapper forwards, instead of guessing the .env nesting depth.
            model_xml = env.get_state()["model"]
            policy.start_episode()
            queue_of(policy).clear()
            states, actions = [], []
            success = False
            for t in range(args.horizon):
                states.append(env.get_state()["states"])
                # standard receding-horizon execution (policy's own Ta via queue)
                a = policy(ob=obs)
                actions.append(np.asarray(a))
                obs, _, done, _ = env.step(a)
                if env.is_success()["task"]:
                    success = True
                    break
                if done:
                    break
            g = grp.create_group(f"rollout_{ep}")
            g.attrs["model_file"] = model_xml
            g.attrs["success"] = bool(success)
            g.create_dataset("states", data=np.asarray(states))
            g.create_dataset("actions", data=np.asarray(actions))
            n_succ += int(success)
        grp.attrs["meta"] = json.dumps({
            "ckpt": args.ckpt, "n": args.n, "horizon": args.horizon,
            "seed": args.seed, "success_rate": n_succ / args.n})
    print(f"COLLECTED {args.n} rollouts ({n_succ} successes, "
          f"{args.n - n_succ} failures) -> {args.out}")


if __name__ == "__main__":
    main()
