"""Closed-loop FTLE label generator (secondary/diagnostic channel).

Implements the closed-loop variant from implementation_plan §1: same perturb-and-replay
pairs as the open-loop labeler, but in each perturbed branch the TRAINED POLICY acts
from its own observations with replanning at every step (k=1), instead of replaying the
recorded actions. lambda_cl(t) is the OLS slope of mean log divergence between the
policy branch and the nominal demo-replay branch.

The lambda_cl vs lambda_ol comparison per segment verifies the theory's mechanism:
feedback injects estimation error in contracting segments (cl worse than ol there) and
corrects errors in expansive segments (ordering flips).

Cost note: every perturbed step is a policy inference (N x K per stamp), so this runs
on GPU and on a SUBSAMPLE (default stride 10, 50 demos, N=4) — the CL channel is
diagnostic, not the primary training label.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python ftle_labeler_cl.py \
      --dataset /mnt/scratch/lh/data/robomimic/square/ph/image_v15.hdf5 \
      --ckpt <best.pth> --output labels_cl_square.npz --sigma-u <policy pos rmse>
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ftle_labeler import fit_slope, task_vec  # noqa: E402

DIV_FLOOR = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="image hdf5 (policy needs pixels)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sigma-u", type=float, required=True)
    ap.add_argument("--sigma-u-source", default="policy_rmse")
    ap.add_argument("--k-horizon", type=int, default=16)
    ap.add_argument("--n-probe", type=int, default=4)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-demos", type=int, default=50)
    ap.add_argument("--pos-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "eval"))
    from eval_k_sweep import load_policy, queue_of
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = load_policy(args.ckpt, device)
    cfg = json.loads(ckpt_dict["config"])
    obs_keys = (cfg["observation"]["modalities"]["obs"]["low_dim"]
                + cfg["observation"]["modalities"]["obs"]["rgb"])
    n_stack = cfg["train"]["frame_stack"]

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(obs=dict(
        low_dim=cfg["observation"]["modalities"]["obs"]["low_dim"],
        rgb=cfg["observation"]["modalities"]["obs"]["rgb"])))
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)

    rng = np.random.default_rng(args.seed)
    n_pert = 3 if args.pos_only else 6
    K = args.k_horizon

    demo_ids, stamps, lam_cl_task, lam_cl_full = [], [], [], []
    t_start = time.time()
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        demos = demos[: args.max_demos]
        for dk in demos:
            g = f[f"data/{dk}"]
            states = g["states"][()]
            actions = g["actions"][()]
            T = actions.shape[0]
            env.reset()
            env.reset_to({"model": g.attrs["model_file"]})
            for t0 in range(0, T - K, args.stride):
                # nominal branch: demo action replay (identical to the OL labeler)
                env.reset_to({"states": states[t0]})
                nom_task, nom_full = [], []
                for j in range(K):
                    obs, _, _, _ = env.step(actions[t0 + j])
                    nom_task.append(task_vec(obs))
                    nom_full.append(env.get_state()["states"][1:])
                nom_task = np.asarray(nom_task); nom_full = np.asarray(nom_full)

                log_d_task = np.empty((args.n_probe, K))
                log_d_full = np.empty((args.n_probe, K))
                for n in range(args.n_probe):
                    env.reset_to({"states": states[t0]})
                    obs = env.get_observation()
                    hist = [obs] * n_stack          # frame history for the policy
                    policy.start_episode()
                    a0 = actions[t0].copy()
                    a0[:n_pert] += args.sigma_u * rng.standard_normal(n_pert)
                    for j in range(K):
                        if j == 0:
                            a = a0                  # identical injection to OL labeler
                        else:
                            ob = {k: np.stack([h[k] for h in hist[-n_stack:]])
                                  for k in obs_keys}
                            queue_of(policy).clear()   # k=1: fresh inference each step
                            a = policy(ob=ob)
                        obs, _, _, _ = env.step(np.asarray(a))
                        hist.append(obs)
                        dt_ = np.linalg.norm(task_vec(obs) - nom_task[j])
                        df_ = np.linalg.norm(env.get_state()["states"][1:] - nom_full[j])
                        log_d_task[n, j] = np.log(max(dt_, DIV_FLOOR))
                        log_d_full[n, j] = np.log(max(df_, DIV_FLOOR))

                demo_ids.append(int(dk.split("_")[1]))
                stamps.append(t0)
                lam_cl_task.append(fit_slope(log_d_task))
                lam_cl_full.append(fit_slope(log_d_full))
            done = len(set(demo_ids))
            print(f"[{done}/{len(demos)}] {dk}: {len(stamps)} stamps total, "
                  f"{time.time()-t_start:.0f}s elapsed", flush=True)

    lam = np.array(lam_cl_task)
    np.savez(
        args.output,
        demo_id=np.array(demo_ids), t=np.array(stamps),
        lambda_cl_task=lam, lambda_cl_full=np.array(lam_cl_full),
        meta=json.dumps({
            "dataset": args.dataset, "ckpt": args.ckpt, "K": K, "N": args.n_probe,
            "stride": args.stride, "max_demos": args.max_demos,
            "pos_only": args.pos_only, "sigma_u": args.sigma_u,
            "sigma_u_source": args.sigma_u_source, "seed": args.seed,
            "mode": "closed_loop (policy replans every step in perturbed branch)",
        }))
    print(f"DONE {len(stamps)} stamps in {time.time()-t_start:.0f}s -> {args.output}")
    print(f"lambda_cl_task: mean={lam.mean():.4f} frac>0={100*(lam>0).mean():.1f}% "
          f"p5={np.percentile(lam,5):.3f} p95={np.percentile(lam,95):.3f}")


if __name__ == "__main__":
    main()
