"""Shadow rollout instrumented with competing replan-trigger proxies
(2026-07-30). Always executes full 16-step chunks (no switching); at every
replan it measures, from the SAME state:
  lam    - open-loop divergence rate (the oracle yardstick)
  spread - policy sample variance: 3 extra chunk samples at the same obs,
           mean pairwise per-step L2 over the first 8 steps ("denoising
           variance" proxy)
  speed  - end-effector speed at the replan step ("speed minima" proxy)
and mid-chunk (8 steps in) queries the policy once more:
  cons   - disagreement between the fresh chunk's first 8 actions and the
           executing chunk's remaining 8 ("chunk consistency" proxy),
           attached to the current replan record
Output mirrors the shadow cells; analysis (Spearman/AUC vs lam) is offline.

Run (groot155, fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_proxy_shadow.py \
    --env-name X --delta <d> --n-episodes 50 --out out.json
"""
import argparse
import json
import os
import time

import numpy as np

from groot_oracle_rollout import (ACTION_KEYS, CKPT, arm_pos_index,
                                  measure_lambda, policy_obs, task_vec,
                                  to_env_action)

N_SPREAD = 3


def chunk_mat(chunk, n):
    return np.stack([np.concatenate([np.ravel(np.asarray(chunk[k])[j])
                                     for k in ACTION_KEYS])
                     for j in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from gr00t.eval.simulation import (MultiStepConfig, SimulationConfig,
                                       VideoConfig, _create_single_env)
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    dc = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(
        model_path=CKPT, modality_config=dc.modality_config(),
        modality_transform=dc.transform(), embodiment_tag="new_embodiment",
        denoising_steps=4)

    horizon = get_task_horizon(args.env_name)
    cfg = SimulationConfig(
        env_name=f"robocasa/{args.env_name}", split=args.split,
        n_episodes=args.n_episodes, n_envs=1,
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=1, max_episode_steps=horizon))
    env = _create_single_env(cfg, 0)
    genv = env.unwrapped
    kenv = genv.env
    p0 = arm_pos_index(kenv)
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        chunk, chunk_len, executed = None, 0, 0
        replan_log = []
        success, steps = False, 0
        prev_eef = task_vec(kenv)[:3]
        for t in range(horizon):
            eef = task_vec(kenv)[:3]
            speed = float(np.linalg.norm(eef - prev_eef))
            prev_eef = eef
            if chunk is None or executed >= chunk_len:
                pobs = policy_obs(obs)
                chunk = policy.get_action(pobs)
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                mat = chunk_mat(chunk, min(8, chunk_len))
                extras = [chunk_mat(policy.get_action(pobs), mat.shape[0])
                          for _ in range(N_SPREAD)]
                all_m = [mat] + extras
                pair = [float(np.linalg.norm(a - b, axis=1).mean())
                        for i, a in enumerate(all_m)
                        for b in all_m[i + 1:]]
                acts = [to_env_action(
                    genv, kenv,
                    {k: np.asarray(chunk[k])[j] for k in ACTION_KEYS})
                    for j in range(chunk_len)]
                lam = measure_lambda(kenv, acts, p0, rng)
                # PACE-style signal: speed valleys of the PREDICTED chunk --
                # per-step commanded eef translation magnitude, smoothed
                prof = np.linalg.norm(
                    np.asarray(chunk["action.end_effector_position"]), axis=1)
                if len(prof) >= 3:
                    prof = np.convolve(prof, np.ones(3) / 3, mode="valid")
                replan_log.append({"t": t, "lam": round(lam, 5),
                                   "spread": round(float(np.mean(pair)), 5),
                                   "speed": round(speed, 5),
                                   "cmd_speed_min": round(float(prof.min()), 5),
                                   "cmd_speed_mean": round(float(prof.mean()), 5)})
            if executed == 8 and chunk_len >= 16:
                fresh = chunk_mat(policy.get_action(policy_obs(obs)), 8)
                tail = chunk_mat(chunk, 16)[8:16]
                replan_log[-1]["cons"] = round(
                    float(np.linalg.norm(fresh - tail, axis=1).mean()), 5)
                # SGAC-style signal: cosine similarity between the queued
                # chunk's remainder and the freshly sampled chunk's head
                fa, ta = fresh.ravel(), tail.ravel()
                den = float(np.linalg.norm(fa) * np.linalg.norm(ta))
                replan_log[-1]["cons_cos"] = round(
                    float(fa @ ta / den) if den > 0 else 0.0, 5)
            act = {k: np.asarray(chunk[k])[executed: executed + 1]
                   for k in ACTION_KEYS}
            obs, _r, term, trunc, info = env.step(act)
            executed += 1
            steps = t + 1
            if bool(np.ravel(info.get("success", False))[0]):
                success = True
                break
            if term or trunc:
                break
        episodes.append({"episode": ep, "success": bool(success),
                         "steps": steps, "replans": replan_log})
        n_u = sum(1 for x in replan_log if x["lam"] > args.delta)
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps}, "
              f"{len(replan_log)} replans, {n_u} unstable "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    env.close()
    result = {"env": args.env_name, "mode": "proxy_shadow",
              "delta": args.delta, "seed": args.seed,
              "n_episodes": args.n_episodes, "horizon": horizon,
              "n_spread": N_SPREAD,
              "success_rate": float(np.mean([e["success"] for e in episodes])),
              "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
