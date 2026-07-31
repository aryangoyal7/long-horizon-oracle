"""Inference-only replan triggers from the literature, run as controllers
(RoboCasa + GR00T N1.5, 2026-07-31).

Same executor as our own switching runs: play k_stable actions open-loop,
drop to k_unstable when the trigger fires. Only the trigger changes, so the
comparison isolates the signal rather than the surrounding system. None of
these needs the simulator or any retraining; each is computed from the chunk
the policy already produced, or from a small number of extra policy samples.

  spread         cross-sample variance of repeated chunk draws (DVAC family;
                 DVAC itself reads variance inside one denoising pass, which
                 needs the flow head's intermediates, so this is the
                 sampling-based stand-in)
  cons           L2 disagreement between the queued chunk tail and a fresh
                 sample (BID family)
  cons_cos       cosine between those two (SGAC), fires when LOW
  cmd_speed_min  smallest smoothed commanded speed in the predicted chunk
                 (PACE), fires when LOW

Thresholds come from results/robocasa/proxy/baseline_thresholds.json, set at
the quantile of each signal's own online distribution that reproduces the
oracle's firing rate for that task, so every controller gets an equal
intervention budget.

Run (groot155, from the fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_baseline_switch.py \
    --env-name TurnOnSinkFaucet --signal spread --out out.json
"""

import argparse
import json
import os
import time

import numpy as np

from groot_oracle_rollout import ACTION_KEYS, CKPT, policy_obs

N_SPREAD = 3


def chunk_mat(chunk, n):
    return np.stack([np.concatenate([np.ravel(np.asarray(chunk[k])[j])
                                     for k in ACTION_KEYS])
                     for j in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--signal", required=True,
                    choices=["spread", "cons", "cons_cos", "cmd_speed_min"])
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    tpath = args.thresholds or os.path.join(
        root, "results", "robocasa", "proxy", "baseline_thresholds.json")
    tab = json.load(open(tpath))[f"{args.env_name}|{args.signal}"]
    thr, direction = float(tab["thresh"]), tab["dir"]

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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        chunk, chunk_len, executed = None, 0, 0
        k_current = args.k_stable
        k_log, replan_log = [], []
        success, steps = False, 0
        pending_cons = None
        for t in range(horizon):
            if chunk is None or executed >= min(k_current, chunk_len):
                pobs = policy_obs(obs)
                chunk = policy.get_action(pobs)
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                if args.signal == "spread":
                    mat = chunk_mat(chunk, min(8, chunk_len))
                    all_m = [mat] + [chunk_mat(policy.get_action(pobs),
                                               mat.shape[0])
                                     for _ in range(N_SPREAD)]
                    pair = [float(np.linalg.norm(a - b, axis=1).mean())
                            for i, a in enumerate(all_m) for b in all_m[i + 1:]]
                    score = float(np.mean(pair))
                elif args.signal == "cmd_speed_min":
                    prof = np.linalg.norm(
                        np.asarray(chunk["action.end_effector_position"]),
                        axis=1)
                    if len(prof) >= 3:
                        prof = np.convolve(prof, np.ones(3) / 3, mode="valid")
                    score = float(prof.min())
                else:
                    # consistency signals compare the queued chunk against a
                    # fresh sample drawn at the same observation
                    n = min(8, chunk_len)
                    fresh = chunk_mat(policy.get_action(pobs), n)
                    tail = chunk_mat(chunk, n)
                    if args.signal == "cons":
                        score = float(np.linalg.norm(fresh - tail, axis=1).mean())
                    else:
                        fa, ta = fresh.ravel(), tail.ravel()
                        den = float(np.linalg.norm(fa) * np.linalg.norm(ta))
                        score = float(fa @ ta / den) if den > 0 else 0.0
                fire = score > thr if direction == "hi" else score < thr
                k_current = args.k_unstable if fire else args.k_stable
                replan_log.append({"t": t, "score": round(score, 5),
                                   "k": int(k_current)})
            act = {k: np.asarray(chunk[k])[executed: executed + 1]
                   for k in ACTION_KEYS}
            obs, _r, term, trunc, info = env.step(act)
            executed += 1
            k_log.append(k_current)
            steps = t + 1
            if bool(np.ravel(info.get("success", False))[0]):
                success = True
                break
            if term or trunc:
                break
        fired = [r for r in replan_log if r["k"] == args.k_unstable]
        episodes.append({"episode": ep, "success": bool(success), "steps": steps,
                         "mean_k": float(np.mean(k_log)),
                         "frac_unstable_calls": len(fired) / max(len(replan_log), 1),
                         "replans": replan_log})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps}, "
              f"mean_k {episodes[-1]['mean_k']:.1f}, fired "
              f"{episodes[-1]['frac_unstable_calls']:.2f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
    env.close()
    rate = float(np.mean([e["success"] for e in episodes]))
    json.dump({"env": args.env_name, "mode": "baseline_switch",
               "signal": args.signal, "thresh": thr, "dir": direction,
               "k_stable": args.k_stable, "k_unstable": args.k_unstable,
               "seed": args.seed, "n_episodes": args.n_episodes,
               "horizon": horizon, "success_rate": rate,
               "mean_k": float(np.mean([e["mean_k"] for e in episodes])),
               "mean_frac_unstable": float(np.mean(
                   [e["frac_unstable_calls"] for e in episodes])),
               "episodes": episodes}, open(args.out, "w"), indent=2)
    print(f"SUCCESS_RATE {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
