"""Paired fork test: does intervening exactly at the oracle's fired moments
cause the win?

Runs GR00T in oracle (16,1) mode. At the FIRST replan whose measured lambda
exceeds delta, the sim state is snapshotted and two continuations run from
the identical state with the identical chunk:
  branch B (control): the chunk executes fully open-loop and the episode
    continues in pure fixed-16 mode, no further lambda measurement;
  branch A (intervention): state restored, normal oracle behavior continues
    (k=1 while lambda > delta).
Both branches get the same remaining step budget. Episodes that never fire
report a single outcome (the no-fire stratum). Cross-run pairing is invalid
on RoboCasa (episode inits are not index-deterministic), so this is the
only clean way to isolate the effect to the flagged segments: same init,
same prefix, same state, same chunk - the branches differ only in whether
we act on the detection.
"""
import argparse
import json
import os
import time

import numpy as np

from groot_oracle_rollout import (ACTION_KEYS, CKPT, arm_pos_index,
                                  measure_lambda, policy_obs, to_env_action)


def slice_act(chunk, j):
    return {k: np.asarray(chunk[k])[j: j + 1] for k in ACTION_KEYS}


def step_success(info):
    return bool(np.ravel(info.get("success", False))[0])


def run_branch_b(env, policy, chunk, budget):
    """Open-loop continuation: finish this chunk, then pure fixed-16."""
    chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
    executed, t, success, obs = 0, 0, False, None
    while t < budget and not success:
        if executed >= chunk_len:
            chunk = policy.get_action(policy_obs(obs))
            chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
            executed = 0
        obs, _r, _term, _trunc, info = env.step(slice_act(chunk, executed))
        executed += 1
        t += 1
        if step_success(info):
            success = True
    return success, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=1)
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
        multistep=MultiStepConfig(n_action_steps=1,
                                  max_episode_steps=3 * horizon))
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
        k_current = args.k_stable
        replan_log, fork = [], None
        success, t = False, 0
        while t < horizon and not success:
            if chunk is None or executed >= min(k_current, chunk_len):
                chunk = policy.get_action(policy_obs(obs))
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                acts = [to_env_action(
                    genv, kenv,
                    {k: np.asarray(chunk[k])[j] for k in ACTION_KEYS})
                    for j in range(chunk_len)]
                lam = measure_lambda(kenv, acts, p0, rng)
                fire = lam > args.delta
                k_current = args.k_unstable if fire else args.k_stable
                replan_log.append({"t": t, "lam": round(lam, 5),
                                   "k": int(k_current)})
                if fire and fork is None:
                    flat0 = kenv.sim.get_state().flatten()
                    succ_b, used_b = run_branch_b(env, policy, chunk,
                                                  horizon - t)
                    kenv.sim.set_state_from_flattened(flat0)
                    kenv.sim.forward()
                    fork = {"t_fire": t, "lam_fire": round(lam, 5),
                            "success_openloop": bool(succ_b),
                            "steps_openloop": used_b}
            obs, _r, _term, _trunc, info = env.step(slice_act(chunk, executed))
            executed += 1
            t += 1
            if step_success(info):
                success = True
        episodes.append({"episode": ep, "success_oracle": bool(success),
                         "steps": t, "fork": fork, "replans": replan_log})
        tag = (f"fired t={fork['t_fire']} A={int(success)} "
               f"B={int(fork['success_openloop'])}" if fork
               else f"nofire A={int(success)}")
        print(f"ep {ep}: {tag} ({time.time()-t0:.0f}s elapsed)", flush=True)

    env.close()
    fired = [e for e in episodes if e["fork"]]
    nofire = [e for e in episodes if not e["fork"]]
    a = [e["success_oracle"] for e in fired]
    b = [e["fork"]["success_openloop"] for e in fired]
    pairs = {"n11": sum(x and y for x, y in zip(a, b)),
             "n10": sum(x and not y for x, y in zip(a, b)),
             "n01": sum((not x) and y for x, y in zip(a, b)),
             "n00": sum((not x) and (not y) for x, y in zip(a, b))}
    result = {"env": args.env_name, "mode": "fork", "delta": args.delta,
              "k_stable": args.k_stable, "k_unstable": args.k_unstable,
              "seed": args.seed, "n_episodes": args.n_episodes,
              "horizon": horizon,
              "n_fired": len(fired), "n_nofire": len(nofire),
              "rate_oracle_fired": float(np.mean(a)) if a else None,
              "rate_openloop_fired": float(np.mean(b)) if b else None,
              "rate_nofire": (float(np.mean([e["success_oracle"]
                                             for e in nofire]))
                              if nofire else None),
              "paired": pairs, "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"FORK_DONE fired={len(fired)} A={result['rate_oracle_fired']} "
          f"B={result['rate_openloop_fired']} pairs={pairs}", flush=True)


if __name__ == "__main__":
    main()
