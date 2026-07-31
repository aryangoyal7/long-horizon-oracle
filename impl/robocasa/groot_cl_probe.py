"""Open-loop AND closed-loop stability, measured at the SAME states, online
(RoboCasa + GR00T N1.5, 2026-07-31).

Why this exists: the robomimic closed-loop labeler (impl/labeler/ftle_labeler_cl.py)
compares a policy-controlled perturbed branch against a DEMO-REPLAY nominal, so its
lambda_cl mixes perturbation growth with the policy's ordinary drift away from the
demonstration. That makes lambda_cl and lambda_ol incomparable and leaves the
"closed-loop stable" bin essentially empty by construction. Here both branches use
the SAME control mode and differ only by the injected perturbation:

  lambda_ol   nominal = chunk replayed open-loop from s0
              probes  = same chunk, first action perturbed
  lambda_cl   nominal = policy replanning EVERY step from s0 (unperturbed)
              probes  = policy replanning every step, first action perturbed
  lambda_ctrl nominal = as above; probes = policy replanning every step, NO
              perturbation. Two stochastic rollouts from one state already diverge,
              so this is the sampling floor that lambda_cl must be read against.

The episode itself always executes full 16-step chunks (pure shadow, no switching),
so the state distribution is the deployed one. State is saved and restored exactly
around every probe.

Run (groot155, from the fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_cl_probe.py \
    --env-name TurnOnSinkFaucet --n-episodes 25 --out out.json
"""

import argparse
import json
import os
import time

import numpy as np

from groot_oracle_rollout import (ACTION_KEYS, CKPT, arm_pos_index, fit_slope,
                                  gripper_on_movable, measure_lambda,
                                  policy_obs, task_vec, to_env_action)

DIV_FLOOR = 1e-12
SIGMA_U = 0.05          # matches the open-loop labeling campaign and the oracle


def make_obs_shaper(ref):
    """The wrapper stacks a leading step axis onto every array. Record the
    reference ndim per key so probe observations rebuilt straight from the
    sim match what the policy saw at reset."""
    ndim = {k: (np.asarray(v).ndim if not isinstance(v, str) else None)
            for k, v in ref.items()}

    def shape(o):
        out = {}
        for k, v in o.items():
            if isinstance(v, str) or ndim.get(k) is None:
                out[k] = v
                continue
            a = np.asarray(v)
            while a.ndim < ndim[k]:
                a = a[None]
            out[k] = a
        return out
    return shape


def cl_branch(genv, kenv, policy, shaper, flat0, K, first_action=None):
    """Restore s0 and run the policy closed-loop (replan every step) for K
    steps. If first_action is given it is executed at step 0 instead of the
    policy's. Returns the task-space trajectory."""
    kenv.sim.set_state_from_flattened(flat0)
    kenv.sim.forward()
    traj = []
    for j in range(K):
        if j == 0 and first_action is not None:
            a = first_action
        else:
            raw = kenv._get_observations(force_update=True)
            o = shaper(genv.get_observation(raw))
            chunk = policy.get_action(policy_obs(o))
            a = to_env_action(genv, kenv,
                              {k: np.asarray(chunk[k])[0] for k in ACTION_KEYS})
        kenv.step(a)
        traj.append(task_vec(kenv))
    return np.asarray(traj)


def measure_lambda_cl(genv, kenv, policy, shaper, acts, p0, rng,
                      n_probe, n_ctrl):
    """Closed-loop divergence rate, plus the policy's own sampling floor."""
    flat0 = kenv.sim.get_state().flatten()
    K = len(acts)
    nom = cl_branch(genv, kenv, policy, shaper, flat0, K)

    def slope_against_nom(traj):
        d = np.linalg.norm(traj - nom, axis=1)
        return np.log(np.maximum(d, DIV_FLOOR))

    logs = []
    for _ in range(n_probe):
        a0 = acts[0].copy()
        a0[p0:p0 + 3] += SIGMA_U * rng.standard_normal(3)
        logs.append(slope_against_nom(
            cl_branch(genv, kenv, policy, shaper, flat0, K, first_action=a0)))
    ctrl = []
    for _ in range(n_ctrl):
        ctrl.append(slope_against_nom(
            cl_branch(genv, kenv, policy, shaper, flat0, K)))

    kenv.sim.set_state_from_flattened(flat0)
    kenv.sim.forward()
    lam = fit_slope(np.asarray(logs))
    lam_ctrl = fit_slope(np.asarray(ctrl)) if ctrl else float("nan")
    return lam, lam_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--n-episodes", type=int, default=25)
    ap.add_argument("--n-probe-cl", type=int, default=4)
    ap.add_argument("--n-ctrl", type=int, default=2)
    ap.add_argument("--cl-every", type=int, default=1,
                    help="probe closed-loop on every Nth replan (cost control)")
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
    shaper = None
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        if shaper is None:
            shaper = make_obs_shaper(obs)
        chunk, chunk_len, executed = None, 0, 0
        probes = []
        success, steps = False, 0
        n_replan = 0
        for t in range(horizon):
            if chunk is None or executed >= chunk_len:
                chunk = policy.get_action(policy_obs(obs))
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                acts = [to_env_action(
                    genv, kenv,
                    {k: np.asarray(chunk[k])[j] for k in ACTION_KEYS})
                    for j in range(chunk_len)]
                lam_ol = measure_lambda(kenv, acts, p0, rng)
                rec = {"t": t, "lam_ol": round(lam_ol, 5),
                       "contact": bool(gripper_on_movable(kenv))}
                if n_replan % args.cl_every == 0:
                    lam_cl, lam_ctrl = measure_lambda_cl(
                        genv, kenv, policy, shaper, acts, p0, rng,
                        args.n_probe_cl, args.n_ctrl)
                    rec["lam_cl"] = round(lam_cl, 5)
                    rec["lam_ctrl"] = round(lam_ctrl, 5)
                n_replan += 1
                probes.append(rec)
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
                         "steps": steps, "probes": probes})
        nc = sum(1 for r in probes if "lam_cl" in r)
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps}, "
              f"{len(probes)} replans ({nc} cl-probed) "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        with open(args.out, "w") as f:      # checkpoint after every episode
            json.dump({"env": args.env_name, "mode": "cl_probe",
                       "sigma_u": SIGMA_U, "n_probe_cl": args.n_probe_cl,
                       "n_ctrl": args.n_ctrl, "cl_every": args.cl_every,
                       "seed": args.seed, "horizon": horizon,
                       "n_episodes_done": len(episodes),
                       "n_episodes": args.n_episodes,
                       "success_rate": float(np.mean([e["success"] for e in episodes])),
                       "episodes": episodes}, f, indent=2)
    env.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
