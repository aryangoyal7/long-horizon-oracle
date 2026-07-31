"""GR00T RoboCasa switching on the MEASURED stability signal (oracle).

At each replan the policy's 16-step chunk is replayed open-loop from the
current sim state: once unperturbed, then N_PROBE times with the first
action's arm delta-position dims perturbed by sigma_u (exactly the labeler's
procedure, K limited to the chunk length). lambda = OLS slope of the mean log
divergence of eef_pos+gripper_qpos. The executed chunk length is k_unstable
if lambda > delta else k_stable. State is restored exactly before execution.

--control contact instead switches on the privileged regime flag: gripper in
contact with a movable body (the carry phase signal from labeling).

Run (groot155, from the fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_oracle_rollout.py \
    --env-name PickPlaceCounterToCabinet --control oracle --delta 0.036 \
    --k-stable 16 --k-unstable 4 --n-episodes 50 --out out.json
"""

import argparse
import json
import os
import time

import numpy as np

CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"
ACTION_KEYS = ["action.gripper_close", "action.end_effector_position",
               "action.end_effector_rotation", "action.base_motion",
               "action.control_mode"]
N_PROBE = 8
SIGMA_U = 0.05           # matches the labeling campaign
DIV_FLOOR = 1e-12


def policy_obs(obs):
    out = {}
    for k, v in obs.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = [v.item()]
        else:
            out[k] = v
    return out


def fit_slope(log_d):
    taus = np.arange(1, log_d.shape[-1] + 1, dtype=np.float64)
    y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def to_env_action(genv, kenv, action_dict):
    """Replicate RoboCasaGymEnv.step's dict -> flat robosuite vector."""
    from robosuite.controllers.composite.composite_controller import HybridMobileBase
    ad = genv.key_converter.unmap_action(dict(action_dict))
    env_action = []
    for robot in kenv.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        action = np.zeros(cc.action_limits[0].shape)
        for part_name, controller in cc.part_controllers.items():
            s, e = cc._action_split_indexes[part_name]
            action[s:e] = ad.pop(f"{pf}{part_name}")
        if isinstance(cc, HybridMobileBase):
            action[-1] = ad.pop(f"{pf}base_mode")
        env_action.append(action)
    return np.concatenate(env_action)


def arm_pos_index(kenv):
    cc = kenv.robots[0].composite_controller
    split = cc._action_split_indexes
    for name in ("right", "arm_right", "right_arm", "arm"):
        if name in split:
            return int(split[name][0])
    for name, (s, e) in split.items():
        if not any(x in name for x in ("gripper", "base", "torso", "mode")):
            return int(s)
    raise RuntimeError(f"cannot resolve arm dims; split={split}")


def task_vec(kenv):
    obs = kenv._get_observations(force_update=False)
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).ravel(),
                           np.asarray(obs["robot0_gripper_qpos"]).ravel()])


def set_images_active(kenv, active):
    try:
        for name in list(kenv._observables):
            if "image" in name:
                kenv.modify_observable(name, "active", active)
    except Exception:
        pass


def gripper_on_movable(kenv):
    sim = kenv.sim
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = (sim.model.geom_id2name(c.geom1) or f"id{c.geom1}").lower()
        n2 = (sim.model.geom_id2name(c.geom2) or f"id{c.geom2}").lower()
        g1 = "gripper" in n1 or "finger" in n1
        g2 = "gripper" in n2 or "finger" in n2
        if not (g1 or g2) or (g1 and g2):
            continue
        other = c.geom2 if g1 else c.geom1
        body = sim.model.geom_bodyid[other]
        if sim.model.body_dofnum[sim.model.body_rootid[body]] >= 6:
            return True
    return False


def measure_lambda(kenv, acts, p0, rng):
    """Open-loop FTLE of the chunk from the current sim state; restores it."""
    flat0 = kenv.sim.get_state().flatten()
    set_images_active(kenv, False)
    K = len(acts)
    nom = []
    for a in acts:
        kenv.step(a)
        nom.append(task_vec(kenv))
    nom = np.asarray(nom)
    log_d = np.empty((N_PROBE, K))
    for n in range(N_PROBE):
        a0 = acts[0].copy()
        a0[p0:p0 + 3] += SIGMA_U * rng.standard_normal(3)
        kenv.sim.set_state_from_flattened(flat0)
        kenv.sim.forward()
        for j in range(K):
            kenv.step(a0 if j == 0 else acts[j])
            d = np.linalg.norm(task_vec(kenv) - nom[j])
            log_d[n, j] = np.log(max(d, DIV_FLOOR))
    kenv.sim.set_state_from_flattened(flat0)
    kenv.sim.forward()
    set_images_active(kenv, True)
    return fit_slope(log_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--control", choices=["oracle", "contact"],
                    default="oracle")
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None,
                    help="record per-episode videos (steps_per_render=1)")
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
        video=VideoConfig(video_dir=args.video_dir,
                          steps_per_render=1) if args.video_dir
        else VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=1,
                                  max_episode_steps=horizon))
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
        k_log, replan_log = [], []
        success, steps = False, 0
        for t in range(horizon):
            replan = (chunk is None or executed >= min(k_current, chunk_len))
            if replan:
                chunk = policy.get_action(policy_obs(obs))
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                if args.control == "oracle":
                    acts = [to_env_action(
                        genv, kenv,
                        {k: np.asarray(chunk[k])[j] for k in ACTION_KEYS})
                        for j in range(chunk_len)]
                    lam = measure_lambda(kenv, acts, p0, rng)
                    fire = lam > args.delta
                    replan_log.append({"t": t, "lam": round(lam, 5),
                                       "k": int(args.k_unstable if fire
                                                else args.k_stable)})
                else:
                    fire = gripper_on_movable(kenv)
                    replan_log.append({"t": t, "contact": bool(fire),
                                       "k": int(args.k_unstable if fire
                                                else args.k_stable)})
                k_current = args.k_unstable if fire else args.k_stable
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
        episodes.append({
            "episode": ep, "success": bool(success), "steps": steps,
            "mean_k": float(np.mean(k_log)),
            "frac_unstable_calls": len(fired) / max(len(replan_log), 1),
            "replans": replan_log})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps}, "
              f"mean_k {episodes[-1]['mean_k']:.1f}, fired "
              f"{episodes[-1]['frac_unstable_calls']:.2f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    if args.video_dir:
        env.reset()   # finalizes the last recorded video file
    env.close()
    rate = float(np.mean([e["success"] for e in episodes]))
    result = {"env": args.env_name, "mode": args.control,
              "sigma_u": SIGMA_U, "n_probe": N_PROBE,
              "delta": args.delta, "k_stable": args.k_stable,
              "k_unstable": args.k_unstable, "seed": args.seed,
              "n_episodes": args.n_episodes, "horizon": horizon,
              "success_rate": rate,
              "mean_k": float(np.mean([e["mean_k"] for e in episodes])),
              "mean_frac_unstable": float(np.mean(
                  [e["frac_unstable_calls"] for e in episodes])),
              "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"SUCCESS_RATE {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
