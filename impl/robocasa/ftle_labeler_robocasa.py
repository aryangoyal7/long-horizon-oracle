"""RoboCasa variant of the open-loop FTLE labeler (see ftle_labeler.py).

Differences:
- every demo carries its own kitchen scene (model_file + ep_meta attrs); the
  per-demo reset follows robocasa's playback recipe:
  set_ep_meta -> reset -> edit_model_xml -> reset_from_xml_string
- the robot is PandaOmron with a 12-dim HYBRID_MOBILE_BASE action; the
  perturbed dims are the arm end-effector delta-position dims, resolved from
  the composite controller's action split at runtime (never assumed)
- task_vec: eef pos + gripper qpos (kitchen envs have no single "object" obs)
- contact flags: free space = gripper touching nothing; the gripper/fixture
  channels record gripper-vs-movable vs gripper-vs-static contact
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

K_HORIZON = 24
N_PROBE = 8
STRIDE = 2
DIV_FLOOR = 1e-12


def fit_slope(log_d):
    taus = np.arange(1, log_d.shape[-1] + 1, dtype=np.float64)
    y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def make_env(env_meta):
    # robocasa's own robomimic-style wrapper: stock robomimic queries the action
    # spec before robosuite 1.5 initializes composite controllers and crashes
    # the wrapper uses robocasa's vendored obs_utils, not robomimic's
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(
        obs=dict(low_dim=["robot0_eef_pos", "robot0_eef_quat",
                          "robot0_gripper_qpos", "robot0_base_pos"], rgb=[])))
    from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.pop("env_name", None)
    return EnvRobocasa(env_name=env_meta["env_name"], render=False,
                       render_offscreen=False, use_image_obs=False,
                       postprocess_visual_obs=False, **kwargs)


def arm_pos_indices(env):
    """Start index of the arm's 3 delta-position action dims."""
    robot = env.env.robots[0]
    cc = getattr(robot, "composite_controller", None)
    split = getattr(cc, "_action_split_indexes", None) if cc else None
    if split:
        for name in ("right", "arm_right", "right_arm", "arm"):
            if name in split:
                s, e = split[name]
                return int(s)
        # fall back to the first non-gripper, non-base part
        for name, (s, e) in split.items():
            if not any(k in name for k in ("gripper", "base", "torso", "mode")):
                return int(s)
    raise RuntimeError(f"cannot resolve arm action dims; split={split}")


def reset_demo(env, model_xml, ep_meta_json):
    env.reset_to({"model": model_xml, "ep_meta": ep_meta_json})


def set_state(env, flat_state):
    env.reset_to({"states": np.asarray(flat_state)})


def task_vec(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).ravel(),
                           np.asarray(obs["robot0_gripper_qpos"]).ravel()])


def contact_flags(env):
    """(gripper-movable, unused, gripper-static): free space = no gripper contact."""
    sim = env.env.sim
    g_mov = g_stat = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = (sim.model.geom_id2name(c.geom1) or f"id{c.geom1}").lower()
        n2 = (sim.model.geom_id2name(c.geom2) or f"id{c.geom2}").lower()
        g1 = "gripper" in n1 or "finger" in n1
        g2 = "gripper" in n2 or "finger" in n2
        if not (g1 or g2) or (g1 and g2):
            continue
        other_geom = c.geom2 if g1 else c.geom1
        body = sim.model.geom_bodyid[other_geom]
        # movable = body has a free joint root; static fixtures/cabinet frames don't
        if sim.model.body_dofnum[sim.model.body_rootid[body]] >= 6:
            g_mov = True
        else:
            g_stat = True
    return g_mov, False, g_stat


def lerobot_to_env(a):
    """rc_*.hdf5 actions are LeRobot-ordered [base3, torso1, mode1, arm6,
    grip1]; the composite controller consumes [arm6, grip1, base3, torso1,
    mode1]. Validated by demo replay: raw fails ~1 m off, remapped succeeds
    within 4-7 cm."""
    out = np.empty_like(a)
    out[..., 0:6] = a[..., 5:11]
    out[..., 6] = a[..., 11]
    out[..., 7:10] = a[..., 0:3]
    out[..., 10] = a[..., 3]
    out[..., 11] = a[..., 4]
    return out


def process_demo(args):
    (env_meta, demo_key, model_xml, ep_meta_json, states, actions, sigma_u,
     seed, k_horizon) = args
    env = _WORKER_ENV[0]
    if env is None:
        env = make_env(env_meta)
        _WORKER_ENV[0] = env
    rng = np.random.default_rng(seed)

    reset_demo(env, model_xml, ep_meta_json)
    p0 = arm_pos_indices(env)

    T = actions.shape[0]
    out = []
    for t0 in range(0, T - k_horizon, STRIDE):
        set_state(env, states[t0])
        flags = contact_flags(env)

        nom_task, nom_full = [], []
        win_flags = np.array(flags, dtype=bool)
        set_state(env, states[t0])
        for j in range(k_horizon):
            obs, _, _, _ = env.step(actions[t0 + j])
            nom_task.append(task_vec(obs))
            nom_full.append(env.env.sim.get_state().flatten()[1:])
            win_flags |= np.array(contact_flags(env), dtype=bool)
        nom_task = np.asarray(nom_task)
        nom_full = np.asarray(nom_full)

        log_d_task = np.empty((N_PROBE, k_horizon))
        log_d_full = np.empty((N_PROBE, k_horizon))
        for n in range(N_PROBE):
            a0 = actions[t0].copy()
            a0[p0:p0 + 3] += sigma_u * rng.standard_normal(3)
            set_state(env, states[t0])
            for j in range(k_horizon):
                a = a0 if j == 0 else actions[t0 + j]
                obs, _, _, _ = env.step(a)
                dt_ = np.linalg.norm(task_vec(obs) - nom_task[j])
                df_ = np.linalg.norm(env.env.sim.get_state().flatten()[1:] - nom_full[j])
                log_d_task[n, j] = np.log(max(dt_, DIV_FLOOR))
                log_d_full[n, j] = np.log(max(df_, DIV_FLOOR))

        out.append((t0, fit_slope(log_d_task), fit_slope(log_d_full),
                    float(np.exp(log_d_task[:, 0]).mean()),
                    float(np.exp(log_d_task[:, -1]).mean()),
                    *flags, *win_flags.tolist()))
    return demo_key, out


_WORKER_ENV = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sigma-u", type=float, default=0.05)
    ap.add_argument("--sigma-u-source", default="placeholder")
    ap.add_argument("--n-demos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-horizon", type=int, default=K_HORIZON)
    args = ap.parse_args()

    jobs = []
    with h5py.File(args.dataset, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if args.n_demos:
            demos = demos[: args.n_demos]
        for i, d in enumerate(demos):
            g = f["data"][d]
            jobs.append((env_meta, d, g.attrs["model_file"], g.attrs["ep_meta"],
                         g["states"][()], lerobot_to_env(g["actions"][()]),
                         args.sigma_u, args.seed + i, args.k_horizon))

    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(process_demo, jobs, chunksize=1)

    demo_ids, stamps, lam_task, lam_full = [], [], [], []
    d_first, d_last = [], []
    c_grip, c_tab, c_fix, w_grip, w_tab, w_fix = [], [], [], [], [], []
    for demo_key, rows in results:
        for (t0, lt, lf, d0, dK, g, tb, fx, wg, wt, wf) in rows:
            demo_ids.append(int(demo_key.split("_")[1]))
            stamps.append(t0); lam_task.append(lt); lam_full.append(lf)
            d_first.append(d0); d_last.append(dK)
            c_grip.append(g); c_tab.append(tb); c_fix.append(fx)
            w_grip.append(wg); w_tab.append(wt); w_fix.append(wf)

    np.savez(
        args.output,
        demo_id=np.array(demo_ids), t=np.array(stamps),
        lambda_task=np.array(lam_task), lambda_full=np.array(lam_full),
        d_first=np.array(d_first), d_last=np.array(d_last),
        contact_gripper=np.array(c_grip), contact_table=np.array(c_tab),
        contact_fixture=np.array(c_fix),
        win_contact_gripper=np.array(w_grip), win_contact_table=np.array(w_tab),
        win_contact_fixture=np.array(w_fix),
        meta=json.dumps({
            "dataset": args.dataset, "K": args.k_horizon, "N": N_PROBE,
            "stride": STRIDE, "sigma_u": args.sigma_u,
            "sigma_u_source": args.sigma_u_source, "seed": args.seed,
            "pos_only": True, "robot": "PandaOmron/HYBRID_MOBILE_BASE",
            "metric": "eef_pos+gripper_qpos (task), full sim state ex-time (full)",
            "contact_semantics": "gripper-movable / unused / gripper-static",
        }))
    n = len(stamps)
    lam = np.array(lam_task)
    print(f"DONE {n} stamps from {len(results)} demos in {time.time()-t_start:.0f}s")
    if n:
        print(f"lambda_task: mean={lam.mean():.4f} frac>0={100*(lam>0).mean():.1f}% "
              f"p5={np.percentile(lam,5):.3f} p95={np.percentile(lam,95):.3f}")


if __name__ == "__main__":
    main()
