"""Open-loop FTLE label generator (perturb-and-replay oracle).

Implements Section 3 of stability_label_generation.md on robomimic datasets:

for each demo, each timestep t (stride):
  1. reset sim to the demo state at t
  2. nominal branch: replay the demo's recorded actions a_{t:t+K}
  3. N perturbed branches: Gaussian noise on the FIRST action only (arm dims, not
     gripper), then the SAME remaining actions
  4. d_n(tau) = task-space distance between branches (eef pos + object obs vector)
  5. lambda_ol(t) = OLS slope of mean_n log d_n(tau) vs tau

Also logs contact structure at t (gripper-object / object-table / object-fixture) as a
sanity channel, and the full-state divergence slope as a secondary metric.

sigma_u note: the method doc sets sigma_u = trained policy validation action RMSE.  Until
policies finish training this runs with --sigma-u as a placeholder; the output records
the value and source so final labels can be regenerated identically.

Output: one .npz per dataset with per-stamp arrays.

Run (no rendering needed — physics only):
  python ftle_labeler.py --dataset .../low_dim_v15.hdf5 --output .../labels_ol.npz \
      --workers 40
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

K_HORIZON = 16
N_PROBE = 8
STRIDE = 2
DIV_FLOOR = 1e-12


def make_env(env_meta):
    try:
        import mimicgen  # noqa: F401 — registers MimicGen envs when labeling their data
    except ImportError:
        pass
    try:
        import libero.libero.envs  # noqa: F401 — registers LIBERO envs
        # dataset env_args carry generation-time bddl paths (wrong machine, wrong
        # suite dir) — re-resolve by basename against the installed repo
        bddl = env_meta.get("env_kwargs", {}).get("bddl_file_name")
        if bddl and not os.path.exists(bddl):
            import glob as _glob
            from libero.libero import get_libero_path
            hits = _glob.glob(os.path.join(get_libero_path("bddl_files"),
                                           "*", os.path.basename(bddl)))
            if hits:
                env_meta["env_kwargs"]["bddl_file_name"] = hits[0]
    except ImportError:
        pass
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(
        obs=dict(low_dim=["robot0_eef_pos", "robot0_eef_quat",
                          "robot0_gripper_qpos", "object"], rgb=[])))
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)


def task_vec(obs):
    """Task-space vector for the divergence metric: ee pos + object obs."""
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).ravel(),
                           np.asarray(obs["object"]).ravel()])


# Task-object geom-name patterns (lowercased substring match). Contact pairs NOT
# involving the task object are arena/distractor noise and must be ignored —
# e.g. square's unused RoundNut rests on the floor all episode, can's env keeps
# Bread/Cereal/Milk on the floor, tool_hang's stand always touches the table.
OBJ_PATTERNS = {
    "Lift": ["cube"],
    "PickPlaceCan": ["can_g"],
    "NutAssemblySquare": ["squarenut"],
    "ToolHang": ["frame_", "tool_"],
    # MimicGen envs (contact flags are a sanity channel only)
    "Stack_D0": ["cube"], "Stack_D1": ["cube"],
    "Square_D0": ["squarenut"], "Square_D1": ["squarenut"],
    "Threading_D0": ["needle"], "Threading_D1": ["needle"],
    "Coffee_D0": ["pod"], "Coffee_D1": ["pod"],
}
# fallback for env names not listed (generic task-object name fragments)
OBJ_PATTERNS_DEFAULT = ["cube", "nut", "pod", "needle", "piece", "mug",
                        "frame_", "tool_", "can_g"]


def contact_flags(env, obj_pats):
    """(gripper-object, object-table, object-fixture) booleans at current sim state.

    Only pairs involving the task object count. 'fixture' = object touching anything
    that is neither gripper nor table (peg, stand, bin walls incl. unnamed geoms).
    """
    sim = env.env.sim
    g_obj = o_tab = o_fix = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = (sim.model.geom_id2name(c.geom1) or f"id{c.geom1}").lower()
        n2 = (sim.model.geom_id2name(c.geom2) or f"id{c.geom2}").lower()
        o1 = any(p in n1 for p in obj_pats)
        o2 = any(p in n2 for p in obj_pats)
        if not (o1 or o2):
            continue
        other = n2 if o1 else n1
        if o1 and o2:
            o_fix = True                      # object-object (e.g. tool vs frame)
        elif "gripper" in other or "finger" in other:
            g_obj = True
        elif "table" in other:
            o_tab = True
        else:
            o_fix = True                      # peg / stand / bin (unnamed) / floor
    return g_obj, o_tab, o_fix


def fit_slope(log_d):
    """OLS slope of mean log divergence vs tau (tau = 1..K)."""
    taus = np.arange(1, log_d.shape[-1] + 1, dtype=np.float64)
    y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def process_demo(args):
    (env_meta, demo_key, model_xml, states, actions, sigma_u, seed, flags_only,
     k_horizon, pos_only) = args
    global K_HORIZON
    K_HORIZON = k_horizon
    n_pert_dims = 3 if pos_only else 6
    env = _WORKER_ENV[0]
    if env is None:
        env = make_env(env_meta)
        _WORKER_ENV[0] = env
    rng = np.random.default_rng(seed)
    obj_pats = OBJ_PATTERNS.get(env_meta["env_name"], OBJ_PATTERNS_DEFAULT)

    env.reset()
    env.reset_to({"model": model_xml})

    T = actions.shape[0]
    out = []
    for t0 in range(0, T - K_HORIZON, STRIDE):
        # contact flags at the stamp itself
        env.reset_to({"states": states[t0]})
        flags = contact_flags(env, obj_pats)

        # nominal branch (also aggregate contact flags over the whole window — the
        # FTLE label is forward-looking, so alignment analysis needs window contacts)
        nom_task, nom_full = [], []
        win_flags = np.array(flags, dtype=bool)
        env.reset_to({"states": states[t0]})
        for j in range(K_HORIZON):
            obs, _, _, _ = env.step(actions[t0 + j])
            nom_task.append(task_vec(obs))
            nom_full.append(env.get_state()["states"][1:])  # drop sim time
            win_flags |= np.array(contact_flags(env, obj_pats), dtype=bool)
        nom_task = np.asarray(nom_task); nom_full = np.asarray(nom_full)

        if flags_only:
            out.append((t0, 0.0, 0.0, 0.0, 0.0, *flags, *win_flags.tolist()))
            continue

        log_d_task = np.empty((N_PROBE, K_HORIZON))
        log_d_full = np.empty((N_PROBE, K_HORIZON))
        for n in range(N_PROBE):
            a0 = actions[t0].copy()
            a0[:n_pert_dims] += sigma_u * rng.standard_normal(n_pert_dims)  # never gripper
            env.reset_to({"states": states[t0]})
            for j in range(K_HORIZON):
                a = a0 if j == 0 else actions[t0 + j]
                obs, _, _, _ = env.step(a)
                dt_ = np.linalg.norm(task_vec(obs) - nom_task[j])
                df_ = np.linalg.norm(env.get_state()["states"][1:] - nom_full[j])
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
    ap.add_argument("--sigma-u-source", default="placeholder",
                    help="'placeholder' or 'policy_rmse'")
    ap.add_argument("--n-demos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flags-only", action="store_true",
                    help="recompute contact flag channels only and merge them into "
                         "the existing --output npz (lambda values preserved)")
    ap.add_argument("--k-horizon", type=int, default=K_HORIZON)
    ap.add_argument("--pos-only", action="store_true",
                    help="perturb position dims only (skip rotation dims)")
    args = ap.parse_args()

    import robomimic.utils.file_utils as FileUtils
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)

    jobs = []
    with h5py.File(args.dataset, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if args.n_demos:
            demos = demos[: args.n_demos]
        for i, d in enumerate(demos):
            g = f["data"][d]
            jobs.append((env_meta, d, g.attrs["model_file"],
                         g["states"][()], g["actions"][()],
                         args.sigma_u, args.seed + i, args.flags_only,
                         args.k_horizon, args.pos_only))

    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(process_demo, jobs, chunksize=1)

    if args.flags_only:
        z = dict(np.load(args.output, allow_pickle=True))
        new = {}
        for demo_key, rows in results:
            did = int(demo_key.split("_")[1])
            for (t0, _, _, _, _, g, tb, fx, wg, wt, wf) in rows:
                new[(did, t0)] = (g, tb, fx, wg, wt, wf)
        keys = list(zip(z["demo_id"].tolist(), z["t"].tolist()))
        missing = [k for k in keys if k not in new]
        assert not missing, f"{len(missing)} stamps missing from repair pass"
        for i, name in enumerate(["contact_gripper", "contact_table", "contact_fixture",
                                  "win_contact_gripper", "win_contact_table",
                                  "win_contact_fixture"]):
            z[name] = np.array([new[k][i] for k in keys])
        np.savez(args.output, **z)
        print(f"FLAGS_REPAIRED {len(keys)} stamps in {time.time()-t_start:.0f}s "
              f"-> {args.output}")
        return

    demo_ids, stamps, lam_task, lam_full = [], [], [], []
    d_first, d_last = [], []
    c_grip, c_tab, c_fix = [], [], []
    w_grip, w_tab, w_fix = [], [], []
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
            "pos_only": args.pos_only,
            "stride": STRIDE, "sigma_u": args.sigma_u,
            "sigma_u_source": args.sigma_u_source, "seed": args.seed,
            "metric": "eef_pos+object (task), full sim state ex-time (full)",
        }))
    n = len(stamps)
    lam = np.array(lam_task)
    print(f"DONE {n} stamps from {len(results)} demos in {time.time()-t_start:.0f}s")
    print(f"lambda_task: mean={lam.mean():.4f}  frac>0={100*(lam>0).mean():.1f}%  "
          f"p5={np.percentile(lam,5):.3f}  p95={np.percentile(lam,95):.3f}")


if __name__ == "__main__":
    main()
