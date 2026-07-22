"""Open-loop FTLE labeler for LIBERO datasets (native OffScreenRenderEnv path).

Why a separate entrypoint: LIBERO replay is only deterministic through the native
env wrapper (set_init_state = set_state + forward + post-process + forced observable
update), and only from the SECOND rollout after a context change onward (the first
rollout canonicalizes solver/controller internals). Protocol per stamp therefore:
  warm-up rollout (discarded) -> nominal rollout -> N perturbed rollouts.
Verified bit-exact (diff ~1e-15) on libero_10 KITCHEN_SCENE3.

Same output schema as ftle_labeler.py (analyze_labels.py-compatible).
sigma_u note: no LIBERO policy is trained yet; pass the robomimic-mean pos-RMSE as
an interim value (recorded in meta) and regenerate after a LIBERO policy exists.

Usage (mg venv, PYTHONPATH=<LIBERO repo>):
  MUJOCO_GL=egl python ftle_labeler_libero.py --dataset <demo.hdf5> \
      --output labels.npz --sigma-u 0.203 --workers 10
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

DIV_FLOOR = 1e-12
_WORKER = {"env": None, "bddl": None, "obj_qpos_idx": None, "eef_site": None}


def fit_slope(log_d):
    taus = np.arange(1, log_d.shape[-1] + 1, dtype=np.float64)
    y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def resolve_bddl(raw):
    from libero.libero import get_libero_path
    raw = raw.decode() if isinstance(raw, bytes) else raw
    hits = glob.glob(os.path.join(get_libero_path("bddl_files"), "*",
                                  os.path.basename(raw)))
    return hits[0] if hits else raw


def get_env(bddl):
    if _WORKER["bddl"] == bddl:
        return _WORKER["env"]
    from libero.libero.envs import OffScreenRenderEnv
    if _WORKER["env"] is not None:
        _WORKER["env"].close()
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=84,
                             camera_widths=84, ignore_done=True)
    env.reset()
    sim = env.sim
    # task-space metric pieces: eef site + free-joint object qpos
    import mujoco
    eef = None
    for cand in ["gripper0_grip_site", "gripper0_eef", "grip_site"]:
        try:
            eef = sim.model.site_name2id(cand)
            break
        except Exception:
            continue
    idx = []
    for j in range(sim.model.njnt):
        if sim.model.jnt_type[j] == 0:              # mjJNT_FREE
            adr = sim.model.jnt_qposadr[j]
            idx.extend(range(adr, adr + 7))
    _WORKER.update(env=env, bddl=bddl, obj_qpos_idx=np.array(idx, dtype=int),
                   eef_site=eef)
    return env


def task_vec(env):
    sim = env.sim
    parts = [sim.data.qpos[_WORKER["obj_qpos_idx"]].ravel()]
    if _WORKER["eef_site"] is not None:
        parts.insert(0, sim.data.site_xpos[_WORKER["eef_site"]].ravel())
    return np.concatenate(parts)


def contact_flags(env):
    sim = env.sim
    g_obj = o_tab = o_fix = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = (sim.model.geom_id2name(c.geom1) or f"id{c.geom1}").lower()
        n2 = (sim.model.geom_id2name(c.geom2) or f"id{c.geom2}").lower()
        grip1 = "gripper" in n1 or "finger" in n1
        grip2 = "gripper" in n2 or "finger" in n2
        rob1 = grip1 or "robot" in n1
        rob2 = grip2 or "robot" in n2
        tab1 = "table" in n1 or "floor" in n1
        tab2 = "table" in n2 or "floor" in n2
        if (grip1 and not rob2 and not tab2) or (grip2 and not rob1 and not tab1):
            g_obj = True
        elif (tab1 and not rob2 and not tab2) or (tab2 and not rob1 and not tab1):
            o_tab = True
        elif not (rob1 or rob2 or tab1 or tab2):
            o_fix = True
    return g_obj, o_tab, o_fix


def rollout(env, states, actions, t0, K, a0=None):
    env.set_init_state(states[t0])
    task, full, wflags = [], [], np.zeros(3, dtype=bool)
    for j in range(K):
        a = a0 if (j == 0 and a0 is not None) else actions[t0 + j]
        env.step(a)
        task.append(task_vec(env))
        full.append(env.sim.get_state().flatten()[1:].copy())
        wflags |= np.array(contact_flags(env))
    return np.asarray(task), np.asarray(full), wflags


def process_demo(args):
    (bddl, demo_key, states, actions, sigma_u, seed, K, stride, n_probe) = args
    env = get_env(bddl)
    rng = np.random.default_rng(seed)
    T = actions.shape[0]
    out = []
    for t0 in range(0, T - K, stride):
        env.set_init_state(states[t0])
        flags = contact_flags(env)
        rollout(env, states, actions, t0, K)          # warm-up, discarded
        nom_task, nom_full, wflags = rollout(env, states, actions, t0, K)
        wflags |= np.array(flags)
        log_dt = np.empty((n_probe, K)); log_df = np.empty((n_probe, K))
        for n in range(n_probe):
            a0 = actions[t0].copy()
            a0[:3] += sigma_u * rng.standard_normal(3)   # pos-only, never gripper
            p_task, p_full, _ = rollout(env, states, actions, t0, K, a0=a0)
            log_dt[n] = np.log(np.maximum(
                np.linalg.norm(p_task - nom_task, axis=1), DIV_FLOOR))
            log_df[n] = np.log(np.maximum(
                np.linalg.norm(p_full - nom_full, axis=1), DIV_FLOOR))
        out.append((t0, fit_slope(log_dt), fit_slope(log_df),
                    float(np.exp(log_dt[:, 0]).mean()),
                    float(np.exp(log_dt[:, -1]).mean()),
                    *flags, *wflags.tolist()))
    return demo_key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sigma-u", type=float, required=True)
    ap.add_argument("--sigma-u-source", default="policy_rmse_robomimic_mean_interim")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--k-horizon", type=int, default=24)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--n-demos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = []
    with h5py.File(args.dataset, "r") as f:
        bddl = resolve_bddl(f["data"].attrs["bddl_file_name"])
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        if args.n_demos:
            demos = demos[: args.n_demos]
        for i, d in enumerate(demos):
            g = f["data"][d]
            jobs.append((bddl, d, g["states"][()], g["actions"][()],
                         args.sigma_u, args.seed + i,
                         args.k_horizon, args.stride, args.n_probe))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(process_demo, jobs, chunksize=1)

    demo_ids, stamps, lam_t, lam_f = [], [], [], []
    d_first, d_last = [], []
    cg, ct, cf, wg, wt, wf = [], [], [], [], [], []
    for demo_key, rows in results:
        for (s, lt, lf, d0, dK, g_, tb, fx, wg_, wt_, wf_) in rows:
            demo_ids.append(int(demo_key.split("_")[1])); stamps.append(s)
            lam_t.append(lt); lam_f.append(lf); d_first.append(d0); d_last.append(dK)
            cg.append(g_); ct.append(tb); cf.append(fx)
            wg.append(wg_); wt.append(wt_); wf.append(wf_)

    np.savez(
        args.output,
        demo_id=np.array(demo_ids), t=np.array(stamps),
        lambda_task=np.array(lam_t), lambda_full=np.array(lam_f),
        d_first=np.array(d_first), d_last=np.array(d_last),
        contact_gripper=np.array(cg), contact_table=np.array(ct),
        contact_fixture=np.array(cf),
        win_contact_gripper=np.array(wg), win_contact_table=np.array(wt),
        win_contact_fixture=np.array(wf),
        meta=json.dumps({
            "dataset": args.dataset, "bddl": bddl, "K": args.k_horizon,
            "N": args.n_probe, "pos_only": True, "stride": args.stride,
            "sigma_u": args.sigma_u, "sigma_u_source": args.sigma_u_source,
            "seed": args.seed,
            "protocol": "libero-native set_init_state + warm-up rollout discard",
            "metric": "eef site + free-joint object qpos (task), full state (full)"}))
    lam = np.array(lam_t)
    print(f"DONE {len(stamps)} stamps from {len(results)} demos "
          f"in {time.time()-t0:.0f}s")
    print(f"lambda_task: mean={lam.mean():.4f} frac>0={100*(lam>0).mean():.1f}% "
          f"p5={np.percentile(lam,5):.3f} p95={np.percentile(lam,95):.3f}")


if __name__ == "__main__":
    main()
