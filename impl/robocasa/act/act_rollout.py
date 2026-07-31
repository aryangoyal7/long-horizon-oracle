"""ACT rollout on RoboCasa: fixed chunk length or oracle switching.

Fixed mode executes k steps of every predicted 16-chunk. Oracle mode runs
the labeling procedure live at each replan (save sim state, replay the
chunk unperturbed and in 8 perturbed branches, fit the divergence slope,
restore) and drops to k_unstable while measured lambda exceeds delta.
Episode inits are seeded by episode index so runs are comparable.
"""
import argparse
import io
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from act_model import ACT, CHUNK  # noqa: E402
from render_act_dataset import make_env, proprio_vec  # noqa: E402

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
N_PROBE = 8
SIGMA_U = 0.05
DIV_FLOOR = 1e-12


def fit_slope(log_d):
    taus = np.arange(1, log_d.shape[-1] + 1, dtype=np.float64)
    y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


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


def measure_lambda(kenv, acts, p0, rng):
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


class Policy:
    def __init__(self, ckpt, camera, px):
        d = torch.load(ckpt, map_location="cuda")
        self.model = ACT().cuda().eval()
        self.model.load_state_dict(d["model"])
        self.s = {k: np.asarray(v, np.float32) for k, v in d["stats"].items()}
        self.camera, self.px = camera, px

    @torch.no_grad()
    def chunk(self, env, obs):
        fr = np.asarray(env.render(mode="rgb_array", height=self.px,
                                   width=self.px, camera_name=self.camera),
                        np.float32) / 255.0
        img = ((fr - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)
        prop = (proprio_vec(obs) - self.s["p_mean"]) / self.s["p_std"]
        a, _, _ = self.model(
            torch.from_numpy(img).unsqueeze(0).cuda(),
            torch.from_numpy(prop.astype(np.float32)).unsqueeze(0).cuda())
        out = a[0].cpu().numpy() * self.s["a_std"] + self.s["a_mean"]
        # in demo order dim 11 is the gripper and dim 4 the hybrid-base mode
        # flag, both discrete in {-1, +1}: snap them, or regression noise
        # opens the gripper / hands control to the mobile base at random
        out[:, 11] = np.where(out[:, 11] > 0, 1.0, -1.0)
        out[:, 4] = np.where(out[:, 4] > 0, 1.0, -1.0)
        return np.clip(lerobot_to_env(out), -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True, help="labeled demo hdf5 (env meta)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["fixed", "oracle"], default="fixed")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=1)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "oracle":
        assert args.delta is not None, "--delta required for oracle"

    with h5py.File(args.hdf5, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    env = make_env(env_meta)
    kenv = env.env
    p0 = None   # robot controllers exist only after the first reset
    policy = Policy(args.ckpt, args.camera, args.px)
    rng = np.random.default_rng(0)

    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        np.random.seed(1000 + ep)
        obs = env.reset()
        if p0 is None:
            p0 = arm_pos_index(kenv)
        success, steps, t = False, 0, 0
        k_log, replan_log = [], []
        while t < args.horizon and not success:
            acts = policy.chunk(env, obs)
            if args.mode == "fixed":
                k = args.k
            else:
                lam = measure_lambda(kenv, acts, p0, rng)
                k = (args.k_unstable if lam > args.delta
                     else args.k_stable)
                replan_log.append({"t": t, "lam": round(lam, 5), "k": k})
            for j in range(min(k, args.horizon - t)):
                obs, _r, done, _info = env.step(acts[j])
                t += 1
                k_log.append(k)
                if env.is_success()["task"]:
                    success = True
                    break
            steps = t
        fired = [r for r in replan_log if r["k"] == args.k_unstable]
        episodes.append({
            "episode": ep, "success": bool(success), "steps": steps,
            "mean_k": float(np.mean(k_log)) if k_log else 0.0,
            "frac_unstable_calls": (len(fired) / max(len(replan_log), 1)
                                    if args.mode == "oracle" else 0.0),
            "replans": replan_log})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps}, "
              f"mean_k {episodes[-1]['mean_k']:.1f}, fired "
              f"{episodes[-1]['frac_unstable_calls']:.2f} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    rate = float(np.mean([e["success"] for e in episodes]))
    with open(args.out, "w") as f:
        json.dump({"env": env_meta["env_name"], "policy": "act",
                   "mode": args.mode, "k": args.k,
                   "k_stable": args.k_stable, "k_unstable": args.k_unstable,
                   "delta": args.delta, "horizon": args.horizon,
                   "n_episodes": args.n_episodes, "success_rate": rate,
                   "episodes": episodes}, f, indent=2)
    print(f"SUCCESS_RATE {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
