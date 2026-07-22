"""Stage 1a/1c evaluation: executed-chunk-length control on trained DP checkpoints.

Modes (--mode):
  fixed        constant executed k for the whole episode
  oracle_seg   per-segment k: k_stable in free space, k_unstable during contact
               segments (oracle segment signal = sim contact flags w/ hysteresis)
  ftle_probe   adaptive controller driven by a LIVE open-loop FTLE probe: at each
               replan, save sim state, perturb-and-replay the policy's own predicted
               chunk (N probes x K steps), restore, pick k by sign of lambda.
               This is the oracle upper bound for the video predictor.
  predictor    Stage 1c: adaptive controller driven by the trained V-JEPA stability
               head. Inference runs in a vjepa-venv subprocess (predictor_bridge.py)
               because robomimic pins an old transformers in this venv. At each
               replan the last 16 camera frames + proprio go to the bridge; the
               returned lambda_hat is thresholded against --pred-delta.

The DP checkpoint is loaded with action_horizon = prediction_horizon (16) so the
internal queue holds the full predicted chunk; executed length is enforced by popping
j actions then clearing the queue (receding-horizon with arbitrary prefix k).

Output: JSON rows (one per episode): success, horizon, k profile, segment stats.
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def load_policy(ckpt_path, device):
    import robomimic.utils.file_utils as FileUtils
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    # force full-chunk queueing; executed prefix is our runtime knob
    cfg = json.loads(ckpt_dict["config"])
    cfg["algo"]["horizon"]["action_horizon"] = cfg["algo"]["horizon"]["prediction_horizon"]
    ckpt_dict["config"] = json.dumps(cfg)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=False)
    return policy, ckpt_dict


def make_env_from_ckpt(ckpt_dict):
    import robomimic.utils.file_utils as FileUtils
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False)
    return env


def contact_signal(env):
    """True if gripper-object or object-fixture contact (oracle segment signal)."""
    # unwrap FrameStackWrapper -> EnvRobosuite -> robosuite env (the one with .sim)
    e = env
    while not hasattr(e, "sim") and hasattr(e, "env"):
        e = e.env
    sim = e.sim
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        lo = ((sim.model.geom_id2name(c.geom1) or "") + " "
              + (sim.model.geom_id2name(c.geom2) or "")).lower()
        grip = "gripper" in lo or "finger" in lo
        robot = "robot" in lo
        table = "table" in lo
        if grip and not table:
            return True
        if (not grip) and (not robot) and (not table):
            return True
    return False


def queue_of(policy):
    return policy.policy.action_queue


def ftle_probe(env, chunk, n_probe, k_win, sigma_u, rng):
    """Open-loop FTLE of the plant under the policy's own predicted chunk."""
    saved = env.get_state()["states"]
    acts = list(chunk)[:k_win]
    k_win = len(acts)
    env.reset_to({"states": saved})
    nom = []
    for a in acts:
        env.step(a)
        s = env.get_state()["states"][1:]
        nom.append(s)
    log_d = np.empty((n_probe, k_win))
    for n in range(n_probe):
        env.reset_to({"states": saved})
        for j, a in enumerate(acts):
            aa = a.copy()
            if j == 0:
                aa[:6] += sigma_u * rng.standard_normal(6)
            env.step(aa)
            d = np.linalg.norm(env.get_state()["states"][1:] - nom[j])
            log_d[n, j] = np.log(max(d, 1e-12))
    env.reset_to({"states": saved})
    taus = np.arange(1, k_win + 1); y = log_d.mean(axis=0)
    tc = taus - taus.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


PRED_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def _frame_from_obs(obs, cam_key):
    """obs image (possibly framestacked, CHW float [0,1]) -> HWC uint8."""
    arr = np.asarray(obs[cam_key])
    if arr.ndim == 4:                                  # (stack, ...) -> latest
        arr = arr[-1]
    if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr


class PredictorClient:
    """Talks to predictor_bridge.py (vjepa venv) over a line protocol."""

    def __init__(self, bridge_py, head_path, cam_key, clip_len=16):
        import collections
        import subprocess
        self.cam_key = cam_key
        self.buf = collections.deque(maxlen=clip_len)
        self.tmp = f"/dev/shm/predbridge_{os.getpid()}.npz"
        bridge = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "predictor_bridge.py")
        self.proc = subprocess.Popen(
            [bridge_py, bridge, "--head", head_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline().split()
        assert line and line[0] == "READY", f"bridge failed to start: {line}"
        self.train_delta = float(line[1])

    def _proprio(self, obs):
        parts = []
        for k in PRED_PROPRIO_KEYS:
            v = np.asarray(obs[k])
            parts.append(v[-1] if v.ndim == 2 else v)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, obs):
        self.buf.clear()
        self.observe(obs)

    def observe(self, obs):
        self.buf.append(_frame_from_obs(obs, self.cam_key))
        self.last_obs = obs

    def query(self):
        np.savez(self.tmp, frames=np.stack(self.buf),
                 proprio=self._proprio(self.last_obs))
        self.proc.stdin.write(f"REQ {self.tmp}\n")
        self.proc.stdin.flush()
        res = self.proc.stdout.readline().split()
        assert res and res[0] == "RES", f"bridge died: {res}"
        return float(res[1]), float(res[2])            # lambda_hat, p_unstable

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


def run_episode(env, policy, mode, k_fixed, k_stable, k_unstable, horizon,
                probe_cfg, rng, pred=None):
    obs = env.reset()
    policy.start_episode()
    queue_of(policy).clear()
    if pred is not None:
        pred.reset(obs)
    in_contact_seg = False
    switch_count = 0          # hysteresis counter
    executed, k_current = 0, (k_fixed if mode == "fixed" else k_stable)
    k_log, probe_log = [], []
    success = False
    for t in range(horizon):
        if len(queue_of(policy)) == 0 or executed >= k_current:
            queue_of(policy).clear()
            a = policy(ob=obs)          # runs inference, fills queue with Tp actions
            executed = 1
            if mode == "oracle_seg":
                sig = contact_signal(env)
                if sig != in_contact_seg:
                    switch_count += 1
                    if switch_count >= 2:      # hysteresis: 2 agreeing replans
                        in_contact_seg = sig
                        switch_count = 0
                else:
                    switch_count = 0
                k_current = k_unstable if in_contact_seg else k_stable
            elif mode == "ftle_probe":
                def to_np(c):
                    return (c.detach().cpu().numpy()
                            if torch.is_tensor(c) else np.asarray(c))
                chunk = [to_np(a)] + [to_np(c) for c in queue_of(policy)]
                lam = ftle_probe(env, np.array(chunk),
                                 probe_cfg["n_probe"], probe_cfg["k_win"],
                                 probe_cfg["sigma_u"], rng)
                probe_log.append(lam)
                k_current = k_unstable if lam > probe_cfg["delta"] else k_stable
            elif mode == "predictor":
                lam_hat, _p = pred.query()
                probe_log.append(lam_hat)
                k_current = (k_unstable if lam_hat > probe_cfg["delta"]
                             else k_stable)
        else:
            a = policy(ob=obs)          # pops from queue, no inference
            executed += 1
        k_log.append(k_current)
        obs, _, done, _ = env.step(a)
        if pred is not None:
            pred.observe(obs)
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break
    return {"success": bool(success), "steps": t + 1,
            "mean_k": float(np.mean(k_log)),
            "probe_lambdas": probe_log if probe_log else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode",
                    choices=["fixed", "oracle_seg", "ftle_probe", "predictor"],
                    default="fixed")
    ap.add_argument("--pred-head", default=None,
                    help="predictor mode: path to head.pt")
    ap.add_argument("--pred-py",
                    default="/mnt/scratch/lh/envs/vjepa/bin/python",
                    help="predictor mode: python of the venv running the bridge")
    ap.add_argument("--pred-cam", default="agentview_image",
                    help="predictor mode: obs key of the camera used in training")
    ap.add_argument("--pred-delta", type=float, default=None,
                    help="predictor mode: lambda_hat threshold "
                         "(default: the delta the head was trained with)")
    ap.add_argument("--k", type=int, default=8, help="k for fixed mode")
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=6)
    ap.add_argument("--probe-k", type=int, default=12)
    ap.add_argument("--probe-sigma-u", type=float, default=0.05)
    ap.add_argument("--probe-delta", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = load_policy(args.ckpt, device)
    env = make_env_from_ckpt(ckpt_dict)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    probe_cfg = {"n_probe": args.probe_n, "k_win": args.probe_k,
                 "sigma_u": args.probe_sigma_u, "delta": args.probe_delta}
    pred = None
    if args.mode == "predictor":
        assert args.pred_head, "--pred-head required for predictor mode"
        pred = PredictorClient(args.pred_py, args.pred_head, args.pred_cam)
        probe_cfg["delta"] = (args.pred_delta if args.pred_delta is not None
                              else pred.train_delta)
    rows = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        rows.append(run_episode(env, policy, args.mode, args.k, args.k_stable,
                                args.k_unstable, args.horizon, probe_cfg, rng,
                                pred=pred))
    if pred is not None:
        pred.close()
    sr = float(np.mean([r["success"] for r in rows]))
    result = {"ckpt": args.ckpt, "mode": args.mode, "k": args.k,
              "k_stable": args.k_stable, "k_unstable": args.k_unstable,
              "pred_head": args.pred_head,
              "pred_delta": probe_cfg["delta"] if args.mode == "predictor" else None,
              "seed": args.seed, "n_episodes": args.n_episodes,
              "success_rate": sr, "elapsed_s": time.time() - t0, "episodes": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{args.mode} k={args.k} (ks={args.k_stable},ku={args.k_unstable}) "
          f"-> success {sr:.2%} [{args.n_episodes} eps, {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
