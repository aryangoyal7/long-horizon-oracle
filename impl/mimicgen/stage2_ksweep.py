"""Stage 2 k-sweep and predictor-switch rollouts on MimicGen tasks.

Runs in the mg venv (robosuite 1.4 + mimicgen). Owns the environment and two
subprocess servers:
  - policy_bridge.py (lh venv, --full-chunk): serves DP actions from a queue
    holding the full predicted chunk; CLEARQ forces re-inference.
  - predictor_bridge.py (vjepa venv, predictor mode only): serves lambda_hat
    from the frozen run1 head on the last 16 agentview frames + proprio.

The episode loop mirrors eval_k_sweep.run_episode exactly: replan when the
queue is empty or the executed count reaches k_current; in predictor mode
k_current switches between k_stable and k_unstable by lambda_hat vs delta at
each replan.

Run (mg venv):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<gpu> CUDA_VISIBLE_DEVICES=<gpu> \
    python stage2_ksweep.py --dataset <task>_image.hdf5 --checkpoint <ckpt> \
      --mode fixed --k 16 --n-episodes 50 --horizon 650 --seed 0 --out out.json
  ... --mode predictor --k-stable 16 --k-unstable 4 [--delta from-head]
"""

import argparse
import collections
import json
import os
import subprocess
import time

import h5py
import numpy as np

LHPY = "/mnt/scratch/lh/envs/lh/bin/python"
VJPY = "/mnt/scratch/lh/envs/vjepa/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_BRIDGE = os.path.join(HERE, "policy_bridge.py")
PRED_BRIDGE = os.path.join(HERE, "..", "eval", "predictor_bridge.py")

OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PRED_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def to_raw(k, v):
    # v0.3's env returns images processed (CHW float [0,1]); send raw HWC uint8
    if k.endswith("_image") and v.ndim == 3 and v.shape[0] == 3:
        return (np.transpose(v, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    return v


class LineServer:
    """Line-protocol subprocess with robomimic stdout noise filtering."""

    def __init__(self, cmd, env=None):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, env=env)

    def expect(self, prefix):
        while True:
            line = self.proc.stdout.readline()
            assert line, f"server died waiting for {prefix!r}"
            if line.strip().startswith(prefix):
                return line.strip()

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def close(self):
        try:
            self.send("QUIT")
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


class PredictorClient:
    """Frame/proprio buffer + predictor_bridge query (from eval_k_sweep.py)."""

    def __init__(self, head_path, cam_key, tmp_dir, clip_len=16):
        self.cam_key = cam_key
        self.buf = collections.deque(maxlen=clip_len)
        self.tmp = os.path.join(tmp_dir, f"predobs_{os.getpid()}.npz")
        self.srv = LineServer([VJPY, PRED_BRIDGE, "--head", head_path])
        ready = self.srv.expect("READY").split()
        self.train_delta = float(ready[1])

    def _frame(self, obs):
        arr = np.asarray(obs[self.cam_key])
        if arr.ndim == 4:
            arr = arr[-1]
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        return arr

    def _proprio(self, obs):
        return np.concatenate(
            [np.asarray(obs[k]) for k in PRED_PROPRIO_KEYS]).astype(np.float32)

    def reset(self, obs):
        self.buf.clear()
        self.observe(obs)

    def observe(self, obs):
        self.buf.append(self._frame(obs))
        self.last_obs = obs

    def query(self):
        np.savez(self.tmp, frames=np.stack(self.buf),
                 proprio=self._proprio(self.last_obs))
        self.srv.send(f"REQ {self.tmp}")
        res = self.srv.expect("RES ").split()
        return float(res[1]), float(res[2])

    def close(self):
        self.srv.close()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["fixed", "predictor"], required=True)
    ap.add_argument("--k", type=int, default=None, help="fixed mode")
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=4)
    ap.add_argument("--delta", type=float, default=None,
                    help="default: the head's training delta")
    ap.add_argument("--pred-head", default=(
        "/mnt/batch/tasks/shared/LS_root/mounts/clusters/garyan181/code/"
        "Users/garyan18/long-horizon/results/predictor/run1_panda_pool/head.pt"))
    ap.add_argument("--pred-cam", default="agentview_image")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stack", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-gpu", default=None)
    args = ap.parse_args()
    if args.mode == "fixed":
        assert args.k is not None, "--k required for fixed mode"

    import mimicgen  # noqa: F401
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    with h5py.File(args.dataset, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": PRED_PROPRIO_KEYS,
                 "rgb": ["agentview_image", "robot0_eye_in_hand_image"]}})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True,
        use_image_obs=True)

    penv = dict(os.environ)
    if args.policy_gpu is not None:
        penv["CUDA_VISIBLE_DEVICES"] = args.policy_gpu
    pol = LineServer(
        [LHPY, POLICY_BRIDGE, "--checkpoint", args.checkpoint, "--full-chunk"],
        env=penv)
    pol.expect("READY")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    pred = None
    delta = args.delta
    if args.mode == "predictor":
        pred = PredictorClient(args.pred_head, args.pred_cam, out_dir)
        if delta is None:
            delta = pred.train_delta
    tmp = os.path.join(out_dir, f"obs_tmp_{os.getpid()}.npz")

    np.random.seed(args.seed)
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        pol.send("RESET"); pol.expect("ACK")
        obs = env.reset()
        hist = {k: [to_raw(k, obs[k])] * args.frame_stack for k in OBS_KEYS}
        if pred is not None:
            pred.reset(obs)
        qlen = 0
        executed = 0
        k_current = args.k if args.mode == "fixed" else args.k_stable
        k_log, lam_log = [], []
        success = False
        steps = 0
        for t in range(args.horizon):
            replan = (qlen == 0 or executed >= k_current)
            if replan:
                pol.send("CLEARQ"); pol.expect("ACK")
                if pred is not None:
                    lam_hat, _p = pred.query()
                    lam_log.append(lam_hat)
                    k_current = (args.k_unstable if lam_hat > delta
                                 else args.k_stable)
            np.savez(tmp, **{k: np.stack(hist[k]) for k in OBS_KEYS})
            pol.send(f"REQ {tmp}")
            res = pol.expect("RES ").split()
            qlen = int(res[1])
            act = np.array([float(x) for x in res[2:]])
            executed = 1 if replan else executed + 1
            obs, _, done, _ = env.step(act)
            for k in OBS_KEYS:
                hist[k] = hist[k][1:] + [to_raw(k, obs[k])]
            if pred is not None:
                pred.observe(obs)
            k_log.append(k_current)
            steps = t + 1
            if env.is_success()["task"]:
                success = True
                break
            if done:
                break
        episodes.append({
            "episode": ep, "success": bool(success), "steps": steps,
            "mean_k": float(np.mean(k_log)),
            "frac_unstable_calls": (float(np.mean(np.array(lam_log) > delta))
                                    if lam_log else None)})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps} steps, "
              f"mean_k {episodes[-1]['mean_k']:.1f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    pol.close()
    if pred is not None:
        pred.close()
    if os.path.exists(tmp):
        os.remove(tmp)

    rate = float(np.mean([e["success"] for e in episodes]))
    result = {"dataset": args.dataset, "checkpoint": args.checkpoint,
              "mode": args.mode, "k": args.k, "k_stable": args.k_stable,
              "k_unstable": args.k_unstable, "delta": delta,
              "n_episodes": args.n_episodes, "horizon": args.horizon,
              "seed": args.seed, "success_rate": rate,
              "mean_k": float(np.mean([e["mean_k"] for e in episodes])),
              "mean_steps_success": (float(np.mean(
                  [e["steps"] for e in episodes if e["success"]]))
                  if rate > 0 else None),
              "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"SUCCESS_RATE {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
