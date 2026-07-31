"""LIBERO-Long k-sweep and predictor-switch rollouts (native env).

Runs in the mg venv with PYTHONPATH=<LIBERO repo>. Owns the env and the two
subprocess servers used by stage2_ksweep.py:
  - policy_bridge.py (lh venv, --full-chunk): DP actions, CLEARQ re-inference
  - predictor_bridge.py (vjepa venv, predictor mode): lambda_hat from a head

Env-side conventions (verified empirically in probe_libero_obs):
  agentview_rgb      = env agentview_image            (NO vertical flip)
  eye_in_hand_rgb    = env robot0_eye_in_hand_image
  ee_pos             = env robot0_eef_pos
  ee_ori             = quat2axisangle(env robot0_eef_quat)
  gripper_states     = env robot0_gripper_qpos
Episodes use the benchmark's fixed init states (episode i -> state i mod N)
with a settle period of noop steps after set_init_state, per LIBERO practice.

Run:
  PYTHONPATH=/mnt/scratch/lh/repos/LIBERO MUJOCO_GL=egl \
  CUDA_VISIBLE_DEVICES=<gpu> python libero_ksweep.py \
    --dataset <task>_demo.hdf5 --checkpoint <ckpt> --mode fixed --k 16 \
    --n-episodes 50 --horizon 600 --seed 0 --out out.json
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
POLICY_BRIDGE = os.path.join(HERE, "..", "mimicgen", "policy_bridge.py")
PRED_BRIDGE = os.path.join(HERE, "..", "eval", "predictor_bridge.py")

POLICY_OBS_KEYS = ["agentview_rgb", "eye_in_hand_rgb",
                   "ee_pos", "ee_ori", "gripper_states"]
NOOP_SETTLE_STEPS = 10


class LineServer:
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
    def __init__(self, head_path, tmp_dir, clip_len=16):
        self.buf = collections.deque(maxlen=clip_len)
        self.tmp = os.path.join(tmp_dir, f"predobs_{os.getpid()}.npz")
        self.srv = LineServer([VJPY, PRED_BRIDGE, "--head", head_path])
        ready = self.srv.expect("READY").split()
        self.train_delta = float(ready[1])

    def reset(self, mapped_obs):
        self.buf.clear()
        self.observe(mapped_obs)

    def observe(self, mapped_obs):
        self.buf.append(mapped_obs["agentview_rgb"])
        self.last = mapped_obs

    def query(self):
        # Heads train on the harmonized 9-dim layout (pos3, quat_xyzw4, grip2);
        # rebuild the quat from axis-angle exactly as harmonize_libero_proprio.
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_rotvec(self.last["ee_ori"]).as_quat()
        prop = np.concatenate([self.last["ee_pos"], quat,
                               self.last["gripper_states"]]).astype(np.float32)
        np.savez(self.tmp, frames=np.stack(self.buf), proprio=prop)
        self.srv.send(f"REQ {self.tmp}")
        res = self.srv.expect("RES ").split()
        return float(res[1]), float(res[2])

    def close(self):
        self.srv.close()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


def map_obs(obs):
    import robosuite.utils.transform_utils as T
    return {
        "agentview_rgb": np.asarray(obs["agentview_image"], dtype=np.uint8),
        "eye_in_hand_rgb": np.asarray(obs["robot0_eye_in_hand_image"],
                                      dtype=np.uint8),
        "ee_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        "ee_ori": np.asarray(
            T.quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
            dtype=np.float32),
        "gripper_states": np.asarray(obs["robot0_gripper_qpos"],
                                     dtype=np.float32),
    }


def task_name_from_dataset(path):
    base = os.path.basename(path)
    return base[: -len("_demo.hdf5")] if base.endswith("_demo.hdf5") else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LIBERO *_demo.hdf5")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["fixed", "predictor"], required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=4)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--pred-head", default=None)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-stack", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "fixed":
        assert args.k is not None, "--k required for fixed mode"
    if args.mode == "predictor":
        assert args.pred_head, "--pred-head required for predictor mode"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_name = task_name_from_dataset(args.dataset)
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    task_id = next(i for i in range(bm.n_tasks)
                   if bm.get_task(i).name == task_name)
    # bm.get_task_init_states uses a bare torch.load, which fails under
    # torch>=2.6 (weights_only default). Load the same file ourselves.
    import torch
    init_path = os.path.join(get_libero_path("init_states"),
                             bm.get_task(task_id).problem_folder,
                             bm.get_task(task_id).init_states_file)
    init_states = torch.load(init_path, weights_only=False)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        bm.get_task(task_id).problem_folder,
                        bm.get_task(task_id).bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=128, camera_widths=128)

    pol = LineServer(
        [LHPY, POLICY_BRIDGE, "--checkpoint", args.checkpoint, "--full-chunk"])
    pol.expect("READY")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    pred, delta = None, args.delta
    if args.mode == "predictor":
        pred = PredictorClient(args.pred_head, out_dir)
        if delta is None:
            delta = pred.train_delta
    tmp = os.path.join(out_dir, f"obs_tmp_{os.getpid()}.npz")

    np.random.seed(args.seed)
    noop = np.zeros(7); noop[-1] = -1.0
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        pol.send("RESET"); pol.expect("ACK")
        env.reset()
        env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(NOOP_SETTLE_STEPS):
            obs, _, _, _ = env.step(noop)
        m = map_obs(obs)
        hist = {k: [m[k]] * args.frame_stack for k in POLICY_OBS_KEYS}
        if pred is not None:
            pred.reset(m)
        qlen, executed = 0, 0
        k_current = args.k if args.mode == "fixed" else args.k_stable
        k_log, lam_log = [], []
        success, steps = False, 0
        for t in range(args.horizon):
            replan = (qlen == 0 or executed >= k_current)
            if replan:
                pol.send("CLEARQ"); pol.expect("ACK")
                if pred is not None:
                    lam_hat, _p = pred.query()
                    lam_log.append(lam_hat)
                    k_current = (args.k_unstable if lam_hat > delta
                                 else args.k_stable)
            np.savez(tmp, **{k: np.stack(hist[k]) for k in POLICY_OBS_KEYS})
            pol.send(f"REQ {tmp}")
            res = pol.expect("RES ").split()
            qlen = int(res[1])
            act = np.array([float(x) for x in res[2:]])
            executed = 1 if replan else executed + 1
            obs, _, done, _ = env.step(act)
            m = map_obs(obs)
            for k in POLICY_OBS_KEYS:
                hist[k] = hist[k][1:] + [m[k]]
            if pred is not None:
                pred.observe(m)
            k_log.append(k_current)
            steps = t + 1
            if env.check_success():
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

    env.close()
    pol.close()
    if pred is not None:
        pred.close()
    if os.path.exists(tmp):
        os.remove(tmp)

    rate = float(np.mean([e["success"] for e in episodes]))
    result = {"dataset": args.dataset, "task": task_name,
              "checkpoint": args.checkpoint, "mode": args.mode, "k": args.k,
              "k_stable": args.k_stable, "k_unstable": args.k_unstable,
              "delta": delta, "n_episodes": args.n_episodes,
              "horizon": args.horizon, "seed": args.seed,
              "success_rate": rate,
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
