"""GR00T N1.5 predictor-switching rollouts on RoboCasa.

Owns a single robocasa gym env (fork's wrapper, MultiStepWrapper with
n_action_steps=1 so this loop controls how much of each 16-step chunk is
executed) and the in-process Gr00tPolicy. A predictor_bridge.py subprocess
(vjepa venv) serves lambda_hat from a RoboCasa head; at each replan the
executed chunk length switches between k_stable and k_unstable.

Predictor inputs mirror head training exactly: 16-frame window of
video.robot0_agentview_left (256px) and a 16-step window of the 16-dim
observation.state vector (base_position, base_rotation, eef pos/rot relative,
gripper_qpos - the modality.json order).

Run (groot155, from the fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_switch_rollout.py \
    --env-name TurnOnSinkFaucet --pred-head <dir>/head.pt --k-unstable 1 \
    --n-episodes 50 --out out.json
"""

import argparse
import collections
import json
import os
import subprocess
import time

import numpy as np

VJPY = "/mnt/scratch/lh/envs/vjepa/bin/python"
PRED_BRIDGE = ("/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/"
               "impl/eval/predictor_bridge.py")
CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"
CAM_KEY = "video.robot0_agentview_left"
STATE_KEYS = ["state.base_position", "state.base_rotation",
              "state.end_effector_position_relative",
              "state.end_effector_rotation_relative", "state.gripper_qpos"]
ACTION_KEYS = ["action.gripper_close", "action.end_effector_position",
               "action.end_effector_rotation", "action.base_motion",
               "action.control_mode"]
CLIP_LEN = 16


def policy_obs(obs):
    """The policy's batching transform indexes every entry; plain strings and
    0-dim arrays (the env's language annotation) must become 1-element lists,
    matching the offline dataset format."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = [v.item()]
        else:
            out[k] = v
    return out


class LineServer:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)

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


class Predictor:
    def __init__(self, head_path, tmp_dir, bridge=PRED_BRIDGE, window=CLIP_LEN):
        # window > CLIP_LEN feeds a longer frame history to a sequence bridge
        self.frames = collections.deque(maxlen=window)
        self.props = collections.deque(maxlen=CLIP_LEN)
        self.tmp = os.path.join(tmp_dir, f"predobs_{os.getpid()}.npz")
        self.srv = LineServer([VJPY, bridge, "--head", head_path])
        self.train_delta = float(self.srv.expect("READY").split()[1])

    @staticmethod
    def _strip(v):
        v = np.asarray(v)
        return v[-1] if v.ndim > 1 and v.shape[0] == 1 else v

    def observe(self, obs):
        frame = np.asarray(obs[CAM_KEY])
        while frame.ndim > 3:
            frame = frame[-1]
        self.frames.append(frame.astype(np.uint8))
        prop = np.concatenate([np.ravel(self._strip(obs[k]))
                               for k in STATE_KEYS]).astype(np.float32)
        self.props.append(prop)

    def reset(self, obs):
        self.frames.clear()
        self.props.clear()
        self.observe(obs)

    def query(self):
        np.savez(self.tmp, frames=np.stack(self.frames),
                 proprio_seq=np.stack(self.props), proprio=self.props[-1])
        self.srv.send(f"REQ {self.tmp}")
        res = self.srv.expect("RES ").split()
        return float(res[1]), float(res[2])

    def close(self):
        self.srv.close()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--pred-head", required=True)
    ap.add_argument("--bridge", default=PRED_BRIDGE,
                    help="predictor server script (vjepa venv); use "
                         "predictor_bridge_seq.py for sequence heads")
    ap.add_argument("--frames-window", type=int, default=CLIP_LEN,
                    help="frame-history buffer length sent to the bridge")
    ap.add_argument("--fire-on", choices=["lam", "prob"], default="lam",
                    help="statistic compared against --delta. Regression heads "
                         "compress lambda_hat toward the mean, so the label-space "
                         "delta never fires; calibrate_seq_head.py emits a "
                         "quantile-matched threshold for either statistic.")
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=4)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None,
                    help="record per-episode videos (steps_per_render=1)")
    ap.add_argument("--control", default="predictor",
                    choices=["predictor", "random", "shadow"],
                    help="predictor: switch on lambda_hat; random: switch at "
                         "--fire-rate ignoring lambda_hat (still logged); "
                         "shadow: never switch, only log lambda_hat")
    ap.add_argument("--fire-rate", type=float, default=None,
                    help="per-replan probability of k_unstable (random mode)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.control == "random":
        assert args.fire_rate is not None, "--fire-rate required for random"
    rng = np.random.RandomState(args.seed)

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

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    pred = Predictor(args.pred_head, out_dir, args.bridge, args.frames_window)
    delta = args.delta if args.delta is not None else pred.train_delta

    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        pred.reset(obs)
        chunk, chunk_len, executed = None, 0, 0
        k_current = args.k_stable
        k_log, lam_log, replan_log = [], [], []
        success, steps = False, 0
        for t in range(horizon):
            replan = (chunk is None or executed >= min(k_current, chunk_len))
            if replan:
                lam_hat, p_hat = pred.query()
                stat = p_hat if args.fire_on == "prob" else lam_hat
                lam_log.append(stat)
                if args.control == "predictor":
                    fire = stat > delta
                elif args.control == "random":
                    fire = bool(rng.random() < args.fire_rate)
                else:   # shadow: log lambda_hat, never switch
                    fire = False
                k_current = args.k_unstable if fire else args.k_stable
                replan_log.append({"t": t, "lam": round(float(lam_hat), 5),
                                   "p": round(float(p_hat), 6),
                                   "k": int(k_current)})
                chunk = policy.get_action(policy_obs(obs))
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
            act = {k: np.asarray(chunk[k])[executed: executed + 1]
                   for k in ACTION_KEYS}
            obs, _r, term, trunc, info = env.step(act)
            executed += 1
            pred.observe(obs)
            k_log.append(k_current)
            steps = t + 1
            if bool(np.ravel(info.get("success", False))[0]):
                success = True
                break
            if term or trunc:
                break
        episodes.append({
            "episode": ep, "success": bool(success), "steps": steps,
            "mean_k": float(np.mean(k_log)),
            "frac_unstable_calls": float(np.mean(np.array(lam_log) > delta)),
            "replans": replan_log})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps} steps, "
              f"mean_k {episodes[-1]['mean_k']:.1f}, "
              f"unstable calls {episodes[-1]['frac_unstable_calls']:.2f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    if args.video_dir:
        env.reset()   # finalizes the last recorded video file
    env.close()
    pred.close()
    rate = float(np.mean([e["success"] for e in episodes]))
    result = {"env": args.env_name, "mode": "predictor",
              "control": args.control, "fire_rate": args.fire_rate,
              "fire_on": args.fire_on,
              "seed": args.seed,
              "pred_head": args.pred_head, "k_stable": args.k_stable,
              "k_unstable": args.k_unstable, "delta": delta,
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
