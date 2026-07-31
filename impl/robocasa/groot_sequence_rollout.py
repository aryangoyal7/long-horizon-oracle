"""Composed long-horizon task on RoboCasa: OpenCabinet -> PickPlaceCounterToCabinet.

One PickPlaceCounterToCabinet env instance. After each reset the cabinet doors
are forced closed, so the episode has two stages driven by the language
instruction given to the generalist policy:
  stage 1: "Open the <cabinet> door(s)."  success = cab.is_open (same
           predicate and 0.90 threshold as the atomic OpenCabinet task)
  stage 2: the env's own instruction ("Pick the X from the counter and place
           it in the cabinet.")  success = the env's own success predicate
Episode success requires finishing both stages within the summed atomic
horizons. Chunk-length control is either fixed k or predictor switching with
the stage-matched head (OpenCabinet head in stage 1, PickPlace head in
stage 2), each at its own training delta.

Run (groot155, from the fork repo root):
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> python groot_sequence_rollout.py \
    --mode fixed --k 16 --n-episodes 50 --out out.json
"""

import argparse
import collections
import json
import os
import subprocess
import time

import numpy as np

VJPY = "/mnt/scratch/lh/envs/vjepa/bin/python"
IMPL = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/impl"
PRED_BRIDGE = os.path.join(IMPL, "eval", "predictor_bridge.py")
PRED_BRIDGE_SEQ = os.path.join(IMPL, "eval", "predictor_bridge_seq.py")
SEQ_HEADS = ("/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/"
             "results/predictor/rc_seq_v3")
ONLINE_THR = os.path.join(SEQ_HEADS, "online_thresholds.json")
# v3 auto-delta per stage task, used by the oracle
ORACLE_DELTA = {"OpenCabinet": 0.0848, "PickPlaceCounterToCabinet": 0.0649}
CKPT = "/mnt/scratch/lh/data/robocasa/ckpt/gr00t_n1-5/multitask_learning/checkpoint-120000"
HEADS = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/predictor/rc_indomain"
CAM_KEY = "video.robot0_agentview_left"
LANG_KEY = "annotation.human.task_description"
STATE_KEYS = ["state.base_position", "state.base_rotation",
              "state.end_effector_position_relative",
              "state.end_effector_rotation_relative", "state.gripper_qpos"]
ACTION_KEYS = ["action.gripper_close", "action.end_effector_position",
               "action.end_effector_rotation", "action.base_motion",
               "action.control_mode"]
CLIP_LEN = 16


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
    def __init__(self, head_path, tmp_dir, tag, bridge=None, window=None):
        # window > CLIP_LEN feeds a longer frame history to the sequence bridge
        self.frames = collections.deque(maxlen=window or CLIP_LEN)
        self.props = collections.deque(maxlen=CLIP_LEN)
        self.tmp = os.path.join(tmp_dir, f"predobs_{tag}_{os.getpid()}.npz")
        self.srv = LineServer([VJPY, bridge or PRED_BRIDGE, "--head", head_path])
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
    ap.add_argument("--mode", choices=["fixed", "predictor", "oracle"],
                    required=True)
    ap.add_argument("--head-variant", choices=["full", "h1", "h4"],
                    default="full",
                    help="full: the original single-clip heads trained on the "
                         "v2 labels. h1/h4: heads trained on the corrected v3 "
                         "labels, thresholded at their online-calibrated "
                         "operating point.")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--k-stable", type=int, default=16)
    ap.add_argument("--k-unstable", type=int, default=4)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None)
    args = ap.parse_args()
    if args.mode == "fixed":
        assert args.k is not None, "--k required for fixed mode"

    from robocasa.models.fixtures.cabinets import HingeCabinet
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

    horizon = (get_task_horizon("OpenCabinet")
               + get_task_horizon("PickPlaceCounterToCabinet"))
    cfg = SimulationConfig(
        env_name="robocasa/PickPlaceCounterToCabinet", split=args.split,
        n_episodes=args.n_episodes, n_envs=1,
        video=VideoConfig(video_dir=args.video_dir,
                          steps_per_render=1) if args.video_dir
        else VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=1,
                                  max_episode_steps=horizon))
    env = _create_single_env(cfg, 0)
    genv = env.unwrapped                    # RoboCasaGymEnv
    kenv = genv.env                         # robosuite Kitchen env

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    preds, deltas = {}, {}
    if args.mode == "predictor":
        thr_tab = (json.load(open(ONLINE_THR))
                   if args.head_variant in ("h1", "h4") else None)
        for stage, task in ((1, "OpenCabinet"),
                            (2, "PickPlaceCounterToCabinet")):
            if args.head_variant == "full":
                preds[stage] = Predictor(
                    os.path.join(HEADS, f"R_{task}_full_s0", "head.pt"),
                    out_dir, f"s{stage}")
                deltas[stage] = preds[stage].train_delta
            else:
                hist = int(args.head_variant[1:])
                preds[stage] = Predictor(
                    os.path.join(SEQ_HEADS,
                                 f"S_{task}_h{hist}_s0", "head.pt"),
                    out_dir, f"s{stage}",
                    bridge=PRED_BRIDGE_SEQ, window=hist * CLIP_LEN)
                # each stage fires at the budget its own atomic task used
                deltas[stage] = float(
                    thr_tab[f"{task}_h{hist}"]["thresh"])
    elif args.mode == "oracle":
        import sys
        sys.path.insert(0, os.path.join(IMPL, "robocasa"))
        from groot_oracle_rollout import (arm_pos_index, measure_lambda,
                                          to_env_action)
        p0 = arm_pos_index(kenv)
        rng = np.random.default_rng(0)
        deltas = {1: ORACLE_DELTA["OpenCabinet"],
                  2: ORACLE_DELTA["PickPlaceCounterToCabinet"]}

    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        # force the cabinet closed, then refresh the observation through the
        # normal wrapper path (one zero-action step) so image formatting
        # matches what the policy transform expects
        kenv.cab.close_door(env=kenv)
        kenv.sim.forward()
        noop = {"action.gripper_close": np.zeros((1, 1), np.float32),
                "action.end_effector_position": np.zeros((1, 3), np.float32),
                "action.end_effector_rotation": np.zeros((1, 3), np.float32),
                "action.base_motion": np.zeros((1, 4), np.float32),
                "action.control_mode": np.zeros((1, 1), np.float32)}
        obs, _r, _te, _tr, _info = env.step(noop)
        assert kenv.cab.is_closed(env=kenv, th=0.05), "door did not close"
        door_word = ("doors" if isinstance(kenv.cab, HingeCabinet)
                     else "door")
        stage1_lang = f"Open the {kenv.cab.nat_lang} {door_word}."
        stage2_lang = str(np.ravel(obs.get(LANG_KEY, ""))[0]) \
            if not isinstance(obs.get(LANG_KEY, ""), str) else obs[LANG_KEY]

        stage = 1
        for p in preds.values():
            p.reset(obs)
        chunk, chunk_len, executed = None, 0, 0
        k_current = args.k if args.mode == "fixed" else args.k_stable
        k_log, replan_log = [], []
        stage1_end, success, steps = None, False, 0
        for t in range(horizon):
            replan = (chunk is None or executed >= min(k_current, chunk_len))
            if replan:
                if args.mode == "predictor":
                    lam_hat, _p = preds[stage].query()
                    k_current = (args.k_unstable
                                 if lam_hat > deltas[stage]
                                 else args.k_stable)
                    replan_log.append(
                        {"t": t, "stage": stage,
                         "lam": round(float(lam_hat), 5),
                         "k": int(k_current)})
                obs = dict(obs)
                obs[LANG_KEY] = stage1_lang if stage == 1 else stage2_lang
                chunk = policy.get_action(policy_obs(obs))
                chunk_len = len(np.asarray(chunk[ACTION_KEYS[1]]))
                executed = 0
                if args.mode == "oracle":
                    # measure lambda on the chunk just produced, then restore
                    acts = [to_env_action(
                        genv, kenv,
                        {k: np.asarray(chunk[k])[j] for k in ACTION_KEYS})
                        for j in range(chunk_len)]
                    lam = measure_lambda(kenv, acts, p0, rng)
                    k_current = (args.k_unstable if lam > deltas[stage]
                                 else args.k_stable)
                    replan_log.append(
                        {"t": t, "stage": stage, "lam": round(lam, 5),
                         "k": int(k_current)})
            act = {k: np.asarray(chunk[k])[executed: executed + 1]
                   for k in ACTION_KEYS}
            obs, _r, term, trunc, info = env.step(act)
            executed += 1
            for p in preds.values():
                p.observe(obs)
            k_log.append(k_current)
            steps = t + 1
            if stage == 1 and kenv.cab.is_open(env=kenv):
                stage = 2
                stage1_end = steps
                chunk = None      # force replan under the new instruction
            elif stage == 2 and bool(np.ravel(
                    info.get("success", False))[0]):
                success = True
                break
            if term or trunc:
                break
        episodes.append({
            "episode": ep, "success": bool(success), "steps": steps,
            "stage1_success": stage1_end is not None,
            "stage1_end": stage1_end,
            "mean_k": float(np.mean(k_log)),
            "replans": replan_log})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps} "
              f"(stage1_end {stage1_end}), mean_k "
              f"{episodes[-1]['mean_k']:.1f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        # checkpoint after every episode: the oracle mode costs ~20 min per
        # episode on this horizon, so a run must survive interruption
        with open(args.out, "w") as _f:
            json.dump({"env": "composed_OpenCabinet_PickPlaceCounterToCabinet",
                       "mode": args.mode, "k": args.k,
                       "head_variant": getattr(args, "head_variant", None),
                       "k_stable": args.k_stable, "k_unstable": args.k_unstable,
                       "deltas": {str(k): v for k, v in deltas.items()},
                       "n_episodes_done": len(episodes),
                       "n_episodes": args.n_episodes, "horizon": horizon,
                       "success_rate": float(np.mean([e["success"] for e in episodes])),
                       "stage1_success_rate": float(np.mean([e["stage1_success"] for e in episodes])),
                       "mean_k": float(np.mean([e["mean_k"] for e in episodes])),
                       "episodes": episodes}, _f, indent=2)

    if args.video_dir:
        env.reset()
    env.close()
    for p in preds.values():
        p.close()
    rate = float(np.mean([e["success"] for e in episodes]))
    s1rate = float(np.mean([e["stage1_success"] for e in episodes]))
    result = {"env": "seq_OpenCabinet+PickPlaceCounterToCabinet",
              "mode": args.mode, "k": args.k, "k_stable": args.k_stable,
              "k_unstable": args.k_unstable,
              "deltas": {str(s): d for s, d in deltas.items()},
              "n_episodes": args.n_episodes, "horizon": horizon,
              "success_rate": rate, "stage1_success_rate": s1rate,
              "mean_k": float(np.mean([e["mean_k"] for e in episodes])),
              "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"SUCCESS_RATE {rate:.3f} STAGE1 {s1rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
