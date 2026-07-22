"""Stage 2 rollout evaluation through the cross-venv bridge.

Runs in the mg venv (robosuite 1.4 + mimicgen): owns the MimicGen environment,
spawns policy_bridge.py in the lh venv as a subprocess (the same pattern
eval_k_sweep.py used for the predictor bridge), and streams observations to it
one step at a time.

Env metadata comes from the converted image hdf5 (data attrs env_args), not
from the checkpoint, so this process never has to unpickle an lh-venv torch
checkpoint.

Run (mg venv):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<gpu> CUDA_VISIBLE_DEVICES=<gpu> \
    python bridge_eval.py --dataset <task>_image.hdf5 --checkpoint <ckpt.pth> \
      --n-episodes 50 --horizon 650 --seed 0 --out results.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

LHPY = "/mnt/scratch/lh/envs/lh/bin/python"
BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_bridge.py")

OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def to_raw(k, v):
    # v0.3's env returns images processed (CHW float [0,1]); the main-branch
    # RolloutPolicy on the other side processes again, so send raw HWC uint8
    if k.endswith("_image") and v.ndim == 3 and v.shape[0] == 3:
        return (np.transpose(v, (1, 2, 0)) * 255.0).round().astype(np.uint8)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="converted <task>_image.hdf5")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-gpu", default=None,
                    help="CUDA_VISIBLE_DEVICES for the policy server "
                         "(default: inherit)")
    ap.add_argument("--frame-stack", type=int, default=2,
                    help="train.frame_stack of the checkpoint; obs are sent "
                         "stacked (fs, ...) like FrameStackWrapper provides")
    args = ap.parse_args()

    import mimicgen  # noqa: F401  registers the envs
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    with h5py.File(args.dataset, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat",
                             "robot0_gripper_qpos"],
                 "rgb": ["agentview_image", "robot0_eye_in_hand_image"]}})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True,
        use_image_obs=True)

    penv = dict(os.environ)
    if args.policy_gpu is not None:
        penv["CUDA_VISIBLE_DEVICES"] = args.policy_gpu
    srv = subprocess.Popen(
        [LHPY, BRIDGE, "--checkpoint", args.checkpoint],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=penv)

    def expect(prefix):
        # robomimic prints warnings to stdout; skip anything non-protocol
        while True:
            line = srv.stdout.readline()
            assert line, f"policy server died waiting for {prefix!r}"
            if line.strip().startswith(prefix):
                return line.strip()

    expect("READY")

    tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                       f"obs_tmp_{os.getpid()}.npz")
    np.random.seed(args.seed)
    episodes = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        srv.stdin.write("RESET\n"); srv.stdin.flush()
        expect("ACK")
        obs = env.reset()
        # FrameStackWrapper semantics: history starts as fs copies of the
        # reset obs, then slides one frame per step
        hist = {k: [to_raw(k, obs[k])] * args.frame_stack for k in OBS_KEYS}
        success, steps = False, 0
        for t in range(args.horizon):
            np.savez(tmp, **{k: np.stack(hist[k]) for k in OBS_KEYS})
            srv.stdin.write(f"REQ {tmp}\n"); srv.stdin.flush()
            res = expect("RES ")
            # RES <qlen> <a0> ... <a6>
            act = np.array([float(x) for x in res.split()[2:]])
            obs, _, done, _ = env.step(act)
            for k in OBS_KEYS:
                hist[k] = hist[k][1:] + [to_raw(k, obs[k])]
            steps = t + 1
            if env.is_success()["task"]:
                success = True
                break
        episodes.append({"episode": ep, "success": bool(success),
                         "steps": steps})
        print(f"ep {ep}: {'SUCCESS' if success else 'fail'} at {steps} steps "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    srv.stdin.write("QUIT\n"); srv.stdin.flush()
    srv.wait(timeout=30)
    if os.path.exists(tmp):
        os.remove(tmp)

    rate = float(np.mean([e["success"] for e in episodes]))
    out = {"dataset": args.dataset, "checkpoint": args.checkpoint,
           "n_episodes": args.n_episodes, "horizon": args.horizon,
           "seed": args.seed, "success_rate": rate, "episodes": episodes}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"SUCCESS_RATE {rate:.3f}", flush=True)


if __name__ == "__main__":
    main()
