"""Diagnose the ACT 0.0: replay demo actions (validates env + success
detection), then roll the trained ACT from demo inits vs random resets."""
import io
import json
import sys

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/impl/robocasa/act")
from act_rollout import make_env, proprio_vec, Policy, lerobot_to_env

HDF5 = "/mnt/scratch/lh/data/robocasa/hdf5/rc_OpenCabinet.hdf5"
CKPT = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/training/act_rc_OpenCabinet/act_final.pt"
SP = sys.argv[1]

src = h5py.File(HDF5, "r")
env_meta = json.loads(src["data"].attrs["env_args"])
env = make_env(env_meta)
policy = Policy(CKPT, "robot0_agentview_left", 224)

def demo_init(name):
    d = src["data"][name]
    env.reset_to({"model": d.attrs["model_file"], "ep_meta": d.attrs["ep_meta"]})
    obs = env.reset_to({"states": d["states"][()][0]})
    return d, obs

print("== A: replay demo actions from demo inits ==", flush=True)
for name in ["demo_1", "demo_2", "demo_3"]:
    d, obs = demo_init(name)
    acts = lerobot_to_env(d["actions"][()])
    succ = False
    for a in acts:
        obs, _r, _dn, _i = env.step(a)
        if env.is_success()["task"]:
            succ = True
            break
    print(f"  {name}: replay {'SUCCESS' if succ else 'FAIL'} (T={len(acts)})",
          flush=True)

def rollout(obs, tag, save_frames=False):
    frames = []
    succ, t = False, 0
    while t < 1050 and not succ:
        acts = policy.chunk(env, obs)
        for j in range(16):
            obs, _r, _dn, _i = env.step(acts[j])
            t += 1
            if save_frames and t % 40 == 0:
                frames.append(env.render(mode="rgb_array", height=224,
                                         width=224,
                                         camera_name="robot0_agentview_left"))
            if env.is_success()["task"]:
                succ = True
                break
    print(f"  {tag}: ACT {'SUCCESS' if succ else 'fail'} at {t}", flush=True)
    if save_frames and frames:
        n = len(frames)
        cols = min(n, 6)
        rows = (n + cols - 1) // cols
        grid = np.zeros((rows * 224, cols * 224, 3), np.uint8)
        for i, f in enumerate(frames):
            r, c = divmod(i, cols)
            grid[r*224:(r+1)*224, c*224:(c+1)*224] = f
        Image.fromarray(grid).save(f"{SP}/diag_{tag}.png")

print("== B: ACT from demo inits ==", flush=True)
for i, name in enumerate(["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]):
    _d, obs = demo_init(name)
    rollout(obs, f"demoinit_{name}", save_frames=(i == 0))

print("== C: ACT from random resets ==", flush=True)
for ep in range(2):
    np.random.seed(1000 + ep)
    obs = env.reset()
    rollout(obs, f"random_{ep}", save_frames=(ep == 0))
print("DIAG_DONE", flush=True)
