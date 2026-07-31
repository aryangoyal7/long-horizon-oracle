"""Render an ACT training set from the labeled RoboCasa demos.

The demo hdf5 stores only sim states and raw 12-dim actions. This replays
each state through the same EnvRobocasa wrapper the labeler used and saves,
per timestep: a JPEG agentview frame, 9-dim proprio (eef pos + quat +
gripper qpos), and the raw action. Output: data/demo_i/{frames,proprio,
actions} in one hdf5 per task.
"""
import argparse
import io
import json
import time

import h5py
import numpy as np
from PIL import Image


def make_env(env_meta):
    import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(
        obs=dict(low_dim=["robot0_eef_pos", "robot0_eef_quat",
                          "robot0_gripper_qpos", "robot0_base_pos"], rgb=[])))
    from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.pop("env_name", None)
    return EnvRobocasa(env_name=env_meta["env_name"], render=False,
                       render_offscreen=True, use_image_obs=False,
                       postprocess_visual_obs=False, **kwargs)


def proprio_vec(obs):
    return np.concatenate([np.asarray(obs["robot0_eef_pos"]).ravel(),
                           np.asarray(obs["robot0_eef_quat"]).ravel(),
                           np.asarray(obs["robot0_gripper_qpos"]).ravel()]
                          ).astype(np.float32)


def jpeg(frame, quality=90):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=224)
    ap.add_argument("--camera", default="robot0_agentview_left")
    ap.add_argument("--n-demos", type=int, default=500)
    args = ap.parse_args()

    src = h5py.File(args.hdf5, "r")
    env_meta = json.loads(src["data"].attrs["env_args"])
    env = make_env(env_meta)
    vlen = h5py.special_dtype(vlen=np.uint8)
    t0 = time.time()
    names = sorted(src["data"].keys(), key=lambda s: int(s.split("_")[1]))
    names = names[: args.n_demos]
    with h5py.File(args.out, "w") as dst:
        g = dst.create_group("data")
        done = 0
        for name in names:
            demo = src["data"][name]
            states = demo["states"][()]
            actions = demo["actions"][()].astype(np.float32)
            ok = False
            for attempt in range(3):
                try:
                    env.reset_to({"model": demo.attrs["model_file"],
                                  "ep_meta": demo.attrs["ep_meta"]})
                    ok = True
                    break
                except Exception as e:
                    print(f"{name}: reset failed try {attempt + 1}: "
                          f"{type(e).__name__}", flush=True)
                    try:
                        env.env.close()
                    except Exception:
                        pass
                    env = make_env(env_meta)
            if not ok:
                print(f"{name}: SKIPPED after 3 attempts", flush=True)
                continue
            T = len(states)
            frames, props = [], []
            for t in range(T):
                obs = env.reset_to({"states": states[t]})
                props.append(proprio_vec(obs))
                fr = env.render(mode="rgb_array", height=args.px,
                                width=args.px, camera_name=args.camera)
                frames.append(jpeg(np.asarray(fr)))
            gd = g.create_group(name)
            fds = gd.create_dataset("frames", (T,), dtype=vlen)
            for t in range(T):
                fds[t] = frames[t]
            gd.create_dataset("proprio", data=np.asarray(props))
            gd.create_dataset("actions", data=actions)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(names)} demos ({time.time()-t0:.0f}s)",
                      flush=True)
        g.attrs["n_demos"] = done
        g.attrs["camera"] = args.camera
        g.attrs["px"] = args.px
    print(f"RENDER_DONE {done} demos ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
