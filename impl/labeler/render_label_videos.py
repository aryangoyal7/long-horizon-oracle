"""Render label sanity-check videos: replay demos in the simulator and tint each
frame's border with the FTLE label active at that timestep.

Border color: RED = unstable (lambda > +delta), GREEN = stable (lambda < -delta),
GRAY = deadband. The current lambda value and window contacts are printed on the
frame. Labels are forward-looking (stamp t0 covers [t0, t0+K)), so frame t shows the
label of the latest stamp <= t.

Usage:
  MUJOCO_GL=egl python render_label_videos.py \
      --dataset .../low_dim_v15.hdf5 --labels final_ol_square.npz \
      --out-dir results/label_videos/square --delta 0.052 \
      --camera agentview --n-demos 5
"""

import argparse
import json
import os

import h5py
import imageio
import numpy as np

RED, GREEN, GRAY = (220, 50, 50), (60, 200, 90), (150, 150, 150)
BW = 10  # border width px


def annotate(img, lam, delta, wflags):
    img = img.copy()
    color = GRAY if lam is None else (
        RED if lam > delta else GREEN if lam < -delta else GRAY)
    img[:BW, :] = color; img[-BW:, :] = color
    img[:, :BW] = color; img[:, -BW:] = color
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        txt = "no stamp" if lam is None else (
            f"lam={lam:+.3f} "
            + ("UNSTABLE" if lam > delta else "STABLE" if lam < -delta else "deadband"))
        if wflags is not None:
            txt += f"  c[g{int(wflags[0])} t{int(wflags[1])} f{int(wflags[2])}]"
        d.rectangle([BW, BW, img.shape[1] - BW, BW + 14], fill=(0, 0, 0))
        d.text((BW + 3, BW + 1), txt, fill=(255, 255, 255))
        img = np.asarray(im)
    except ImportError:
        pass
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="low_dim hdf5 (replay source)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-demos", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    try:
        import mimicgen  # noqa: F401
    except ImportError:
        pass
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils

    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dict(
        obs=dict(low_dim=["robot0_eef_pos"], rgb=[])))
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True)

    z = np.load(args.labels, allow_pickle=True)
    lab_did, lab_t, lab_lam = z["demo_id"], z["t"], z["lambda_task"]
    wg = z.get("win_contact_gripper"); wt = z.get("win_contact_table")
    wf = z.get("win_contact_fixture")

    os.makedirs(args.out_dir, exist_ok=True)
    picked = sorted(set(lab_did.tolist()))[: args.n_demos]
    with h5py.File(args.dataset, "r") as f:
        for did in picked:
            dk = f"demo_{did}"
            g = f[f"data/{dk}"]
            states, actions = g["states"][()], g["actions"][()]
            sel = lab_did == did
            ts_d = lab_t[sel]; lam_d = lab_lam[sel]
            fl_d = (np.stack([wg[sel], wt[sel], wf[sel]], 1)
                    if wg is not None else None)
            order = np.argsort(ts_d)
            ts_d, lam_d = ts_d[order], lam_d[order]
            if fl_d is not None:
                fl_d = fl_d[order]
            env.reset()
            env.reset_to({"model": g.attrs["model_file"], "states": states[0]})
            frames = []
            for t in range(actions.shape[0]):
                env.step(actions[t])
                img = env.render(mode="rgb_array", height=args.size,
                                 width=args.size, camera_name=args.camera)
                i = np.searchsorted(ts_d, t, side="right") - 1
                lam = float(lam_d[i]) if i >= 0 else None
                fl = fl_d[i] if (fl_d is not None and i >= 0) else None
                frames.append(annotate(np.asarray(img), lam, args.delta, fl))
            out = os.path.join(args.out_dir, f"{dk}.mp4")
            imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)
            print(f"saved {out} ({len(frames)} frames)", flush=True)
    meta = {"labels": args.labels, "delta": args.delta,
            "legend": "RED=unstable GREEN=stable GRAY=deadband; "
                      "c[g,t,f]=window contact gripper/table/fixture"}
    with open(os.path.join(args.out_dir, "README.json"), "w") as fo:
        json.dump(meta, fo, indent=2)
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
