"""Label sanity videos for LIBERO datasets (native env path — the robomimic wrapper
path is nondeterministic for LIBERO, see ftle_labeler_libero.py).

Same annotation scheme as render_label_videos.py: border RED = unstable,
GREEN = stable, GRAY = deadband; lambda + window contacts printed on-frame.
"""

import argparse
import json
import os
import sys

import h5py
import imageio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "labeler"))
from render_label_videos import annotate  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ftle_labeler_libero import resolve_bddl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-demos", type=int, default=5)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        conv = f["data"].attrs.get("macros_image_convention", b"opencv")
        conv = conv.decode() if isinstance(conv, bytes) else conv
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = conv          # match the dataset's frame orientation

    from libero.libero.envs import OffScreenRenderEnv

    z = np.load(args.labels, allow_pickle=True)
    lab_did, lab_t, lab_lam = z["demo_id"], z["t"], z["lambda_task"]
    wg, wt, wf = (z.get("win_contact_gripper"), z.get("win_contact_table"),
                  z.get("win_contact_fixture"))

    os.makedirs(args.out_dir, exist_ok=True)
    with h5py.File(args.dataset, "r") as f:
        bddl = resolve_bddl(f["data"].attrs["bddl_file_name"])
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.size,
                                 camera_widths=args.size, ignore_done=True)
        env.reset()
        picked = sorted(set(lab_did.tolist()))[: args.n_demos]
        for did in picked:
            dk = f"demo_{did}"
            g = f[f"data/{dk}"]
            states, actions = g["states"][()], g["actions"][()]
            sel = lab_did == did
            order = np.argsort(lab_t[sel])
            ts_d = lab_t[sel][order]; lam_d = lab_lam[sel][order]
            fl_d = (np.stack([wg[sel], wt[sel], wf[sel]], 1)[order]
                    if wg is not None else None)
            env.set_init_state(states[0])
            frames = []
            for t in range(actions.shape[0]):
                obs, _, _, _ = env.step(actions[t])
                img = obs.get("agentview_image")
                if img is None:  # fall back to any image key
                    key = [k for k in obs if k.endswith("_image")][0]
                    img = obs[key]
                i = np.searchsorted(ts_d, t, side="right") - 1
                lam = float(lam_d[i]) if i >= 0 else None
                fl = fl_d[i] if (fl_d is not None and i >= 0) else None
                frames.append(annotate(np.ascontiguousarray(img), lam,
                                       args.delta, fl))
            out = os.path.join(args.out_dir, f"{dk}.mp4")
            imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)
            print(f"saved {out} ({len(frames)} frames)", flush=True)
    with open(os.path.join(args.out_dir, "README.json"), "w") as fo:
        json.dump({"labels": args.labels, "delta": args.delta,
                   "legend": "RED=unstable GREEN=stable GRAY=deadband"}, fo,
                  indent=2)
    print("RENDER_DONE")


if __name__ == "__main__":
    main()
