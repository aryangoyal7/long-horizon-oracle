"""Add proprio_seq (N, CLIP_LEN, 9) to feature files that only carry the single-stamp
proprio vector.

Cached feature files predate the switch from a proprio snapshot at t to the proprio
window over the same 16 frames the encoder sees. Re-extracting them would mean re-running
the frozen V-JEPA encoder over every stamp, which is GPU-days; but proprio is just a few
obs arrays, so it can be recomputed from the source hdf5 in minutes. The visual tokens
are copied through untouched, so the vision half of the file is bit-identical.

Windowing and start-of-episode padding match vjepa_extract.py exactly.

Usage:
  python backfill_proprio_seq.py --features /mnt/scratch/lh/features/feat2_square.npz \
      --dataset /mnt/scratch/lh/data/robomimic/square/ph/image_v15.hdf5
  # LIBERO files store different obs keys:
  python backfill_proprio_seq.py --features feat_libero10_X.npz --dataset X.hdf5 \
      --proprio-keys ee_pos ee_ori gripper_states
"""

import argparse
import os

import h5py
import numpy as np

CLIP_LEN = 16
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--proprio-keys", nargs="+", default=None)
    ap.add_argument("--demo-prefix", default="demo")
    ap.add_argument("--clip-len", type=int, default=CLIP_LEN)
    ap.add_argument("--out", default=None, help="default: overwrite in place")
    args = ap.parse_args()

    keys = args.proprio_keys or PROPRIO_KEYS
    z = dict(np.load(args.features, allow_pickle=True))
    if "proprio_seq" in z:
        print(f"SKIP {args.features} (already has proprio_seq)")
        return

    demo_ids, ts = z["demo_id"], z["t"]
    out = np.empty((len(ts), args.clip_len, z["proprio"].shape[1]), dtype=np.float32)

    with h5py.File(args.dataset, "r") as f:
        order = np.lexsort((ts, demo_ids))
        cur, prop_all = None, None
        for idx in order:
            d, t = int(demo_ids[idx]), int(ts[idx])
            if d != cur:
                g = f[f"data/{args.demo_prefix}_{d}"]
                prop_all = np.concatenate([g[f"obs/{k}"][()] for k in keys], axis=1)
                cur = d
            lo = max(0, t - args.clip_len + 1)
            s = prop_all[lo: t + 1]
            if s.shape[0] < args.clip_len:
                s = np.concatenate(
                    [np.repeat(s[:1], args.clip_len - s.shape[0], 0), s])
            out[idx] = s

    # the stored snapshot must equal the last frame of the window, or the two
    # halves of the file describe different timesteps
    drift = float(np.abs(out[:, -1] - z["proprio"]).max())
    if drift > 1e-4:
        raise SystemExit(f"FAIL {args.features}: last frame of window disagrees with "
                         f"stored proprio by {drift:.3g} — wrong dataset or keys?")

    z["proprio_seq"] = out
    dest = args.out or args.features
    tmp = dest + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, dest)
    print(f"BACKFILLED {dest}  seq{out.shape}  (last-frame drift {drift:.2g})")


if __name__ == "__main__":
    main()
