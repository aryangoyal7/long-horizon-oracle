"""Convert a robocasa365 LeRobot dataset to a robomimic-style hdf5 for the
FTLE labeler. The lerobot extras already ship per-step MuJoCo states
(states.npz), the per-episode compiled scene (model.xml.gz) and ep_meta, so
this is pure repackaging - no replay needed.

Output layout (robomimic convention + robocasa per-demo scene attrs):
  data.attrs["env_args"]            robomimic env metadata json
  data/demo_i/states  (T, S)
  data/demo_i/actions (T, A)
  data/demo_i.attrs["model_file"]   episode scene MJCF (str)
  data/demo_i.attrs["ep_meta"]      episode meta json (str, includes lang)
  data/demo_i.attrs["num_samples"]  T
"""
import argparse, gzip, json, os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-demos", type=int, default=None)
    args = ap.parse_args()

    d = Path(args.lerobot_dir)
    info = json.load(open(d / "meta" / "info.json"))
    n = info["total_episodes"] if args.n_demos is None else min(
        args.n_demos, info["total_episodes"])
    meta = json.load(open(d / "extras" / "dataset_meta.json"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept = skipped = 0
    with h5py.File(args.out, "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps(meta["env_args"])
        g.attrs["source"] = str(d)
        total = 0
        for i in range(n):
            ep = d / "extras" / f"episode_{i:06d}"
            pq = d / "data" / f"chunk-{i // info.get('chunks_size', 1000):03d}" / f"episode_{i:06d}.parquet"
            try:
                states = np.load(ep / "states.npz")["states"]
                df = pd.read_parquet(pq)
                actions = np.stack(df["action"].values).astype(np.float32)
                ep_meta = json.load(open(ep / "ep_meta.json"))
                with gzip.open(ep / "model.xml.gz", "rb") as fx:
                    xml = fx.read().decode("utf-8")
            except Exception as e:
                skipped += 1
                print(f"skip ep {i}: {type(e).__name__} {e}", flush=True)
                continue
            T = min(len(states), len(actions))
            dg = g.create_group(f"demo_{kept}")
            dg.create_dataset("states", data=states[:T])
            dg.create_dataset("actions", data=actions[:T])
            dg.attrs["model_file"] = xml
            dg.attrs["ep_meta"] = json.dumps(ep_meta)
            dg.attrs["num_samples"] = T
            kept += 1
            total += T
        g.attrs["total"] = total
    print(f"DONE kept={kept} skipped={skipped} steps={total} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
