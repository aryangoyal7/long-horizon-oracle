"""V-JEPA features for RoboCasa stamps, reading frames from the LeRobot mp4s
and proprio from the parquet observation.state (the converted hdf5 has no
image obs). Output schema matches vjepa_extract.py exactly, so train_head.py
consumes it unchanged.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "predictor"))
from vjepa_extract import encode_batch, MODEL_ID, CLIP_LEN, POOL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="observation.images.robot0_agentview_left")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import imageio.v3 as iio
    import pandas as pd

    info = json.load(open(os.path.join(args.lerobot_dir, "meta", "info.json")))
    chunks = info.get("chunks_size", 1000)

    device = "cuda"
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval().to(device)

    z = np.load(args.labels, allow_pickle=True)
    demo_ids, ts = z["demo_id"], z["t"]
    order = np.lexsort((ts, demo_ids))

    out_feats = [None] * len(ts)
    out_prop = [None] * len(ts)
    cur, frames, prop_all = None, None, None
    bc, bp, bpos = [], [], []

    def flush():
        nonlocal bc, bp, bpos
        if not bc:
            return
        enc = encode_batch(model, np.stack(bc), device)
        for i, pos in enumerate(bpos):
            out_feats[pos] = enc[i]
            out_prop[pos] = bp[i]
        bc, bp, bpos = [], [], []

    for idx in order:
        d, t = int(demo_ids[idx]), int(ts[idx])
        if d != cur:
            flush()
            mp4 = os.path.join(args.lerobot_dir, "videos",
                               f"chunk-{d // chunks:03d}", args.cam,
                               f"episode_{d:06d}.mp4")
            frames = iio.imread(mp4)
            pq = os.path.join(args.lerobot_dir, "data",
                              f"chunk-{d // chunks:03d}", f"episode_{d:06d}.parquet")
            prop_all = np.stack(pd.read_parquet(pq)["observation.state"].values)
            cur = d
        lo = max(0, t - CLIP_LEN + 1)
        clip = frames[lo: t + 1]
        if clip.shape[0] < CLIP_LEN:
            clip = np.concatenate(
                [np.repeat(clip[:1], CLIP_LEN - clip.shape[0], 0), clip])
        pseq = prop_all[lo: t + 1]
        if pseq.shape[0] < CLIP_LEN:
            pseq = np.concatenate(
                [np.repeat(pseq[:1], CLIP_LEN - pseq.shape[0], 0), pseq])
        bc.append(clip)
        bp.append(pseq)
        bpos.append(idx)
        if len(bc) == args.batch:
            flush()
    flush()

    extra = {k: z[k] for k in ["win_contact_gripper", "win_contact_table",
                               "win_contact_fixture"] if k in z.files}
    np.savez(
        args.out,
        features=np.stack(out_feats),
        proprio_seq=np.stack(out_prop).astype(np.float32),
        proprio=np.stack(out_prop)[:, -1].astype(np.float32),
        demo_id=demo_ids, t=ts,
        lambda_task=z["lambda_task"],
        **extra,
        meta=json.dumps({"model": MODEL_ID, "clip_len": CLIP_LEN, "pool": POOL,
                         "cam": args.cam, "labels": args.labels,
                         "labels_meta": str(z["meta"])}),
    )
    print(f"EXTRACTED {len(ts)} stamps -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
