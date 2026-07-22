"""Extract frozen DINOv2 single-frame features for the Stage 1b ablation.

Stamps (demo_id, t) and label targets are read from an EXISTING V-JEPA feature
npz, so the DINOv2 set is stamp-for-stamp aligned with run1's training data and
the pinned run1 validation split applies unchanged. Only the frame AT t is
encoded (no clip), which is the point of the ablation: does the head need the
16-frame motion context, or is a single appearance frame enough?

Tokens stored per stamp: CLS + 16x16 patch grid avg-pooled to 8x8 = 65 x 1024
fp16 (the attentive head accepts any token count).

Run in the vjepa venv:
  CUDA_VISIBLE_DEVICES=4 python dino_extract.py \
      --stamps /mnt/scratch/lh/features/feat2_lift.npz \
      --dataset /mnt/scratch/lh/data/robomimic/lift/ph/image_v15.hdf5 \
      --cam-key agentview_image --out /mnt/scratch/lh/features/dino_lift.npz
"""

import argparse
import json

import h5py
import numpy as np
import torch

MODEL_ID = "facebook/dinov2-large"
POOL_TO = 8            # 16x16 patch grid -> 8x8
LABEL_KEYS = ["lambda_task", "win_contact_gripper", "win_contact_table",
              "win_contact_fixture"]


@torch.no_grad()
def encode_batch(model, frames, device):
    """frames: uint8 (B, H, W, 3) -> (B, 1 + POOL_TO^2, 1024) fp16."""
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear",
                                        align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = model(pixel_values=(x - mean) / std).last_hidden_state  # B,257,D
    cls, patches = out[:, :1], out[:, 1:]
    B, N, D = patches.shape
    g = int(N ** 0.5)                                             # 16
    patches = patches.view(B, g, g, D).permute(0, 3, 1, 2)
    patches = torch.nn.functional.avg_pool2d(patches, g // POOL_TO)
    patches = patches.permute(0, 2, 3, 1).reshape(B, POOL_TO * POOL_TO, D)
    return torch.cat([cls, patches], 1).half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamps", required=True,
                    help="existing V-JEPA feature npz supplying stamps + labels")
    ap.add_argument("--dataset", required=True, help="image hdf5")
    ap.add_argument("--cam-key", required=True)
    ap.add_argument("--demo-prefix", default="demo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(device)

    z = np.load(args.stamps, allow_pickle=True)
    demo_ids, ts, prop = z["demo_id"], z["t"], z["proprio"]

    out_feats = [None] * len(ts)
    order = np.lexsort((ts, demo_ids))
    with h5py.File(args.dataset, "r") as f:
        cur_demo, frames = None, None
        batch_f, batch_pos = [], []

        def flush():
            nonlocal batch_f, batch_pos
            if not batch_f:
                return
            enc = encode_batch(model, np.stack(batch_f), device)
            for i, pos in enumerate(batch_pos):
                out_feats[pos] = enc[i]
            batch_f, batch_pos = [], []

        for idx in order:
            d, t = int(demo_ids[idx]), int(ts[idx])
            if d != cur_demo:
                flush()
                frames = f[f"data/{args.demo_prefix}_{d}/obs/{args.cam_key}"][()]
                cur_demo = d
            batch_f.append(frames[t])
            batch_pos.append(idx)
            if len(batch_f) == args.batch:
                flush()
        flush()

    np.savez(
        args.out,
        features=np.stack(out_feats),
        proprio=prop.astype(np.float32),
        demo_id=demo_ids, t=ts,
        **{k: z[k] for k in LABEL_KEYS if k in z.files},
        meta=json.dumps({"model": MODEL_ID, "single_frame": True,
                         "cam": args.cam_key, "stamps_from": args.stamps}),
    )
    print(f"EXTRACTED {len(ts)} stamps -> {args.out} "
          f"({np.stack(out_feats).nbytes/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
