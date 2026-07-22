"""Extract frozen V-JEPA 2 features for every labeled stamp.

For each stamp (demo_id, t) in the labels npz: take the last 16 frames of the main
camera ending at t (padded at episode start), resize to 256, run the frozen V-JEPA 2
ViT-L encoder, and store a compact token grid: 8 temporal x (4x4 spatially pooled)
x 1024 = 128 tokens per stamp (fp16). The attentive-probe head trains on these.

Also stores proprio (eef pos/quat/gripper) at t and the label targets copied from the
labels npz, making the output a self-contained training set for the predictor.

Run (one GPU per task):
  CUDA_VISIBLE_DEVICES=4 python vjepa_extract.py --task square \
      --labels /mnt/scratch/lh/labels/final_ol_square.npz \
      --out /mnt/scratch/lh/features/feat_square.npz
"""

import argparse
import json

import h5py
import numpy as np
import torch

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
CLIP_LEN = 16          # frames fed to the encoder (tubelet 2 -> 8 temporal tokens)
POOL = 4               # spatial pooling of the 16x16 patch grid -> 4x4
DATA_ROOT = "/mnt/scratch/lh/data/robomimic"
MAIN_CAM = {"lift": "agentview", "can": "agentview", "square": "agentview",
            "tool_hang": "sideview"}
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


@torch.no_grad()
def encode_batch(model, clips, device):
    """clips: uint8 (B, T, H, W, 3) -> pooled tokens (B, 8*POOL*POOL, D) fp16."""
    x = torch.from_numpy(clips).to(device)
    x = x.permute(0, 1, 4, 2, 3).float() / 255.0            # B,T,C,H,W
    x = torch.nn.functional.interpolate(
        x.flatten(0, 1), size=(256, 256), mode="bilinear", align_corners=False
    ).unflatten(0, (clips.shape[0], CLIP_LEN))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    x = (x - mean) / std
    out = model(pixel_values_videos=x).last_hidden_state     # B, 8*16*16, D
    B, N, D = out.shape
    T = CLIP_LEN // 2
    g = int((N // T) ** 0.5)                                  # 16
    out = out.view(B, T, g, g, D).permute(0, 1, 4, 2, 3)      # B,T,D,g,g
    out = torch.nn.functional.avg_pool2d(out.flatten(0, 1), g // POOL)
    out = out.unflatten(0, (B, T)).permute(0, 1, 3, 4, 2)     # B,T,4,4,D
    return out.reshape(B, T * POOL * POOL, D).half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="camera lookup key, or use --cam to override")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dataset", default=None,
                    help="explicit image hdf5 path (overrides DATA_ROOT/task layout)")
    ap.add_argument("--cam", default=None,
                    help="camera name (overrides MAIN_CAM[task])")
    ap.add_argument("--cam-key", default=None,
                    help="full obs key for frames (e.g. agentview_rgb for LIBERO)")
    ap.add_argument("--proprio-keys", nargs="+", default=None,
                    help="obs keys for proprio (LIBERO: ee_pos ee_ori gripper_states)")
    ap.add_argument("--demo-prefix", default="demo",
                    help="hdf5 demo group prefix (rollout files use 'rollout')")
    ap.add_argument("--lambda-key", default="lambda_task",
                    help="labels npz key for the regression target "
                         "(CL labels use lambda_cl_task)")
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval().to(device)

    z = np.load(args.labels, allow_pickle=True)
    demo_ids, ts = z["demo_id"], z["t"]
    cam_key = args.cam_key or f"{(args.cam or MAIN_CAM[args.task])}_image"
    proprio_keys = args.proprio_keys or PROPRIO_KEYS

    feats, proprio = [], []
    ds_path = args.dataset or f"{DATA_ROOT}/{args.task}/ph/image_v15.hdf5"
    with h5py.File(ds_path, "r") as f:
        order = np.lexsort((ts, demo_ids))                    # group stamps by demo
        cur_demo, frames, prop_all = None, None, None
        batch_clips, batch_prop, batch_pos = [], [], []
        out_feats = [None] * len(ts)
        out_prop = [None] * len(ts)

        def flush():
            nonlocal batch_clips, batch_prop, batch_pos
            if not batch_clips:
                return
            enc = encode_batch(model, np.stack(batch_clips), device)
            for i, pos in enumerate(batch_pos):
                out_feats[pos] = enc[i]
                out_prop[pos] = batch_prop[i]
            batch_clips, batch_prop, batch_pos = [], [], []

        for idx in order:
            d, t = int(demo_ids[idx]), int(ts[idx])
            if d != cur_demo:
                flush()
                g = f[f"data/{args.demo_prefix}_{d}"]
                frames = g[f"obs/{cam_key}"][()]              # T,H,W,3 uint8
                prop_all = np.concatenate(
                    [g[f"obs/{k}"][()] for k in proprio_keys], axis=1)
                cur_demo = d
            lo = max(0, t - CLIP_LEN + 1)
            clip = frames[lo: t + 1]
            if clip.shape[0] < CLIP_LEN:                      # pad episode start
                clip = np.concatenate(
                    [np.repeat(clip[:1], CLIP_LEN - clip.shape[0], 0), clip])
            batch_clips.append(clip)
            batch_prop.append(prop_all[t])
            batch_pos.append(idx)
            if len(batch_clips) == args.batch:
                flush()
        flush()

    extra = {k: z[k] for k in ["win_contact_gripper", "win_contact_table",
                               "win_contact_fixture"] if k in z.files}
    np.savez(
        args.out,
        features=np.stack(out_feats),                        # N,128,1024 fp16
        proprio=np.stack(out_prop).astype(np.float32),
        demo_id=demo_ids, t=ts,
        lambda_task=z[args.lambda_key],
        **extra,
        meta=json.dumps({"model": MODEL_ID, "clip_len": CLIP_LEN, "pool": POOL,
                         "cam": cam_key, "labels": args.labels,
                         "lambda_key": args.lambda_key,
                         "labels_meta": str(z["meta"])}),
    )
    print(f"EXTRACTED {len(ts)} stamps -> {args.out} "
          f"({np.stack(out_feats).nbytes/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
