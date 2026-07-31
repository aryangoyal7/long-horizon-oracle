"""Inference server for the sequence-history head (train_head_seq.py).

Same line protocol as predictor_bridge.py (READY <delta> / REQ <npz> /
RES <lambda_hat> <p>), but the npz frames buffer may hold up to
hist*16 frames; it is sliced into `hist` causal 16-frame clips at 16-frame
spacing (padded at the front by repeating the first frame, matching both
training-time clamping and episode starts).

Run: CUDA_VISIBLE_DEVICES=<gpu> python predictor_bridge_seq.py --head <dir>/head.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

IMPL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(IMPL, "predictor"))
from train_head_seq import SeqHead                     # noqa: E402
from vjepa_extract import CLIP_LEN, MODEL_ID, encode_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    args = ap.parse_args()
    device = "cuda"

    from transformers import AutoModel
    enc = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    enc.eval().to(device)

    ck = torch.load(args.head, map_location=device, weights_only=False)
    hist = int(ck["hist"])
    stride = int(ck.get("hist_stride", CLIP_LEN))
    prop_win = int(ck.get("prop_win", 16))
    head = SeqHead(prop_dim=len(ck["prop_mu"]), hist=hist)
    head.load_state_dict(ck["model"])
    head.eval().to(device)
    mu = torch.tensor(np.asarray(ck["prop_mu"]), dtype=torch.float32, device=device)
    sd = torch.tensor(np.asarray(ck["prop_sd"]), dtype=torch.float32, device=device)

    print(f"READY {ck.get('delta', 0.0)}", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line == "QUIT":
            break
        if not line.startswith("REQ "):
            continue
        z = np.load(line[4:])
        frames = z["frames"]                            # (T, H, W, 3)
        T = frames.shape[0]
        clips = []
        for j in range(hist):                           # oldest -> newest
            end = T - 1 - (hist - 1 - j) * stride
            end = max(end, 0)
            clip = frames[max(0, end - CLIP_LEN + 1): end + 1]
            if clip.shape[0] < CLIP_LEN:
                clip = np.concatenate(
                    [np.repeat(clip[:1], CLIP_LEN - clip.shape[0], 0), clip])
            clips.append(clip)
        with torch.no_grad():
            tokens = encode_batch(enc, np.stack(clips), device)  # (hist,128,D)
            tokens = torch.from_numpy(tokens).float().to(device)[None]
            ps = z["proprio_seq"]
            if ps.shape[0] < prop_win:
                ps = np.concatenate(
                    [np.repeat(ps[:1], prop_win - ps.shape[0], 0), ps])
            prop = torch.from_numpy(ps[-prop_win:][None]).float().to(device)
            prop = (prop - mu) / (sd + 1e-8)
            logit, lam_hat = head(tokens, prop)
        # p is often saturated near 1 for the tail class; print enough digits
        # that a quantile-matched threshold remains comparable
        print(f"RES {lam_hat[0].item():.6f} {torch.sigmoid(logit[0]).item():.8f}",
              flush=True)


if __name__ == "__main__":
    main()
