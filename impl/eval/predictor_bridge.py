"""Stability-predictor inference server for Stage 1c evaluation.

Runs in the vjepa venv (new transformers); eval_k_sweep (lh venv, old transformers
pinned by robomimic) talks to it over a line protocol:

  bridge stdout:  READY <delta>            after models are loaded
  eval  -> stdin: REQ <npz_path>           npz: frames (T,H,W,3) uint8, proprio (P,)
  bridge stdout:  RES <lambda_hat> <p_unstable>
  eval  -> stdin: QUIT

The clip is padded at the front by repeating the first frame if T < 16, matching
the training-time episode-start padding in vjepa_extract.py.

Run: CUDA_VISIBLE_DEVICES=<gpu> python predictor_bridge.py --head <dir>/head.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

IMPL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(IMPL, "predictor"))
from train_head import AttentiveHead              # noqa: E402
from vjepa_extract import CLIP_LEN, MODEL_ID, encode_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="path to head.pt")
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    enc.eval().to(device)

    ck = torch.load(args.head, map_location=device, weights_only=False)
    head = AttentiveHead(prop_dim=len(ck["prop_mu"]))
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
        clip = z["frames"]
        if clip.shape[0] < CLIP_LEN:
            clip = np.concatenate(
                [np.repeat(clip[:1], CLIP_LEN - clip.shape[0], 0), clip])
        with torch.no_grad():
            tokens = encode_batch(enc, clip[None], device)     # 1,128,1024 fp16
            tokens = torch.from_numpy(tokens).float().to(device)
            prop = torch.from_numpy(z["proprio"][None]).float().to(device)
            prop = (prop - mu) / (sd + 1e-8)
            logit, lam_hat = head(tokens, prop)
        p = torch.sigmoid(logit[0]).item()
        print(f"RES {lam_hat[0].item():.6f} {p:.4f}", flush=True)


if __name__ == "__main__":
    main()
