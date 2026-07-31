"""Operating-point calibration for a trained sequence head (2026-07-31).

The online cells fired on (lambda_hat > delta), where delta is the LABEL-space
threshold. Regression heads compress toward the mean, so lambda_hat almost
never crosses it: every online predictor cell ran at mean_k 16.0 with a 0.00
fire rate, i.e. it never switched. Ranking was fine (AUC .84-.90); only the
threshold was wrong.

This recomputes, on the held-out demos of the SAME split the head trained
with, the thresholds on p_unstable and on lambda_hat that reproduce a target
fire rate (default: the true positive rate of the labels), and writes them
next to the head as calib.json. No retraining.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_head_seq import SeqHead, build_hist_index, auto_delta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--target-rate", type=float, default=None,
                    help="fire rate to calibrate to; default = label positive rate")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    ha = ck["args"]
    hist, stride = int(ck["hist"]), int(ck.get("hist_stride", 16))
    seed, val_frac = int(ha["seed"]), float(ha["val_frac"])

    z = np.load(args.features, allow_pickle=True)
    F = z["features"]; P = z["proprio_seq"].astype(np.float32)
    fd, ft = z["demo_id"], z["t"]
    lz = np.load(args.labels, allow_pickle=True)
    lab = {(int(d), int(s)): float(l) for d, s, l in
           zip(lz["demo_id"], lz["t"], lz["lambda_task"])}
    lam = np.array([lab.get((int(d), int(s)), np.nan) for d, s in zip(fd, ft)])
    keep = ~np.isnan(lam)
    hist_idx = build_hist_index(fd, ft, hist, stride)
    rows = np.where(keep)[0]
    lam_k = lam[rows].astype(np.float32)
    demo_k = fd[rows]
    delta = auto_delta(lam_k)

    rng = np.random.default_rng(seed)          # same first draw as training
    demos = np.unique(demo_k)
    val_demos = set(rng.choice(demos, int(len(demos) * val_frac), replace=False))
    val_m = np.isin(demo_k, list(val_demos))
    tr_m = ~val_m

    mu = np.asarray(ck["prop_mu"]); sd = np.asarray(ck["prop_sd"])
    P = (P - mu) / sd

    model = SeqHead(dim=F.shape[-1], prop_dim=P.shape[-1], hist=hist).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    vi = np.where(val_m)[0]
    outs = []
    with torch.no_grad():
        for i in range(0, len(vi), 256):
            r = rows[vi[i: i + 256]]
            tok = torch.from_numpy(F[hist_idx[r]].astype(np.float32)).to(dev)
            pr = torch.from_numpy(P[r]).to(dev)
            lg, lh = model(tok, pr)
            outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
    o = np.concatenate(outs)
    p_hat, lam_hat = o[:, 0], o[:, 1]
    lam_val = lam_k[val_m]
    y = lam_val > delta
    target = args.target_rate if args.target_rate is not None else float(y.mean())

    q = 100 * (1 - target)
    p_thresh = float(np.percentile(p_hat, q))
    lamhat_thresh = float(np.percentile(lam_hat, q))
    fired_p = p_hat > p_thresh
    fired_l = lam_hat > lamhat_thresh
    calib = {
        "delta_label": delta, "target_rate": target,
        "p_thresh": p_thresh, "lamhat_thresh": lamhat_thresh,
        "lamhat_min": float(lam_hat.min()), "lamhat_max": float(lam_hat.max()),
        "lamhat_frac_over_delta": float((lam_hat > delta).mean()),
        "n_val": int(len(lam_val)),
        "recall_at_p_thresh": float(fired_p[y].mean()) if y.any() else None,
        "precision_at_p_thresh": float(y[fired_p].mean()) if fired_p.any() else None,
        "false_fire_at_p_thresh": float(fired_p[~y].mean()),
        "recall_at_lamhat_thresh": float(fired_l[y].mean()) if y.any() else None,
        "precision_at_lamhat_thresh": float(y[fired_l].mean()) if fired_l.any() else None,
        "false_fire_at_lamhat_thresh": float(fired_l[~y].mean()),
    }
    out = os.path.join(os.path.dirname(args.head), "calib.json")
    with open(out, "w") as f:
        json.dump(calib, f, indent=2)
    print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
