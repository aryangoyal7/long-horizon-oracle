"""Compare run1 (Panda pool) and run3 (Panda + LIBERO) heads on the IDENTICAL
Panda validation stamps, so the LIBERO in/out decision is not confounded by
different val splits.

run1's val split is reconstructed exactly: same feature list, same order, same
seed, same rng call sequence as train_head.py. Both heads are then scored on
those stamps with run1's training delta.
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/impl/predictor")
from train_head import AttentiveHead, auroc, load_features

F_DIR = "/mnt/scratch/lh/features"
RUN1_FEATURES = [f"{F_DIR}/feat2_lift.npz", f"{F_DIR}/feat2_can.npz",
                 f"{F_DIR}/feat2_square.npz", f"{F_DIR}/feat2_tool_hang.npz",
                 f"{F_DIR}/feat2_rollouts_lift.npz", f"{F_DIR}/feat2_rollouts_can.npz",
                 f"{F_DIR}/feat_mg_stack_d0.npz", f"{F_DIR}/feat_mg_square_d0.npz"]
P_DIR = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/predictor"

dev = "cuda" if torch.cuda.is_available() else "cpu"
F, P, lam, demo = load_features(RUN1_FEATURES)

# exact replay of train_head's split logic for run1 (seed 0, val_frac 0.2)
delta = float(np.percentile(np.abs(lam[lam < np.median(lam)]), 95))
rng = np.random.default_rng(0)
demos = np.unique(demo)
val_demos = set(rng.choice(demos, int(len(demos) * 0.2), replace=False))
val_m = np.isin(demo, list(val_demos))
y = (lam > delta).astype(np.float32)
conf = np.abs(lam) > delta
print(f"panda val stamps: {val_m.sum()}  delta {delta:.4f}  "
      f"frac_unstable {y[val_m].mean():.3f}")

for name in (sys.argv[1:] or ["run1_panda_pool", "run3_panda_plus_libero"]):
    ck = torch.load(f"{P_DIR}/{name}/head.pt", map_location=dev, weights_only=False)
    head = AttentiveHead(prop_dim=len(ck["prop_mu"]))
    head.load_state_dict(ck["model"]); head.eval().to(dev)
    Pn = (P - np.asarray(ck["prop_mu"])) / np.asarray(ck["prop_sd"])
    scores, lams_hat = [], []
    with torch.no_grad():
        idx = np.where(val_m)[0]
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            lg, lh = head(torch.from_numpy(F[b]).float().to(dev),
                          torch.from_numpy(Pn[b]).float().to(dev))
            scores.append(lg.cpu().numpy()); lams_hat.append(lh.cpu().numpy())
    s = np.concatenate(scores); lh = np.concatenate(lams_hat)
    yv, cv, lv = y[val_m], conf[val_m], lam[val_m]
    print(json.dumps({
        "head": name,
        "panda_val_auroc_all": auroc(s, yv),
        "panda_val_auroc_confident": auroc(s[cv], yv[cv]),
        "panda_val_lambda_mae": float(np.abs(lh - lv).mean()),
    }))
print("COMPARE_DONE")
