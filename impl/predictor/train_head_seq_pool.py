"""Pooled multi-task stability head (2026-07-31).

train_head_seq.py trains one head per task. For a composed episode that
crosses several tasks, using per-task heads requires knowing which task the
robot is currently in, which on the composed benchmark comes from the
environment's own stage predicate. That is privileged information the
controller would not have on a real robot, and only the predictor cells
benefit from it, so it biases the comparison. This trains a single head over
all tasks so no stage signal is needed at run time.

Pooling detail: each task has its own noise floor, with auto-delta ranging
from .058 to .085 on RoboCasa, so raw lambda is not comparable across tasks.
We train on lambda scaled by each task's own delta, which puts every task's
decision boundary at 1.0 and gives the pooled head a single threshold.

Usage:
  python train_head_seq_pool.py --hist 4 --seed 0 --out <dir> \
      --features feat_a.npz feat_b.npz --labels lab_a.npz lab_b.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_head import auroc, spearman, auto_delta          # noqa: E402
from train_head_seq import SeqHead, build_hist_index        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--hist", type=int, default=4)
    ap.add_argument("--hist-stride", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert len(args.features) == len(args.labels), "features/labels must pair"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    F_all, P_all, y_all, demo_all, hist_all, task_all = [], [], [], [], [], []
    per_task_delta, offset, row_base = {}, 0, 0
    for fi, (ff, lf) in enumerate(zip(args.features, args.labels)):
        z = np.load(ff, allow_pickle=True)
        F = z["features"]; P = z["proprio_seq"].astype(np.float32)
        fd, ft = z["demo_id"], z["t"]
        lz = np.load(lf, allow_pickle=True)
        lab = {(int(d), int(s)): float(l) for d, s, l in
               zip(lz["demo_id"], lz["t"], lz["lambda_task"])}
        lam = np.array([lab.get((int(d), int(s)), np.nan)
                        for d, s in zip(fd, ft)])
        keep = ~np.isnan(lam)
        assert keep.mean() > 0.9, f"{ff}: stamp overlap only {keep.mean():.2%}"
        d = auto_delta(lam[keep])
        name = os.path.basename(lf).replace(".npz", "")
        per_task_delta[name] = d
        # scale so every task's decision boundary sits at 1.0
        hidx = build_hist_index(fd, ft, args.hist, args.hist_stride) + row_base
        rows = np.where(keep)[0]
        F_all.append(F); P_all.append(P)
        hist_all.append(hidx[rows])
        y_all.append((lam[rows] / d).astype(np.float32))
        demo_all.append(fd[rows] + offset)
        task_all.append(np.full(len(rows), fi))
        offset += int(fd.max()) + 1
        row_base += len(fd)

    F = np.concatenate(F_all); P = np.concatenate(P_all).astype(np.float32)
    hist_idx = np.concatenate(hist_all)
    lam_s = np.concatenate(y_all)            # lambda / delta, boundary at 1
    demo = np.concatenate(demo_all); task = np.concatenate(task_all)
    del F_all, P_all, y_all, demo_all, hist_all

    DELTA = 1.0
    y = (lam_s > DELTA).astype(np.float32)
    confident = np.abs(lam_s) > DELTA

    rng = np.random.default_rng(args.seed)
    demos = np.unique(demo)
    val_demos = set(rng.choice(demos, int(len(demos) * args.val_frac),
                               replace=False))
    val_m = np.isin(demo, list(val_demos)); tr_m = ~val_m

    mu, sd = P.mean((0, 1)), P.std((0, 1)) + 1e-6
    P = (P - mu) / sd

    model = SeqHead(dim=F.shape[-1], prop_dim=P.shape[-1],
                    hist=args.hist).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos_w = torch.tensor([(y[tr_m & confident] == 0).sum()
                          / max((y[tr_m & confident] == 1).sum(), 1)]).to(dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    hub = nn.HuberLoss(delta=0.5)

    def gather(sel):
        tok = torch.from_numpy(F[hist_idx[sel]].astype(np.float32)).to(dev)
        pr = torch.from_numpy(P[hist_idx[sel][:, -1]]).to(dev)
        return tok, pr

    tr_idx = np.where(tr_m)[0]
    for ep in range(args.epochs):
        model.train(); rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), args.batch):
            b = tr_idx[i: i + args.batch]
            tok, pr = gather(b)
            lg, lh = model(tok, pr)
            m = torch.from_numpy(confident[b]).to(dev)
            loss = hub(lh, torch.from_numpy(lam_s[b]).to(dev))
            if m.any():
                loss = loss + bce(lg[m], torch.from_numpy(y[b]).to(dev)[m])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval(); vi = np.where(val_m)[0]; outs = []
    with torch.no_grad():
        for i in range(0, len(vi), 256):
            tok, pr = gather(vi[i: i + 256])
            lg, lh = model(tok, pr)
            outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
    o = np.concatenate(outs)
    lv, yv, cv = lam_s[val_m], y[val_m], confident[val_m]
    res = {"tag": args.tag, "hist": args.hist, "seed": args.seed,
           "pooled_tasks": args.labels, "per_task_delta": per_task_delta,
           "delta_scaled": DELTA, "epochs": args.epochs,
           "n_train": int(tr_m.sum()), "n_val": int(val_m.sum()),
           "val_auroc_all": auroc(o[:, 0], yv),
           "val_auroc_confident": auroc(o[:, 0][cv], yv[cv]),
           "val_lambda_spearman": spearman(o[:, 1], lv),
           "val_frac_unstable": float(yv.mean())}
    # per-task breakdown on the pooled validation split
    tv = task[val_m]
    for fi, lf in enumerate(args.labels):
        m = tv == fi
        if m.sum() > 50 and yv[m].any() and (~yv[m].astype(bool)).any():
            res[f"auroc_{os.path.basename(lf).replace('.npz','')}"] = \
                auroc(o[:, 0][m], yv[m])
    torch.save({"model": model.state_dict(), "prop_mu": mu, "prop_sd": sd,
                "delta": DELTA, "args": vars(args), "seq": True,
                "hist": args.hist, "hist_stride": args.hist_stride,
                "per_task_delta": per_task_delta,
                "prop_seq": True, "prop_win": int(P.shape[1])},
               os.path.join(args.out, "head.pt"))
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
