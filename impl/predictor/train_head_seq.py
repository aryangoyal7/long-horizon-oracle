"""Sequence-history stability predictor (2026-07-30, user request).

The single-clip head sees one causal 16-frame V-JEPA clip ending at t. This
variant stacks the clips of the last `hist` replan-spaced moments
(t, t-16, t-32, t-48 for hist=4, ~3.2 s of history) and attends over all of
them, with a learned per-clip position embedding. hist=1 through the same
code is the matched single-clip baseline.

Labels come from a SEPARATE npz (--labels, the v3 relabeling): the
lambda_task stored inside the feature files is v2 (action-order bug) and is
ignored. Features are joined to labels on (demo_id, t); features themselves
are frame-only and unaffected by the bug.

Usage:
  python train_head_seq.py --features feat2_rc_X.npz --labels final3_env_rc_X.npz \
      --hist 4 --seed 0 --out results/predictor/rc_seq_v3/S_X_h4_s0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_head import auroc, spearman, auto_delta  # noqa: E402


class SeqHead(nn.Module):
    def __init__(self, dim=1024, prop_dim=16, hidden=256, heads=8, hist=4):
        super().__init__()
        self.hist = hist
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.clip_emb = nn.Parameter(torch.randn(hist, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.prop_rnn = nn.GRU(prop_dim, 64, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim + 64, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2))

    def forward(self, tokens, prop):
        # tokens (B, hist, K, D); prop (B, T, P)
        B, H, K, D = tokens.shape
        tokens = (tokens + self.clip_emb[None]).reshape(B, H * K, D)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)
        _, h = self.prop_rnn(prop)
        out = self.mlp(torch.cat([self.norm(pooled[:, 0]), h[-1]], dim=1))
        return out[:, 0], out[:, 1]


def build_hist_index(demo, t, hist, stride):
    """Row indices of the clips at t, t-stride, ..., clamped to each demo's
    earliest stamp (matches the episode-start padding used at extraction)."""
    idx = np.empty((len(t), hist), dtype=np.int64)
    for d in np.unique(demo):
        rows = np.where(demo == d)[0]
        rows = rows[np.argsort(t[rows])]
        ts = t[rows]
        for j in range(hist):
            want = t[rows] - (hist - 1 - j) * stride
            pos = np.searchsorted(ts, want, side="right") - 1
            idx[rows, j] = rows[np.clip(pos, 0, len(rows) - 1)]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
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
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.features, allow_pickle=True)
    F = z["features"]                                  # (N, K, D) fp16
    if "proprio_seq" in z.files:
        P = z["proprio_seq"].astype(np.float32)        # (N, win, P)
    else:
        # older feature files store only the stamp snapshot. Tile it to the
        # window so the proprio GRU has a valid input shape; a constant
        # sequence makes the GRU equivalent to the snapshot MLP, which is the
        # right control here because the h1 vs h4 question is about VISUAL
        # history, not proprioceptive history.
        P = np.repeat(z["proprio"].astype(np.float32)[:, None, :], 16, axis=1)
        print("[proprio] no proprio_seq in features; tiling the snapshot")
    fd, ft = z["demo_id"], z["t"]
    lz = np.load(args.labels, allow_pickle=True)
    lab = {(int(d), int(s)): float(l) for d, s, l in
           zip(lz["demo_id"], lz["t"], lz["lambda_task"])}
    lam = np.array([lab.get((int(d), int(s)), np.nan) for d, s in zip(fd, ft)])
    keep = ~np.isnan(lam)
    frac = keep.mean()
    assert frac > 0.9, f"feature/label stamp overlap only {frac:.2%}"
    # keep every feature row (history sources) but train/eval on matched stamps
    hist_idx = build_hist_index(fd, ft, args.hist, args.hist_stride)
    rows = np.where(keep)[0]
    lam_k = lam[rows].astype(np.float32)
    demo_k = fd[rows]

    delta = auto_delta(lam_k)
    y = (lam_k > delta).astype(np.float32)
    confident = np.abs(lam_k) > delta

    rng = np.random.default_rng(args.seed)
    demos = np.unique(demo_k)
    val_demos = set(rng.choice(demos, int(len(demos) * args.val_frac),
                               replace=False))
    val_m = np.isin(demo_k, list(val_demos))
    tr_m = ~val_m

    mu, sd = P[rows][tr_m].mean((0, 1)), P[rows][tr_m].std((0, 1)) + 1e-6
    P = (P - mu) / sd

    model = SeqHead(dim=F.shape[-1], prop_dim=P.shape[-1],
                    hist=args.hist).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos_w = torch.tensor([(y[tr_m & confident] == 0).sum()
                          / max((y[tr_m & confident] == 1).sum(), 1)]).to(dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    hub = nn.HuberLoss(delta=0.1)

    def gather(sel):
        """sel: indices into the matched (rows) arrays -> (tok, prop)."""
        r = rows[sel]
        tok = torch.from_numpy(F[hist_idx[r]].astype(np.float32)).to(dev)
        pr = torch.from_numpy(P[r]).to(dev)
        return tok, pr

    tr_idx = np.where(tr_m)[0]
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), args.batch):
            b = tr_idx[i: i + args.batch]
            tok, pr = gather(b)
            lg, lh = model(tok, pr)
            m = torch.from_numpy(confident[b]).to(dev)
            loss = hub(lh, torch.from_numpy(lam_k[b]).to(dev))
            if m.any():
                loss = loss + bce(lg[m], torch.from_numpy(y[b]).to(dev)[m])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    vi = np.where(val_m)[0]
    outs = []
    with torch.no_grad():
        for i in range(0, len(vi), 256):
            tok, pr = gather(vi[i: i + 256])
            lg, lh = model(tok, pr)
            outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
    o = np.concatenate(outs)
    lv, yv, cv = lam_k[val_m], y[val_m], confident[val_m]
    fire = o[:, 1] > delta
    tail = yv.astype(bool)
    res = {
        "tag": args.tag, "hist": args.hist, "hist_stride": args.hist_stride,
        "seed": args.seed, "delta": delta, "epochs": args.epochs,
        "n_train": int(tr_m.sum()), "n_val": int(val_m.sum()),
        "stamp_overlap": float(frac),
        "val_auroc_all": auroc(o[:, 0], yv),
        "val_auroc_confident": auroc(o[:, 0][cv], yv[cv]),
        "val_lambda_mae": float(np.abs(o[:, 1] - lv).mean()),
        "val_lambda_spearman": spearman(o[:, 1], lv),
        "val_frac_unstable": float(yv.mean()),
        # tail = the open-loop-unstable stratum; the rest = the stable bulk
        "val_tail_recall": float(fire[tail].mean()) if tail.any() else None,
        "val_tail_precision": float(tail[fire].mean()) if fire.any() else None,
        "val_stable_false_fire": float(fire[~tail].mean()),
    }
    torch.save({"model": model.state_dict(), "prop_mu": mu, "prop_sd": sd,
                "delta": delta, "args": vars(args), "seq": True,
                "hist": args.hist, "hist_stride": args.hist_stride,
                "prop_seq": True, "prop_win": int(P.shape[1])},
               os.path.join(args.out, "head.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
