"""Train the stability predictor head on frozen V-JEPA 2 features.

Architecture (V-JEPA attentive-probe recipe, small):
  learnable query -> cross-attention over the 128 stored tokens -> pooled vector
  proprio -> 2-layer MLP -> concat -> MLP -> [logit_unstable, lambda_hat]

Losses: class-weighted BCE on y = (lambda > delta), deadband stamps (|lambda| <= delta)
EXCLUDED from the BCE (ambiguous by construction) but included in the Huber regression
on lambda. Split is BY DEMO to prevent temporal leakage.

Usage:
  python train_head.py --features feat_lift.npz [feat_can.npz ...] \
      --out results/predictor/lift --epochs 40
Options: --proprio-only (baseline), --holdout-features X.npz (cross-task eval).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


class AttentiveHead(nn.Module):
    def __init__(self, dim=1024, prop_dim=9, hidden=256, heads=8, proprio_only=False):
        super().__init__()
        self.proprio_only = proprio_only
        if not proprio_only:
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            feat_out = dim
        else:
            feat_out = 0
        self.prop = nn.Sequential(nn.Linear(prop_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.mlp = nn.Sequential(
            nn.Linear(feat_out + 64, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2))

    def forward(self, tokens, prop):
        parts = [self.prop(prop)]
        if not self.proprio_only:
            q = self.query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.attn(q, tokens, tokens)
            parts.insert(0, self.norm(pooled[:, 0]))
        out = self.mlp(torch.cat(parts, dim=1))
        return out[:, 0], out[:, 1]                     # logit, lambda_hat


def load_features(paths):
    F, P, L, D = [], [], [], []
    offset = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        F.append(z["features"]); P.append(z["proprio"])
        L.append(z["lambda_task"]); D.append(z["demo_id"] + offset)
        offset += int(z["demo_id"].max()) + 1           # keep demo ids unique across tasks
    return (np.concatenate(F), np.concatenate(P).astype(np.float32),
            np.concatenate(L).astype(np.float32), np.concatenate(D))


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y.astype(bool); np_, nn_ = p.sum(), (~p).sum()
    if np_ == 0 or nn_ == 0:
        return float("nan")
    return float((r[p].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--holdout-features", nargs="+", default=[],
                    help="extra files evaluated only (cross-task transfer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=None,
                    help="deadband; default = p95 |lambda| of low-lambda half per file")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-demos-json", default=None,
                    help="explicit global demo ids for validation (overrides the "
                         "random split; use to pin val across pool variants)")
    ap.add_argument("--proprio-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    F, P, lam, demo = load_features(args.features)
    delta = args.delta if args.delta is not None else float(
        np.percentile(np.abs(lam[lam < np.median(lam)]), 95))
    y = (lam > delta).astype(np.float32)
    confident = np.abs(lam) > delta

    rng = np.random.default_rng(args.seed)
    demos = np.unique(demo)
    if args.val_demos_json:
        with open(args.val_demos_json) as f:
            val_demos = set(json.load(f))
    else:
        val_demos = set(rng.choice(demos, int(len(demos) * args.val_frac),
                                   replace=False))
    val_m = np.isin(demo, list(val_demos))
    tr_m = ~val_m

    prop_mu, prop_sd = P[tr_m].mean(0), P[tr_m].std(0) + 1e-6
    P = (P - prop_mu) / prop_sd

    model = AttentiveHead(dim=F.shape[-1], prop_dim=P.shape[-1],
                          proprio_only=args.proprio_only).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos_w = torch.tensor([(y[tr_m & confident] == 0).sum()
                          / max((y[tr_m & confident] == 1).sum(), 1)]).to(dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    hub = nn.HuberLoss(delta=0.1)

    tr_idx = np.where(tr_m)[0]
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr_idx)
        tot = 0.0
        for i in range(0, len(tr_idx), args.batch):
            b = tr_idx[i: i + args.batch]
            tok = torch.from_numpy(F[b]).float().to(dev)
            pr = torch.from_numpy(P[b]).to(dev)
            lg, lh = model(tok, pr)
            m = torch.from_numpy(confident[b]).to(dev)
            loss = hub(lh, torch.from_numpy(lam[b]).to(dev))
            if m.any():
                loss = loss + bce(lg[m], torch.from_numpy(y[b]).to(dev)[m])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()

    def evaluate(Fe, Pe, lame, tag):
        ye = (lame > delta).astype(np.float32)
        conf = np.abs(lame) > delta
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Fe), 512):
                tok = torch.from_numpy(Fe[i:i+512]).float().to(dev)
                pr = torch.from_numpy(Pe[i:i+512]).to(dev)
                lg, lh = model(tok, pr)
                outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
        o = np.concatenate(outs)
        res = {
            f"{tag}_auroc_all": auroc(o[:, 0], ye),
            f"{tag}_auroc_confident": auroc(o[:, 0][conf], ye[conf]),
            f"{tag}_lambda_mae": float(np.abs(o[:, 1] - lame).mean()),
            f"{tag}_frac_unstable": float(ye.mean()),
            f"{tag}_n": int(len(ye)),
        }
        return res, o

    results = {"delta": delta, "features": args.features,
               "proprio_only": args.proprio_only,
               "n_train": int(tr_m.sum()), "n_val": int(val_m.sum())}
    r, _ = evaluate(F[val_m], P[val_m], lam[val_m], "val")
    results.update(r)
    for hp in args.holdout_features:
        Fh, Ph, lamh, _ = load_features([hp])
        Ph = (Ph - prop_mu) / prop_sd
        r, _ = evaluate(Fh, Ph, lamh, "xfer_" + os.path.basename(hp).split(".")[0])
        results.update(r)

    torch.save({"model": model.state_dict(), "prop_mu": prop_mu, "prop_sd": prop_sd,
                "delta": delta, "args": vars(args)},
               os.path.join(args.out, "head.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
