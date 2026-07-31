"""Train the stability predictor head on frozen V-JEPA 2 or DINOv2 features.

Architecture (V-JEPA attentive-probe recipe, small):
  learnable query -> cross-attention over the stored tokens -> pooled vector
  proprio -> 2-layer MLP -> concat -> MLP -> [logit_unstable, lambda_hat]

Losses: class-weighted BCE on y = (lambda > delta), deadband stamps (|lambda| <= delta)
EXCLUDED from the BCE (ambiguous by construction) but included in the Huber regression
on lambda. Split is BY DEMO to prevent temporal leakage.

Usage:
  python train_head.py --features feat_lift.npz [feat_can.npz ...] \
      --out results/predictor/lift --epochs 40
Ablation options:
  --proprio-only        drop the visual tokens (proprioception floor)
  --no-proprio          drop the proprio vector (vision-only)
  --holdout-features    extra files evaluated only (cross-task transfer)

Holdout sets are scored twice: once against the training delta (the deployed
threshold) and once against a delta refit on the holdout set itself, which
separates a ranking failure from a calibration failure. Spearman rank
correlation between lambda_hat and true lambda is reported for the same reason:
it is threshold-free.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


class AttentiveHead(nn.Module):
    def __init__(self, dim=1024, prop_dim=9, hidden=256, heads=8, proprio_only=False,
                 no_proprio=False, prop_seq=False):
        super().__init__()
        self.proprio_only = proprio_only
        self.no_proprio = no_proprio
        self.prop_seq = prop_seq
        if not proprio_only:
            self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            feat_out = dim
        else:
            feat_out = 0
        self.prop = self.prop_rnn = None
        if not no_proprio:
            if prop_seq:
                # A GRU over the proprio window, so the head sees the trajectory
                # (hence velocity and gripper closing) rather than one snapshot.
                self.prop_rnn = nn.GRU(prop_dim, 64, batch_first=True)
            else:
                self.prop = nn.Sequential(nn.Linear(prop_dim, 64), nn.GELU(),
                                          nn.Linear(64, 64))
            prop_out = 64
        else:
            prop_out = 0
        self.mlp = nn.Sequential(
            nn.Linear(feat_out + prop_out, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2))

    def forward(self, tokens, prop):
        parts = []
        if not self.no_proprio:
            if self.prop_rnn is not None:
                if prop.dim() == 2:                     # legacy snapshot input
                    prop = prop.unsqueeze(1)
                _, h = self.prop_rnn(prop)
                parts.append(h[-1])
            else:
                if prop.dim() == 3:                     # snapshot head, seq input
                    prop = prop[:, -1]
                parts.append(self.prop(prop))
        if not self.proprio_only:
            q = self.query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.attn(q, tokens, tokens)
            parts.insert(0, self.norm(pooled[:, 0]))
        out = self.mlp(torch.cat(parts, dim=1))
        return out[:, 0], out[:, 1]                     # logit, lambda_hat


def load_features(paths, seq=True):
    """Proprio comes back as (N, T, 9) when every file carries proprio_seq at a common
    window length, else as (N, 9) snapshots. Mixed pools are collapsed to the last
    frame rather than silently padded, and the collapse is printed."""
    F, P, L, D = [], [], [], []
    offset = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        F.append(z["features"])
        P.append(z["proprio_seq"] if (seq and "proprio_seq" in z.files)
                 else z["proprio"][:, None, :])
        L.append(z["lambda_task"]); D.append(z["demo_id"] + offset)
        offset += int(z["demo_id"].max()) + 1           # keep demo ids unique across tasks
    lens = {a.shape[1] for a in P}
    if len(lens) > 1:
        print(f"[proprio] mixed window lengths {sorted(lens)}; using last frame only")
        P = [a[:, -1:] for a in P]
    P = np.concatenate(P).astype(np.float32)
    if P.shape[1] == 1:
        P = P[:, 0]                                     # legacy (N, 9)
    return (np.concatenate(F), P,
            np.concatenate(L).astype(np.float32), np.concatenate(D))


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y.astype(bool); np_, nn_ = p.sum(), (~p).sum()
    if np_ == 0 or nn_ == 0:
        return float("nan")
    return float((r[p].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def rankdata(x):
    o = np.argsort(x); r = np.empty(len(x), dtype=np.float64)
    r[o] = np.arange(1, len(x) + 1)
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auto_delta(lam):
    return float(np.percentile(np.abs(lam[lam < np.median(lam)]), 95))


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
    ap.add_argument("--proprio-last-only", action="store_true",
                    help="ignore proprio_seq and use the stamp snapshot only "
                         "(the pre-sequence behaviour; matched ablation control)")
    ap.add_argument("--no-proprio", action="store_true",
                    help="vision-only ablation: drop the proprio vector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="free-text label recorded in metrics.json")
    args = ap.parse_args()
    if args.proprio_only and args.no_proprio:
        ap.error("--proprio-only and --no-proprio are mutually exclusive")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    F, P, lam, demo = load_features(args.features, seq=not args.proprio_last_only)
    delta = args.delta if args.delta is not None else auto_delta(lam)
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

    # Per-dimension stats pooled over the window, not per (timestep, dim): normalising
    # each timestep separately would rescale the very differences between timesteps
    # that carry the velocity signal. Shape is (9,) either way, so the saved stats stay
    # compatible with the snapshot head and the runtime bridge.
    ax = (0, 1) if P.ndim == 3 else 0
    prop_mu, prop_sd = P[tr_m].mean(ax), P[tr_m].std(ax) + 1e-6
    P = (P - prop_mu) / prop_sd

    prop_is_seq = (P.ndim == 3)                   # recorded now; P is freed at line ~254
    prop_window = int(P.shape[1]) if prop_is_seq else 1
    model = AttentiveHead(dim=F.shape[-1], prop_dim=P.shape[-1],
                          proprio_only=args.proprio_only,
                          no_proprio=args.no_proprio,
                          prop_seq=prop_is_seq).to(dev)
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

    def predict(Fe, Pe):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Fe), 512):
                tok = torch.from_numpy(Fe[i:i+512]).float().to(dev)
                pr = torch.from_numpy(Pe[i:i+512]).to(dev)
                lg, lh = model(tok, pr)
                outs.append(torch.stack([torch.sigmoid(lg), lh], 1).cpu().numpy())
        return np.concatenate(outs)

    def evaluate(Fe, Pe, lame, tag, own_delta=False):
        o = predict(Fe, Pe)
        ye = (lame > delta).astype(np.float32)
        conf = np.abs(lame) > delta
        res = {
            f"{tag}_auroc_all": auroc(o[:, 0], ye),
            f"{tag}_auroc_confident": auroc(o[:, 0][conf], ye[conf]),
            f"{tag}_lambda_mae": float(np.abs(o[:, 1] - lame).mean()),
            f"{tag}_lambda_spearman": spearman(o[:, 1], lame),
            f"{tag}_frac_unstable": float(ye.mean()),
            f"{tag}_n": int(len(ye)),
        }
        if own_delta:
            d2 = auto_delta(lame)
            y2 = (lame > d2).astype(np.float32)
            c2 = np.abs(lame) > d2
            res[f"{tag}_own_delta"] = d2
            res[f"{tag}_auroc_all_own_delta"] = auroc(o[:, 0], y2)
            res[f"{tag}_auroc_confident_own_delta"] = auroc(o[:, 0][c2], y2[c2])
        return res

    results = {"delta": delta, "features": args.features,
               "holdout_features": args.holdout_features,
               "proprio_only": args.proprio_only, "no_proprio": args.no_proprio,
               "proprio_seq": prop_is_seq, "proprio_window": prop_window,
               "seed": args.seed, "tag": args.tag, "epochs": args.epochs,
               "n_train": int(tr_m.sum()), "n_val": int(val_m.sum())}
    results.update(evaluate(F[val_m], P[val_m], lam[val_m], "val"))
    del F, P
    for hp in args.holdout_features:
        Fh, Ph, lamh, _ = load_features([hp], seq=not args.proprio_last_only)
        Ph = (Ph - prop_mu) / prop_sd
        results.update(evaluate(Fh, Ph, lamh,
                                "xfer_" + os.path.basename(hp).split(".")[0],
                                own_delta=True))
        del Fh, Ph

    torch.save({"model": model.state_dict(), "prop_mu": prop_mu, "prop_sd": prop_sd,
                "delta": delta, "args": vars(args),
                # the runtime bridge needs this to rebuild the right proprio encoder
                "prop_seq": prop_is_seq, "prop_win": prop_window},
               os.path.join(args.out, "head.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
