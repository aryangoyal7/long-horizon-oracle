"""Sanity analysis of FTLE labels (Section 6 of stability_label_generation.md).

Produces, per task:
  1. lambda histogram split by window-contact category + suggested deadband delta
     (delta = 95th percentile of |lambda| over free-space stamps = the noise floor).
  2. Contact alignment stats: mean lambda per category, AUROC of lambda ranking
     contact-in-window, label fractions after deadband + median filter.
  3. Per-demo lambda(t) profiles with contact shading (first 6 demos).

Usage: python analyze_labels.py --labels labels_ol_square.npz --out results/labels/square
"""

import argparse
import json
import os

import numpy as np


def auroc(scores, labels):
    """Rank-based AUROC without sklearn."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def median_filter_per_demo(demo_id, t, y, k=3):
    out = y.copy()
    for d in np.unique(demo_id):
        m = demo_id == d
        idx = np.argsort(t[m])
        yy = y[m][idx]
        f = yy.copy()
        for i in range(len(yy)):
            lo, hi = max(0, i - k // 2), min(len(yy), i + k // 2 + 1)
            f[i] = np.median(yy[lo:hi])
        tmp = out[m]; tmp[idx] = f; out[m] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.labels, allow_pickle=True)
    lam = z["lambda_task"]; demo_id = z["demo_id"]; t = z["t"]
    wg, wt_, wf = z["win_contact_gripper"], z["win_contact_table"], z["win_contact_fixture"]
    meta = json.loads(str(z["meta"]))

    contact_any = wg | wf                       # gripper-object or object-fixture
    free = ~contact_any
    delta = float(np.percentile(np.abs(lam[free]), 95)) if free.sum() else 0.05

    lam_f = median_filter_per_demo(demo_id, t, lam)
    unstable = lam_f > delta
    stable = lam_f < -delta

    stats = {
        "meta": meta,
        "n_stamps": int(len(lam)),
        "frac_free_space": float(free.mean()),
        "mean_lambda_free": float(lam[free].mean()) if free.sum() else None,
        "mean_lambda_gripper_contact": float(lam[wg].mean()) if wg.sum() else None,
        "mean_lambda_fixture_contact": float(lam[wf].mean()) if wf.sum() else None,
        "suggested_deadband_delta_p95_free": delta,
        "auroc_lambda_predicts_window_contact": float(auroc(lam, contact_any)),
        "label_fracs_after_filter": {
            "unstable": float(unstable.mean()),
            "stable": float(stable.mean()),
            "deadband": float((~unstable & ~stable).mean()),
        },
        "frac_contact_labeled_unstable": float(unstable[contact_any].mean())
        if contact_any.sum() else None,
        "frac_free_labeled_unstable": float(unstable[free].mean()) if free.sum() else None,
    }
    with open(os.path.join(args.out, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(np.percentile(lam, 0.5), np.percentile(lam, 99.5), 60)
    axes[0].hist(lam[free], bins=bins, alpha=0.6, density=True, label="free space")
    if contact_any.sum():
        axes[0].hist(lam[contact_any], bins=bins, alpha=0.6, density=True,
                     label="contact in window")
    axes[0].axvline(delta, ls="--", c="k", lw=1, label=f"±δ={delta:.3f}")
    axes[0].axvline(-delta, ls="--", c="k", lw=1)
    axes[0].set_xlabel("λ_ol (per step)"); axes[0].legend(fontsize=8)
    axes[0].set_title(os.path.basename(args.labels))

    shown = np.unique(demo_id)[:6]
    for d in shown:
        m = demo_id == d
        idx = np.argsort(t[m])
        axes[1].plot(t[m][idx], lam_f[m][idx], lw=1)
        cm = m & contact_any
        axes[1].plot(t[cm], lam_f[cm], "r.", ms=3)
    axes[1].axhline(delta, ls="--", c="k", lw=0.7); axes[1].axhline(-delta, ls="--", c="k", lw=0.7)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("λ (median-filtered)")
    axes[1].set_title("per-demo profiles (red = contact in window)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "label_analysis.png"), dpi=150)
    print("saved", os.path.join(args.out, "label_analysis.png"))


if __name__ == "__main__":
    main()
