"""Divergence-curve graphs: perturbed vs non-perturbed error over the probe window,
at the most contractive and most expansive labeled stamps of a task.

For each selected stamp this recomputes the perturb-and-replay probe (N branches,
noise on the first action only) and plots ||x_pert(tau) - x_nom(tau)|| on a log scale
against tau: thin gray = individual branches, bold = mean, dashed = the OLS
exponential fit whose slope is lambda. Contractive segments show the error funneling
back toward zero; expansive segments show it growing.

Output: <out-dir>/{stable,unstable}_<rank>_demo<did>_t<t0>.png + selection.json.

Usage:
  MUJOCO_GL=egl python plot_divergence_curves.py --dataset <low_dim.hdf5> \
      --labels final2_ol_square.npz --out-dir .../divergence_curves \
      --sigma-u 0.22 --delta 0.052 [--n-per-class 3] [--k-horizon 24]
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ftle_labeler import make_env, task_vec, fit_slope  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def probe_curves(env, states, actions, t0, K, sigma_u, n_probe, rng, pos_only=True):
    n_pert = 3 if pos_only else 6
    env.reset_to({"states": states[t0]})
    nom = []
    for j in range(K):
        obs, _, _, _ = env.step(actions[t0 + j])
        nom.append(task_vec(obs))
    nom = np.asarray(nom)
    D = np.empty((n_probe, K))
    for n in range(n_probe):
        a0 = actions[t0].copy()
        a0[:n_pert] += sigma_u * rng.standard_normal(n_pert)
        env.reset_to({"states": states[t0]})
        for j in range(K):
            obs, _, _, _ = env.step(a0 if j == 0 else actions[t0 + j])
            D[n, j] = np.linalg.norm(task_vec(obs) - nom[j])
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sigma-u", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--k-horizon", type=int, default=24)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import robomimic.utils.file_utils as FileUtils
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = make_env(env_meta)
    rng = np.random.default_rng(args.seed)

    z = np.load(args.labels, allow_pickle=True)
    did, ts, lam = z["demo_id"], z["t"], z["lambda_task"]

    # extremes, one stamp per demo for variety
    order = np.argsort(lam)
    def pick(idx_iter, n):
        out, seen = [], set()
        for i in idx_iter:
            if did[i] in seen:
                continue
            out.append(int(i)); seen.add(did[i])
            if len(out) == n:
                break
        return out
    stable_idx = pick(order, args.n_per_class)                    # most negative
    unstable_idx = pick(order[::-1], args.n_per_class)            # most positive

    os.makedirs(args.out_dir, exist_ok=True)
    sel_log = []
    with h5py.File(args.dataset, "r") as f:
        for cls, idxs in [("stable", stable_idx), ("unstable", unstable_idx)]:
            for rank, i in enumerate(idxs):
                d_i, t0, lam_lab = int(did[i]), int(ts[i]), float(lam[i])
                g = f[f"data/demo_{d_i}"]
                states, actions = g["states"][()], g["actions"][()]
                K = min(args.k_horizon, actions.shape[0] - t0)
                env.reset()
                env.reset_to({"model": g.attrs["model_file"]})
                D = probe_curves(env, states, actions, t0, K,
                                 args.sigma_u, args.n_probe, rng)
                log_d = np.log(np.maximum(D, 1e-12))
                lam_fit = fit_slope(log_d)
                taus = np.arange(1, K + 1)
                y = log_d.mean(axis=0)
                b = y.mean() - lam_fit * taus.mean()

                fig, ax = plt.subplots(figsize=(7, 4.2))
                for n in range(D.shape[0]):
                    ax.semilogy(taus, D[n], color="gray", alpha=0.45, lw=1)
                ax.semilogy(taus, np.exp(y), color="#1f77b4", lw=2.5,
                            label="mean over branches")
                ax.semilogy(taus, np.exp(b + lam_fit * taus), "--",
                            color="#ff7f0e", lw=2,
                            label=f"exp fit: lambda = {lam_fit:+.3f}")
                ax.axhline(args.sigma_u, color="k", lw=0.8, ls=":",
                           label=f"injected size sigma_u = {args.sigma_u:.3f}")
                ax.set_xlabel("steps after perturbation (tau)")
                ax.set_ylabel("|| perturbed - nominal ||  (task space)")
                verdict = ("EXPANSIVE / unstable" if lam_fit > args.delta else
                           "CONTRACTIVE / stable" if lam_fit < -args.delta
                           else "deadband")
                ax.set_title(f"demo {d_i}  t0={t0}  {verdict}   "
                             f"(label lambda={lam_lab:+.3f})")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3, which="both")
                fig.tight_layout()
                out = os.path.join(args.out_dir,
                                   f"{cls}_{rank}_demo{d_i}_t{t0}.png")
                fig.savefig(out, dpi=130)
                plt.close(fig)
                sel_log.append({"class": cls, "demo": d_i, "t0": t0,
                                "lambda_label": lam_lab,
                                "lambda_recomputed": lam_fit, "png": out})
                print(f"saved {out} (lam label {lam_lab:+.3f} "
                      f"recomputed {lam_fit:+.3f})", flush=True)

    with open(os.path.join(args.out_dir, "selection.json"), "w") as fo:
        json.dump({"sigma_u": args.sigma_u, "delta": args.delta,
                   "stamps": sel_log}, fo, indent=2)
    print("CURVES_DONE")


if __name__ == "__main__":
    main()
