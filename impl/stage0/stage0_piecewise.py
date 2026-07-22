"""Stage 0: piecewise-linear synthetic validation of segment-dependent chunk length.

System (2-D state represented as one complex number; rotations become complex scalars):
    x_{t+1} = A_ol(t) x_t + u_t + w_t
with three time segments:  [0,50) contracting, [50,100) expansive, [100,150) contracting.
    A_ol = s * e^{i*theta},  s = RHO_C (<1) in contracting, LAM_E (>1) in expansive.

Expert: closed-loop A_cl = RHO_CL * e^{i*theta} everywhere via u* = -K_s x, K_s = A_ol - A_cl.
Demo tube: x* = 0.

Learned chunking policy (deterministic smooth imitator, worst-case error within the
J_DEMO ball, per the lower-bound construction regime of Simchowitz et al. 2503.09722):
at a replan from state x0 it emits the action sequence
    u_i = -(K_s * A_cl^i + M_i) x0 ,   i = 0..k-1
where M_i is a fixed error map with |M_i| = EPS_s (relative regression error, small on the
demo tube).  Worst-case alignment: M_i = -EPS_s * e^{i*theta} (+ small per-trial jitter).
EPS is per-segment: large in contracting (plain BC), small in expansive (emulating
noise-injected data per Zhang et al. Thm 2 — without it no policy class succeeds there).

Predictions under test:
  P1  contracting segments: success degrades as k -> 1 (floor k*); analytic k* ~ 3-4 here.
  P2  expansive segment: success degrades for large k (ceiling ~ 1/log(LAM_E) ~ 8.8).
  P3  no constant k matches the per-segment profile (heatmap corner beats diagonal).
  P4  the FTLE labeler recovers the true regimes: lambda_ol ~ log s per segment; the
      closed-loop FTLE flips sign vs open-loop exactly as the hypothesis requires.

Run:  python3 stage0_piecewise.py --out <results_dir>
"""

import argparse
import json
import os

import numpy as np

# ----------------------------- system constants -----------------------------
THETA = 0.35                      # rotation angle (genericity)
RHO_C = 0.50                      # open-loop contraction factor (contracting segments)
LAM_E = 1.12                      # open-loop expansion factor (expansive segment)
RHO_CL = 0.75                     # expert closed-loop factor (everywhere)
EPS_C = 0.35                      # relative regression error, contracting segments
EPS_E = 0.10                      # relative regression error, expansive segment
SEGS = [(0, 50, "C1"), (50, 100, "E"), (100, 150, "C2")]
T = 150
TOL = 0.30                        # failure: |x_t| > TOL at any t   (demo tube ~0.03)
SIGMA_A = 0.005                   # absolute per-action noise (graded failures)
SIGMA_ETA = 0.01                  # observation noise at replan
SIGMA_X0 = 0.03                   # initial state scale
JIT_SCALE = 0.05                  # per-trial jitter on |M_i|
JIT_PHASE = 0.10                  # per-trial jitter on arg(M_i)
K_SET = [1, 2, 3, 4, 6, 8, 12, 16]
ROT = np.exp(1j * THETA)
A_CL = RHO_CL * ROT

SEG_PARAMS = {
    "C1": (RHO_C * ROT, EPS_C),
    "E": (LAM_E * ROT, EPS_E),
    "C2": (RHO_C * ROT, EPS_C),
}


def seg_of(t):
    for a, b, name in SEGS:
        if a <= t < b:
            return name
    raise ValueError(t)


def crandn(rng, *shape):
    """Standard complex normal (unit expected squared magnitude)."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


# ------------------------------ rollout engine ------------------------------
def run_trials(k_by_seg, n_trials, seed):
    """Vectorised rollout of n_trials episodes with per-segment executed chunk length.

    Replans occur every k_seg steps and at every segment boundary (identical schedule
    across trials).  Returns success mask and per-trial max |x|.
    """
    rng = np.random.default_rng(seed)
    x = SIGMA_X0 * crandn(rng, n_trials)
    max_abs = np.abs(x).copy()

    # fixed per-trial error maps M_i for each segment type (the "trained policy"):
    # worst-case aligned base, small per-trial draw jitter.  One map per chunk index,
    # shared across all k (execute-prefix semantics, as with a real chunking policy).
    kmax = max(K_SET)
    M = {}
    for s in ("C1", "E", "C2"):
        eps = SEG_PARAMS[s][1]
        scale = eps * (1.0 + JIT_SCALE * rng.standard_normal((n_trials, kmax)))
        phase = THETA + JIT_PHASE * rng.standard_normal((n_trials, kmax))
        M[s] = -scale * np.exp(1j * phase)

    x0_cur = x.copy()          # state snapshot at the last replan (with obs noise)
    i_in_chunk = 0
    seg_cur = "C1"
    for t in range(T):
        s = seg_of(t)
        a_ol, _ = SEG_PARAMS[s]
        k_exec = k_by_seg[s]
        if s != seg_cur or i_in_chunk >= k_exec:
            i_in_chunk = 0
            seg_cur = s
        if i_in_chunk == 0:
            x0_cur = x + SIGMA_ETA * crandn(rng, n_trials)
        K_s = a_ol - A_CL
        u = -(K_s * A_CL**i_in_chunk + M[s][:, i_in_chunk]) * x0_cur \
            + SIGMA_A * crandn(rng, n_trials)
        x = a_ol * x + u
        np.maximum(max_abs, np.abs(x), out=max_abs)
        i_in_chunk += 1

    return max_abs < TOL, max_abs


# ------------------------------- FTLE labeler -------------------------------
def ftle_labels(n_probe=8, k_win=12, sigma_u=0.02, stride=2, seed=0):
    """Exact synthetic analogue of the real labeler (stability_label_generation.md).

    Open-loop: perturb the first action, replay the SAME remaining demo actions, OLS
    slope of mean log divergence.  Closed-loop: the learned k=1 policy reacts in the
    perturbed branch.  Returns per-stamp t, lambda_ol, lambda_cl.
    """
    rng = np.random.default_rng(seed)

    # one demo: expert closed loop from a random start, with process noise
    x_demo = np.empty(T + 1, dtype=complex)
    u_demo = np.empty(T, dtype=complex)
    x_demo[0] = SIGMA_X0 * crandn(rng)
    for t in range(T):
        a_ol, _ = SEG_PARAMS[seg_of(t)]
        u_demo[t] = -(a_ol - A_CL) * x_demo[t] + SIGMA_A * crandn(rng)
        x_demo[t + 1] = a_ol * x_demo[t] + u_demo[t]

    # learned k=1 policy for the closed-loop branch (jitter-free worst-case M_0)
    def learned_u(xv, s):
        a_ol, eps = SEG_PARAMS[s]
        m0 = -eps * ROT
        return -((a_ol - A_CL) + m0) * xv

    stamps, lam_ol, lam_cl = [], [], []
    for t0 in range(0, T - k_win, stride):
        div_ol = np.zeros((n_probe, k_win))
        div_cl = np.zeros((n_probe, k_win))
        for n in range(n_probe):
            du = sigma_u * crandn(rng)
            xa = x_demo[t0]          # nominal branch state
            xo = x_demo[t0]          # open-loop perturbed branch
            xc = x_demo[t0]          # closed-loop perturbed branch
            for j in range(k_win):
                t = t0 + j
                s = seg_of(t)
                a_ol, _ = SEG_PARAMS[s]
                pert = du if j == 0 else 0.0
                xa = a_ol * xa + u_demo[t]
                xo = a_ol * xo + u_demo[t] + pert
                xc = a_ol * xc + learned_u(xc, s) + pert
                div_ol[n, j] = abs(xo - xa)
                div_cl[n, j] = abs(xc - xa)
        taus = np.arange(1, k_win + 1)
        for divs, out in ((div_ol, lam_ol), (div_cl, lam_cl)):
            y = np.log(np.maximum(divs, 1e-300)).mean(axis=0)
            out.append(np.polyfit(taus, y, 1)[0])
        stamps.append(t0)
    return np.array(stamps), np.array(lam_ol), np.array(lam_cl)


# ---------------------------------- driver ----------------------------------
def main():
    global EPS_C, LAM_E, SEG_PARAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps-c", type=float, default=EPS_C,
                    help="regression error in contracting segments (raises floor k*)")
    ap.add_argument("--lam-e", type=float, default=LAM_E,
                    help="expansion factor in expansive segment (lowers ceiling)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    EPS_C, LAM_E = args.eps_c, args.lam_e
    SEG_PARAMS = {"C1": (RHO_C * ROT, EPS_C), "E": (LAM_E * ROT, EPS_E),
                  "C2": (RHO_C * ROT, EPS_C)}
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "results", "stage0")
    os.makedirs(out, exist_ok=True)
    res = {"params": {k: v for k, v in globals().items()
                      if k.isupper() and isinstance(v, (int, float, list))}}

    # --- 1D sweeps -----------------------------------------------------------
    sweep_c, sweep_e = {}, {}
    for k in K_SET:
        ok, _ = run_trials({"C1": k, "E": 1, "C2": k}, args.n_trials, args.seed)
        sweep_c[k] = float(ok.mean())
        ok, _ = run_trials({"C1": 8, "E": k, "C2": 8}, args.n_trials, args.seed + 1)
        sweep_e[k] = float(ok.mean())
    res["sweep_contracting_k(kE=1)"] = sweep_c
    res["sweep_expansive_k(kC=8)"] = sweep_e

    # --- 2D heatmap (k_c applied to C1 and C2, k_e to E) ----------------------
    heat = np.zeros((len(K_SET), len(K_SET)))
    for i, kc in enumerate(K_SET):
        for j, ke in enumerate(K_SET):
            ok, _ = run_trials({"C1": kc, "E": ke, "C2": kc},
                               args.n_trials, args.seed + 10 + 7 * i + j)
            heat[i, j] = ok.mean()
    res["heatmap_success"] = heat.tolist()
    diag = {k: float(heat[i, i]) for i, k in enumerate(K_SET)}
    res["constant_k_diagonal"] = diag
    res["best_constant_k"] = max(diag, key=diag.get)
    res["best_constant_k_success"] = diag[res["best_constant_k"]]
    bi, bj = np.unravel_index(np.argmax(heat), heat.shape)
    res["best_adaptive"] = {"k_contracting": K_SET[bi], "k_expansive": K_SET[bj],
                            "success": float(heat[bi, bj])}

    # --- FTLE labeler recovery ------------------------------------------------
    stamps, lam_ol, lam_cl = ftle_labels(seed=args.seed)
    seg_names = np.array([seg_of(t) for t in stamps])
    interior = np.array([  # stamps whose whole window stays inside one segment
        seg_of(t) == seg_of(min(t + 12, T - 1)) for t in stamps])
    ftle = {}
    for s, (a_ol, eps) in SEG_PARAMS.items():
        m = (seg_names == s) & interior
        ftle[s] = {
            "lambda_ol_measured": float(lam_ol[m].mean()),
            "lambda_ol_true": float(np.log(abs(a_ol))),
            "lambda_cl_measured": float(lam_cl[m].mean()),
            "lambda_cl_true(k=1 learned loop)": float(np.log(abs(A_CL + eps * ROT))),
        }
    res["ftle_recovery"] = ftle
    res["analytic_notes"] = {
        "k_star_contracting_pred": "3-4 (|G_k|<1 first at k=3..4)",
        "ceiling_expansive_pred": f"~1/log(LAM_E) = {1.0 / np.log(LAM_E):.1f}",
    }

    with open(os.path.join(out, "results" + args.tag + ".json"), "w") as f:
        json.dump(res, f, indent=2)

    # --- plots ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    ax.plot(K_SET, [sweep_c[k] for k in K_SET], "o-", label="contracting segs (k_E=1)")
    ax.plot(K_SET, [sweep_e[k] for k in K_SET], "s-", label="expansive seg (k_C=8)")
    ax.axvline(4, ls=":", c="gray"); ax.text(4.1, 0.05, "k* pred", fontsize=8)
    ax.axvline(1.0 / np.log(LAM_E), ls="--", c="gray")
    ax.text(1.0 / np.log(LAM_E) + .1, 0.05, "1/λ pred", fontsize=8)
    ax.set_xlabel("executed chunk length k in the swept segment")
    ax.set_ylabel("success rate"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
    ax.set_title("P1+P2: opposite k constraints per regime")

    ax = axes[1]
    im = ax.imshow(heat, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(K_SET)), K_SET); ax.set_yticks(range(len(K_SET)), K_SET)
    ax.set_xlabel("k expansive"); ax.set_ylabel("k contracting")
    for i in range(len(K_SET)):
        for j in range(len(K_SET)):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="w" if heat[i, j] < 0.6 else "k")
    ax.plot(range(len(K_SET)), range(len(K_SET)), "r--", lw=1, label="constant k")
    ax.plot(bj, bi, "r*", ms=14, label="best adaptive")
    ax.legend(fontsize=7, loc="upper left"); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("P3: adaptive corner beats constant-k diagonal")

    ax = axes[2]
    ax.plot(stamps, lam_ol, ".-", label="λ̂ open-loop (labeler)")
    ax.plot(stamps, lam_cl, ".-", label="λ̂ closed-loop (learned k=1)")
    for a, b, name in SEGS:
        a_ol, _ = SEG_PARAMS[name]
        ax.hlines(np.log(abs(a_ol)), a, b, colors="k", ls=":", lw=1)
    ax.axhline(0, c="gray", lw=0.5)
    for a, b, name in SEGS:
        ax.axvspan(a, b, alpha=0.08,
                   color=("tab:blue" if name != "E" else "tab:red"))
    ax.set_xlabel("t"); ax.set_ylabel("λ̂ (per step)"); ax.legend(fontsize=8)
    ax.set_title("P4: FTLE recovers regimes; CL flips sign vs OL")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stage0_money_plots" + args.tag + ".png"), dpi=160)

    print(json.dumps({k: res[k] for k in
                      ("sweep_contracting_k(kE=1)", "sweep_expansive_k(kC=8)",
                       "constant_k_diagonal", "best_constant_k",
                       "best_constant_k_success", "best_adaptive",
                       "ftle_recovery")}, indent=2))


if __name__ == "__main__":
    main()
