import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = "/mnt/scratch/lh/labels/final2_ol_rc_{}.npz"
HEADS = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/predictor/rc_indomain"
TASKS = ["OpenCabinet", "OpenDrawer", "PickPlaceCounterToCabinet", "TurnOnSinkFaucet"]
NICE = {"OpenCabinet": "OpenCabinet", "OpenDrawer": "OpenDrawer",
        "PickPlaceCounterToCabinet": "PickPlaceCounterToCab", "TurnOnSinkFaucet": "TurnOnSinkFaucet"}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4 (light)
GRAY = "#8a8a8a"; INK = "#333333"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "figure.dpi": 150})

data, stats = {}, {}
for t in TASKS:
    z = np.load(LABELS.format(t), allow_pickle=True)
    lam, dem, ts = z["lambda_task"], z["demo_id"], z["t"]
    cg = z["win_contact_gripper"]
    delta = json.load(open(f"{HEADS}/R_{t}_full_s0/metrics.json"))["delta"]
    # normalized progress per stamp within its demo
    prog = np.zeros(len(ts), float)
    for d in np.unique(dem):
        m = dem == d
        tmax = ts[m].max()
        prog[m] = ts[m] / max(tmax, 1)
    data[t] = dict(lam=lam, dem=dem, ts=ts, prog=prog, cg=cg, delta=delta)
    stats[t] = dict(n_stamps=int(len(lam)), n_demos=int(len(np.unique(dem))),
        mean=float(lam.mean()), std=float(lam.std()), min=float(lam.min()), max=float(lam.max()),
        p5=float(np.percentile(lam, 5)), p50=float(np.percentile(lam, 50)),
        p95=float(np.percentile(lam, 95)), frac_pos=float((lam > 0).mean()),
        frac_unstable=float((lam > delta).mean()), delta=float(delta),
        frac_gripper_contact=float((cg > 0.5).mean()),
        mean_ep_len=float(np.mean([ts[dem == d].max() for d in np.unique(dem)])))

json.dump(stats, open("figs/stats.json", "w"), indent=1)

# Fig 1: lambda distributions, 2x2 small multiples, single hue
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
for ax, t in zip(axes.flat, TASKS):
    d = data[t]
    ax.hist(d["lam"], bins=80, range=(-0.15, 0.25), color=C[0], alpha=0.85, edgecolor="none")
    ax.axvline(0, color=GRAY, lw=1.0, ls="--")
    ax.axvline(d["delta"], color=C[1], lw=1.2)
    ax.set_title(NICE[t], fontsize=9.5, color=INK)
    ax.text(0.98, 0.86, f"$\\lambda>0$: {100*stats[t]['frac_pos']:.0f}%\n$\\lambda>\\delta$: {100*stats[t]['frac_unstable']:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
for ax in axes[1]: ax.set_xlabel("$\\lambda$ (FTLE slope, per step)")
for ax in axes[:, 0]: ax.set_ylabel("stamps")
fig.tight_layout(); fig.savefig("figs/fig_lambda_dist.pdf"); fig.savefig("figs/fig_lambda_dist.png", dpi=110); plt.close(fig)

# Fig 2: temporal profile, median + IQR vs progress, 2x2
BINS = np.linspace(0, 1, 21); MID = 0.5 * (BINS[1:] + BINS[:-1])
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
for ax, t in zip(axes.flat, TASKS):
    d = data[t]; q1, q2, q3 = [], [], []
    for i in range(20):
        m = (d["prog"] >= BINS[i]) & (d["prog"] < BINS[i+1])
        v = d["lam"][m]
        q1.append(np.percentile(v, 25)); q2.append(np.percentile(v, 50)); q3.append(np.percentile(v, 75))
    ax.fill_between(MID, q1, q3, color=C[0], alpha=0.22, lw=0)
    ax.plot(MID, q2, color=C[0], lw=2)
    ax.axhline(0, color=GRAY, lw=1.0, ls="--")
    ax.axhline(d["delta"], color=C[1], lw=1.0)
    ax.set_title(NICE[t], fontsize=9.5, color=INK)
for ax in axes[1]: ax.set_xlabel("episode progress")
for ax in axes[:, 0]: ax.set_ylabel("$\\lambda$ (median, IQR)")
fig.tight_layout(); fig.savefig("figs/fig_temporal_profile.pdf"); fig.savefig("figs/fig_temporal_profile.png", dpi=110); plt.close(fig)

# Fig 3: fraction unstable vs progress, 4 lines, direct labels
fig, ax = plt.subplots(figsize=(7.0, 3.4))
for i, t in enumerate(TASKS):
    d = data[t]; f = []
    for j in range(20):
        m = (d["prog"] >= BINS[j]) & (d["prog"] < BINS[j+1])
        f.append((d["lam"][m] > d["delta"]).mean())
    ax.plot(MID, f, color=C[i], lw=2, label=NICE[t])
ax.set_xlabel("episode progress"); ax.set_ylabel("fraction $\\lambda>\\delta$")
ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig("figs/fig_frac_unstable.pdf"); fig.savefig("figs/fig_frac_unstable.png", dpi=110); plt.close(fig)

# Fig 4: one representative episode trace per task with gripper-contact shading
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
for ax, t in zip(axes.flat, TASKS):
    d = data[t]
    # pick the demo whose frac-unstable is closest to the task median profile
    fracs = {int(dd): (d["lam"][d["dem"] == dd] > d["delta"]).mean() for dd in np.unique(d["dem"])}
    target = stats[t]["frac_unstable"]
    pick = min(fracs, key=lambda k: abs(fracs[k] - target))
    m = d["dem"] == pick
    ts, lam, cg = d["ts"][m], d["lam"][m], d["cg"][m]
    o = np.argsort(ts); ts, lam, cg = ts[o], lam[o], cg[o]
    incontact = cg > 0.5
    lo = lam.min() - 0.02; hi = lam.max() + 0.02
    ax.fill_between(ts, lo, hi, where=incontact, color=C[2], alpha=0.15, lw=0)
    ax.plot(ts, lam, color=C[0], lw=1.4)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color=GRAY, lw=1.0, ls="--")
    ax.axhline(d["delta"], color=C[1], lw=1.0)
    ax.set_title(f"{NICE[t]} (demo {pick})", fontsize=9.5, color=INK)
    ax.set_xlabel("timestep"); ax.set_ylabel("$\\lambda$")
fig.tight_layout(); fig.savefig("figs/fig_example_traces.pdf"); fig.savefig("figs/fig_example_traces.png", dpi=110); plt.close(fig)
print(json.dumps(stats, indent=1))
