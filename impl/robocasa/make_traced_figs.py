"""Regime-timeline figures from the traced switching cells.

Per task:
  <task>_timelines.pdf: lambda_hat traces at each replan, successes on top,
    failures below, delta line, firing marks, episode-end markers.
  Combined firing_by_progress.pdf: fraction of unstable calls per episode
    progress bin, success vs failure, one panel per task.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACED = "/home/azureuser/cloudfiles/code/Users/garyan18/long-horizon/results/robocasa/switch_traced"
OUT = os.path.join(TRACED, "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

for path in sorted(glob.glob(os.path.join(TRACED, "*_traced_*.json"))):
    d = json.load(open(path))
    task = d["env"]
    delta = d["delta"]
    name = os.path.basename(path).replace(".json", "")
    succ = [e for e in d["episodes"] if e["success"]]
    fail = [e for e in d["episodes"] if not e["success"]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, eps, lab, col in ((axes[0], succ, "success", GREEN),
                              (axes[1], fail, "failure", ORANGE)):
        for e in eps:
            ts = [r["t"] for r in e["replans"]]
            ls = [r["lam"] for r in e["replans"]]
            ax.plot(ts, ls, color=col, alpha=0.35, lw=0.8)
            fire_t = [r["t"] for r in e["replans"] if r["lam"] > delta]
            fire_l = [r["lam"] for r in e["replans"] if r["lam"] > delta]
            ax.scatter(fire_t, fire_l, color=BLUE, s=14, zorder=3)
            ax.axvline(e["steps"], color=col, alpha=0.12, lw=0.6)
        ax.axhline(delta, color="k", ls="--", lw=1,
                   label=rf"$\delta$ = {delta:.3f}")
        ax.set_ylabel(r"$\hat\lambda$")
        ax.set_title(f"{lab} episodes (n={len(eps)})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("timestep")
    fig.suptitle(f"{name}: predicted stability at each replan "
                 "(blue dots = head fired)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}_timelines.pdf"))
    plt.close(fig)
    print("wrote", f"{name}_timelines.pdf")

# aggregate: firing rate by episode progress, success vs failure
paths = sorted(glob.glob(os.path.join(TRACED, "*_traced_*.json")))
fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 3.2),
                         sharey=True)
BINS = np.linspace(0, 1, 11)
for ax, path in zip(np.atleast_1d(axes), paths):
    d = json.load(open(path))
    delta = d["delta"]
    name = os.path.basename(path).replace(".json", "").replace("_traced", "")
    for flag, lab, col in ((True, "success", GREEN),
                           (False, "failure", ORANGE)):
        eps = [e for e in d["episodes"] if e["success"] == flag]
        if not eps:
            continue
        prog, fired = [], []
        for e in eps:
            horizon = max(e["steps"], 1)
            for r in e["replans"]:
                prog.append(min(r["t"] / horizon, 1.0))
                fired.append(1.0 if r["lam"] > delta else 0.0)
        prog, fired = np.array(prog), np.array(fired)
        idx = np.digitize(prog, BINS) - 1
        rate = [fired[idx == b].mean() if (idx == b).any() else np.nan
                for b in range(len(BINS) - 1)]
        ax.plot(0.5 * (BINS[:-1] + BINS[1:]), rate, "-o", ms=3,
                color=col, label=lab)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("episode progress")
    ax.legend(fontsize=8)
np.atleast_1d(axes)[0].set_ylabel("fraction of replans fired")
fig.suptitle("Where the head fires: success vs failure episodes", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "firing_by_progress.pdf"))
print("wrote firing_by_progress.pdf")
