"""Pick the six demonstration stamps per task shown in the label-video grid.

Source: the K=24 demonstration-stamp labels behind the paper's open-loop
tables (results/labels_bothK_20260902/bothK_ol_<ds>.npz, k24_* columns, copied
verbatim from the labeling runs), primary gate setting m=2.0, t=2.365.

  buttons 1, 2  UNSTABLE  highest lambda_bar among stamps clearing both gates
  buttons 3, 4  DEADBAND  lambda_bar nearest zero
  buttons 5, 6  STABLE    most negative lambda_bar among stamps clearing both gates

The two picks of a class come from different demonstrations (the second pick
is the best stamp from any other demo); consecutive stamps of one demo are two
steps apart and render as near-identical clips. --allow-same-demo restores the
plain top-2. A task with fewer than two stable stamps gets what exists and the
shortfall is recorded for the button tooltip.

usage: python select_stamps.py [--allow-same-demo] > stamps.json
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOTHK = f"{ROOT}/results/labels_bothK_20260902"
LH = "/mnt/scratch/lh"
COL = "k24_label_m2.0_t2.365"

# (platform, task as shown, label dataset key, raw label file, sim dataset)
TASKS = [
    ("robomimic", "lift", "lift"),
    ("robomimic", "can", "can"),
    ("robomimic", "square", "square"),
    ("robomimic", "tool_hang", "tool_hang"),
    ("mimicgen", "coffee", "mg_coffee_preparation_d0"),
    ("mimicgen", "nut_assembly", "mg_nut_assembly_d0"),
    ("robocasa", "OpenCabinet", "rc_OpenCabinet"),
    ("robocasa", "OpenDrawer", "rc_OpenDrawer"),
    ("robocasa", "PickPlaceCounterToCabinet", "rc_PickPlaceCounterToCabinet"),
    ("robocasa", "TurnOnSinkFaucet", "rc_TurnOnSinkFaucet"),
]


def raw_source(platform, key):
    """The labeling run's own output file (per-branch slopes, seed, sigma)."""
    if platform == "robocasa":
        t = key[3:]
        return (f"{ROOT}/results/rc_relabel_20260831/labels/final4_rmse_rc_{t}.npz",
                f"{LH}/data/robocasa/hdf5/rc_{t}.hdf5")
    lab = f"{ROOT}/artifacts/labels/regen2/final2b_ol_{key}.npz"
    if platform == "mimicgen":
        return lab, f"{LH}/data/mimicgen/{key[3:]}.hdf5"
    return lab, f"{LH}/data/robomimic/{key}/ph/low_dim_v15.hdf5"


def pick(idx, order_key, demo, n, distinct):
    out = []
    for i in idx[np.argsort(order_key, kind="stable")]:
        if distinct and any(demo[i] == demo[j] for j in out):
            continue
        out.append(i)
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-same-demo", action="store_true")
    args = ap.parse_args()
    distinct = not args.allow_same_demo
    sel = []
    for platform, task, key in TASKS:
        z = np.load(f"{BOTHK}/bothK_ol_{key}.npz", allow_pickle=True)
        lam, se, lab = z["k24_lambda_mean"], z["k24_lambda_se"], z[COL]
        demo, t = z["demo_id"], z["t"]
        raw_path, sim_path = raw_source(platform, key)
        raw = np.load(raw_path, allow_pickle=True)
        meta = json.loads(str(raw["meta"]))
        # the label file behind the tables must be the labeling run's output
        rl = raw["lambda_mean"]
        rk = dict(zip(zip(raw["demo_id"].tolist(), raw["t"].tolist()), range(len(rl))))
        u, d, s = (np.where(lab == c)[0] for c in (1, 0, -1))
        groups = [("unstable", pick(u, -lam[u], demo, 2, distinct)),
                  ("deadband", pick(d, np.abs(lam[d]), demo, 2, distinct)),
                  ("stable", pick(s, lam[s], demo, 2, distinct))]
        clips, button = [], 0
        for cls, ids in groups:
            for k in range(2):
                button += 1
                if k >= len(ids):
                    clips.append({"button": button, "class": cls, "available": False,
                                  "n_class": int({"unstable": len(u), "deadband": len(d),
                                                  "stable": len(s)}[cls])})
                    continue
                i = ids[k]
                j = rk[(int(demo[i]), int(t[i]))]
                assert abs(rl[j] - lam[i]) < 5e-6, (task, rl[j], lam[i])
                clips.append({
                    "button": button, "class": cls, "idx": k + 1, "available": True,
                    "demo_id": int(demo[i]), "t": int(t[i]),
                    "lambda_bar": float(lam[i]), "lambda_se": float(se[i]),
                    "label": int(lab[i]),
                    "file": f"{platform}_{task}_{cls}_{k + 1}.mp4"})
        sel.append({
            "platform": platform, "task": task, "label_key": key,
            "n_stamps": int(len(lam)),
            "n_by_class": {"unstable": int(len(u)), "deadband": int(len(d)),
                           "stable": int(len(s))},
            "labels_table_file": f"{BOTHK}/bothK_ol_{key}.npz",
            "labels_run_file": raw_path, "sim_dataset": sim_path,
            "K": int(meta["K"]), "N": int(meta["N"]), "stride": int(meta["stride"]),
            "sigma_u": float(meta["sigma_u"]), "seed": meta.get("seed"),
            "rng": meta.get("rng", "default_rng(seed + demo index), sequential per stamp"),
            "gate": {"m": 2.0, "t": 2.365, "column": COL},
            "distinct_demos": distinct, "clips": clips})
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
