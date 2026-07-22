"""Rewrite cached feature files with targets from the new (true sigma_u) labels.

Features are indexed by (demo_id, t) and label-independent; only the embedded
lambda/contact target fields need replacing. Output: feat2_<task>.npz.
"""

import json

import numpy as np

for t in ["lift", "can", "square", "tool_hang"]:
    fz = dict(np.load(f"/mnt/scratch/lh/features/feat_{t}.npz", allow_pickle=True))
    lz = np.load(f"/mnt/scratch/lh/labels/final2_ol_{t}.npz", allow_pickle=True)
    lut = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(lz["demo_id"], lz["t"]))}
    idx = np.array([lut[(int(d), int(s))] for d, s in zip(fz["demo_id"], fz["t"])])
    for k in ["lambda_task", "win_contact_gripper", "win_contact_table",
              "win_contact_fixture"]:
        fz[k] = lz[k][idx]
    fz["meta"] = json.dumps({"rejoined_labels": f"final2_ol_{t}.npz",
                             "orig_meta": str(fz.get("meta"))})
    np.savez(f"/mnt/scratch/lh/features/feat2_{t}.npz", **fz)
    print(f"rejoined {t}: {len(idx)} stamps", flush=True)
print("REJOIN_DONE")
