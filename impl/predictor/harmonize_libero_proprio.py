"""Convert LIBERO feature files' 8-dim proprio (pos3, axis-angle ori3, grip2) to
the robomimic 9-dim layout (pos3, quat_xyzw4, grip2) so the files can join the
pooled head training. Output: featq_libero10_*.npz."""

import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation

for p in sorted(glob.glob("/mnt/scratch/lh/features/feat_libero10_*.npz")):
    z = dict(np.load(p, allow_pickle=True))
    pr = z["proprio"]
    assert pr.shape[1] == 8, (p, pr.shape)
    quat = Rotation.from_rotvec(pr[:, 3:6]).as_quat()      # xyzw, matches robosuite
    z["proprio"] = np.concatenate(
        [pr[:, :3], quat.astype(np.float32), pr[:, 6:8]], axis=1)
    out = p.replace("feat_libero10_", "featq_libero10_")
    np.savez(out, **z)
    print(os.path.basename(out), z["proprio"].shape, flush=True)
print("HARMONIZE_DONE")
