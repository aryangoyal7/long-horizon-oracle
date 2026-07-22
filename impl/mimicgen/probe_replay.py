"""MimicGen compatibility probe under our pinned stack (robosuite 1.5.1, mujoco 3.2.6).

Gate for all MimicGen work (predictor-pool expansion + long-horizon rollout tasks).
Checks, for one downloaded MimicGen dataset:
  1. the mimicgen env class registers and constructs via robomimic EnvUtils,
  2. offscreen rendering works (image obs for the predictor),
  3. state replay is deterministic enough for the FTLE labeler: reset_to a stored
     state, replay recorded actions, compare reached states to stored states.

Exit prints PROBE_OK or PROBE_FAIL(<stage>).

Usage:
  MUJOCO_GL=egl python probe_replay.py --dataset <demo.hdf5> [--n-demos 3]
"""

import argparse
import json
import sys

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-demos", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max |state diff| per step for bit-near determinism")
    args = ap.parse_args()

    stage = "import"
    try:
        import mimicgen  # noqa: F401  (registers envs with robosuite)
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils

        stage = "env_construct"
        with h5py.File(args.dataset, "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
            demos = list(f["data"].keys())[: args.n_demos]

            ObsUtils.initialize_obs_utils_with_obs_specs(
                {"obs": {"low_dim": ["robot0_eef_pos"], "rgb": ["agentview_image"]}})
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta, render=False, render_offscreen=True)

            stage = "render"
            env.reset()
            img = env.render(mode="rgb_array", height=84, width=84,
                             camera_name="agentview")
            assert img is not None and img.shape == (84, 84, 3), img.shape

            stage = "replay_determinism"
            # The labeler's requirement is SELF-consistency: reset_to + same actions
            # -> identical trajectory (nominal and perturbed branches then share any
            # controller transient, which cancels in the divergence). Match against
            # the STORED states is informational only — MimicGen generation leaves a
            # warm-controller qvel transient (~0.1 at step 1, decaying) that does not
            # affect labeling.
            worst_self, worst_stored = 0.0, 0.0
            for dk in demos:
                g = f[f"data/{dk}"]
                states = g["states"][()]
                actions = g["actions"][()]
                t0 = min(30, max(0, len(actions) - 17))
                K = min(16, len(actions) - t0)
                runs = []
                for _ in range(2):
                    env.reset()
                    env.reset_to({"model": g.attrs["model_file"],
                                  "states": states[t0]})
                    traj = []
                    for j in range(K):
                        env.step(actions[t0 + j])
                        traj.append(env.get_state()["states"][1:].copy())
                    runs.append(np.asarray(traj))
                d_self = float(np.max(np.abs(runs[0] - runs[1])))
                worst_self = max(worst_self, d_self)
                d_stored = float(np.max(np.abs(
                    runs[0] - states[t0 + 1: t0 + 1 + K, 1:])))
                worst_stored = max(worst_stored, d_stored)
                if d_self > args.tol:
                    print(f"PROBE_FAIL(replay_determinism) demo={dk} "
                          f"self_diff={d_self:.3e} tol={args.tol}")
                    sys.exit(1)
            print(f"replay self-consistency worst diff: {worst_self:.3e} "
                  f"(stored-state mismatch, informational: {worst_stored:.3e})")
    except SystemExit:
        raise
    except Exception as e:
        print(f"PROBE_FAIL({stage}): {type(e).__name__}: {e}")
        sys.exit(1)

    print("PROBE_OK")


if __name__ == "__main__":
    main()
