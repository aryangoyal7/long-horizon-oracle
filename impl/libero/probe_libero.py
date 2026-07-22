"""LIBERO compatibility probe: dataset layout + env construction + replay
self-consistency (the FTLE labeler's requirement).

Tries the robomimic path first (env_args in the hdf5), then the LIBERO-native path
(OffScreenRenderEnv + set_init_state). Prints findings either way — this is a probe.
"""

import argparse
import json
import traceback

import h5py
import numpy as np


def self_consistency(reset_fn, step_fn, get_state_fn, actions, t0, K):
    runs = []
    for _ in range(2):
        reset_fn()
        traj = []
        for j in range(K):
            step_fn(actions[t0 + j])
            traj.append(np.asarray(get_state_fn()).copy())
        runs.append(np.asarray(traj))
    return float(np.max(np.abs(runs[0] - runs[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    f = h5py.File(args.dataset, "r")
    print("data attrs:", sorted(f["data"].attrs.keys()))
    demos = sorted(f["data"].keys())[:1]
    g = f[f"data/{demos[0]}"]
    print("demo keys:", sorted(g.keys()))
    states = g["states"][()] if "states" in g else None
    actions = g["actions"][()]
    print("states:", None if states is None else states.shape,
          "actions:", actions.shape)
    t0 = min(30, max(0, len(actions) - args.k - 1))

    def resolve_bddl(raw):
        # dataset attrs carry generation-time relative paths (e.g. "chiliocosm/...");
        # resolve <suite>/<name>.bddl against the installed repo's bddl root
        import os
        from libero.libero import get_libero_path
        raw = raw.decode() if isinstance(raw, bytes) else raw
        base = get_libero_path("bddl_files")
        cand = os.path.join(base, os.path.basename(os.path.dirname(raw)),
                            os.path.basename(raw))
        return cand if os.path.exists(cand) else raw

    # ---- path A: robomimic-style ---------------------------------------------------
    try:
        em = json.loads(f["data"].attrs["env_args"])
        print("env_args env_name:", em.get("env_name"), "type:", em.get("type"))
        try:
            import libero.libero.envs  # noqa: F401 — registers LIBERO envs
            print("libero envs registered")
            if "bddl_file_name" in em.get("env_kwargs", {}):
                em["env_kwargs"]["bddl_file_name"] = resolve_bddl(
                    em["env_kwargs"]["bddl_file_name"])
                print("resolved bddl:", em["env_kwargs"]["bddl_file_name"])
        except Exception as e:
            print("libero env registration failed:", type(e).__name__, str(e)[:80])
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        ObsUtils.initialize_obs_utils_with_obs_specs(
            {"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
        env = EnvUtils.create_env_from_metadata(
            env_meta=em, render=False, render_offscreen=False)
        model_xml = g.attrs.get("model_file")
        def reset():
            env.reset()
            init = {"states": states[t0]}
            if model_xml is not None:
                init["model"] = model_xml
            env.reset_to(init)
        d = self_consistency(reset, lambda a: env.step(a),
                             lambda: env.get_state()["states"][1:],
                             actions, t0, args.k)
        print(f"PROBE_OK(robomimic-path) self-consistency diff={d:.3e}"
              if d < 1e-5 else
              f"PROBE_FAIL(robomimic-replay) self diff={d:.3e}")
        return
    except Exception:
        print("robomimic path failed:")
        traceback.print_exc(limit=3)

    # ---- path B: LIBERO-native -----------------------------------------------------
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import os
        bddl = f["data"].attrs.get("bddl_file_name")
        if isinstance(bddl, bytes):
            bddl = bddl.decode()
        print("bddl_file_name attr:", bddl)
        bddl = resolve_bddl(bddl)
        print("resolved:", bddl)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        env.reset()
        def reset():
            env.reset()
            env.set_init_state(states[t0])
        d = self_consistency(reset, lambda a: env.step(a),
                             lambda: env.sim.get_state().flatten()[1:],
                             actions, t0, args.k)
        print(f"PROBE_OK(libero-path) self-consistency diff={d:.3e}"
              if d < 1e-5 else
              f"PROBE_FAIL(libero-replay) self diff={d:.3e}")
    except Exception:
        print("PROBE_FAIL(libero-path):")
        traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()
