"""Sanity checks required before the FTLE labeler can be trusted.

1. Headless EGL rendering works (needed for image obs generation + predictor inputs).
2. get_state/set_state replay determinism: resetting to a saved sim state and replaying
   the SAME actions must reproduce the SAME states (bit-exact or to float noise floor).
   The open-loop FTLE label is meaningless without this.
3. Perturbation propagates: a small first-action perturbation produces diverging or
   contracting |state difference| — the raw signal the labeler fits.

Run inside the lh venv with MUJOCO_GL=egl.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def make_env():
    cfg = load_composite_controller_config(controller="BASIC")
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        hard_reset=False,
    )


def flat_state(env):
    return np.concatenate([env.sim.data.qpos.copy(), env.sim.data.qvel.copy()])


def main():
    env = make_env()
    obs = env.reset()
    img = obs["agentview_image"]
    print(f"[1] EGL render OK: agentview_image {img.shape} dtype={img.dtype} "
          f"mean={img.mean():.1f} (nonzero={img.mean() > 1.0})")

    # roll 10 warmup steps with fixed small actions
    rng = np.random.default_rng(0)
    warm = [0.1 * rng.standard_normal(env.action_dim) for _ in range(10)]
    for a in warm:
        env.step(a)

    saved = env.sim.get_state().flatten()
    actions = [0.1 * rng.standard_normal(env.action_dim) for _ in range(20)]

    # branch A
    states_a = []
    for a in actions:
        env.step(a)
        states_a.append(flat_state(env))

    # branch B: restore and replay identical actions
    env.sim.set_state_from_flattened(saved)
    env.sim.forward()
    states_b = []
    for a in actions:
        env.step(a)
        states_b.append(flat_state(env))

    diffs = [np.abs(a - b).max() for a, b in zip(states_a, states_b)]
    print(f"[2] replay determinism: max|state diff| over 20 steps = {max(diffs):.3e} "
          f"({'PASS (bit-exact)' if max(diffs) == 0.0 else ('PASS (noise floor)' if max(diffs) < 1e-10 else 'FAIL')})")

    # branch C: perturb only the first action, replay rest identically
    env.sim.set_state_from_flattened(saved)
    env.sim.forward()
    states_c = []
    for j, a in enumerate(actions):
        ap = a.copy()
        if j == 0:
            ap[:3] += 0.05  # small ee-delta perturbation
        env.step(ap)
        states_c.append(flat_state(env))
    div = [np.linalg.norm(a - c) for a, c in zip(states_a, states_c)]
    print(f"[3] perturbation propagates: |div| t=1: {div[0]:.4f}  t=10: {div[9]:.4f}  "
          f"t=20: {div[19]:.4f}")
    print("    (free space + gravity-compensated arm => expect decay or slow drift)")
    env.close()
    print("ALL SANITY CHECKS DONE")


if __name__ == "__main__":
    main()
