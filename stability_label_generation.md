# Generating Ground-Truth Stability Labels for Long-Horizon Demonstrations

**Purpose.** This document describes, step by step, how we generate the supervised training
data for the video stability predictor: per-timestep labels saying whether the current
segment of a long-horizon task is **open-loop stable** or **open-loop unstable** (plus a
secondary **closed-loop** label). These labels are the ground truth; the video model is
trained afterwards to predict them from pixels alone.

Companion documents: `adaptive_chunking_hypothesis_v0.4.tex` (theory),
`implementation_plan_v0.1.md` (full pipeline).

---

## 1. The idea in plain words

A segment is **open-loop stable** if a small mistake dies out on its own: you can perturb the
system, keep playing the *same* actions, and the perturbed trajectory converges back to the
nominal one. A segment is **open-loop unstable** if the same small mistake grows.

So the measurement is simple: **make a small mistake on purpose, replay the same actions,
and watch whether the gap shrinks or grows.** The rate at which the gap grows or shrinks is
the label.

This measured rate is a textbook quantity: the **finite-time Lyapunov exponent (FTLE)**
[3, 4]. We are not inventing an estimator — we are applying a standard one per-timestep
along robot demonstrations, which nobody has done for this purpose.

Why this is the *right* ground truth for our hypothesis: the theory papers [1, 2] define the
problem as error compounding — execution error J_TRAJ growing like C^T times the training
(regression) error J_DEMO. They *assume* stability mathematically (EISS, Assumption 3.1 in
[2]) and never measure it. By choosing the size of our deliberate mistake to equal the
policy's actual regression error (see Step 3), our measurement becomes exactly their
question: *does an error of the size the policy really makes get amplified or absorbed?*

## 2. What we need before starting

| Ingredient | Where it comes from |
|---|---|
| Simulator with state save/restore | MuJoCo via robomimic (`get_state` / `set_state`) |
| Demonstrations | robomimic ph datasets (e.g. Square, Tool-Hang: 200 demos each) |
| A trained chunking policy | Diffusion Policy trained on the same demos (needed for the perturbation scale, the closed-loop label, and rollout coverage) |
| The policy's validation action RMSE | One number, computed once on held-out demos; call it **σ_u** |

## 3. Step-by-step: the OPEN-LOOP label

For each trajectory (demo or policy rollout) and each timestep `t` on a stride of 2–5 steps:

**Step 1 — Reset.** Restore the simulator to the exact state at time `t`
(`set_state`). Let `a_t, a_{t+1}, ..., a_{t+K-1}` be the actions that were actually recorded
from `t` onward (the demo's actions, or the policy's executed actions for rollout data).
Horizon: `K = 16` steps.

**Step 2 — Nominal branch.** Replay those K actions unchanged. Record the state trajectory
`x_t, ..., x_{t+K}` (state = end-effector pose + object pose).

**Step 3 — Perturbed branches.** Repeat N = 8 times: restore the state at `t` again, add
Gaussian noise **only to the first action**, `a_t + ε_n` with `ε_n ~ N(0, σ_u² I)`, then
replay the remaining K−1 actions **unchanged**. Record each perturbed state trajectory.

Two deliberate choices here:
- *Perturb the action, not the state.* This matches the input channel of the EISS
  definition in [1, 2] (errors enter through actions) and guarantees the perturbation is
  physically reachable.
- *σ_u = the policy's validation action RMSE.* We probe with a mistake of exactly the size
  the policy will actually make at execution time. That is what ties the label to the
  error-compounding theory instead of being an arbitrary stability test.

**Step 4 — Measure the gap over time.** For each perturbed branch `n` and each step
`τ = 1..K`, compute the state distance
`d_n(τ) = || x_{t+τ}^{perturbed,n} − x_{t+τ}^{nominal} ||`.

**Step 5 — Fit the growth rate.** Average over branches, then fit a line to the log-gap:

```
λ_ol(t) = OLS slope of  mean_n log d_n(τ)   versus   τ = 1..K
```

This slope is the open-loop finite-time Lyapunov exponent at time `t`:
- `λ_ol < 0` → gap shrinks → **contracting** (open-loop stable)
- `λ_ol > 0` → gap grows → **expansive** (open-loop unstable)

**Step 6 — Threshold with a deadband.** With margin `δ` (start: δ such that |λ| < δ covers
the noise floor measured on obviously-free-space frames):

```
y_ol(t) = UNSTABLE  if λ_ol(t) > +δ
y_ol(t) = STABLE    if λ_ol(t) < −δ
y_ol(t) = previous label otherwise   (deadband: inherit)
```

**Step 7 — Smooth.** Median-filter `y_ol` over ~3 consecutive labeled stamps so labels form
contiguous segments instead of flickering at boundaries. Also log the simulator's contact
flag at each `t` — not used as a label, but as a sanity channel (labels should roughly align
with contact on/off).

## 4. Step-by-step: the CLOSED-LOOP label

Identical to Section 3 with **one change in Step 3**: in each perturbed branch, after the
initial action perturbation, the remaining K−1 actions are **not** replayed — instead the
trained policy observes the perturbed states and acts, replanning every step (k = 1).

The fitted slope `λ_cl(t)` and thresholded label `y_cl(t)` then say whether **per-step
feedback with this policy** damps the injected error or amplifies it.

Notes:
- This label is *policy-conditional* by definition (a better policy ⇒ more closed-loop
  stable states). That is intended: it is the operationally relevant quantity. Say this
  explicitly in the paper.
- It is also the mechanism check from the theory [1]: in open-loop-stable segments we
  expect λ_cl > λ_ol (per-step feedback re-injects the policy's estimation error); in
  open-loop-unstable segments the ordering should flip.

## 5. What each timestep contributes to the training set

```
input  : last 16 frames ending at t  +  proprioception
targets: y_ol(t)  [main, BCE]      λ_ol(t)  [auxiliary, Huber]
         y_cl(t)  [aux, BCE]       λ_cl(t)  [auxiliary, Huber]
```

**Coverage rule:** run the whole procedure on BOTH the demonstrations and rollouts of the
trained policy (including failed rollouts). The predictor must be trained on the state
distribution it will see at runtime, not only the narrow tube of successful demos.

## 6. Cross-checks (cheap, all standard)

1. **Jacobian check** (contraction analysis [5]): on a subsample of states,
   finite-difference the one-step sim map, take the log of the largest singular value of the
   Jacobian → an independent local contraction rate. Should correlate with λ_ol.
2. **Open-loop replay test** (the imitation-learning community's coarse standard): from a
   perturbed start, replay the full remaining demo open-loop; task success should be high in
   trajectories dominated by STABLE labels and low otherwise.
3. **Contact alignment:** fraction of UNSTABLE stamps that carry the contact flag (and vice
   versa). No exact match expected — some contacts are stabilizing (e.g. object settled in a
   grasp) — but gross misalignment means a bug.

## 7. Parameters (finalized empirically 2026-07-12 — 6-variant probe study on square)

| Parameter | Final value | Why |
|---|---|---|
| Horizon K | **24 steps** (1.2 s @ 20 Hz) | lowers free-space noise floor to lift-level (p95 0.051) |
| Branches N | 8 | averages out perturbation-direction luck |
| Perturbed dims | **position only (a[0:3])** | rotation-dim noise injects lever-arm divergence in free space that swamps the signal; pos-only doubled free-vs-contact separation |
| Perturbation scale σ_u | policy validation action RMSE (0.05 placeholder until trained) | ties label to error-compounding theory |
| Label stride | every 2 steps | labels vary slowly; saves compute |
| Deadband δ | p95 of \|λ\| on free-space stamps (≈0.05) | avoids coin-flip labels near 0 |
| Median filter | 3 stamps | contiguous segments, no flicker |
| State metric | ee pos + object obs (task space) | full-state metric discriminates worse (checked) |

Probe-study outcome (square, target = fixture-contact-in-window): baseline 6-dof/K16 →
12.7% of fixture stamps above deadband vs 4.4% of free stamps; final pos-only/K24 →
**36.6% vs 1.2%** (30:1). Notes: (i) contact-AUROC is a sanity channel, not the
objective — the hypothesis itself says contact and instability correlate but are not
identical (funnel-type contacts stabilize), so Stage 1a success-vs-k is the decisive
test; (ii) **contact classification must be object-centric** — only pairs involving the
task object count (square's unused RoundNut lies on the floor all episode; can's env
keeps Bread/Cereal/Milk around; tool_hang's stand always touches the table).

Cost: pure simulator rollouts, no gradients, embarrassingly parallel across (trajectory, t,
branch). For 200 demos + 200 rollouts this is hours on CPU workers.

## 8. What is standard vs. what is ours

- **Standard (cite, don't defend):** the FTLE estimator itself [3, 4]; perturb-and-replay
  divergence measurement; contraction-rate cross-check [5]; the J_TRAJ vs J_DEMO
  error-compounding framing [1, 2].
- **Ours (the contribution):** applying the estimator *per-timestep along long-horizon
  manipulation demonstrations* (the literature only classifies whole environments, and
  [1, 2] only ever assume stability); the policy-in-the-loop closed-loop variant; using the
  policy's own regression error as the perturbation scale; distilling the labels into a
  video predictor.

## References

1. Simchowitz, Pfrommer, Jadbabaie. *The Pitfalls of Imitation Learning when Actions are
   Continuous.* arXiv:2503.09722. https://arxiv.org/abs/2503.09722
   — proves exponential compounding through the feedback channel even for open-loop-stable
   dynamics; source of the EISS framing.
2. Zhang, Pfrommer, Pan, Matni, Simchowitz. *Action Chunking and Exploratory Data Collection
   Yield Exponential Improvements in Behavior Cloning for Continuous Control.*
   arXiv:2507.09061. https://arxiv.org/abs/2507.09061
   — the k > k* chunking guarantee (Thm 1) and noise-injection result (Thm 2). Note: assumes
   EISS (Assumption 3.1); contains **no** empirical stability-classification procedure —
   verified by reading the paper, which is why this labeling protocol has to exist.
3. Benettin, Galgani, Giorgilli, Strelcyn. *Lyapunov Characteristic Exponents for smooth
   dynamical systems; a method for computing all of them.* Meccanica 15, 1980.
   — the classical algorithm; our Step 3–5 is its finite-time, finite-perturbation form.
4. *Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach.*
   arXiv:2410.10674. https://arxiv.org/html/2410.10674v1
   — recent precedent for FTLE-style perturbation-growth measurement on learned policies;
   also *FTLE Analysis of MPC and RL* (Krishna et al.),
   https://www.researchgate.net/publication/374894326 — both classify whole policies/
   systems, not segments.
5. Lohmiller, Slotine. *On Contraction Analysis for Non-linear Systems.* Automatica 34(6),
   1998. — contraction-rate view behind the Jacobian cross-check (Section 6.1).
