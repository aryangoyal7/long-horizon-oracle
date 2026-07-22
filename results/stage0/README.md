# Stage 0 results — synthetic piecewise-linear validation

Code: `impl/stage0/stage0_piecewise.py`. 2000 trials/cell. System: 3 segments
(contract ρ=0.5 / expand λ / contract), expert closed-loop 0.75, worst-case-within-ball
regression error (per-segment ε), executed chunk length k set per segment.
Success = trajectory stays within 10× the demo tube for all 150 steps.

## Headline run (`results.json`, `stage0_money_plots.png`): ε_c=0.55, λ_e=1.35

**P1 — floor in contracting segments (the sign prediction the literature can't make):**
success collapses as k→1: k∈{1,2,3} → 0%, k=4 → 20%, k=6 → 99.9%, k≥8 → 100%.
Short execution is not "safe" here — it is fatal. Empirical k* ≈ 5–6.

**P2 — ceiling in expansive segment:** k≤4 → 100%, k=6 → 6%, k≥8 → 0%.
Empirical ceiling ≈ 4–5, tracking the 1/log λ prediction (3.3).

**P3 — no constant k works when floor > ceiling:**
best constant k (k=6): **3.1%** success. Best adaptive profile (k_c=6, k_e=2): **100%**.
The heatmap corner beats the diagonal by 97 points.

**P4 — the FTLE labeler is exact and the closed-loop label flips as predicted:**

| segment | λ_ol measured | λ_ol true | λ_cl measured | λ_cl true |
|---|---|---|---|---|
| C1 (contract) | −0.6931 | −0.6931 | **+0.263** | +0.262 |
| E (expand)    | +0.3001 | +0.3001 | **−0.102** | −0.163 |
| C2 (contract) | −0.6931 | −0.6931 | **+0.264** | +0.262 |

Open-loop FTLE recovers the true rate to 4 decimals. The closed-loop FTLE (learned k=1
policy in the loop) has the OPPOSITE sign in both regimes: per-step feedback destabilizes
the contracting segments (the Simchowitz feedback-channel mechanism) and stabilizes the
expansive one. This is the full three-case decision logic observed in one plot.

## Secondary run (`results_window.json`): ε_c=0.35, λ_e=1.12

Floor (≈3) < ceiling (≈12) → a constant k∈[3,8] satisfies both constraints and reaches
100%. Adaptive ties but cannot beat it. **Interpretation:** the theory predicts adaptive
chunking wins exactly when the per-segment constraints are disjoint (k*_i > 1/λ_j).
Whether real manipulation tasks are in the disjoint regime is precisely what Stage 1
measures — this run documents that the claim is falsifiable, not tautological.

## Notes
- Regression error uses the worst-case-within-J_DEMO-ball alignment (the lower-bound
  regime of arXiv:2503.09722), with per-trial jitter; expansive-segment error is smaller
  (0.10), emulating noise-injected data per arXiv:2507.09061 Thm 2 — without which no
  imitator succeeds there under any k.
- Replans forced at segment boundaries for all methods (isolates within-segment k effect).
- k executes a prefix of one shared 16-step chunk map, mirroring Diffusion Policy usage.
