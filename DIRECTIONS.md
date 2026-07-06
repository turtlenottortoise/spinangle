# DIRECTIONS — heuristics-free attack map & next-session queue

Status ledger for the "beat SPHERE-JEPA cleanly / extend to dynamics" program.
Every direction is tagged with the theorem behind it and its empirical status.
CPU proxy evidence: `scripts/cpu_reg_bench.py` → `results/cpu_reg_bench.json`
(toy JEPA on vMF clusters, best-of-λ, 3 seeds; selects candidates only — proves
nothing at scale, per decision rule #7).

## CPU bench snapshot (2026-07-06)

| method              | kNN   | linear | clump | note |
|---------------------|-------|--------|-------|------|
| none (control)      | .703  | .778   | .757  | collapses into a cap (same geometry as TwoRoom h) |
| sigreg (N(0,1))     | .742  | .787   | .011  | LeJEPA target, mismatched on sphere |
| sigreg → N(0,1/d)   | .776  | .811   | .007  | ≈ analytic SUSReg; retargeting = +3.4 pts free |
| susreg (sampled)    | .773  | .801   | .010  | ties analytic version |
| mmd_energy (determ.)| .684  | .804   | .038  | their follow-up's winner LOSES at toy scale |
| h2 moments          | .652  | .775   | .014  | spreads but doesn't structure |
| codesphere (random) | .729  | .788   | .086  | random prototypes — fails rule #1 |
| **codesphere_etf**  | **.781** | .815 | .016 | **cross-polytope prototypes: best pure method, deterministic, 0 hyperparams** |
| local_density       | .575  | .767   | .464  | diagnostic only (rule #3 confirmed) |
| infonce (τ=0.2)     | .812  | .847   | .006  | wins, but heuristic (negatives + temperature) |
| vmf_mle (κ by MLE)  | .704  | .800   | .192  | naive de-heuristicized NCE FAILS: κ≈150 ≫ κ(τ=.2)≈5 |
| codesphere+weak nce | .815  | .840   | .038  | combos ≈ infonce; NCE does the lifting |

Gradient-noise microtest: sliced methods rel-noise 1.3–2.2 (noise > signal);
all deterministic methods exactly 0. Their follow-up's motivation reproduces.

## A. Heuristics-free directions (static SSL / their turf)

1. **Finite-geometry optimality (WON at proxy scale).** Batches/codebooks are
   finite point sets → the right target is frame theory, not continuous density:
   Welch bound, tight frames, Cohn–Kumar universal optimality, neural-collapse
   simplex-ETF. `codesphere_etf` (rotated cross-polytope, K=2d) beat every pure
   regularizer deterministically with zero free parameters. Next: simplex-ETF
   banks, multi-scale (coarse semantic / fine instance), learned-rotation-only
   prototypes. → REAL-SCALE TEST #1.
2. **Diaconis–Freedman critique (theory, untested writeup).** Almost all 1-D
   projections of any high-d dataset look Gaussian ⇒ sliced tests (SIGReg AND
   SUSReg) provably lose power with d. Frame as the reason deterministic/finite
   methods are needed; their follow-up half-concedes without naming it.
3. **The marginal hole (our sharpest critique, empirically shown).** Their
   minimax theorem constrains p(z) only; encoders with identical uniform
   marginals span .575–.815 kNN in our bench. Uniformity is nearly free and not
   the binding constraint — objectives must touch the conditional/joint.
4. **Principled discriminative objective (OPEN; naive version failed).**
   vmf_mle showed the NCE temperature is NOT the positives' vMF concentration.
   Correct version = full mixture likelihood (κ from joint fit, not positives'
   resultant). Do not oversell; park until A1 lands.
5. **Uniformity–robustness tradeoff (unclaimed paper).** Intrinsic dim k ≪ d−1
   ⇒ encoder image is measure-zero on S^{d−1}; exact uniformity unattainable;
   pushing uniformity beyond Lipschitz resolution must inflate the Lipschitz
   constant. Testable: uniformity weight ↑ ⇒ robustness ↓.

## B. World-model directions (our turf)

6. **SO(d) transitions — the "vGPT" (STRONGEST theorem-backed contender).**
   z_{t+1} = exp(A(a, z_t))·z_t, A skew-symmetric. By construction: uniform
   marginal preserved under rollout (dynamics cannot collapse), pairwise angles
   preserved (no rollout clumping — the TwoRoom failure), ‖z‖ ≡ 1 (no drift),
   invertible (backward planning), unit-circle eigenvalues ⇒ gradient norms
   preserved over horizon (uRNN truth). Step size = ‖A‖ explicit in the Lie
   algebra ⇒ gate death is not an attractor. Honest cost: rotations preserve
   information; add penalized contractive correction. → next contender run.
7. **SUSReg directly on h** (drop the projector loophole behind the TwoRoom
   cap): implement as sigreg with target N(0,1/d) on normalized h (~30 lines in
   objective.py). Pairs with memory-NCE (already in code).
8. **Transition grounding** (anti-identity): supervise predicted step size with
   true state deltas from the dataset (poor-man's solver-derived correction).
   Direct fix for the Push-T gate-death/identity floor. Pair with
   gate_bias_init=2.0 and/or α floor.
9. **SphericalGRU / streaming core** (O(1)/step, filter-corrected,
   surprise-triggered replanning). Frontier claims: success vs planner
   wall-clock, success vs sensing-gap. CSV already has plan_latency_s columns.
10. **Unified residual streams (depth = time).** World models have two residual
    streams: depth-wise (standard attnres, what nGPT geometrized) and time-wise
    (the state update, what we geometrize). Tie them: predictor depth = k
    weight-tied iterations of the same tangent/SO(d) step (unrolled spherical
    flow). Collapse-safety rule: any new attention connection must flow from
    CONTEXT ONLY — never give the predictor access to target-encoder features
    (copy shortcut ⇒ identity collapse; we have met this failure personally).

## C. Tomorrow's real-scale queue

1. Static SSL small-real test (their currency, one L4/A100 session):
   codesphere_etf vs susreg vs sigreg vs mmd_energy on CIFAR-10/STL-10-scale
   JEPA; metrics: kNN, linear probe, clumping, eff-rank + gradient-noise.
2. World-model grid (needs matched official Push-T baseline finishing):
   tangent+fixes (8) retrain OR SO(d) contender (6) — pick ONE; SUSReg-on-h (7)
   rides along as the regularizer swap.
3. Diagnostics backlog: Push-T controllability probe on collapsed tangent
   (20 min, for the record); TwoRoom memory-NCE eval if final ckpt present.

## D. Contact plan (Centre Borelli / SPHERE-JEPA authors)

- Email draft agreed (see session notes): thesis = "you made representations
  spherical; we make transitions spherical"; leads with TwoRoom result +
  transition-collapse finding; asks 30 min at Centre Borelli (Paris ✓).
- Before sending: results README (TwoRoom table, Push-T collapse probes, CPU
  bench table + gradient-noise reproduction of their claim, one diagnostics
  plot). The CPU bench is the icebreaker artifact: reproduces their motivation,
  shows one anomaly (MMD losing at small scale), proposes the joint experiment
  (their regularizers, our dynamics testbed).
- Do NOT claim: "beats LeWM" (one seed, saturated task), rollout-error
  superiority (geometry-incomparable), or novelty of spherical JEPA per se.
