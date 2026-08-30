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
| codesphere_etf      | .781  | .815   | .016  | cross-polytope (K=2d), deterministic |
| **codesphere_simplex** | **.782** | .810 | .017 | **simplex ETF (K=d+1): ties cross with half the prototypes, best eff-rank (26.0) of the family** |
| codesphere_ms       | .782  | .812   | .042  | coarse simplex + fine tight-frame bank; flat at toy scale (10 classes — no hierarchy to exploit; real-scale question) |
| local_density       | .575  | .767   | .464  | diagnostic only (rule #3 confirmed) |
| infonce (τ=0.2)     | .812  | .847   | .006  | wins among singles, but heuristic (negatives + temperature) |
| vmf_mle (κ by MLE)  | .704  | .800   | .192  | naive de-heuristicized NCE FAILS: κ≈150 ≫ κ(τ=.2)≈5 |
| codesphere+weak nce | .815  | .840   | .038  | combos ≈ infonce; NCE does the lifting |
| **codesphere_simplex+nce** | **.819** | .837 | .027 | **best overall: optimal prototypes + weak discriminative** |
| codesphere_ms+nce   | .817  | .843   | .032  | second overall |

Honesty notes: 3 seeds ⇒ ±~0.01 noise, top-of-family differences are ties;
CodeSphere is *not* zero-knob (Sinkhorn assignment τ=0.1 and λ remain) — its
claim is theorem-fixed TARGET + deterministic gradients, not zero parameters.
The whole ETF family (.781–.782) sits above every continuous-density
regularizer, sliced or deterministic.

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
2. **Diaconis–Freedman critique (DEMONSTRATED, `cpu_paper_tests.py` T1).**
   Uniform-on-half-subsphere (measure zero, projection variance exactly 1/d):
   sliced SUSReg-style discrimination z-score collapses 150→46→13→**3.5** as
   d goes 8→512 (n=2048, R=64), while the H2 moment/frame statistic stays flat
   at ~1400σ. Sliced tests provably and now measurably go blind with dimension;
   moment/finite-configuration statistics do not.
3. **The marginal hole (our sharpest critique, empirically shown).** Their
   minimax theorem constrains p(z) only; encoders with identical uniform
   marginals span .575–.815 kNN in our bench. Uniformity is nearly free and not
   the binding constraint — objectives must touch the conditional/joint.
4. **Principled discriminative objective (OPEN; naive version failed).**
   vmf_mle showed the NCE temperature is NOT the positives' vMF concentration.
   Correct version = full mixture likelihood (κ from joint fit, not positives'
   resultant). Do not oversell; park until A1 lands.
5. **Uniformity–robustness tradeoff (CONFIRMED at toy scale, with a KNEE —
   `cpu_paper_tests.py` T2).** k=4 manifold in R^64, sweeping simplex-reg λ:
   λ 0→32 drives |mean| 0.65→0.10 and clump 0.43→0.01, but empirical Lipschitz
   inflates **12×** (0.054→0.62) and the clean-vs-noisy kNN gap **4×**
   (0.058→0.224) while clean acc barely moves. Sharper than predicted: there
   is a KNEE — λ=0.5 buys most of the uniformity at zero Lipschitz cost;
   λ≥2 crosses the manifold's resolution and pays. "Resolution-limited vs
   pathological regime" is the paper's figure 1.

## B. World-model directions (our turf)

6. **SO(d) transitions — the "vGPT" (TOY-VALIDATED, GPU-READY).**
   z_{t+1} = k exact plane rotations of z_t (planes+angles from heads; raw
   linear θ, zero-init ⇒ starts AT identity with healthy gradients). Toy test
   (`scripts/cpu_rotation_bench.py`, reconstructs the Push-T smooth-dynamics
   collapse regime): **10-step rollout error 3–5× lower than tangent/gated**
   (.058 vs .195/.273), top future-retrieval (.84), step size tracks true
   motion (76%). NEW FINDING: gated/tangent training *shrinks true latent
   motion* (encoder co-adapts to make copying cheap; true step .056–.069 vs
   rotation's .111–.113) — identity collapse has an **encoder accomplice**;
   isometric transitions remove the incentive. Wired end-to-end:
   `+experiment=rotation_spherical_simplex` (mode=rotation predictor,
   simplex-proto loss on h, cosine+stop-grad); smoke tests pass incl.
   θ-gradient-at-identity check and exact-unit-norm planner rollouts.
   Honest cost: rotations preserve information; contractive correction is the
   known follow-up. → THE next contender training run.
   Parameterization truths measured (`cpu_paper_tests.py` T3, random nontrivial
   weights): gradient norm through a T-step rollout is DEAD by T=16 for
   simple/gated/tangent (0.000–0.015 rel.) and conserved for rotation (1.16 at
   T=16, 1.86 at T=64 — uRNN property); pairwise-angle distortion after 32
   steps: rotation 0.12 vs 0.72–0.91 (6–7× better structure preservation);
   Cayley low-rank skew (Woodbury, d=192, k=8): orthogonal to 1e-7, pairwise
   cosines to 3e-8, **0.18 ms/step for B=256 on CPU** — cheap enough for
   real-time control loops without a GPU.
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

## B3. Architecture toy results — contact world (`scripts/cpu_arch_bench.py`)

Two-body toy mirroring Push-T: pusher (action-driven) + block that moves ONLY
on contact (sparse events), block weakly observable. All variants share the
contender recipe (cosine + stop-grad + simplex proto). 3 seeds:

| mode      | roll10 | retr@1 | R²push | R²block | params |
|-----------|--------|--------|--------|---------|--------|
| tangent   | .404   | **.099** | **.343** | .030  | 8.6k |
| mono_rot  | .049   | .878   | .138   | −.109   | 9.1k |
| prod_rot  | **.041** | .862 | .178   | −.120   | 9.2k |
| tied3     | .066   | .859   | .228   | −.012   | 9.1k |
| untied3   | .055   | .893   | .142   | .032    | 27.3k |
| mem_rot   | .059   | .859   | .182   | −.101   | 14.9k |

Verdicts on the three ideas:
- **Product-of-spheres (idea 1): weak-positive.** Best rollout (−16% vs mono)
  at matched params; keep as the spherical-patch design for the Push-T retrain
  (real spatial structure; a 2-patch toy can't show more).
- **Depth=time tying (idea 2): benefit regime untested.** Depth added nothing
  here — the toy's true dynamics ARE one rotation, so extra steps are
  unnecessary by construction. Tying ≈ untied at 1/3 params (efficiency
  confirmed). Needs composition-demanding dynamics to test properly. Parked.
- **Memory residual (idea 3): negative at toy scale.** No gain anywhere; the
  bottleneck is not access to history (see finding below). Superseded by
  objective-level fixes (#8 grounding, contact-weighted losses).

Two NEW findings (both strengthen the paper story):
- **The objective blind spot.** EVERY architecture discards the contact-driven
  block (R²_block ≤ .03 across the board; step sizes uncorrelated with true
  contact). The cosine-JEPA loss barely rewards encoding a body that moves
  rarely and weakly — architecture cannot rescue what the objective does not
  value. Sparse-event information loss is an OBJECTIVE-level problem →
  grounding/contact-weighted losses (#8) outrank any architecture tweak.
- **Probes anti-correlate with rollout competence, third instance.** tangent
  posts the best state-probe R² (.343) and the worst rollout/retrieval
  (.404/.099) — same inversion as TwoRoom retrieval-vs-planning and toy-1
  act_frac. Also: tangent is BRITTLE across settings (retr .857→.099 between
  bench configs) while rotation cores were stable in every configuration
  tested. (v1 of this bench had an unnormalized-cosine metric bug for the
  product latent; v2 numbers above are authoritative.)

## B2. Tiering — production-grade vs publishable (2026-07-06)

**Production-grade today** (deploy, no research bet):
- **Retargeted SIGReg** (`target N(0,1/d)` on normalized embeddings): one-line
  change, +3.4 kNN at proxy scale, zero new machinery.
- **Simplex-prototype regularizer** (`codesphere_simplex`): deterministic
  gradients, K=d+1 (tiny, O(BdD)), theorem-fixed target. Most industrial
  application: **VQ codebook dead-code prevention for speech/audio codecs**
  (regularize/init codebooks toward tight frames).
- **H2 mean-penalty as a collapse alarm**: ‖mean(z)‖² is a cap detector — a
  monitoring rail for any embedding pipeline, not a paper.
- **Gradient-noise microtest as CI** for regularizer implementations.

**Publishable on top of SPHERE-JEPA** (contingent on real-scale runs):
1. *Finite beats continuous*: regularize toward universally optimal finite
   configurations (simplex/cross-polytope/tight frames), not continuous
   densities. Theory: Cohn–Kumar + Benedetto–Fickus + neural-collapse ETF;
   critique: Diaconis–Freedman + the marginal hole (.575–.815 kNN at matched
   uniformity). Their-turf paper.
2. *Uniformity–robustness tradeoff*: topology/Lipschitz argument with a cheap
   testable prediction (λ_uniformity ↑ ⇒ robustness ↓). Standalone.
3. *SO(d) dynamics* (flagship, collaboration paper): transitions in the
   sphere's symmetry group — rollout collapse impossible by measure
   preservation; + our probes≠planning and transition-collapse findings as
   motivation. Implementation path: Cayley transform (I−A/2)⁻¹(I+A/2) or
   low-rank skew A = UVᵀ−VUᵀ (O(dk)) instead of full matrix-exp.
4. (Parked) full-mixture-likelihood contrastive — naive vMF-MLE failed;
   κ-from-joint is the open thread.

## B4. Tokenizer / VQ-codebook finding (`scripts/cpu_vq_codebook_bench.py`)

Spherical VQ autoencoder, K=256 codes on S^15, 96 true data modes. Two regimes
expose that **codebook occupancy and continuous-marginal uniformity are
different axes**:

REGIME A (end-to-end, encoder co-collapses; vanilla z_clump 0.997):
  marginal regs HELP occupancy by spreading z — sigreg_h 219/256, simplex_h 79.
REGIME B (frozen encoder, z fixed & spread at clump 0.61; = the speech setup):
  vanilla 51, ema 44, **sigreg_h 51, simplex_h 51 (identical to vanilla)**,
  entropy 51, **sinkhorn 232, reinit 223**.

Taxonomy of anti-collapse mechanisms (the real result):
- act on the continuous embedding z (SIGReg/SUSReg/MMD/simplex, AND a
  usage-entropy penalty routed through softmax(z·C)): help occupancy ONLY to
  the extent the collapse is *encoder-driven*. When z is already spread
  (codebook-side death) they are literal no-ops. This is Leonard's speech
  finding, now with a mechanism.
- act on the discrete assignment / codebook geometry directly (Sinkhorn/
  balanced assignment, dead-code reinit): fix occupancy in BOTH regimes
  (4–4.5× here). This is why "Sph-KL helped" in the speech run — it anchors on
  the codebook, i.e. it is secretly a codebook-side method (halfway to
  CodeSphere), not a marginal method.
- DIAGNOSTIC: measure z_clump / |mean| of the *pre-quant* embedding. Low clump
  + dead codes ⇒ codebook-side ⇒ only assignment/reinit help. High clump ⇒
  encoder-driven ⇒ spreading z (marginal reg) also helps.
- CAVEAT (honesty): the usage-entropy no-op is specific to the frozen-encoder
  isolation (it backprops only into the frozen encoder). With a trainable
  encoder it becomes an encoder-reshaping method (Regime-A-like). Sinkhorn/
  reinit are the only mechanisms that work regardless.
Publishable line for the Borelli group: "representation uniformity regularizers
do not transfer to discrete tokenizer collapse; occupancy is an
assignment-histogram problem." Product: a drop-in Sinkhorn-balanced spherical
VQ (their KL, made codebook-anchored) for speech/audio codecs.
NOTE: only finding #1 of the user's three is captured here; #2/#3 pending.

## B5. Bingham directional-uncertainty head (`scripts/cpu_bingham_pose.py`)

Deep-Bingham rotation head (orthonormal M + concentrations Z, MC normalizer on
a fixed S^3 grid) vs geodesic quaternion regression; 3D rotation from 3 noisy
matrix measurements (60/25/15 clean/medium/occluded). Verdict on the "swap the
head, get calibrated uncertainty for free" pitch:

CONFIRMED (real, keep):
- antipodal q==-q gap EXACTLY 0 by construction (no min(+-q) hack ever);
- mode accuracy BETTER than geodesic regression (median 5.9 vs 7.9 deg) --
  head swap is accuracy-positive;
- the normalization constant is differentiable + stable via fixed-grid MC
  (the flagged bottleneck is tractable; 8192->32768 grid barely moves results,
  so MC is NOT the limiting factor here).

NOT SUPPORTED / deflators (the honest part):
- concentration is ANTI-calibrated to occlusion: model is MORE certain on
  occluded samples (sep clean-occl = -5 to -9, robust to readout & grid). This
  is the heteroscedastic-NLL pitfall (Seitzer et al. ICLR'22); the beta-NLL
  remedy made it WORSE (sep -5 -> -14), not better.
- as an error-ranker the concentration works but is MEDIOCRE and is BEATEN by
  the geodesic baseline's FREE pre-norm-magnitude proxy (rank-corr 0.26 vs
  0.42; risk-coverage@25% 8.6 vs 8.2 deg). You can get comparable confidence
  for free without Bingham in this regime.
- CAVEAT that saves the idea: this toy uses Gaussian measurement noise, which
  is vMF/Gaussian's home, NOT Bingham's. Bingham's real edge is MULTIMODAL /
  SYMMETRY-induced ambiguity (symmetric object -> genuinely bimodal posterior),
  where a single-quaternion regressor catastrophically averages. Single Bingham
  natively covers only the ANTIPODAL case; general object symmetry needs a
  Bingham/Kent MIXTURE -> that is a research program, NOT a plug-and-play swap.

REVISED SCORING (vs the 9/10 pitch): antipodal+accuracy are free and real
(~7/10 engineering, low risk). "Instant calibrated uncertainty from a naive
head swap" is over-claimed and contradicted here. The genuine 10/10 ("universal
directional-uncertainty output layer with downstream planning gains") is real
but requires mixtures + symmetry benchmarks + the heteroscedastic fixes --
consistent with Leonard's own "why not 10/10". NEXT TEST if pursued: object-
symmetry ambiguity with a Bingham mixture vs regressor (the home-turf test this
one deliberately was not). Ties to our SO(d) world model: same directional-
output-layer question, and the mode-vs-average failure is the pose-domain
cousin of our transition-collapse finding.

## B6. Constraint-form anti-collapse — the mechanism that survived
(`cpu_dual_constraint_test.py`; motivated by the cap-economics falsification)

Derived-and-tested mechanisms (each aimed at a measured failure):
- M1 kinetic floor (encoder temporal step ≥ c·state delta): PRIOR ART —
  Temporally-Centered SIGReg (2607.26924) regularizes temporal residuals.
  Our test adds: even in constraint form it binds exactly yet fails to
  de-collapse (satisfiable inside the cap). Proxy constraint.
- M2 counterfactual action-dispersion floor (act_frac as a loss): open as a
  trained loss (only metrics/offline weights exist: 2606.24152, 2608.06706)
  but INCONCLUSIVE here — floor met at init in this toy; needs a
  controllability-broken setting (Push-T) to test.
- M3 **constraint-form anti-collapse via dual ascent** (open in SSL; penalty
  form is universal): VALIDATED in two rounds.
  Round 1: constraints bind EXACTLY at a measured price (λ*=0.22) — dual
  mechanics work; my proxy targets were the failure ("you get what you
  constrain, not what you hope").
  Round 2 (spread_dual): constraining the diagnostic itself (batch clump ≤
  0.10) with zero fixed weights de-caps the no-regularizer rotation model:
  clump .80→.05, |mean| .89→.21, λ* → 2.77 = the price of information,
  discovered automatically. Caveats: erank stays low (dimensional collapse
  needs its own constraint), motion partially restored.

The salvaged framework — **diagnostic-constrained SSL**: pick the health
diagnostics you actually care about (clump, erank, temporal motion,
dispersion), enforce each as a constraint with its own dual-ascended
multiplier, ZERO λ tuning, and read the converged λ* vector as interpretable
prices. Honest grade ~7/10: mechanics proven at toy scale; needs the erank+
motion constraint-set extension, then real scale (CIFAR bench / Push-T).
Fits the "collapse is repricing" theory chapter exactly: dual ascent is
automated repricing.

## C. Tomorrow's real-scale queue

1. Static SSL small-real test (their currency, one L4/A100 session): READY —
   `scripts/gpu_cifar_reg_bench.py` (SimSiam-style ResNet-18 on CIFAR-10,
   none/sigreg/sigreg_1overd/mmd_energy/simplex/simplex_nce; kNN + linear +
   clumping + NOISY-kNN robustness gap + Lipschitz probe, so it also scale-
   tests the tradeoff paper). ~17 min/method/40ep on L4 ⇒ ~1.5–2 h total;
   idempotent per-method JSONs. CPU smoke-tested.
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
