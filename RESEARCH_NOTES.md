# nGPT-JEPA — Mechanism Notes & Research Directions

Working theory for *why* (and *where*) spherical / nGPT geometry should help a JEPA
world model, the predictions it makes, and the open directions. Companion to the
ablation harness; every claim here is meant to be falsified by a run.

## 0. The organizing principle: normalize what *recurs*

A world model's latent `h_t` is a **recurrent state**: `h_{t+1}=T(h_t,a_t)`, fed back
each rollout step and read by the loss each step. `T`'s internal activations are
recomputed fresh per step and never accumulate. Therefore:

- normalizing `h` (the recurrent state) is **objective- and stability-critical**
  (enters the cosine loss; accumulates as rollout drift);
- normalizing `T`'s internals is **optimization-only** (resets each step; pays the
  −23% effective-rank cost for no rollout benefit).

LMs have no recurrent state (residual stream read once at the end), so nGPT's only
site is the feedforward internals. **The world model has a new site — the recurrent
latent — where spherical geometry is load-bearing; that is the core reason it is more
valuable here, and it says put the geometry on `h`, not inside `T`.**

## 1. Objective–geometry alignment (the clean identity)

Cosine loss `L = 1 − ĥ·t` has gradient `∂L/∂h = −(1/‖h‖) P_ĥ t`, `P_ĥ = I − ĥĥᵀ`.
The `F.normalize` Jacobian is `P_x̂/‖x‖`. **Same projector**, idempotent, so loss and
layer compose with no wasted radial component: **nGPT-normalize + cosine-JEPA =
Riemannian gradient descent on Sᵈ⁻¹.** In an LM the softmax + learned logit scales
`s` reinject magnitude and *decode the sphere away*, so the constraint is a gauge `s`
compensates for. JEPA has no such decode — the sphere is the objective.

## 2. SIGReg on z vs on h — a scale-mismatch (well-posedness) result

Take `h` uniform on the unit sphere. A random 1-D projection `w·h` has variance
`wᵀE[hhᵀ]w = 1/d`, and the hard cap `tr(Cov) ≤ 1` bounds the *total* variance budget
over all `d` directions at 1. SIGReg (Epps–Pulley vs **standard** normal) targets
per-direction variance **1** → off by a factor `d`, and **unsatisfiable** for unit
vectors. Trace the gradient: SIGReg pushes to widen the projections (`|h_i·a_j|↑`);
summed over random `a_j` the dominant component is **radial** (`+h_i`, inflate the
norm), which `normalize` then projects out.

> **SIGReg directly on unit-norm `h` is ill-posed and *inert*, not collapse-inducing**
> (earlier note overstated this). Its useful gradient is radial and the sphere deletes
> it; the weak tangential residual is, if anything, mildly anti-collapse. Corrected
> prediction for ablation #13: it behaves like **≈ no regularizer** (collapses like the
> no-reg variant F), while projector-SIGReg(z) actually prevents collapse — so "#13
> hurts" only *relative to G*.

The fix is the projector. `z = Wh` with `‖W‖ ~ √d` **amplifies** the unit shell
(per-dir var `1/d`) to the Gaussian shell (per-dir var `~1`, `‖z‖≈√d`), so the
projections genuinely *can* be `N(0,1)`: the gradient is no longer a wasted radial term
and does real isotropic shaping in z, which backprops (`Wᵀ∂L/∂z`) to spread `h`
angularly. (The sphere image is intrinsically `(d−1)`-dim so `z` can't be a *perfect*
Gaussian, but in high `d` the missing dimension is negligible for the projection test —
the **scale**, not the dimension, is the fix.)

**Design fork this exposes.** Two principled regularizers: (1) project + standard
SIGReg(z) [current variant G]; (2) keep `h` but target the **uniform-sphere** law
`N(0, 1/d)` (or standardize projections first) — a scale-free "sphere-native SIGReg"
that needs no projector. Ablation (1) vs (2): if (2) matches G, the projector is just
rescaling; if G wins, the projector's extra capacity is doing real work.

## 3. DINO vs LeWM — anti-collapse as a moment-order spectrum

Both prevent collapse on the sphere, by controlling different moments of the angular
distribution:

| mechanism | what it controls | moment order | isotropy | alignment exp. `a` |
|---|---|---|---|---|
| **DINO centering** | mean direction (kill 1st moment) | 1 | light | high (collapse-risky) |
| pairwise-cosine / VICReg | off-diagonal covariance | 2 | medium | medium |
| **LeWM SIGReg(z)** | full distribution → isotropic Gaussian | all | strong | low (collapse-safe) |

DINO also adds **stop-grad + EMA teacher** (asymmetry); LeWM is **symmetric**
(no stop-grad). Two consequences:

1. **The anti-collapse mechanisms form an ordered spectrum by moment order**, and
   (from §6) higher moment order ⇒ more isotropy ⇒ *lower* alignment exponent ⇒ worse
   width-transfer but safer from collapse. **The optimum is interior** — predicts an
   inverted-U in rollout quality vs isotropy (sweep SIGReg-on-z weight to find the knee).
2. **Stop-grad matters more on the sphere than it did for Euclidean LeWM.** Normalizing
   `h` removes the radial DOF SIGReg used, shifting the entire anti-collapse burden onto
   the (weaker) angular channel. So a DINO-style asymmetric spherical JEPA may be
   materially more robust than the symmetric LeWM-style one. **Ablation:** spherical-
   symmetric (no stop-grad) vs spherical-asymmetric (stop-grad + predictor), and
   centering vs pairwise-cosine vs SIGReg(z) as the anti-collapse.

## 4. The gate solves a react-vs-drift impossibility

Realized step angle `ψ_t ≈ g_t·φ_t`; total drift `D_H = Σψ_t`. Want both
`D_H = O(1)` (long-horizon stability) and `max_t ψ_t = Ω(1)` (full reaction to a
surprise). A **state-independent** α=c forces `c=O(1/H)` ⇒ kills reactivity. A
**state-dependent** gate satisfies both **iff surprise is sparse**: `O(1)` full-size
jumps + a tail of `O(1/H)` steps ⇒ `D_H=O(1)` and `max ψ=O(1)`.

> **Predicts the gate's advantage over fixed-α (F vs E) is largest in punctuated
> environments** (Two-Room room-transitions, Push-T contacts) and smallest in smooth
> control (Reacher). Test: histogram learned `g_t` against event times (probe wired in).

This is the depth↔horizon transposition of νGPT's `α_init ∝ 1/depth`: the rollout
horizon is the world model's effective depth.

## 5. LERP / residual / value-residual interactions

The nGPT LERP `h_new = normalize(h + α·b)` is a retraction; the gate is a per-state,
per-channel `α`. Interaction taxonomy:

- **Score-residual** (RealFormer: residual on attention *logits*): acts in logit
  space, **orthogonal** to nGPT (which acts on activations) → composes cleanly. Low
  value in a shallow (6-layer) predictor.
- **Value-residual** (inject layer-0 value into each block): acts on the residual
  *stream* → competes with the LERP budget: `normalize(h + α·block + β·v₀)`; `α,β`
  co-tuned. Helps depth; predictor too shallow to need it.
- **Recurrent / goal anchor** (this work): transpose the value-residual from the depth
  axis to the **horizon** axis:
  ```
  h_{t+1} = normalize((1-g)·h_t + g·u + β·h_anchor)
  ```
  anchor = window-start (implemented, `predictor.anchor_beta`) or the **goal latent**
  (deeper variant, needs planner threading). This anchors long rollouts and is the
  WM-native residual. **General principle: transpose depth-axis transformer tricks to
  the horizon axis** (stochastic-depth → stochastic-rollout, layerwise-LR → horizon-
  wise gate schedule, …).
- **Muon / spectral optimizers**: equalize the update's *singular values*; nGPT
  equalizes weight *row-norms*; Adam equalizes *gradient scale*. Stacking all three is
  the "third normalization → over-correct" risk — **do not** bolt Muon onto nGPT+Adam
  without watching effective step size.
- **qk-norm**: local spherical normalization inside attention — in-spirit, synergistic,
  stabilizes attention over long action sequences. **RoPE**: orthogonal.

### → Recurrent-anchor research direction (status: implemented, sweeping)
`+experiment=gated_spherical_anchor model.predictor.anchor_beta={0,0.05,0.1,0.25}`.
Hypotheses: (a) small β reduces long-horizon rollout drift (lower `roll_err_20`,
lower angular drift) without hurting 1-step; (b) a *goal* anchor (next milestone)
improves goal-conditioned planning specifically, by pulling rollouts toward the goal
cap. Deeper build: thread `goal_emb` into the predictor as the anchor and re-test
planning success at fixed CEM budget.

## 6. The alignment exponent is objective-dependent (νGPT extension)

`‖ΔW·h‖` is set by activation clustering: lower effective rank ⇒ larger typical
overlap `|vᵀh|` ⇒ higher alignment exponent `a` (νGPT measures `a=3/4` on the
hypersphere; `1/2`=random, `1`=μP). The JEPA cosine loss **rewards clustering**
(`cos=1` at collapse), pushing `r_eff` *below* an LM's (whose CE forbids collapse).
Therefore:

> **A JEPA transition's alignment exponent is higher than an LM's 3/4** (toward μP's
> 1), so its correct width-LR exponent is **steeper than `d^{-3/4}`** — *measure it,
> don't port it from LMs.* And one scalar (clustering ≈ `1 − r_eff/d`) sets the
> alignment exponent, the width-LR exponent, AND the collapse margin simultaneously.

Corrected note: per-group **LR** (νGPT prescription) **composes** with nGPT+Adam — not
redundant. The redundant move is a *third gradient-magnitude* normalization
(Fisher-adaptive). So `+experiment=ngpt_lr_groups` is a legitimate test.

## 7. What's measurable on the current run (probes wired in)

- training logs: `probe_gate_mean/std/frac_active`, `probe_step_angle_mean`, `emb_norm`.
- `eval_latent_metrics.py` writes `results/per_step_metrics.csv`:
  `roll_err(k), eff_rank(k), clumping(k), step_angle(k), gate_mean(k)`.
- `make_plots.py` → `plots/mechanism_curves.png` (gate(k), r_eff(k), ψ(k)).

The two "killer" plots: **gate `g_t` vs environment events** (sparse spikes ⇒ §4) and
**`r_eff` vs rollout quality** (inverted-U ⇒ §2/§3).

## 8. Paper outline — "Spherical World Models: where nGPT geometry pays off in JEPA"

**Thesis.** Under Adam the only non-redundant nGPT contribution is the geometric
constraint; an LM can't use it (softmax decodes it) but a JEPA *is* it. The value
concentrates on the recurrent latent, and shows up exactly where the mechanism
predicts (long-horizon, sparse-event), not uniformly.

**Contributions.**
1. The recurrent-latent principle (§0) — a structural site LMs lack.
2. Riemannian-GD identity (§1) and the SIGReg-on-h scale-mismatch / well-posedness
   result + the sphere-native-SIGReg fork (§2).
3. Anti-collapse moment-order spectrum unifying DINO and LeWM (§3); the interior
   isotropy optimum.
4. The gate as the unique solver of react-vs-drift for sparse-event dynamics (§4),
   with the depth↔horizon transposition (§5).
5. The objective-dependent alignment exponent (§6): JEPA transitions sit above the
   LM's 3/4; a measurement + corrected width-LR prescription.

**Empirical backbone.** Matched-compute vs **official LeWM** on its own benchmarks +
the ablation table; the two killer plots. *A non-uniform result is the strong result*:
gated-spherical winning on Two-Room/long-horizon but tying on Push-T short-horizon
**confirms** the mechanism more convincingly than a uniform win would.

**Decision rules (unchanged).** Better retrieval/rank alone = representation win, not
a world-model win. Report negatives (simple-spherical collapse, direct-SIGReg-on-h
behaving like no-reg) — they are *predicted* and strengthen the mechanism story.

## References
- Loshchilov et al. (2025), *nGPT*, ICLR 2025, arXiv:2410.01131.
- Shigida, Hanin & Gromov (2026), *Learning Rate Transfer in Normalized Transformers*, arXiv:2604.27077.
- Maes, Le Lidec et al. (2026), *LeWorldModel*, arXiv:2603.19312.
