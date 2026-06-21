# nGPT-JEPA vs. Official LeWM — Report

> Status: **harness complete & CPU-validated; GPU runs pending.** This skeleton is
> filled in from `results/all_runs.csv` + `plots/` as Colab runs complete. No result
> is reported as "beating LeWM" unless it is measured against **official LeWM** under
> the same dataset, eval protocol, training budget, and planner budget.

## Hypothesis
A **gated spherical** nGPT-JEPA transition — `u = normalize(Update(h_t,a_t))`,
`g = sigmoid(Gate(h_t,a_t))`, `h_pred = normalize((1−g)·h_t + g·u)` — improves latent
rollout stability, retrieval, and **planning success / efficiency** over LeWM's
Euclidean/SIGReg latent world model under matched compute.

## Method ladder (config flag → variant)
| # | variant | run | predictor | latent | reg |
|---|---|---|---|---|---|
| A | official LeWM | `+experiment=official_lewm` | ARPredictor | Euclidean | SIGReg(emb) |
| B | LeWM −SIGReg | `+experiment=lewm_nosigreg` | ARPredictor | Euclidean | none |
| C | non-norm MSE JEPA | `+experiment=lewm_nosigreg` | ARPredictor | Euclidean | none |
| D | simple spherical | `+experiment=simple_spherical` | `normalize(U)` | sphere | none |
| E | full-ish residual | `+experiment=fullish_residual` | `norm(h+α·norm(U))` | sphere | none |
| F | **gated spherical** | `+experiment=gated_spherical` | gated | sphere | none |
| G | gated + proj-SIGReg | `+experiment=gated_spherical_projector_sigreg` | gated | sphere | SIGReg(z=Proj(h)) |
| H | gated + memory NCE | `+experiment=gated_spherical_memory` | gated | sphere | +NCE |
| I | gated + selective SSM | `+experiment=gated_spherical_ssm` | keep/write SSM | sphere | none |

Ablations: `gated_direct_sigreg_h` (#13, SIGReg on h), `gated_spherical_anticollapse`,
no-gate=E, no-norm=B/C, no-memory=F, global-LR vs νGPT-LR (Phase 7).

## Headline comparison (per benchmark — fill from results/all_runs.csv)
Push-T (first target):

| variant | success ↑ | return ↑ | final dist ↓ | plan latency ↓ | roll_err@20 ↓ | fut r@1 ↑ | eff.rank | clumping | emb‖·‖ |
|---|---|---|---|---|---|---|---|---|---|
| official LeWM (A) |  |  |  |  |  |  |  |  | ~? |
| non-norm MSE (C) |  |  |  |  |  |  |  |  |  |
| simple spherical (D) |  |  |  |  |  |  |  |  | 1.00 |
| full-ish residual (E) |  |  |  |  |  |  |  |  | 1.00 |
| **gated spherical (F)** |  |  |  |  |  |  |  |  | 1.00 |
| gated + proj-SIGReg (G) |  |  |  |  |  |  |  |  | 1.00 |
| gated + memory (H) |  |  |  |  |  |  |  |  | 1.00 |
| gated + SSM (I) |  |  |  |  |  |  |  |  | 1.00 |

Repeat for Reacher, Two-Room (LeWM may be weaker; long-horizon may favour stable
spherical goals), and OGBench-Cube if feasible.

Plots: `success_vs_steps`, `rollout_error_vs_horizon`, `retrieval_vs_steps`,
`rank_clumping`, `planning_budget_curve`.

## Decision rules applied
A method beats LeWM **only** if it improves official task success or planning
efficiency. Better retrieval/rank/clumping alone ⇒ "representation win", not a
world-model win. If prediction cosine improves but control worsens ⇒ failure. If
non-norm MSE wins ⇒ report it. If D loses but F wins ⇒ the contribution is gated
spherical *dynamics*, not mere normalization.

## Final answers (to complete)
1. **Did official LeWM reproduce?** — _pending Phase 1 (see official_lewm_reproduction.md)._
2. **Did simple spherical JEPA (D) beat it?** — _pending._
3. **Did gated spherical (F) beat it?** — _pending._
4. **Did projector SIGReg (G) help?** — _pending._
5. **Did memory (H) help, and did retrieval gains transfer to planning?** — _pending._
6. **Did SSM (I) help only in the temporal/action transition?** — _pending._
7. **Did any latent improvement translate to control success?** — _pending._
8. **What failed?** — _pending._
9. **Real LeWM win or only a proxy win?** — _pending._

## Final target claim (verdict)
> _Gated spherical nGPT-JEPA improves LeWM's latent transition geometry and yields
> better or more efficient planning on at least one official LeWM benchmark under
> matched compute._ — **VERDICT: pending GPU results.**
