# spinangle — gated spherical nGPT-JEPA vs. official LeWM

Research harness testing whether a **gated spherical** nGPT-JEPA latent transition
beats **official LeWM** (LeWorldModel, Maes et al. 2026) on its *own* benchmarks —
Push-T, Reacher, Two-Room, OGBench-Cube — under matched compute and the same planner.

> **The baseline is official LeWM, vendored unchanged** from
> [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm) (commit pinned in
> `UPSTREAM_COMMIT.txt`). All nGPT-JEPA changes are surgical and behind config flags;
> with defaults the code reproduces LeWM exactly. No toy "LeWM-like" substitute is
> used as the baseline.

## Idea
LeWM keeps latent states **Euclidean** and prevents collapse with **SIGReg**
(isotropic-Gaussian regularization). nGPT-JEPA instead puts the world-model state on
the **unit sphere** and learns transitions as **gated angular updates**:

```
h_t      = normalize(encoder(obs))            # unit latent
u        = normalize(UpdateNet(h_t, a_t))     # proposed direction
g        = sigmoid(GateNet(h_t, a_t))         # how far to move
h_pred   = normalize((1 - g) * h_t + g * u)   # smooth move along the sphere
loss     = 1 - <h_pred, h_target>             # angular prediction loss
```

The bet: smooth angular updates give more stable long-horizon latent rollouts and
better goal/future retrieval, which should translate into **better or cheaper
planning** — the only thing that counts as beating LeWM.

## Where things are
| path | what |
|---|---|
| `jepa.py`, `module.py`, `eval.py`, `utils.py` | vendored LeWM (flag-gated edits only) |
| `objective.py` | variant-aware training loss (torch-only, CPU-testable) |
| `variants.py` | spherical predictors, SIGReg projector, memory NCE |
| `train.py` | LeWM trainer (imports `objective.lejepa_forward`) |
| `config/train/model/*` | per-variant model configs (predictor + norm flags) |
| `config/train/experiment/*` | `+experiment=<variant>` = model + loss in one flag |
| `config/eval/*` | LeWM eval/planner configs (**unchanged**) |
| `metrics.py` | rollout / retrieval / representation metrics (pure, tested) |
| `scripts/` | `eval_latent_metrics.py`, `make_plots.py`, `log_run.py` |
| `smoke_test.py` | CPU validation of every variant (no GPU/data needed) |
| `official_lewm_reproduction.md` | Phase-1 reproduction protocol + upstream pin |
| `RUN_MATRIX.md` | exact train/eval commands for all variants × benchmarks |
| `report.md` | final report skeleton (answers the 9 mandatory questions) |
| `notebooks/colab_runner.ipynb` | one-click Colab driver (GPU) |

## Workflow (this repo is the harness; GPU runs on Colab)
1. **Locally / CI (CPU):** `python smoke_test.py && python metrics.py` — validates all
   model variants and metrics with no data or GPU.
2. **Colab (GPU):** open `notebooks/colab_runner.ipynb` (or run `bootstrap.sh`),
   install `stable-worldmodel[train,env]`, download official data/checkpoints, then
   follow `official_lewm_reproduction.md` (Phase 1) and `RUN_MATRIX.md` (Phases 2-7).
3. Results accumulate in `results/all_runs.csv`; `scripts/make_plots.py` regenerates
   `plots/*.png`; fill `report.md`.

## CPU sanity check
```bash
pip install torch einops numpy            # CPU is fine
python smoke_test.py                       # all variants: fwd/bwd + unit-norm + planner
python metrics.py                          # metric self-checks
```

## Status
Harness complete and CPU-validated. GPU reproduction + variant runs pending (see
`report.md`). Variant claims are made only against official LeWM under identical
dataset / eval / training / planner budgets.

## Credit
Built on official LeWM (`lucas-maes/le-wm`, arXiv 2603.19312) and
`stable-worldmodel` / `stable-pretraining` (galilai-group). See `UPSTREAM_README.md`
and `LICENSE`.
