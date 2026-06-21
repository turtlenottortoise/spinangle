# Official LeWM Reproduction Protocol

This document pins the official baseline and gives the exact commands to reproduce
it **before** any nGPT-JEPA modification. The opponent in every comparison is
*official LeWM* — not a toy substitute.

## Upstream pin

| | |
|---|---|
| Repo | https://github.com/lucas-maes/le-wm (official code base) |
| Paper | LeWorldModel: *Stable End-to-End JEPA from Pixels*, Maes, Le Lidec, Scieur, LeCun, Balestriero (arXiv 2603.19312) |
| Commit | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (see `UPSTREAM_COMMIT.txt`) |
| Checkpoints / data | https://huggingface.co/collections/quentinll/lewm |

The official source files (`jepa.py`, `module.py`, `train.py`, `eval.py`,
`utils.py`, `config/`) are vendored at the repo root. Our changes are **surgical and
flag-gated**; with the default config the code is behaviourally identical to upstream
(verified by `smoke_test.py::reference_official_matches`). The original upstream
README is preserved as `UPSTREAM_README.md`.

### What we changed vs. upstream (and why it is safe)
| File | Change | Official behaviour when default? |
|---|---|---|
| `jepa.py` | `JEPA.__init__` gains `normalize_emb`, `normalize_pred`, `sigreg_projector` (all default off/None); `encode`/`predict` optionally L2-normalize | **Yes** — defaults False ⇒ Euclidean latents, byte-identical math |
| `train.py` | `lejepa_forward` moved to `objective.py` and made flag-aware | **Yes** — `loss.pred.type=mse`, `loss.sigreg.target=emb` ⇒ `MSE + 0.09·SIGReg(emb)` |
| `objective.py` | **new** — variant-aware objective (torch-only, CPU-testable) | n/a |
| `variants.py` | **new** — spherical predictors, SIGReg projector, memory NCE | n/a |
| `config/train/*` | added `loss.pred/anticollapse/memory` keys + variant model/experiment configs | **Yes** — official defaults declared explicitly |
| `eval.py`, planner, data loading | **unchanged** | **Yes** |

The planner cost (`JEPA.criterion`) is **unchanged**: for unit-norm latents,
MSE = 2(1 − cos), which is monotone in cosine, so CEM's top-k ranking is identical
whether the latent is Euclidean or spherical. This is why spherical variants need
**no** change to `eval.py` or the solver.

## Environment (Colab GPU)

```bash
# Python 3.10 venv per upstream
uv venv --python=3.10 && source .venv/bin/activate
uv pip install "stable-worldmodel[train,env]"
# (the repo builds on stable-worldmodel for envs/planning/eval and
#  stable-pretraining for training; both are pulled in by the extra)
```

`bootstrap.sh` automates this; `notebooks/colab_runner.ipynb` runs it on Colab.

## Data

Datasets are HDF5, downloaded from the HuggingFace collection and extracted under
`$STABLEWM_HOME` (default `~/.stable-wm/`):

```bash
export STABLEWM_HOME=/content/stable-wm           # or any persistent path
tar --zstd -xvf <archive>.tar.zst -C $STABLEWM_HOME
```

| Benchmark | train data (config) | eval dataset |
|---|---|---|
| Push-T   | `pusht_expert_train` (`data=pusht`)   | `pusht_expert_train` |
| Reacher  | `dmc/reacher_random` (`data=dmc`)     | `dmc/reacher_random` |
| Two-Room | `tworoom` (`data=tworoom`)            | `tworoom` |
| OGB-Cube | `ogbench/cube_single_expert` (`data=ogb`) | `ogbench/cube_single_expert` |

## Phase 1 — reproduce official LeWM

**1a. Pretrained-checkpoint evaluation** (fastest reproduction; no training).
Convert the HF mirror to the object checkpoint `eval.py` expects (snippet in
`UPSTREAM_README.md`), then:

```bash
python eval.py --config-name=pusht.yaml   policy=pusht/lewm
python eval.py --config-name=reacher.yaml policy=reacher/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
python eval.py --config-name=cube.yaml    policy=cube/lewm
```

Record `metrics` and `evaluation_time` from the printed results / `*_results.txt`
into `results/all_runs.csv` (use `scripts/log_run.py`).

**1b. Short sanity training run** (verify the training loop end-to-end):

```bash
python train.py +experiment=official_lewm data=pusht \
    trainer.max_epochs=2 wandb.enabled=false
```

**1c. Full-budget training** at the official setting (100 epochs, bf16, AdamW
lr 5e-5, batch 128 — see `config/train/lewm.yaml`), then evaluate the produced
checkpoint with the **same** `eval.py` command as 1a.

A reproduction is considered successful when 1a (pretrained) and 1c (our retrain)
land within run-to-run noise of the paper's reported success/return for at least
Push-T or Reacher. Fill the table below.

### Reproduction results (to complete on GPU)

| Benchmark | source | success ↑ | return ↑ | final dist ↓ | plan latency | reproduced? |
|---|---|---|---|---|---|---|
| Push-T | paper |  |  |  |  | — |
| Push-T | pretrained ckpt (1a) |  |  |  |  |  |
| Push-T | our retrain (1c) |  |  |  |  |  |
| Reacher | pretrained ckpt (1a) |  |  |  |  |  |

> **Gate:** do not proceed to Phase 2+ comparisons until at least one official
> benchmark reproduces. All variant claims are made only against the numbers in
> this table, under the identical dataset / eval protocol / training budget /
> planner budget.
