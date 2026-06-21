# Run matrix — exact commands (Colab GPU)

Every run uses LeWM's **own** `train.py` / `eval.py` and the **same** planner budget
(CEM: 300 samples, 30 steps, top-k 30, horizon 5, action_block 5, eval_budget 50,
num_eval 50). Only `+experiment=` and `data=` change.

## Download official data + checkpoints
```bash
export STABLEWM_HOME=/content/stable-wm
# tworoom is the lightest dataset (3.4G) -> do this first to validate the full loop
python scripts/download_assets.py --benchmark tworoom --data --ckpt
# others: pusht (13G), reacher (24G), cube (46G) -- ckpt-only is small if disk is tight
python scripts/download_assets.py --benchmark pusht --ckpt
```
`--ckpt` rebuilds the official weights into `$STABLEWM_HOME/<bench>/lewm_object.ckpt`
(eval with `policy=<bench>/lewm`). Datasets land as `.h5` under `$STABLEWM_HOME`.

## Training (per variant)
```bash
# benchmark in {pusht, dmc(=reacher), tworoom, ogb(=cube)}
V=gated_spherical          # any experiment name below
DATA=pusht
python train.py +experiment=$V data=$DATA wandb.enabled=false
#   short sanity: append  trainer.max_epochs=2
#   matched budget: default 100 epochs
```

Experiments: `official_lewm`, `lewm_nosigreg`, `simple_spherical`,
`fullish_residual`, `gated_spherical`, `gated_spherical_projector_sigreg`,
`gated_spherical_memory`, `gated_spherical_ssm`, `gated_direct_sigreg_h`,
`gated_spherical_anticollapse`.

Checkpoints land in `$STABLEWM_HOME` under `output_model_name` (set per experiment),
e.g. `pusht/gated_spherical`. Move/symlink so eval's `policy=<benchmark>/<variant>`
resolves.

## Planning evaluation (LeWM's eval.py, unchanged)
```bash
python eval.py --config-name=pusht.yaml   policy=pusht/gated_spherical
python eval.py --config-name=reacher.yaml policy=reacher/gated_spherical
python eval.py --config-name=tworoom.yaml policy=tworoom/gated_spherical
python eval.py --config-name=cube.yaml    policy=cube/gated_spherical
# then log into results/all_runs.csv:
python scripts/log_run.py --variant gated_spherical --benchmark pusht --phase 3 \
    --success <S> --return <R> --plan_latency_s <T> --plan_samples 300 --train_epochs 100
```

## Offline latent / retrieval / representation metrics
```bash
python scripts/eval_latent_metrics.py --policy pusht/gated_spherical \
    --data pusht --benchmark pusht --variant gated_spherical --spherical \
    --horizon 20 --num_batches 16
```
(omit `--spherical` for official/Euclidean variants). Appends a row to
`results/all_runs.csv` automatically.

## Plots / report
```bash
python scripts/make_plots.py        # -> plots/*.png
# edit report.md tables from results/all_runs.csv
```

## Planning-budget curve (Phase: planner efficiency)
Sweep CEM samples to compare success-at-equal-compute:
```bash
for N in 50 100 300 600; do
  python eval.py --config-name=pusht.yaml policy=pusht/gated_spherical \
      solver.num_samples=$N
done
```

## Phase 7 — νGPT-style LR / scaling
Three points to compare (stability, speed, final performance):
```bash
python train.py +experiment=gated_spherical data=pusht                 # global LR (baseline)
python train.py +experiment=ngpt_lr         data=pusht                 # global high LR + reduced warmup + no-WD-on-norm
python train.py +experiment=ngpt_lr_groups  data=pusht                 # per-group LR (higher LR on spherical transition nets)
```
`ngpt_lr_groups` assigns params to optimizer groups by **regex on module names**
(spt-native); verify the match on the first run via the printed optimizer groups.
`ngpt_lr`'s `scheduler.warmup_epochs` uses the pl_bolts signature — adjust if spt's
`LinearWarmupCosineAnnealingLR` differs.

## Suggested order
1. Phase 1 reproduce official (`official_lewm`, pretrained ckpt + retrain) — **gate**.
2. Phase 2 `simple_spherical` vs `lewm_nosigreg` vs `official_lewm` on Push-T.
3. Phase 3 `gated_spherical` (+ `fullish_residual` as no-gate ablation).
4. Phase 4 `gated_spherical_projector_sigreg` vs `gated_direct_sigreg_h`.
5. Phase 5 `gated_spherical_memory`. 6. Phase 6 `gated_spherical_ssm`.
7. Then Two-Room, then OGBench-Cube.
