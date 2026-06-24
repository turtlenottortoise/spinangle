#!/usr/bin/env python3
"""Offline latent-world-model metrics for a trained LeWM / nGPT-JEPA checkpoint.

Computes the brief's *latent-rollout*, *retrieval*, and *representation* metrics
(planning/control metrics come from the unchanged ``eval.py``). It mirrors the data
loading of ``train.py`` so sequences are sampled identically, then uses the model's
own ``encode`` + an autoregressive rollout and the pure functions in ``metrics.py``.

This script needs the full stable-worldmodel / stable-pretraining stack and a GPU,
so it runs on Colab (not in the CPU smoke container). Dataset key names differ per
benchmark; the proprio/state probe is skipped automatically when unavailable.

Example
-------
python scripts/eval_latent_metrics.py \
    --policy pusht/gated_spherical --data pusht --benchmark pusht \
    --variant gated_spherical --spherical --horizon 20 --num_batches 16
"""
import argparse
from functools import partial

import hydra
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf, open_dict

import stable_pretraining as spt  # noqa: F401  (registers backbones used by ckpt)
import stable_worldmodel as swm
import torch.nn.functional as F
from pathlib import Path

import metrics as M
from utils import get_column_normalizer, get_img_preprocessor

ROOT_RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_data(data_name, history, horizon, img_size, frameskip=5, batch_size=16):
    """Load the eval dataset with sequences long enough for the rollout horizon,
    using the same transforms as train.py."""
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name="lewm", overrides=[f"data={data_name}"])
    dcfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dcfg.pop("name")
    dcfg["num_steps"] = history + horizon          # longer sequences for rollout
    dcfg["frameskip"] = frameskip
    dataset = swm.data.load_dataset(name, transform=None, **dcfg)

    tfs = [get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        tfs.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*tfs)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       drop_last=True)


@torch.no_grad()
def rollout_latents(model, pixels, actions, history, horizon):
    """Autoregressive free-running rollout (mirrors JEPA.rollout, single sample).

    pixels: (B, H+K, C, H, W), actions: (B, H+K, A). Returns
    (pred, true) latent trajectories of shape (B, K, D) aligned by horizon.
    """
    info = {"pixels": pixels[:, :history], "action": actions[:, :history]}
    info = model.encode(info)
    emb = info["emb"]                                   # (B, H, D)
    act_emb = model.action_encoder(actions)            # (B, H+K, D)

    HS = history
    preds, gates = [], []
    cur = emb
    for k in range(horizon):
        a = act_emb[:, : history + k]
        e_trunc = cur[:, -HS:]
        a_trunc = a[:, -HS:]
        p = model.predict(e_trunc, a_trunc)[:, -1:]    # (B, 1, D)
        preds.append(p)
        cur = torch.cat([cur, p], dim=1)
        probe = getattr(getattr(model, "predictor", None), "probe", {}) or {}
        gates.append(probe.get("gate_mean", float("nan")))   # mechanism probe
    pred = torch.cat(preds, dim=1)                     # (B, K, D)

    true = model.encode({"pixels": pixels, "action": actions})["emb"][:, history:]
    return pred, true[:, :horizon], gates


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="ckpt name relative to STABLEWM_HOME")
    ap.add_argument("--data", required=True, help="train data config name (pusht/dmc/...)")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--spherical", action="store_true", help="use cosine/angular metrics")
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    loader = load_data(args.data, args.history, args.horizon, args.img_size,
                       batch_size=args.batch_size)

    all_pred, all_true, all_h, all_state, all_gates = [], [], [], [], []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        pixels = batch["pixels"].float().to(device)
        actions = torch.nan_to_num(batch["action"].float(), 0.0).to(device)
        pred, true, gates = rollout_latents(model, pixels, actions, args.history, args.horizon)
        all_gates.append(gates)
        all_pred.append(pred.cpu())
        all_true.append(true.cpu())
        all_h.append(true.reshape(-1, true.size(-1)).cpu())
        for key in ("state", "proprio"):
            if key in batch:
                all_state.append(batch[key][:, args.history:args.history + args.horizon]
                                 .reshape(-1, batch[key].size(-1)).float())
                break

    pred = torch.cat(all_pred)            # (N, K, D)
    true = torch.cat(all_true)            # (N, K, D)
    H = torch.cat(all_h)                  # (N*K, D) encoded latents

    sph = args.spherical
    # ---- per-step mechanism curves: r_eff(k), step-angle(k), gate(k), err(k) ----
    # These are the plots the theory lives on (r_eff-vs-horizon inverted-U;
    # gate-vs-step sparse-event spikes; angular drift accumulation).
    import csv as _csv
    import numpy as _np
    gate_curve = _np.nanmean(_np.array(all_gates, dtype=float), axis=0) \
        if all_gates and len(all_gates[0]) else None
    per_step_path = ROOT_RESULTS / "per_step_metrics.csv"
    new = not per_step_path.exists()
    with per_step_path.open("a", newline="") as f:
        w = _csv.writer(f)
        if new:
            w.writerow(["variant", "benchmark", "step", "roll_err",
                        "eff_rank", "clumping", "step_angle", "gate_mean"])
        for k in range(pred.size(1)):
            cloud = pred[:, k]
            if k == 0:
                ang = 0.0
            else:
                a = F.normalize(pred[:, k - 1], dim=-1); b = F.normalize(pred[:, k], dim=-1)
                ang = (a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos().mean().item()
            err = M.rollout_errors(pred, true, horizons=(k + 1,), spherical=sph).get(k + 1)
            gate = float(gate_curve[k]) if gate_curve is not None else ""
            w.writerow([args.variant, args.benchmark, k + 1, err,
                        M.effective_rank(cloud), M.mean_pairwise_cosine(cloud), ang, gate])
    print(f"wrote per-step curves -> {per_step_path}")
    re = M.rollout_errors(pred, true, horizons=(1, 5, 10, 20), spherical=sph)
    # future-state retrieval: predicted final latent must retrieve the true final
    fut = M.retrieval_metrics(pred[:, -1], true[:, -1])
    hard = {n: M.retrieval_metrics(pred[:, -1], true[:, -1], num_candidates=n)["r@1"]
            for n in (32, 128, 256)}

    out = {
        "variant": args.variant, "benchmark": args.benchmark, "phase": "latent",
        "seed": args.seed, "ckpt_path": args.policy, "spherical": int(sph),
        "roll_err_1": re.get(1), "roll_err_5": re.get(5),
        "roll_err_10": re.get(10), "roll_err_20": re.get(20),
        "rollout_drift": M.rollout_drift(pred, true, spherical=sph),
        "norm_drift": M.norm_drift(pred), "angular_drift": M.angular_drift(pred),
        "fut_r1": fut["r@1"], "fut_r5": fut["r@5"], "mrr": fut["mrr"],
        "retr_hard32_r1": hard[32], "retr_hard128_r1": hard[128],
        "retr_hard256_r1": hard[256],
        "eff_rank": M.effective_rank(H), "clumping": M.mean_pairwise_cosine(H),
        "emb_norm_mean": H.norm(dim=-1).mean().item(),
    }
    if all_state:
        S = torch.cat(all_state)
        # discretize continuous state into bins for a kNN probe sanity signal
        labels = (S[:, 0] > S[:, 0].median()).long() if S.size(1) else None
        if labels is not None:
            out["knn_probe"] = M.knn_probe(H, labels)

    print("== latent metrics ==")
    for k, v in out.items():
        print(f"  {k}: {v}")

    # append to results/all_runs.csv via the shared logger
    from log_run import main as log_main
    argv = []
    for k, v in out.items():
        if v is not None:
            argv += [f"--{k}", str(v)]
    log_main(argv)


if __name__ == "__main__":
    main()
