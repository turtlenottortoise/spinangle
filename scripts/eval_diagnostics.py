#!/usr/bin/env python3
"""Behavioral diagnostics: what does a contender do *differently* from LeWM?

Three geometry-fair characterizations for one checkpoint, dumped to
``results/diag_<variant>_<benchmark>.json`` (+ a per-step curve CSV):

  1. controllability  - holding the state fixed, how far does the predicted
     next-latent *direction* move when the action changes (angular radians),
     measured against how far it moves across *different states*. A ~0 action
     fraction means the model ignores actions => unplannable regardless of how
     good its retrieval looks. This is the single most planning-decisive signal
     and is absent from eval_latent_metrics.py.
  2. long-horizon     - retrieval@1, effective rank, clumping, latent norm and
     per-step angle as a function of rollout step k (the stability curve; the
     latent table only saw horizon ~5).
  3. state decodability - R^2 of a ridge-linear map latent -> proprio (a graded
     signal, unlike the binary median-split kNN probe that saturates at ~0.99).

Same data pipeline + rollout as ``eval_latent_metrics.py`` (imported), so the
sequences match train.py. Needs the full stack + GPU (run on Colab).

Example
-------
python scripts/eval_diagnostics.py --policy tworoom/tangent_spherical_sigreg_l4_single \
    --data tworoom --benchmark tworoom --variant tangent_spherical_sigreg \
    --spherical --horizon 20 --frameskip 1
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import stable_pretraining as spt  # noqa: F401  (registers backbones used by ckpt)
import stable_worldmodel as swm

import metrics as M
from eval_latent_metrics import load_data, rollout_latents

ROOT_RESULTS = Path(__file__).resolve().parent.parent / "results"


def _ang_spread(P):
    """Mean angular deviation (rad) of a stack P=(G,B,D) around its per-column
    (per-state) mean direction. Geometry-fair: normalizes before comparing."""
    Pn = F.normalize(P, dim=-1)
    mean_dir = F.normalize(Pn.mean(0), dim=-1)            # (B, D)
    cos = (Pn * mean_dir.unsqueeze(0)).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return cos.arccos().mean().item()


@torch.no_grad()
def controllability(model, pixels, actions, history, n_perturb=8):
    """Does the 1-step prediction respond to the action?

    Holds the encoded history fixed and swaps the most-recent action for other
    samples' actions, measuring the angular spread of the predicted next-latent
    direction (act_rad). Compares it to the spread of predictions across
    different states (state_rad) so the ratio is the fraction of the model's
    dynamic range that the action actually controls.
    """
    info = model.encode({"pixels": pixels[:, :history], "action": actions[:, :history]})
    emb = info["emb"]                                     # (B, H, D)
    act_emb = model.action_encoder(actions)              # (B, H+K, D)
    base_a = act_emb[:, :history].clone()
    p0 = model.predict(emb, base_a)[:, -1]               # (B, D) real-action pred

    B = emb.size(0)
    preds = [p0]
    for _ in range(n_perturb):
        perm = torch.randperm(B, device=emb.device)
        a = base_a.clone()
        a[:, -1] = act_emb[perm, history - 1]            # swap in another action
        preds.append(model.predict(emb, a)[:, -1])
    P = torch.stack(preds, 0)                             # (n_perturb+1, B, D)
    act_rad = _ang_spread(P)                              # action-induced (same state)

    Pn = F.normalize(p0, dim=-1)                          # across-state spread
    gmean = F.normalize(Pn.mean(0, keepdim=True), dim=-1)
    state_rad = (Pn * gmean).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos().mean().item()
    return act_rad, state_rad, act_rad / (state_rad + 1e-9)


def linear_probe_r2(H, S, ridge=1e-2, seed=0):
    """R^2 of a ridge-linear map H -> S on an 80/20 split, averaged over the
    proprio dimensions. Graded analogue of the binary kNN probe."""
    N = H.size(0)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g)
    ntr = max(1, int(0.8 * N))
    Xtr, Xte = H[idx[:ntr]], H[idx[ntr:]]
    Ytr, Yte = S[idx[:ntr]], S[idx[ntr:]]
    if Xte.size(0) < 2:
        return None
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xtr, Xte, Ytr, Yte = Xtr - xm, Xte - xm, Ytr - ym, Yte - ym
    D = Xtr.size(1)
    A = Xtr.t() @ Xtr + ridge * torch.eye(D, dtype=Xtr.dtype)
    W = torch.linalg.solve(A, Xtr.t() @ Ytr)
    pe = Xte @ W
    ss_res = (pe - Yte).pow(2).sum(0)
    ss_tot = Yte.pow(2).sum(0) + 1e-9
    return (1 - ss_res / ss_tot).mean().item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--frameskip", type=int, default=1,
                    help="lower => longer temporal reach per episode (default 1)")
    ap.add_argument("--num_batches", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_perturb", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    loader = load_data(args.data, args.history, args.horizon, 224,
                       frameskip=args.frameskip, batch_size=args.batch_size)

    sph = args.spherical
    all_pred, all_true, all_state, ctrl = [], [], [], []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        pixels = batch["pixels"].float().to(device)
        actions = torch.nan_to_num(batch["action"].float(), 0.0).to(device)
        pred, true, _ = rollout_latents(model, pixels, actions, args.history, args.horizon)
        all_pred.append(pred.cpu())
        all_true.append(true.cpu())
        ctrl.append(controllability(model, pixels, actions, args.history, args.n_perturb))
        for key in ("state", "proprio"):
            if key in batch:
                all_state.append(batch[key][:, args.history:args.history + args.horizon].float())
                break

    if not all_pred:
        import sys
        sys.exit(f"[diag] FATAL: loader yielded 0 batches at horizon={args.horizon} "
                 f"frameskip={args.frameskip}; lower --horizon and/or --frameskip.")

    pred = torch.cat(all_pred)                            # (N, K, D)
    true = torch.cat(all_true)
    K = pred.size(1)

    import numpy as np
    act_rad, state_rad, ratio = np.array(ctrl, dtype=float).mean(0).tolist()

    # ---- per-step long-horizon curves ----
    import csv
    curve_path = ROOT_RESULTS / f"diag_curve_{args.variant}_{args.benchmark}.csv"
    with curve_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "retr_r1", "roll_err", "eff_rank", "clumping",
                    "norm_mean", "step_angle"])
        for k in range(K):
            pk, tk = pred[:, k], true[:, k]
            retr = M.retrieval_metrics(pk, tk)["r@1"]
            err = M.rollout_errors(pred, true, horizons=(k + 1,), spherical=sph).get(k + 1)
            if k == 0:
                ang = 0.0
            else:
                a = F.normalize(pred[:, k - 1], dim=-1)
                b = F.normalize(pred[:, k], dim=-1)
                ang = (a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos().mean().item()
            w.writerow([k + 1, retr, err, M.effective_rank(pk),
                        M.mean_pairwise_cosine(pk), pk.norm(dim=-1).mean().item(), ang])

    # ---- linear state decodability (R^2 latent -> proprio) ----
    probe_r2 = None
    if all_state:
        S = torch.cat(all_state).reshape(-1, all_state[0].size(-1))
        Hflat = true.reshape(-1, true.size(-1))
        n = min(Hflat.size(0), S.size(0))
        if n > 10:
            probe_r2 = linear_probe_r2(Hflat[:n], S[:n])

    out = {
        "variant": args.variant, "benchmark": args.benchmark,
        "horizon": K, "frameskip": args.frameskip, "spherical": int(sph),
        # controllability (the planning-viability signal)
        "ctrl_action_rad": act_rad, "ctrl_state_rad": state_rad,
        "ctrl_action_frac": ratio,
        # long-horizon endpoints
        "retr_r1_first": M.retrieval_metrics(pred[:, 0], true[:, 0])["r@1"],
        "retr_r1_last": M.retrieval_metrics(pred[:, -1], true[:, -1])["r@1"],
        "clumping_first": M.mean_pairwise_cosine(pred[:, 0]),
        "clumping_last": M.mean_pairwise_cosine(pred[:, -1]),
        "norm_drift": M.norm_drift(pred), "angular_drift": M.angular_drift(pred),
        # graded state decodability
        "state_probe_r2": probe_r2,
    }
    sink = ROOT_RESULTS / f"diag_{args.variant}_{args.benchmark}.json"
    sink.write_text(json.dumps(out, indent=2))
    print("== diagnostics ==")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"wrote -> {sink}")
    print(f"wrote curve -> {curve_path}")


if __name__ == "__main__":
    main()
