#!/usr/bin/env python3
"""New anti-collapse MECHANISMS, each built to fix a measured failure.

Motivation (results/cpu_noreg_theorem_test.json): fixed-weight hinges LOSE the
economics tug-of-war -- the collapsed cap costs ~5e-3 in penalty but saves
~1.4e-2 in prediction loss, so Adam keeps the cap. Penalty-form regularization
(VICReg/SIGReg/simplex, all fixed-lambda) has this flaw by construction.

Mechanisms under test (rotation core, NO marginal regularizer):
  kin_fix    encoder-step hinge, fixed w=5      (known to fail -- baseline)
  kin_dual   SAME constraint, Lagrangian dual ascent: lam <- max(0, lam+eta*v).
             The multiplier GROWS until the constraint binds, so the collapsed
             solution cannot stay cheaper. Constraint-form anti-collapse;
             no fixed lambda anywhere. (Novel in SSL per search.)
  disp_fix   counterfactual ACTION-DISPERSION floor: K action-swapped
             predictions per state must fan out by >= tau_d. Attacks
             controllability collapse directly (act_frac as a loss).
  dual_full  both constraints, separate multipliers, zero fixed weights.

References: rot+reg (simplex penalty) and rot_noreg (cap collapse).
Success = restore latent motion & spread (clump ~ rot+reg) without any
marginal regularizer; report converged lambda* = the measured price of
information the economics framing predicts.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_rotation_bench import (D_OBS, D_Z, N_EP_EV, N_EP_TR, T_EP,   # noqa: E402
                                Encoder, Predictor, make_world)
from variants import simplex_proto_loss                                # noqa: E402

torch.set_num_threads(4)
BATCH, STEPS, ETA_DUAL = 128, 600, 2.0


def ang(a, b):
    return (a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()


def spread_of(preds):
    P = torch.stack(preds)                                  # (K, B, D)
    mean_dir = F.normalize(P.mean(0), dim=-1)
    return ang(P, mean_dir.unsqueeze(0)).mean()


def run(cfg, seed):
    torch.manual_seed(seed)
    episodes, g = make_world(seed)
    Xtr, Atr, Str = episodes(N_EP_TR)
    Xev, Aev, _ = episodes(N_EP_EV)
    enc, pred = Encoder(), Predictor("rot_learn", seed=seed)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-3)
    lam1, lam2 = 0.0, 0.0

    for step in range(STEPS):
        ep = torch.randint(0, N_EP_TR, (BATCH,), generator=g)
        t = torch.randint(0, T_EP - 1, (BATCH,), generator=g)
        z_t, z_n = enc(Xtr[ep, t]), enc(Xtr[ep, t + 1])
        zp = pred(z_t, Atr[ep, t])
        loss = 1 - (zp * z_n.detach()).sum(-1).mean()
        if cfg == "rot+reg":
            loss = loss + 1.0 * simplex_proto_loss(z_t)

        ang_true = ang(Str[ep, t], Str[ep, t + 1]).detach()
        if cfg in ("kin_fix", "kin_dual", "dual_full"):
            v1 = F.relu(0.5 * ang_true - ang(z_t, z_n)).mean()
            if cfg == "kin_fix":
                loss = loss + 5.0 * v1
            else:
                loss = loss + lam1 * v1
        if cfg in ("disp_fix", "dual_full"):
            preds = [zp]
            for _ in range(2):
                perm = torch.randperm(BATCH, generator=g)
                preds.append(pred(z_t, Atr[ep, t][perm]))
            v2 = F.relu(0.25 * ang_true.mean() - spread_of(preds))
            if cfg == "disp_fix":
                loss = loss + 5.0 * v2
            else:
                loss = loss + lam2 * v2

        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if cfg in ("kin_dual", "dual_full"):
                lam1 = max(0.0, lam1 + ETA_DUAL * float(v1))
            if cfg == "dual_full":
                lam2 = max(0.0, lam2 + ETA_DUAL * float(v2))

    with torch.no_grad():
        Z = enc(Xev.reshape(-1, D_OBS)).reshape(N_EP_EV, T_EP, D_Z)
        lat_step = ang(Z[:, :-1], Z[:, 1:]).mean().item()
        z_t = Z[:, :-1].reshape(-1, D_Z)
        a = Aev[:, : T_EP - 1].reshape(-1, 2)
        zp = pred(z_t, a)
        z_n = Z[:, 1:].reshape(-1, D_Z)
        pred_loss = (1 - (zp * z_n).sum(-1)).mean().item()
        H = Z.reshape(-1, D_Z)[:4000]
        sim = H @ H.t()
        clump = ((sim.sum() - H.size(0)) / (H.size(0) * (H.size(0) - 1))).item()
        mean_n = H.mean(0).norm().item()
        s = torch.linalg.svdvals(H - H.mean(0))
        p = s / s.sum(); p = p[p > 1e-12]
        erank = torch.exp(-(p * p.log()).sum()).item()
        # controllability: action-swap fan-out on eval states
        zb, ab = z_t[:512], a[:512]
        preds = [pred(zb, ab)]
        for _ in range(4):
            preds.append(pred(zb, ab[torch.randperm(512, generator=g)]))
        act_spread = spread_of(preds).item()
        cur = Z[:, 0]
        for tt in range(10):
            cur = pred(cur, Aev[:, tt])
        roll = (1 - (cur * Z[:, 10]).sum(-1)).mean().item()
        r1 = ((F.normalize(cur, dim=-1) @ F.normalize(Z[:, 10], dim=-1).t()).argmax(1)
              == torch.arange(N_EP_EV)).float().mean().item()

    return dict(pred=pred_loss, lat_step=lat_step, clump=clump, mean_n=mean_n,
                erank=erank, act_spread=act_spread, roll10=roll, retr1=r1,
                lam1=lam1, lam2=lam2)


def main():
    cfgs = ["rot+reg", "rot_noreg", "kin_fix", "kin_dual", "disp_fix", "dual_full"]
    print("rotation core, smooth-dynamics toy; 3 seeds. Success = dual rows restore")
    print("motion/spread (~rot+reg) with NO marginal regularizer.\n")
    print(f"{'config':<10}{'pred':>7}{'latstep':>8}{'clump':>7}{'|mean|':>7}{'erank':>7}"
          f"{'actspr':>8}{'roll10':>8}{'retr@1':>7}{'lam1*':>7}{'lam2*':>7}")
    print("-" * 90)
    out = {}
    for cfg in cfgs:
        rs = [run(cfg, s) for s in range(3)]
        m = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        out[cfg] = m
        print(f"{cfg:<10}{m['pred']:>7.4f}{m['lat_step']:>8.3f}{m['clump']:>7.3f}"
              f"{m['mean_n']:>7.3f}{m['erank']:>7.1f}{m['act_spread']:>8.3f}"
              f"{m['roll10']:>8.4f}{m['retr1']:>7.3f}{m['lam1']:>7.2f}{m['lam2']:>7.2f}")
    Path("results/cpu_dual_constraint_test.json").write_text(json.dumps(out, indent=2))
    print("\nlam* = converged multiplier = measured 'price of information'.")
    print("wrote results/cpu_dual_constraint_test.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
