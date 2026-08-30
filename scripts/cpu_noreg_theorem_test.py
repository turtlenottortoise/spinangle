#!/usr/bin/env python3
"""Stage-1 falsifier for the 'collapse-impossible world models' claim.

Claim: isometric (rotation) transitions + GROUNDED step size make collapse a
non-critical point, so anti-collapse regularizers can be deleted.

Self-critique already found one hole before running: grounding the Lie-algebra
magnitude sum|theta| is gameable (rotate in a plane orthogonal to z => theta>0
yet z fixed). The defensible version grounds the REALIZED angle
arccos<z_t, z_pred> against the true state-delta angle (hinge: move at least
c * true angle). This script measures, on the smooth-dynamics toy that killed
tangent (cpu_rotation_bench world):

  tangent + proto      (reference regularized baseline)
  tangent, NO reg      (control: should degrade -> comparison isn't vacuous)
  rot    + proto       (current contender recipe)
  rot,    NO reg       (the naked claim)
  rot,    NO reg + realized-angle grounding (the theorem's version)

Collapse indicators on eval latents: clump (mean pairwise cos), eff_rank,
|mean|; dynamics quality: pred-vs-floor, realized vs true step, 10-step
rollout, future retrieval. Also reports sum|theta| vs realized angle -- if they
diverge, the orthogonal-spin gaming is happening in the wild.
"""
import json
import math
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
BATCH, STEPS = 128, 600


def realized_angle(z_t, z_p):
    return (z_t * z_p).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()


def run(mode, seed, reg_w, ground):
    torch.manual_seed(seed)
    episodes, g = make_world(seed)
    Xtr, Atr, Str = episodes(N_EP_TR)
    Xev, Aev, Sev = episodes(N_EP_EV)
    enc, pred = Encoder(), Predictor(mode, seed=seed)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-3)

    for step in range(STEPS):
        ep = torch.randint(0, N_EP_TR, (BATCH,), generator=g)
        t = torch.randint(0, T_EP - 1, (BATCH,), generator=g)
        z_t = enc(Xtr[ep, t])
        z_n = enc(Xtr[ep, t + 1])
        zp = pred(z_t, Atr[ep, t])
        loss = 1 - (zp * z_n.detach()).sum(-1).mean()
        if reg_w > 0:
            loss = loss + reg_w * simplex_proto_loss(z_t)
        if ground == "angle":
            s_t, s_n = Str[ep, t], Str[ep, t + 1]
            ang_true = (s_t * s_n).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
            ang_real = realized_angle(z_t, zp)
            # hinge: the realized latent step must be at least half the true
            # state step -- forbids identity/shrinkage, doesn't force overshoot
            loss = loss + 1.0 * F.relu(0.5 * ang_true - ang_real).pow(2).mean()
        elif ground == "enc_angle":
            # refinement 2: ground the ENCODER's own temporal step (reaches the
            # shrinking encoder directly, not via the predictor), harder weight
            s_t, s_n = Str[ep, t], Str[ep, t + 1]
            ang_true = (s_t * s_n).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
            ang_enc = realized_angle(z_t, z_n)
            loss = loss + 5.0 * F.relu(0.5 * ang_true - ang_enc).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Z = enc(Xev.reshape(-1, D_OBS)).reshape(N_EP_EV, T_EP, D_Z)
        floor = (1 - (Z[:, :-1] * Z[:, 1:]).sum(-1)).mean().item()
        true_lat_step = (Z[:, :-1] * Z[:, 1:]).sum(-1).clamp(-1, 1).arccos().mean().item()
        z_t = Z[:, :-1].reshape(-1, D_Z)
        a = Aev[:, : T_EP - 1].reshape(-1, 2)
        zp = pred(z_t, a)
        z_n = Z[:, 1:].reshape(-1, D_Z)
        pred_loss = (1 - (zp * z_n).sum(-1)).mean().item()
        real_ang = realized_angle(z_t, zp).mean().item()
        theta = pred.step_size(z_t[:512], a[:512]) if mode.startswith("rot") else float("nan")

        # collapse indicators on the latent cloud
        H = Z.reshape(-1, D_Z)[:4000]
        sim = H @ H.t()
        clump = ((sim.sum() - H.size(0)) / (H.size(0) * (H.size(0) - 1))).item()
        mean_n = H.mean(0).norm().item()
        s = torch.linalg.svdvals(H - H.mean(0))
        p = s / s.sum(); p = p[p > 1e-12]
        erank = torch.exp(-(p * p.log()).sum()).item()

        cur = Z[:, 0]
        for tt in range(10):
            cur = pred(cur, Aev[:, tt])
        tgt = Z[:, 10]
        roll = (1 - (cur * tgt).sum(-1)).mean().item()
        r1 = ((F.normalize(cur, dim=-1) @ F.normalize(tgt, dim=-1).t()).argmax(1)
              == torch.arange(N_EP_EV)).float().mean().item()

    return dict(pred=pred_loss, floor=floor, xfloor=floor / max(pred_loss, 1e-9),
                theta=theta, real_ang=real_ang, lat_step=true_lat_step,
                clump=clump, mean_n=mean_n, erank=erank, roll10=roll, retr1=r1)


CONFIGS = [
    ("tangent+reg",   "tangent",   1.0, None),
    ("tangent_noreg", "tangent",   0.0, None),
    ("rot+reg",       "rot_learn", 1.0, None),
    ("rot_noreg",     "rot_learn", 0.0, None),
    ("rot_noreg+grd", "rot_learn", 0.0, "angle"),
    ("rot_noreg+egrd", "rot_learn", 0.0, "enc_angle"),
]


def main():
    print("smooth-dynamics toy (the Push-T collapse regime); 3 seeds\n")
    print(f"{'config':<15}{'pred':>7}{'xfloor':>7}{'theta':>7}{'realang':>8}"
          f"{'latstep':>8}{'clump':>7}{'|mean|':>7}{'erank':>7}{'roll10':>8}{'retr@1':>7}")
    print("-" * 95)
    out = {}
    for name, mode, reg_w, ground in CONFIGS:
        rs = [run(mode, s, reg_w, ground) for s in range(3)]
        m = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        out[name] = m
        print(f"{name:<15}{m['pred']:>7.4f}{m['xfloor']:>7.2f}{m['theta']:>7.3f}"
              f"{m['real_ang']:>8.3f}{m['lat_step']:>8.3f}{m['clump']:>7.3f}"
              f"{m['mean_n']:>7.3f}{m['erank']:>7.1f}{m['roll10']:>8.4f}{m['retr1']:>7.3f}")
    Path("results/cpu_noreg_theorem_test.json").write_text(json.dumps(out, indent=2))
    print("\nclaim survives if rot_noreg(+grd) keeps clump/|mean| ~ rot+reg levels")
    print("and rollout/retr do not degrade; theta >> realang = orthogonal-spin gaming.")
    print("wrote results/cpu_noreg_theorem_test.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
