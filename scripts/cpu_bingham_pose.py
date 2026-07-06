#!/usr/bin/env python3
"""CPU test: Bingham directional head vs geodesic quaternion regression.

Claim under test: swapping a rotation head from
    q = normalize(linear(features));  loss = geodesic(q, q*)
to a Bingham head
    A = bingham_head(features);  loss = bingham_nll(A, q*)
buys (a) antipodal correctness q == -q for free, (b) calibrated uncertainty
(concentration tracks true error under noise/occlusion), and (c) a usable
risk-coverage curve ("know when not to trust") -- WITHOUT losing mode accuracy.
The practical bottleneck is the Bingham normalization constant F(A); we make it
differentiable + stable via a Monte-Carlo estimate on a FIXED S^3 grid.

Deep-Bingham parameterization (Gilitschenski/Deng style): network emits an
orthonormal M in R^{4x4} (via QR) and concentrations Z = -softplus(.) <= 0;
A = M diag(0, Z1, Z2, Z3) M^T, so the mode is M[:,0] and |Z| is concentration.
Bingham density p(q) ~ exp(q^T A q) is antipodal by construction: q^T A q =
(-q)^T A (-q).

Task: infer a 3D rotation from 3 independent noisy measurements of its matrix
(heteroscedastic: the per-sample noise level is inferable from measurement
disagreement -> the model CAN learn calibrated uncertainty). Some samples are
"occluded" (high noise). Baseline: normalize(head)->q, loss 1-(q.q*)^2
(antipodal-safe geodesic), confidence proxy = pre-norm magnitude.
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

N, BATCH, STEPS = 8192, 256, 1200
M_GRID = 8192                     # fixed S^3 samples for the MC normalizer
LOG_AREA_S3 = math.log(2 * math.pi ** 2)


def rand_quat(n, g):
    return F.normalize(torch.randn(n, 4, generator=g), dim=-1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1)                                            # (n, 9)


def geodesic_deg(qa, qb):
    d = (F.normalize(qa, dim=-1) * F.normalize(qb, dim=-1)).sum(-1).abs().clamp(max=1)
    return torch.rad2deg(2 * torch.arccos(d))


def make_data(n, g):
    q = rand_quat(n, g)
    R = quat_to_matrix(q)
    # per-sample noise level: 60% clean, 25% medium, 15% occluded
    u = torch.rand(n, 1, generator=g)
    sigma = torch.where(u < 0.60, 0.03, torch.where(u < 0.85, 0.25, 0.7))
    meas = torch.cat([R + sigma * torch.randn(n, 9, generator=g) for _ in range(3)], -1)
    return meas, q, sigma.squeeze(-1)


# --------------------------------------------------------------------------- #
# Bingham head + NLL
# --------------------------------------------------------------------------- #
class BinghamHead(nn.Module):
    def __init__(self, d_in=27, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.to_M = nn.Linear(h, 16)
        self.to_Z = nn.Linear(h, 3)

    def forward(self, x):
        f = self.body(x)
        Mraw = self.to_M(f).reshape(-1, 4, 4)
        M, _ = torch.linalg.qr(Mraw)                  # orthonormal columns
        Z = -F.softplus(self.to_Z(f))                 # (B,3) <= 0
        return M, Z


def bingham_terms(M, Z, q):
    """Return q^T A q for a batch of quaternions q (B,4) under A=M diag(0,Z) M^T."""
    proj = torch.einsum("bd,bdk->bk", q, M)           # (B,4): q . m_k
    return (Z * proj[:, 1:] ** 2).sum(-1)             # z0=0 term dropped


def bingham_nll(M, Z, q_star, grid):
    data = bingham_terms(M, Z, q_star)                # (B,)
    # log F(A) = log mean_j exp(g_j^T A g_j) + log area  (g_j fixed S^3 grid)
    proj = torch.einsum("md,bdk->bmk", grid, M)       # (B, Mgrid, 4)
    quad = (Z.unsqueeze(1) * proj[..., 1:] ** 2).sum(-1)   # (B, Mgrid)
    logF = torch.logsumexp(quad, dim=1) - math.log(grid.size(0)) + LOG_AREA_S3
    return (-data + logF).mean()


@torch.no_grad()
def bingham_readout(M, Z):
    mode = M[:, :, 0]                                 # eigenvector of eigenvalue 0
    conc = (-Z).mean(-1)                              # overall concentration
    amb = (-Z).min(-1).values                         # weakest-constrained axis
    return mode, conc, amb


# --------------------------------------------------------------------------- #
class GeoHead(nn.Module):
    def __init__(self, d_in=27, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU())
        self.out = nn.Linear(h, 4)

    def forward(self, x):
        raw = self.out(self.body(x))
        return raw                                    # un-normalized (magnitude=proxy)


def risk_coverage(err, conf):
    """Mean error over the most-confident coverage fractions."""
    order = conf.argsort(descending=True)
    e = err[order]
    return {c: e[:max(1, int(c * len(e)))].mean().item() for c in (0.25, 0.5, 1.0)}


def train_eval(kind, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Xtr, Qtr, _ = make_data(N, g)
    grid = rand_quat(M_GRID, torch.Generator().manual_seed(0))    # fixed grid
    net = BinghamHead() if kind == "bingham" else GeoHead()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)

    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,), generator=g)
        x, q = Xtr[idx], Qtr[idx]
        if kind == "bingham":
            M, Z = net(x)
            loss = bingham_nll(M, Z, q, grid)
        else:
            raw = net(x)
            qh = F.normalize(raw, dim=-1)
            loss = (1 - (qh * q).sum(-1) ** 2).mean()          # antipodal geodesic
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Xe, Qe, sig = make_data(4096, torch.Generator().manual_seed(seed + 99))
        if kind == "bingham":
            M, Z = net(Xe)
            mode, conf, amb = bingham_readout(M, Z)
            err = geodesic_deg(mode, Qe)
            # antipodal invariance check: NLL(q) == NLL(-q) exactly
            inv = (bingham_nll(M[:64], Z[:64], Qe[:64], grid)
                   - bingham_nll(M[:64], Z[:64], -Qe[:64], grid)).abs().item()
        else:
            raw = net(Xe)
            err = geodesic_deg(raw, Qe)
            conf = raw.norm(dim=-1)                              # magnitude proxy
            inv = float("nan")
        # calibration: does confidence rank errors? (Spearman-ish via rank corr)
        rc = risk_coverage(err, conf)
        cr = err.argsort().argsort().float()
        cf = conf.argsort().argsort().float()
        rank_corr = ((cr - cr.mean()) * (cf - cf.mean())).mean() / (cr.std() * cf.std())
        # correlation of confidence with TRUE noise level (occlusion awareness)
        occ = ((conf - conf.mean()) * (sig - sig.mean())).mean() / (conf.std() * sig.std())
    return dict(kind=kind, med_err=err.median().item(), mean_err=err.mean().item(),
                antipodal_gap=inv, conf_err_rankcorr=-rank_corr.item(),
                conf_vs_noise_corr=-occ.item(),
                rc25=rc[0.25], rc50=rc[0.5], rc100=rc[1.0])


def main():
    print(f"task: 3D rotation from 3 noisy matrix measurements "
          f"(60% clean / 25% medium / 15% occluded), N={N}\n")
    print(f"{'head':<10}{'med_err':>9}{'mean_err':>9}{'antip_gap':>11}"
          f"{'conf~err':>9}{'conf~occl':>10}{'rc@25%':>8}{'rc@50%':>8}{'rc@100%':>9}")
    print("-" * 84)
    rows = {}
    for kind in ["geodesic", "bingham"]:
        rs = [train_eval(kind, s) for s in range(3)]
        a = {k: (sum(r[k] for r in rs) / len(rs) if isinstance(rs[0][k], float) else rs[0][k])
             for k in rs[0] if k != "kind"}
        rows[kind] = a
        print(f"{kind:<10}{a['med_err']:>9.2f}{a['mean_err']:>9.2f}"
              f"{a['antipodal_gap']:>11.1e}{a['conf_err_rankcorr']:>9.3f}"
              f"{a['conf_vs_noise_corr']:>10.3f}{a['rc25']:>8.2f}{a['rc50']:>8.2f}"
              f"{a['rc100']:>9.2f}")
    Path("results/cpu_bingham_pose.json").write_text(__import__("json").dumps(rows, indent=2))
    print("\nconf~err: rank-corr(confidence, error), higher=better calibrated;")
    print("conf~occl: corr(confidence, true noise level); rc@X: mean error (deg)")
    print("  on the X most-confident fraction -- the 'know when not to trust' curve.")
    print("wrote results/cpu_bingham_pose.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
