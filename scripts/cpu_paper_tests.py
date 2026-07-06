#!/usr/bin/env python3
"""CPU evidence for the three paper claims (DIRECTIONS.md tiering).

T1  Diaconis-Freedman blindness: a distribution uniform on a HALF-dimensional
    great subsphere is massively non-uniform (measure zero) yet its 1-D random
    projections have exactly the same variance (1/d) and near-Gaussian shape as
    the full uniform. Sliced statistics (SIGReg/SUSReg family) should lose
    discrimination power as d grows; a second-moment/frame statistic (H2/tight-
    frame residual) sees the rank deficiency at any d. We report discrimination
    z-scores (effect sizes) vs dimension.

T2  Uniformity-robustness tradeoff: data on a k=4-dim manifold in R^64. As the
    uniformity weight rises, the encoder must fold a low-dim manifold over more
    of S^{d-1}, inflating its Lipschitz constant; prediction: empirical
    Lipschitz and the clean-vs-noisy accuracy gap grow with lambda.

T3  SO(d) parameterization truths (architecture-level, random nontrivial
    weights -- these are claims about the map, not about training):
    (a) gradient norm through a T-step rollout: rotation ~ conserved (uRNN),
        additive/gated variants decay or explode;
    (b) pairwise-angle distortion of a batch after 32 rollout steps;
    (c) Cayley transform with low-rank skew A=UV^T-VU^T via Woodbury:
        exact orthogonality check + per-step timing at d=192.
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_rotation_bench import Predictor, plane_rot   # noqa: E402
from variants import simplex_proto_loss               # noqa: E402

torch.set_num_threads(4)


# =========================================================================== #
# T1: sliced tests go blind in high d; moment statistics do not
# =========================================================================== #
def t1_power(n=2048, R=64, trials=20):
    print("== T1: discrimination z-score, uniform vs uniform-on-half-subsphere ==")
    print(f"{'d':>6} | {'sliced (SUSReg-style)':>22} | {'H2 moment/frame':>16}")
    for d in [8, 32, 128, 512]:
        stats = {"sliced": {"null": [], "alt": []}, "h2": {"null": [], "alt": []}}
        for tr in range(trials):
            g = torch.Generator().manual_seed(1000 * d + tr)
            unif = F.normalize(torch.randn(n, d, generator=g), dim=-1)
            # alt: uniform on a random d/2-dim great subsphere (measure zero,
            # projection variance exactly 1/d in expectation)
            V, _ = torch.linalg.qr(torch.randn(d, d // 2, generator=g))
            sub = F.normalize(torch.randn(n, d // 2, generator=g), dim=-1) @ V.t()
            ref = F.normalize(torch.randn(n, d, generator=g), dim=-1)  # test ref

            u = F.normalize(torch.randn(R, d, generator=g), dim=-1)
            r_sorted = (ref @ u.t()).sort(dim=0)[0]
            for name, X in [("null", unif), ("alt", sub)]:
                t_sorted = (X @ u.t()).sort(dim=0)[0]
                stats["sliced"][name].append(
                    (t_sorted - r_sorted).pow(2).mean().item())
                cov = X.t() @ X / n
                stats["h2"][name].append(
                    (cov - torch.eye(d) / d).pow(2).sum().item() * n)
        def z(s):
            null = torch.tensor(s["null"]); alt = torch.tensor(s["alt"])
            return ((alt.mean() - null.mean()) / null.std().clamp_min(1e-12)).item()
        print(f"{d:>6} | {z(stats['sliced']):>22.1f} | {z(stats['h2']):>16.1f}")


# =========================================================================== #
# T2: uniformity weight -> Lipschitz inflation -> robustness loss
# =========================================================================== #
def t2_tradeoff(seeds=3):
    D_IN, D_Z, K_INT, NCLS = 64, 32, 4, 10
    N_TR, N_TE, BATCH, STEPS = 4096, 1024, 256, 300
    print("\n== T2: uniformity-robustness tradeoff (k=4 manifold in R^64) ==")
    print(f"{'lam':>6}{'|mean|':>8}{'clump':>8}{'Lip^':>8}{'knn':>7}"
          f"{'knn+noise':>10}{'gap':>7}")

    def one(lam, seed):
        g = torch.Generator().manual_seed(seed)
        W1 = torch.randn(K_INT, 128, generator=g) / math.sqrt(K_INT)
        W2 = torch.randn(128, D_IN, generator=g) / math.sqrt(128)
        anchors = torch.randn(NCLS, K_INT, generator=g)

        def make(n):
            w = torch.randn(n, K_INT, generator=g)
            y = (w @ anchors.t()).argmax(1)
            x = torch.tanh(w @ W1) @ W2
            return x, y, w
        xtr, ytr, wtr = make(N_TR)
        xte, yte, _ = make(N_TE)
        enc = torch.nn.Sequential(torch.nn.Linear(D_IN, 128), torch.nn.ReLU(),
                                  torch.nn.Linear(128, D_Z))
        f = lambda x: F.normalize(enc(x), dim=-1)
        pred = torch.nn.Sequential(torch.nn.Linear(D_Z, 128), torch.nn.ReLU(),
                                   torch.nn.Linear(128, D_Z))
        opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()),
                               lr=3e-3)
        for step in range(STEPS):
            i = torch.randint(0, N_TR, (BATCH,), generator=g)
            # on-manifold aug (perturb intrinsic coords) + tiny ambient noise
            w1 = wtr[i] + 0.25 * torch.randn(BATCH, K_INT, generator=g)
            w2 = wtr[i] + 0.25 * torch.randn(BATCH, K_INT, generator=g)
            v1 = torch.tanh(w1 @ W1) @ W2 + 0.01 * torch.randn(BATCH, D_IN, generator=g)
            v2 = torch.tanh(w2 @ W1) @ W2 + 0.01 * torch.randn(BATCH, D_IN, generator=g)
            z1, z2 = f(v1), f(v2)
            p1 = F.normalize(pred(z1), dim=-1)
            loss = 1 - (p1 * z2.detach()).sum(-1).mean()
            if lam > 0:
                loss = loss + lam * simplex_proto_loss(z1)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            ztr, zte = f(xtr), f(xte)
            # empirical Lipschitz: angular displacement per unit ambient input
            delta = 0.05 * F.normalize(torch.randn(N_TE, D_IN, generator=g), dim=-1)
            zpert = f(xte + delta)
            ang = (zte * zpert).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
            lip = (ang / delta.norm(dim=-1)).mean().item()

            def knn(q):
                nbr = (q @ ztr.t()).topk(5, dim=1).indices
                return (torch.mode(ytr[nbr], 1).values == yte).float().mean().item()
            clean = knn(zte)
            noisy = knn(f(xte + 0.3 * torch.randn(N_TE, D_IN, generator=g)))
            mean_n = zte.mean(0).norm().item()
            sim = zte @ zte.t()
            clump = ((sim.sum() - N_TE) / (N_TE * (N_TE - 1))).item()
        return mean_n, clump, lip, clean, noisy

    for lam in [0.0, 0.5, 2.0, 8.0, 32.0]:
        rs = [one(lam, s) for s in range(seeds)]
        m = [sum(r[i] for r in rs) / seeds for i in range(5)]
        print(f"{lam:>6.1f}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.3f}{m[3]:>7.3f}"
              f"{m[4]:>10.3f}{m[3]-m[4]:>7.3f}")


# =========================================================================== #
# T3: SO(d) parameterization truths
# =========================================================================== #
def t3_so_d():
    D = 32
    torch.manual_seed(0)
    preds = {}
    for mode in ["simple", "gated", "tangent", "rot_learn"]:
        p = Predictor(mode, seed=0)
        if mode.startswith("rot"):     # nontrivial rotations at init
            torch.nn.init.normal_(p.theta.weight, std=0.05)
            torch.nn.init.normal_(p.theta.bias, std=0.1)
        preds[mode] = p

    print("\n== T3a: gradient norm through T-step rollout (rel. to T=4) ==")
    print(f"{'mode':<10}" + "".join(f"{f'T={t}':>10}" for t in [4, 16, 64]))
    for mode, p in preds.items():
        row = []
        for T in [4, 16, 64]:
            z0 = F.normalize(torch.randn(64, D), dim=-1).requires_grad_(True)
            z = z0
            for t in range(T):
                z = p(z, torch.randn(64, 2))
            tgt = F.normalize(torch.randn(1, D), dim=-1)
            (z * tgt).sum().backward()
            row.append(z0.grad.norm().item())
        base = row[0]
        print(f"{mode:<10}" + "".join(f"{v / base:>10.3f}" for v in row))

    print("\n== T3b: pairwise-angle distortion after 32 rollout steps ==")
    z0 = F.normalize(torch.randn(256, D), dim=-1)
    G0 = z0 @ z0.t()
    for mode, p in preds.items():
        z = z0.clone()
        with torch.no_grad():
            for t in range(32):
                z = p(z, torch.randn(256, 2))
        G = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).t()
        off = ~torch.eye(256, dtype=torch.bool)
        print(f"  {mode:<10} mean |dcos| = {(G - G0)[off].abs().mean():.4f}")

    print("\n== T3c: Cayley low-rank skew (Woodbury) at d=192, k=8 ==")
    d, k, B = 192, 8, 256
    torch.manual_seed(1)
    U, V = torch.randn(d, k) / d ** 0.5, torch.randn(d, k) / d ** 0.5
    P = torch.cat([U, V], 1)                    # A = P Q^T (rank 2k)
    Q = torch.cat([V, -U], 1)

    def cayley_apply(z):
        y = z + 0.5 * (z @ Q) @ P.t()           # (I + A/2) z
        M = torch.eye(2 * k) - 0.5 * (Q.t() @ P)
        w = y + 0.5 * torch.linalg.solve(M, (y @ Q).t()).t() @ P.t()
        return w

    z = F.normalize(torch.randn(B, d), dim=-1)
    w = cayley_apply(z)
    print(f"  norm preservation: max ||Rz|-1| = {(w.norm(dim=-1) - 1).abs().max():.2e}")
    zi, zj = w[:128], w[128:]
    g_before = (z[:128] * z[128:]).sum(-1)
    g_after = (zi * zj).sum(-1)
    print(f"  pairwise-angle preservation: max |dcos| = {(g_after - g_before).abs().max():.2e}")
    t0 = time.time()
    for _ in range(200):
        cayley_apply(z)
    print(f"  time/step (B=256, d=192, k=8, CPU): {(time.time()-t0)/200*1e3:.2f} ms")


if __name__ == "__main__":
    t0 = time.time()
    t1_power()
    t2_tradeoff()
    t3_so_d()
    print(f"\ntotal {time.time()-t0:.1f}s")
