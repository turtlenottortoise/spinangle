#!/usr/bin/env python3
"""CPU proxy bench: spherical uniformity regularizers inside a toy JEPA.

Purpose (decision rule #7 of the SPHERE-JEPA notes): this does NOT prove
superiority over SPHERE-JEPA -- it decides which regularizer deserves an
expensive GPU run, and it mechanistically checks the follow-up paper's claim
that sliced (projection-based) regularizers have noisy gradients while
sphere-native deterministic ones do not.

Setup
-----
Data: 10 vMF-style class clusters in R^64; two "views" per sample = small
independent perturbations (augmentation proxy). Encoder: MLP -> unit sphere in
R^32. Loss: BYOL/JEPA-style cosine prediction with stop-grad target (collapses
without a regularizer) + lambda * Reg(z). Every method gets a small lambda grid
and S seeds; we report the best-of-grid by validation kNN.

Methods
-------
none            collapse control
sigreg          sliced W2 of projections vs N(0,1)      (LeJEPA target, mismatched on the sphere)
sigreg_1overd   sliced W2 vs N(0,1/d)                   (~large-d SUSReg, retargeted SIGReg)
susreg          sliced W2 vs projections of a fresh Uniform(S^{d-1}) sample (MC target)
mmd_energy      deterministic MMD-to-uniform == pairwise Gaussian-kernel energy (Wang-Isola)
h2              ||mean||^2 + ||cov - I/d||_F^2          (cheap moment matching)
codesphere      K fixed uniform prototypes + Sinkhorn-balanced soft assignment
local_density   variance of log kNN angular radius      (local crowding penalty)
infonce         contrastive reference (uses negatives, replaces reg)

Metrics: kNN@5 (cosine), ridge linear probe, clumping (mean pairwise cos),
effective rank, mean-vector norm. All on held-out data.
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(4)

D_IN, D_Z, N_CLASSES = 64, 32, 10
N_TRAIN, N_EVAL, BATCH = 4096, 1024, 256
STEPS = 300


# --------------------------------------------------------------------------- #
# data: vMF-ish clusters + augmentation views
# --------------------------------------------------------------------------- #
def make_data(seed):
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(N_CLASSES, D_IN, generator=g), dim=-1)
    def sample(n):
        y = torch.randint(0, N_CLASSES, (n,), generator=g)
        x = centers[y] + 0.35 * torch.randn(n, D_IN, generator=g)
        return x, y
    xtr, ytr = sample(N_TRAIN)
    xte, yte = sample(N_EVAL)
    return xtr, ytr, xte, yte, g


def views(x, g):
    return x + 0.25 * torch.randn(x.shape, generator=g), \
           x + 0.25 * torch.randn(x.shape, generator=g)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_IN, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, D_Z))
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(D_Z, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, D_Z))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
# regularizers (z: (B, D) unit-norm)
# --------------------------------------------------------------------------- #
def _sliced_w2(z, target_sorted, R, g):
    u = F.normalize(torch.randn(R, z.size(1), generator=g), dim=-1)
    t = z @ u.t()                                    # (B, R)
    t_sorted, _ = t.sort(dim=0)
    return (t_sorted - target_sorted).pow(2).mean()


def reg_sigreg(z, g, sigma=1.0, R=16):
    B = z.size(0)
    q = torch.erfinv(2 * (torch.arange(B) + 0.5) / B - 1) * math.sqrt(2) * sigma
    return _sliced_w2(z, q.unsqueeze(1), R, g)


def reg_susreg(z, g, R=16):
    ref = F.normalize(torch.randn(z.size(0), z.size(1), generator=g), dim=-1)
    u = F.normalize(torch.randn(R, z.size(1), generator=g), dim=-1)
    t, r = (z @ u.t()).sort(dim=0)[0], (ref @ u.t()).sort(dim=0)[0]
    return (t - r).pow(2).mean()


def reg_mmd_energy(z, g=None, t=2.0):
    # deterministic MMD-to-uniform up to a constant: pairwise Gaussian-kernel energy
    sq = torch.cdist(z, z).pow(2)
    B = z.size(0)
    off = ~torch.eye(B, dtype=torch.bool)
    return torch.logsumexp(-t * sq[off], dim=0) - math.log(B * (B - 1))


def reg_h2(z, g=None):
    d = z.size(1)
    mu = z.mean(0)
    cov = (z.t() @ z) / z.size(0)
    return mu.pow(2).sum() + (cov - torch.eye(d) / d).pow(2).sum()


def _rand_rot(gg):
    Q, _ = torch.linalg.qr(torch.randn(D_Z, D_Z, generator=gg))
    return Q


def make_protos(mode, seed):
    """Theorem-backed prototype banks on S^{d-1}.

    cross   : rotated cross-polytope {+-e_i}, K=2d  (universally optimal, Cohn-Kumar)
    simplex : rotated regular simplex, K=d+1, pairwise cos = -1/d
              (neural-collapse ETF; universally optimal)
    union4  : union of 4 rotated orthonormal bases, K=4d -- a unit-norm tight
              frame, i.e. a global minimizer of the frame potential
              (Benedetto-Fickus); the principled *fine* bank.
    random  : iid Gaussian normalized (the naive baseline).
    """
    gg = torch.Generator().manual_seed(seed)
    if mode == "cross":
        Q = _rand_rot(gg)
        return torch.cat([Q, -Q], dim=0)
    if mode == "simplex":
        K = D_Z + 1
        M = torch.eye(K) - torch.full((K, K), 1.0 / K)   # rows span a d-dim subspace
        _, _, Vt = torch.linalg.svd(M)
        C = F.normalize(M @ Vt[:D_Z].t(), dim=-1)        # (d+1, d), cos = -1/d
        off = C @ C.t() - torch.eye(K)
        assert off.abs().max() - 1.0 / D_Z < 1e-4        # verify ETF geometry
        return C @ _rand_rot(gg)
    if mode == "union4":
        return torch.cat([_rand_rot(gg) for _ in range(4)], dim=0)
    return F.normalize(torch.randn(64, D_Z, generator=gg), dim=-1)


class CodeSphere:
    def __init__(self, mode="random", seed=0, tau=0.1):
        self.C = make_protos(mode, seed)
        self.tau = tau

    def __call__(self, z, g=None):
        sim = z @ self.C.t()                          # (B, K)
        with torch.no_grad():                          # Sinkhorn-balanced targets
            Q = torch.exp(sim / self.tau)
            for _ in range(3):
                Q = Q / Q.sum(0, keepdim=True)
                Q = Q / Q.sum(1, keepdim=True)
        return -(Q * sim).sum(1).mean()


class MultiCodeSphere:
    """Multi-scale: coarse semantic bank (simplex, K=d+1) + fine tight-frame
    bank (union of rotated orthobases, K=4d), equal weights."""
    def __init__(self, seed=0):
        self.coarse = CodeSphere("simplex", seed)
        self.fine = CodeSphere("union4", seed + 1)

    def __call__(self, z, g=None):
        return 0.5 * self.coarse(z) + 0.5 * self.fine(z)


def reg_local_density(z, g=None, k=3, eps=1e-3):
    sim = z @ z.t()
    sim.fill_diagonal_(-2.0)
    nn_k = sim.topk(k, dim=1).values[:, -1]           # cos to k-th neighbor
    return torch.log(eps + 1.0 - nn_k).var()


def loss_infonce(z1, z2, tau=0.2):
    B = z1.size(0)
    logits = z1 @ z2.t() / tau
    return F.cross_entropy(logits, torch.arange(B))


def loss_vmf_mle(z1, z2):
    """Temperature-free contrastive: conditional likelihood of a vMF mixture.
    kappa is not a hyperparameter -- it is the closed-form Banerjee et al. (2005)
    MLE estimate from the positives' mean resultant, recomputed each batch."""
    r = (z1 * z2).sum(-1).mean().clamp(0.05, 0.995).detach()
    kappa = r * (z1.size(1) - r ** 2) / (1.0 - r ** 2)
    logits = kappa * (z1 @ z2.t())
    return F.cross_entropy(logits, torch.arange(z1.size(0)))


REGS = {
    "none":          (lambda z, g: z.sum() * 0.0,          [0.0]),
    "sigreg":        (lambda z, g: reg_sigreg(z, g, 1.0),  [1.0, 4.0]),
    "sigreg_1overd": (lambda z, g: reg_sigreg(z, g, 1.0 / math.sqrt(D_Z)), [4.0, 16.0]),
    "susreg":        (reg_susreg,                          [4.0, 16.0]),
    "mmd_energy":    (reg_mmd_energy,                      [0.25, 1.0]),
    "h2":            (reg_h2,                              [4.0, 16.0]),
    "codesphere":    (None,                                [1.0, 4.0]),   # built per-run
    "local_density": (reg_local_density,                   [1.0, 4.0]),
    "infonce":       (None,                                [1.0]),        # replaces pred+reg
    "vmf_mle":       (None,                                [1.0]),        # temp-free NCE
    "codesphere_etf":     (None,                           [1.0, 4.0]),   # cross-polytope
    "codesphere_simplex": (None,                           [1.0, 4.0]),   # neural-collapse ETF
    "codesphere_ms":      (None,                           [1.0, 4.0]),   # coarse+fine banks
}

# prototype construction per codesphere variant
PROTO_MODE = {"codesphere": "random", "codesphere_etf": "cross",
              "codesphere_simplex": "simplex"}

# combos: pred loss + lam*Reg + 0.25*InfoNCE (uniformity + weak discriminative)
COMBOS = {
    "susreg+nce":          [4.0, 16.0],
    "sigreg_1overd+nce":   [4.0, 16.0],
    "mmd_energy+nce":      [0.25, 1.0],
    "codesphere+nce":      [1.0, 4.0],
    "codesphere_simplex+nce": [1.0, 4.0],
    "codesphere_ms+nce":   [1.0, 4.0],
}


# --------------------------------------------------------------------------- #
# eval: kNN, ridge probe, geometry
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(enc, xtr, ytr, xte, yte):
    ztr, zte = enc(xtr), enc(xte)
    sim = zte @ ztr.t()
    nbr = sim.topk(5, dim=1).indices
    knn = (torch.mode(ytr[nbr], dim=1).values == yte).float().mean().item()
    Y = F.one_hot(ytr, N_CLASSES).float()
    A = ztr.t() @ ztr + 1e-2 * torch.eye(D_Z)
    W = torch.linalg.solve(A, ztr.t() @ Y)
    lin = ((zte @ W).argmax(1) == yte).float().mean().item()
    h = zte
    clump = ((h @ h.t()).sum() - h.size(0)) / (h.size(0) * (h.size(0) - 1))
    s = torch.linalg.svdvals(h - h.mean(0))
    p = s / s.sum(); p = p[p > 1e-12]
    erank = torch.exp(-(p * p.log()).sum()).item()
    return dict(knn=knn, linear=lin, clumping=clump.item(), eff_rank=erank,
                mean_norm=h.mean(0).norm().item())


def run_one(method, lam, seed):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte, g = make_data(seed)
    enc = Encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    base = method[:-4] if method.endswith("+nce") else method
    reg_fn = REGS[base][0]
    if base in PROTO_MODE:
        reg_fn = CodeSphere(PROTO_MODE[base], seed=seed)
    elif base == "codesphere_ms":
        reg_fn = MultiCodeSphere(seed=seed)
    for step in range(STEPS):
        idx = torch.randint(0, N_TRAIN, (BATCH,), generator=g)
        v1, v2 = views(xtr[idx], g)
        z1, z2 = enc(v1), enc(v2)
        if method == "infonce":
            loss = loss_infonce(z1, z2)
        elif method == "vmf_mle":
            loss = loss_vmf_mle(F.normalize(enc.pred(z1), dim=-1), z2.detach())
        else:
            p1 = F.normalize(enc.pred(z1), dim=-1)
            pred = 1.0 - (p1 * z2.detach()).sum(-1).mean()
            loss = pred + lam * reg_fn(z1, g)
            if method.endswith("+nce"):          # weak contrastive assist
                loss = loss + 0.25 * loss_infonce(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(enc, xtr, ytr, xte, yte)


# --------------------------------------------------------------------------- #
# gradient-noise microtest: sliced vs deterministic
# --------------------------------------------------------------------------- #
def grad_noise():
    torch.manual_seed(0)
    z0 = F.normalize(torch.randn(BATCH, D_Z), dim=-1)
    out = {}
    for name in ["sigreg", "sigreg_1overd", "susreg", "mmd_energy", "h2",
                 "codesphere", "local_density"]:
        fn = REGS[name][0]
        if name == "codesphere":
            fn = CodeSphere("random", seed=0)
        grads = []
        for rep in range(20):
            g = torch.Generator().manual_seed(1000 + rep)   # resample projections
            z = z0.clone().requires_grad_(True)
            fn(z, g).backward()
            grads.append(z.grad.flatten())
        G = torch.stack(grads)
        gm = G.mean(0)
        rel_noise = (G - gm).norm(dim=1).mean() / (gm.norm() + 1e-12)
        out[name] = rel_noise.item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("== gradient-noise microtest (relative grad std across projection resamples) ==")
    for k, v in grad_noise().items():
        print(f"  {k:<14} {v:8.4f}")

    print("\n== toy JEPA bench (best-of-lambda, mean over seeds) ==")
    print(f"{'method':<20}{'lam':>6}{'knn':>8}{'linear':>8}{'clump':>8}{'erank':>8}{'|mean|':>8}")
    results = {}
    grid = {m: lams for m, (fn, lams) in REGS.items()}
    grid.update(COMBOS)
    for method, lams in grid.items():
        best = None
        for lam in lams:
            accs = [run_one(method, lam, s) for s in range(args.seeds)]
            agg = {k: sum(a[k] for a in accs) / len(accs) for k in accs[0]}
            if best is None or agg["knn"] > best[1]["knn"]:
                best = (lam, agg)
        lam, m = best
        results[method] = {"lam": lam, **m}
        print(f"{method:<20}{lam:>6.2f}{m['knn']:>8.3f}{m['linear']:>8.3f}"
              f"{m['clumping']:>8.3f}{m['eff_rank']:>8.2f}{m['mean_norm']:>8.3f}")
    with open("results/cpu_reg_bench.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/cpu_reg_bench.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time()-t0:.1f}s")
