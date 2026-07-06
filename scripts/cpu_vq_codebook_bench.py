#!/usr/bin/env python3
"""CPU test: VQ codebook collapse is an OCCUPANCY problem, not a marginal one.

A spherical VQ autoencoder on synthetic multi-cluster data (K_true real modes,
K=256 codes on S^{d-1}). We compare anti-collapse mechanisms and measure two
DIFFERENT axes:

  * marginal uniformity of the CONTINUOUS pre-quant code z (clump, |mean|) --
    what SIGReg/SUSReg/MMD/simplex ("Borelli family") actually control;
  * discrete codebook OCCUPANCY (used codes / 256, usage perplexity) -- what
    "codebook collapse" actually is.

Claim under test (Leonard's finding): representation regularizers shape the z
marginal but do NOT fix Voronoi-cell occupancy; you can have a near-uniform z
marginal and 80% dead codes simultaneously. The fixes that work act on the
ASSIGNMENT HISTOGRAM directly (balanced/Sinkhorn assignment, usage-entropy,
dead-code reinit), not on the marginal.

Mechanisms:
  vanilla    commitment + codebook loss (nearest-code STE)
  ema        EMA codebook, commitment only (VQ-VAE-2 standard)
  sigreg_h   vanilla + sliced-Gaussian reg on continuous z  (Borelli: marginal)
  simplex_h  vanilla + simplex-ETF proto on continuous z     (Borelli: marginal)
  entropy    vanilla + penalize -H(batch code-usage)         (occupancy)
  sinkhorn   balanced Sinkhorn assignment for the commitment  (occupancy; the
             "codebook-as-anchors" mechanism == halfway to CodeSphere)
  reinit     vanilla + periodic dead-code reinit from encoder outputs (occupancy)
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from variants import simplex_proto_loss   # noqa: E402

torch.set_num_threads(4)

D_OBS, D_Z, K, K_TRUE = 64, 16, 256, 96
N, BATCH, STEPS = 8192, 256, 1500


def make_data(seed):
    g = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(K_TRUE, D_Z, generator=g), dim=-1)
    W = torch.randn(D_Z, D_OBS, generator=g) / math.sqrt(D_Z)

    def sample(n):
        y = torch.randint(0, K_TRUE, (n,), generator=g)
        z = F.normalize(centers[y] + 0.15 * torch.randn(n, D_Z, generator=g), dim=-1)
        x = torch.tanh(z @ W) + 0.02 * torch.randn(n, D_OBS, generator=g)
        return x
    return sample(N), g


class VQAE(nn.Module):
    def __init__(self, seed):
        super().__init__()
        gg = torch.Generator().manual_seed(seed + 7)
        self.enc = nn.Sequential(nn.Linear(D_OBS, 128), nn.ReLU(), nn.Linear(128, D_Z))
        self.dec = nn.Sequential(nn.Linear(D_Z, 128), nn.ReLU(), nn.Linear(128, D_OBS))
        C = F.normalize(torch.randn(K, D_Z, generator=gg), dim=-1)
        self.register_buffer("C", C)                 # codebook (buffer; EMA/manual)
        self.register_buffer("ema_n", torch.ones(K))
        self.register_buffer("ema_m", C.clone())

    def encode(self, x):
        return F.normalize(self.enc(x), dim=-1)


def usage_stats(assign):
    cnt = torch.bincount(assign, minlength=K).float()
    p = cnt / cnt.sum()
    used = int((cnt > 0).sum())
    perp = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    return used, perp, cnt


def run(mode, seed, freeze_enc=False):
    torch.manual_seed(seed)
    X, g = make_data(seed)
    m = VQAE(seed)
    # freeze_enc keeps encoder params in the graph (so marginal losses still
    # backprop harmlessly) but OUT of the optimizer -> z stays spread and fixed,
    # isolating pure codebook-side occupancy. Marginal regs become no-ops here.
    trained = list(m.dec.parameters())
    if not freeze_enc:
        trained = list(m.enc.parameters()) + trained
    opt = torch.optim.Adam(trained, lr=3e-3)
    ema_decay, beta = 0.99, 0.25

    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,), generator=g)
        x = X[idx]
        z = m.encode(x)                              # (B, D_Z) unit
        sim = z @ m.C.t()                            # cosine to codes

        if mode == "sinkhorn":                       # balanced soft assignment
            with torch.no_grad():
                Q = torch.exp(sim / 0.1)
                for _ in range(3):
                    Q = Q / Q.sum(0, keepdim=True).clamp_min(1e-12)
                    Q = Q / Q.sum(1, keepdim=True).clamp_min(1e-12)
                assign = Q.argmax(1)
            e = m.C[assign]
            commit = beta * (1 - (z * e.detach()).sum(-1)).mean()
            # pull codes toward their soft-assigned mass (codebook update)
            code_loss = -(Q * sim).sum(1).mean()
        else:
            assign = sim.argmax(1)
            e = m.C[assign]
            commit = beta * (1 - (z * e.detach()).sum(-1)).mean()
            code_loss = (1 - (z.detach() * e).sum(-1)).mean() if mode != "ema" else 0.0 * commit

        z_q = z + (e - z).detach()                   # straight-through
        recon = F.mse_loss(m.dec(z_q), x)
        loss = recon + commit + code_loss

        if mode == "sigreg_h":
            B, d = z.shape
            u = F.normalize(torch.randn(64, d), dim=-1)
            t = (z @ u.t()).sort(dim=0)[0]
            q = torch.erfinv(2 * (torch.arange(B) + 0.5) / B - 1) * math.sqrt(2) / math.sqrt(d)
            loss = loss + 4.0 * (t - q.unsqueeze(1)).pow(2).mean()
        elif mode == "simplex_h":
            loss = loss + 1.0 * simplex_proto_loss(z)
        elif mode == "entropy":                      # maximize batch usage entropy
            soft = torch.softmax(sim / 0.1, dim=1).mean(0)
            loss = loss - 0.5 * (-(soft * (soft + 1e-9).log()).sum())

        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            if mode == "ema":                        # EMA codebook update
                oh = F.one_hot(assign, K).float()
                m.ema_n.mul_(ema_decay).add_(oh.sum(0), alpha=1 - ema_decay)
                m.ema_m.mul_(ema_decay).add_(oh.t() @ z, alpha=1 - ema_decay)
                n = m.ema_n.clamp_min(1e-2)
                m.C.copy_(F.normalize(m.ema_m / n.unsqueeze(1), dim=-1))
            else:                                    # gradient-free code move toward z
                oh = F.one_hot(assign, K).float()
                cnt = oh.sum(0).clamp_min(1e-6)
                upd = (oh.t() @ z) / cnt.unsqueeze(1)
                mask = (oh.sum(0) > 0).float().unsqueeze(1)
                m.C.mul_(1 - 0.1 * mask).add_(0.1 * mask * F.normalize(upd, dim=-1))
                m.C.copy_(F.normalize(m.C, dim=-1))
            if mode == "reinit" and step % 200 == 199:   # revive dead codes
                cnt = torch.bincount(assign, minlength=K)
                dead = (cnt == 0).nonzero().flatten()
                if len(dead):
                    src = z[torch.randint(0, BATCH, (len(dead),), generator=g)]
                    m.C[dead] = F.normalize(src + 0.01 * torch.randn_like(src), dim=-1)

    # ---- eval on a held-out pass ----
    with torch.no_grad():
        Xe, _ = make_data(seed + 100)
        z = m.encode(Xe)
        assign = (z @ m.C.t()).argmax(1)
        used, perp, _ = usage_stats(assign)
        recon = F.mse_loss(m.dec(z + (m.C[assign] - z).detach()), Xe).item()
        # continuous z marginal uniformity (what Borelli methods control)
        sub = z[:4000]
        clump = ((sub @ sub.t()).sum() - sub.size(0)) / (sub.size(0) * (sub.size(0) - 1))
        mean_norm = z.mean(0).norm().item()
    return dict(mode=mode, used=used, perp=perp, recon=recon,
                z_clump=clump.item(), z_meannorm=mean_norm)


def table(freeze_enc):
    modes = ["vanilla", "ema", "sigreg_h", "simplex_h", "entropy", "sinkhorn", "reinit"]
    label = "REGIME B: frozen encoder (z spread & fixed -> codebook-side collapse only)" \
        if freeze_enc else "REGIME A: end-to-end (encoder can co-collapse with codebook)"
    print(f"\n== {label} ==")
    print(f"{'mechanism':<11}{'used/256':>9}{'perplex':>9}{'recon':>8}"
          f" | {'z_clump':>8}{'z_|mean|':>9}   axis")
    print("-" * 74)
    out = {}
    for mode in modes:
        rs = [run(mode, s, freeze_enc) for s in range(3)]
        a = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0] if k != "mode"}
        out[mode] = a
        axis = "occupancy" if mode in ("ema", "entropy", "sinkhorn", "reinit") else \
               ("marginal" if mode in ("sigreg_h", "simplex_h") else "baseline")
        print(f"{mode:<11}{a['used']:>9.0f}{a['perp']:>9.1f}{a['recon']:>8.4f}"
              f" | {a['z_clump']:>8.3f}{a['z_meannorm']:>9.3f}   {axis}")
    return out


def main():
    print(f"true modes = {K_TRUE}, codebook K = {K}, S^{D_Z - 1}")
    out = {"regime_A_e2e": table(False), "regime_B_frozen": table(True)}
    Path("results/cpu_vq_codebook_bench.json").write_text(json.dumps(out, indent=2))
    print("\nRegime A (encoder co-collapses): marginal regs help occupancy by")
    print("  spreading z. Regime B (z fixed & spread): marginal regs are no-ops")
    print("  on occupancy; only assignment-histogram methods (entropy/sinkhorn/")
    print("  reinit) revive dead codes. => the transfer failure is conditional on")
    print("  WHICH side collapsed; diagnostic = z_clump of the pre-quant embedding.")
    print("wrote results/cpu_vq_codebook_bench.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
