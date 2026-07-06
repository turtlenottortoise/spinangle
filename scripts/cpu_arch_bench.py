#!/usr/bin/env python3
"""CPU test of three architecture ideas on a two-body CONTACT world.

World (mirrors Push-T structure): two spherical bodies in R^8 --
  pusher s_a: rotated directly by the 2-dim action every step;
  block  s_b: rotates ONLY on contact (sigmoid-gated by cos(s_a, s_b)),
              i.e. sparse-event dynamics;
  obs: pusher seen clearly, block seen WEAKLY (partial observability).

Ideas under test (all share the contender recipe: cosine + stop-grad +
simplex-ETF proto loss on the latent; rotation cores unless noted):
  mono_rot   single sphere S^31, one rotation step        (GPU contender ref)
  prod_rot   product of spheres (S^15)^2, per-patch rotations with
             cross-patch conditioning                     (idea 1)
  tied3      the SAME rotation module applied 3x (unrolled spherical flow:
             depth = integration steps, weight-tied)      (idea 2)
  untied3    3 stacked rotation modules, separate weights (idea 2 control)
  mem_rot    mono + cross-attention read over the last 5 latents
             (architecture-level retrieval)               (idea 3)
  tangent    sigmoid-gated tangent step                   (casualty reference)

Metrics: 10-step free rollout error, future retrieval R@1, ridge-R^2 of
pusher/block state from the latent, corr(|step|, true contact) -- does the
learned step size fire on contact events? -- and predictor param count.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_rotation_bench import plane_rot          # exact plane rotation
from variants import simplex_proto_loss           # dimension-agnostic ETF reg

torch.set_num_threads(4)

D_S, D_OBS, D_Z = 8, 64, 32
T_EP, N_EP_TR, N_EP_EV, BATCH, STEPS = 30, 512, 128, 128, 600
MEM = 5                                            # memory window (idea 3)


# --------------------------------------------------------------------------- #
# contact world
# --------------------------------------------------------------------------- #
def make_world(seed):
    g = torch.Generator().manual_seed(seed)
    Qa, _ = torch.linalg.qr(torch.randn(D_S, D_S, generator=g))
    Qb, _ = torch.linalg.qr(torch.randn(D_S, D_S, generator=g))
    Wa = torch.randn(D_S, 128, generator=g) / math.sqrt(D_S)
    Wb = torch.randn(D_S, 128, generator=g) / math.sqrt(D_S)
    Wo = torch.randn(128, D_OBS, generator=g) / math.sqrt(128)

    def obs(sa, sb):
        return (torch.tanh(sa @ Wa) + 0.6 * torch.tanh(sb @ Wb)) @ Wo

    def episodes(n):
        sa = F.normalize(torch.randn(n, D_S, generator=g), dim=-1)
        sb = F.normalize(torch.randn(n, D_S, generator=g), dim=-1)
        SA, SB, A, CT = [sa], [sb], [], []
        for _ in range(T_EP - 1):
            a = torch.randn(n, 2, generator=g)
            contact = torch.sigmoid(12.0 * ((sa * sb).sum(-1, keepdim=True) - 0.6))
            sa = plane_rot(sa, Qa[:, 0], Qa[:, 1], 0.08 * a[:, :1])
            sa = plane_rot(sa, Qa[:, 2], Qa[:, 3], 0.08 * a[:, 1:])
            sa = F.normalize(sa, dim=-1)
            sb = plane_rot(sb, Qb[:, 0], Qb[:, 1], 0.15 * contact * a[:, :1])
            sb = F.normalize(sb, dim=-1)
            SA.append(sa); SB.append(sb); A.append(a); CT.append(contact)
        SA, SB = torch.stack(SA, 1), torch.stack(SB, 1)
        X = obs(SA.reshape(-1, D_S), SB.reshape(-1, D_S)).reshape(n, T_EP, D_OBS)
        X = X + 0.01 * torch.randn(X.shape, generator=g)
        return X, torch.stack(A, 1), SA, SB, torch.stack(CT, 1)

    return episodes, g


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class RotStep(nn.Module):
    """k exact plane rotations of a dz-dim unit state, driven by `cond`."""

    def __init__(self, dz, dc, k=4):
        super().__init__()
        self.dz, self.k = dz, k
        self.theta = nn.Linear(dc, k)
        nn.init.zeros_(self.theta.weight); nn.init.zeros_(self.theta.bias)
        self.planes = nn.Linear(dc, 2 * k * dz)

    def forward(self, z, cond):
        th = self.theta(cond)
        raw = self.planes(cond).reshape(-1, self.k, 2, self.dz)
        out = z
        for i in range(self.k):
            p = F.normalize(raw[:, i, 0], dim=-1)
            qr = raw[:, i, 1]
            q = F.normalize(qr - (qr * p).sum(-1, keepdim=True) * p, dim=-1)
            out = plane_rot(out, p, q, th[:, i:i + 1])
        return out, th.abs().sum(-1)                  # (B, dz), (B,) step size


class Encoder(nn.Module):
    def __init__(self, product=False):
        super().__init__()
        self.product = product
        self.net = nn.Sequential(nn.Linear(D_OBS, 128), nn.ReLU(),
                                 nn.Linear(128, D_Z))

    def forward(self, x):
        h = self.net(x)
        if self.product:                              # (S^15)^2 product latent
            h = h.reshape(*h.shape[:-1], 2, D_Z // 2)
            return F.normalize(h, dim=-1).reshape(*x.shape[:-1], D_Z)
        return F.normalize(h, dim=-1)


class Predictor(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        half = D_Z // 2
        if mode == "tangent":
            self.net = nn.Sequential(nn.Linear(D_Z + 2, 128), nn.ReLU(),
                                     nn.Linear(128, D_Z))
            self.gate = nn.Linear(D_Z + 2, 1)
            nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        elif mode == "mono_rot":
            self.rot = RotStep(D_Z, D_Z + 2)
        elif mode == "prod_rot":                      # per-patch SO(16), coupled
            self.rot_a = RotStep(half, D_Z + 2)
            self.rot_b = RotStep(half, D_Z + 2)
        elif mode == "tied3":
            self.rot = RotStep(D_Z, D_Z + 2)          # ONE module, applied 3x
        elif mode == "untied3":
            self.rots = nn.ModuleList(RotStep(D_Z, D_Z + 2) for _ in range(3))
        elif mode == "mem_rot":
            dm = 16
            self.q = nn.Linear(D_Z + 2, dm)
            self.kv = nn.Linear(D_Z, 2 * dm)
            self.rot = RotStep(D_Z, D_Z + 2 + dm)
            self.dm = dm

    def forward(self, z, a, mem=None):
        za = torch.cat([z, a], -1)
        if self.mode == "tangent":
            u = self.net(za)
            delta = u - (u * z).sum(-1, keepdim=True) * z
            delta = F.normalize(delta, dim=-1)
            alpha = torch.sigmoid(self.gate(za))
            return F.normalize(z + alpha * delta, dim=-1), alpha.squeeze(-1)
        if self.mode == "mono_rot":
            return self.rot(z, za)
        if self.mode == "prod_rot":
            half = D_Z // 2
            z1, z2 = z[:, :half], z[:, half:]
            o1, t1 = self.rot_a(z1, za)               # cross-patch conditioning
            o2, t2 = self.rot_b(z2, za)
            return torch.cat([o1, o2], -1), torch.stack([t1, t2], -1)  # per-patch probes
        if self.mode == "tied3":
            out, tot = z, 0.0
            for _ in range(3):                        # unrolled spherical flow
                out, th = self.rot(out, torch.cat([out, a], -1))
                tot = tot + th
            return out, tot
        if self.mode == "untied3":
            out, tot = z, 0.0
            for r in self.rots:
                out, th = r(out, torch.cat([out, a], -1))
                tot = tot + th
            return out, tot
        # mem_rot: attention read over past latents, then rotate
        q = self.q(za).unsqueeze(1)                   # (B,1,dm)
        kv = self.kv(mem)                             # (B,M,2dm)
        k, v = kv[..., :self.dm], kv[..., self.dm:]
        att = torch.softmax((q * k).sum(-1) / math.sqrt(self.dm), dim=-1)
        read = (att.unsqueeze(-1) * v).sum(1)         # (B,dm)
        return self.rot(z, torch.cat([za, read], -1))


def ridge_r2(H, S):
    n = H.size(0); ntr = int(0.8 * n)
    Xtr, Xte, Ytr, Yte = H[:ntr], H[ntr:], S[:ntr], S[ntr:]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    A = (Xtr - xm).t() @ (Xtr - xm) + 1e-2 * torch.eye(H.size(1))
    W = torch.linalg.solve(A, (Xtr - xm).t() @ (Ytr - ym))
    pe = (Xte - xm) @ W
    return (1 - (pe - (Yte - ym)).pow(2).sum(0)
            / (Yte - ym).pow(2).sum(0).clamp_min(1e-9)).mean().item()


# --------------------------------------------------------------------------- #
def run(mode, seed):
    torch.manual_seed(seed)
    episodes, g = make_world(seed)
    Xtr, Atr, _, _, _ = episodes(N_EP_TR)
    Xev, Aev, SAe, SBe, CTe = episodes(N_EP_EV)
    enc = Encoder(product=(mode == "prod_rot"))
    pred = Predictor(mode)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-3)

    def proto(z):
        if mode == "prod_rot":
            return simplex_proto_loss(z.reshape(-1, D_Z // 2))
        return simplex_proto_loss(z)

    for step in range(STEPS):
        ep = torch.randint(0, N_EP_TR, (BATCH,), generator=g)
        t = torch.randint(MEM - 1, T_EP - 1, (BATCH,), generator=g)
        z_t = enc(Xtr[ep, t])
        z_n = enc(Xtr[ep, t + 1])
        mem = None
        if mode == "mem_rot":
            past = torch.stack([Xtr[ep, t - i] for i in range(MEM)], 1)
            mem = enc(past.reshape(-1, D_OBS)).reshape(BATCH, MEM, D_Z)
        zp, _ = pred(z_t, Atr[ep, t], mem)
        # normalize both sides: for the product latent (norm sqrt(2)) this makes
        # the loss the mean per-patch cosine; a no-op for single-sphere modes.
        cos = (F.normalize(zp, dim=-1) * F.normalize(z_n.detach(), dim=-1)).sum(-1)
        loss = (1 - cos).mean() + 0.3 * proto(z_t)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        Z = enc(Xev.reshape(-1, D_OBS)).reshape(N_EP_EV, T_EP, D_Z)
        # step-size vs contact correlation on 1-step transitions
        t_idx = torch.arange(MEM - 1, T_EP - 1)
        zs = Z[:, t_idx].reshape(-1, D_Z)
        acts = Aev[:, t_idx].reshape(-1, 2)
        mem = None
        if mode == "mem_rot":
            past = torch.stack([Z[:, t_idx - i] for i in range(MEM)], 2)
            mem = past.reshape(-1, MEM, D_Z)
        _, steps_mag = pred(zs, acts, mem)
        ct = CTe[:, t_idx].reshape(-1)
        if steps_mag.ndim == 1:
            steps_mag = steps_mag.unsqueeze(-1)
        cm = ct - ct.mean()
        corrs = []
        for j in range(steps_mag.size(-1)):           # per-patch for prod_rot
            sm = steps_mag[:, j] - steps_mag[:, j].mean()
            corrs.append(((sm * cm).mean() / (sm.std() * cm.std() + 1e-9)).item())
        corr = max(corrs, key=abs)                    # the "block patch" emerges

        # 10-step free rollout + retrieval (memory fed by own predictions)
        cur = Z[:, MEM - 1]
        buf = [Z[:, max(MEM - 1 - i, 0)] for i in range(MEM)]
        for tt in range(10):
            m = torch.stack(buf[:MEM], 1) if mode == "mem_rot" else None
            cur, _ = pred(cur, Aev[:, MEM - 1 + tt], m)
            buf.insert(0, cur); buf = buf[:MEM]
        tgt = Z[:, MEM - 1 + 10]
        roll = (1 - (F.normalize(cur, dim=-1) * F.normalize(tgt, dim=-1))
                .sum(-1)).mean().item()
        sim = F.normalize(cur, dim=-1) @ F.normalize(tgt, dim=-1).t()
        r1 = (sim.argmax(1) == torch.arange(N_EP_EV)).float().mean().item()

        Hf = Z.reshape(-1, D_Z)
        r2a = ridge_r2(Hf, SAe.reshape(-1, D_S))
        r2b = ridge_r2(Hf, SBe.reshape(-1, D_S))

    n_par = sum(p.numel() for p in pred.parameters())
    return dict(roll10=roll, retr1=r1, r2_pusher=r2a, r2_block=r2b,
                contact_corr=corr, params=n_par)


def main():
    modes = ["tangent", "mono_rot", "prod_rot", "tied3", "untied3", "mem_rot"]
    print(f"{'mode':<10}{'roll10':>8}{'retr@1':>8}{'R2push':>8}{'R2block':>9}"
          f"{'ct-corr':>9}{'params':>9}")
    out = {}
    for mode in modes:
        rs = [run(mode, s) for s in range(3)]
        m = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        out[mode] = m
        print(f"{mode:<10}{m['roll10']:>8.4f}{m['retr1']:>8.3f}{m['r2_pusher']:>8.3f}"
              f"{m['r2_block']:>9.3f}{m['contact_corr']:>9.3f}{int(m['params']):>9d}")
    Path("results/cpu_arch_bench.json").write_text(json.dumps(out, indent=2))
    print("\nct-corr: does the learned step size fire on true contact events?")
    print("wrote results/cpu_arch_bench.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
