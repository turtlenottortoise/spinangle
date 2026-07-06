#!/usr/bin/env python3
"""CPU test of the rotation-core contender against the measured Push-T failure.

Reconstructs the identity-collapse regime in miniature: ground-truth state
lives on a sphere and moves by SMALL rotations each step (~4 deg, like Push-T
at frameskip 5 where consecutive latents are ~3 deg apart), so the identity
map is nearly free under cosine loss. This is exactly the regime where our
tangent predictor's sigmoid gate died (gate -> 0, predictor -> identity).

Contestants (same encoder, same JEPA loss: cosine + stop-grad + simplex reg):
  simple    z' = normalize(MLP(z, a))                       unconstrained
  gated     z' = normalize((1-s)z + s*MLP(z,a)), s=sigmoid  nGPT LERP (variant G)
  tangent   tangent-projected step, alpha=sigmoid scalar     our Push-T casualty
  rot_fixed k plane rotations, fixed planes, learned angles  minimal SO(d)
  rot_learn k plane rotations, planes from (z,a), learned    the contender

The rotation parameterization has NO saturating gate: theta is a raw linear
head (zero-init), the identity is theta=0 where gradients are healthy
(d z'/d theta != 0), and ||z'|| = 1 exactly by construction.

Metrics (eval episodes): pred loss vs the IDENTITY FLOOR (1 - cos of true
consecutive latents), learned step size vs true step size, controllability
(action-swap angular spread / across-state spread), 10-step free rollout
error, retrieval R@1 of the 10-step future.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_reg_bench import CodeSphere  # simplex prototype regularizer (D_Z=32)

torch.set_num_threads(4)

D_S, D_OBS, D_Z = 8, 64, 32
T_EP, N_EP_TR, N_EP_EV, BATCH, STEPS = 30, 512, 128, 128, 600
TRUE_ANGLE = 0.06          # per-action-channel rotation (rad): smooth regime


# --------------------------------------------------------------------------- #
# ground-truth spherical dynamics + observation map
# --------------------------------------------------------------------------- #
def plane_rot(z, p, q, theta):
    """Exact rotation of z by theta in the (p, q) plane (p, q orthonormal)."""
    c1 = (z * p).sum(-1, keepdim=True)
    c2 = (z * q).sum(-1, keepdim=True)
    ct, st = torch.cos(theta), torch.sin(theta)
    return z + (ct - 1) * (c1 * p + c2 * q) + st * (c1 * q - c2 * p)


def make_world(seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D_S, D_S, generator=g))
    planes = [(Q[:, 0], Q[:, 1]), (Q[:, 2], Q[:, 3])]      # two action channels
    W1 = torch.randn(D_S, 128, generator=g) / math.sqrt(D_S)
    W2 = torch.randn(128, D_OBS, generator=g) / math.sqrt(128)

    def obs(s):
        return torch.tanh(s @ W1) @ W2

    def episodes(n):
        s = F.normalize(torch.randn(n, D_S, generator=g), dim=-1)
        S, A = [s], []
        for _ in range(T_EP - 1):
            a = torch.randn(n, 2, generator=g)
            s = plane_rot(s, *planes[0], (TRUE_ANGLE * a[:, :1]))
            s = plane_rot(s, *planes[1], (TRUE_ANGLE * a[:, 1:]))
            s = F.normalize(s, dim=-1)
            S.append(s); A.append(a)
        S = torch.stack(S, 1)                               # (n, T, D_S)
        X = obs(S.reshape(-1, D_S)).reshape(n, T_EP, D_OBS)
        X = X + 0.01 * torch.randn(X.shape, generator=g)
        return X, torch.stack(A, 1), S                      # A: (n, T-1, 2)

    return episodes, g


# --------------------------------------------------------------------------- #
# predictors (z: (B, D_Z) unit; a: (B, 2))
# --------------------------------------------------------------------------- #
class Predictor(torch.nn.Module):
    def __init__(self, mode, k=4, seed=0):
        super().__init__()
        self.mode, self.k = mode, k
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_Z + 2, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, D_Z))
        if mode in ("gated", "tangent"):
            self.gate = torch.nn.Linear(D_Z + 2, 1)
            torch.nn.init.zeros_(self.gate.weight); torch.nn.init.zeros_(self.gate.bias)
        if mode.startswith("rot"):
            self.theta = torch.nn.Linear(D_Z + 2, k)
            torch.nn.init.zeros_(self.theta.weight); torch.nn.init.zeros_(self.theta.bias)
            if mode == "rot_fixed":
                gg = torch.Generator().manual_seed(seed)
                Q, _ = torch.linalg.qr(torch.randn(D_Z, D_Z, generator=gg))
                self.register_buffer("P", Q[:, :2 * k].t().reshape(k, 2, D_Z))
            else:                                            # rot_learn
                self.plane_head = torch.nn.Linear(D_Z + 2, 2 * k * D_Z)

    def step_size(self, z, a):
        za = torch.cat([z, a], -1)
        if self.mode in ("gated", "tangent"):
            return torch.sigmoid(self.gate(za)).mean().item()
        if self.mode.startswith("rot"):
            return self.theta(za).abs().sum(-1).mean().item()
        return float("nan")

    def forward(self, z, a):
        za = torch.cat([z, a], -1)
        if self.mode == "simple":
            return F.normalize(self.net(za), dim=-1)
        if self.mode == "gated":
            u = F.normalize(self.net(za), dim=-1)
            s = torch.sigmoid(self.gate(za))
            return F.normalize((1 - s) * z + s * u, dim=-1)
        if self.mode == "tangent":
            u = self.net(za)
            delta = u - (u * z).sum(-1, keepdim=True) * z
            delta = F.normalize(delta, dim=-1)
            alpha = torch.sigmoid(self.gate(za))
            return F.normalize(z + alpha * delta, dim=-1)
        # rotation modes: sequence of k exact plane rotations
        th = self.theta(za)                                  # (B, k)
        if self.mode == "rot_fixed":
            planes = self.P.unsqueeze(0).expand(z.size(0), -1, -1, -1)
        else:
            raw = self.plane_head(za).reshape(-1, self.k, 2, D_Z)
            planes = raw
        out = z
        for i in range(self.k):
            p = F.normalize(planes[:, i, 0], dim=-1)
            q_raw = planes[:, i, 1]
            q = F.normalize(q_raw - (q_raw * p).sum(-1, keepdim=True) * p, dim=-1)
            out = plane_rot(out, p, q, th[:, i:i + 1])
        return out                                           # exactly unit-norm


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D_OBS, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, D_Z))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
# train + evaluate one variant
# --------------------------------------------------------------------------- #
def run(mode, seed):
    torch.manual_seed(seed)
    episodes, g = make_world(seed)
    Xtr, Atr, _ = episodes(N_EP_TR)
    Xev, Aev, _ = episodes(N_EP_EV)
    enc, pred = Encoder(), Predictor(mode, seed=seed)
    proto = CodeSphere("simplex", seed=seed)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=3e-3)

    for step in range(STEPS):
        ep = torch.randint(0, N_EP_TR, (BATCH,), generator=g)
        t = torch.randint(0, T_EP - 1, (BATCH,), generator=g)
        z_t = enc(Xtr[ep, t])
        z_n = enc(Xtr[ep, t + 1])
        zp = pred(z_t, Atr[ep, t])
        loss = 1 - (zp * z_n.detach()).sum(-1).mean() + 1.0 * proto(z_t)
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- eval ----
    with torch.no_grad():
        Z = enc(Xev.reshape(-1, D_OBS)).reshape(N_EP_EV, T_EP, D_Z)
        floor = (1 - (Z[:, :-1] * Z[:, 1:]).sum(-1)).mean().item()   # identity floor
        true_step = (Z[:, :-1] * Z[:, 1:]).sum(-1).clamp(-1, 1).arccos().mean().item()

        z_t = Z[:, :-1].reshape(-1, D_Z); a = Aev.reshape(-1, 2)
        zp = pred(z_t, a)
        z_n = Z[:, 1:].reshape(-1, D_Z)
        pred_loss = (1 - (zp * z_n).sum(-1)).mean().item()

        # controllability: action swap vs across-state spread
        B0 = min(512, z_t.size(0))
        zb, ab = z_t[:B0], a[:B0]
        preds = [pred(zb, ab)]
        for _ in range(8):
            perm = torch.randperm(B0, generator=g)
            preds.append(pred(zb, ab[perm]))
        P = torch.stack(preds)
        mean_dir = F.normalize(P.mean(0), dim=-1)
        act_rad = (P * mean_dir.unsqueeze(0)).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos().mean().item()
        gdir = F.normalize(preds[0].mean(0, keepdim=True), dim=-1)
        state_rad = (preds[0] * gdir).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos().mean().item()

        # 10-step free rollout + future retrieval
        cur = Z[:, 0]
        for t10 in range(10):
            cur = pred(cur, Aev[:, t10])
        tgt = Z[:, 10]
        roll_err = (1 - (cur * tgt).sum(-1)).mean().item()
        sim = F.normalize(cur, dim=-1) @ F.normalize(tgt, dim=-1).t()
        r1 = (sim.argmax(1) == torch.arange(N_EP_EV)).float().mean().item()

    return dict(mode=mode, pred_loss=pred_loss, identity_floor=floor,
                beats_floor=floor / max(pred_loss, 1e-9),
                learned_step=pred.step_size(z_t[:512], a[:512]),
                true_step=true_step, act_rad=act_rad, state_rad=state_rad,
                act_frac=act_rad / (state_rad + 1e-9),
                roll_err_10=roll_err, retr1_10=r1,
                norm_dev=abs(zp.norm(dim=-1).mean().item() - 1.0))


def main():
    modes = ["simple", "gated", "tangent", "rot_fixed", "rot_learn"]
    print(f"{'mode':<10}{'pred':>8}{'floor':>8}{'x-floor':>8}{'step':>8}"
          f"{'true':>7}{'actfrac':>8}{'roll10':>8}{'retr@1':>8}{'|1-nrm|':>9}")
    out = {}
    for mode in modes:
        rs = [run(mode, s) for s in range(3)]
        m = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0] if k != "mode"}
        out[mode] = m
        print(f"{mode:<10}{m['pred_loss']:>8.4f}{m['identity_floor']:>8.4f}"
              f"{m['beats_floor']:>8.2f}{m['learned_step']:>8.4f}{m['true_step']:>7.3f}"
              f"{m['act_frac']:>8.3f}{m['roll_err_10']:>8.4f}{m['retr1_10']:>8.3f}"
              f"{m['norm_dev']:>9.2e}")
    Path("results").mkdir(exist_ok=True)
    Path("results/cpu_rotation_bench.json").write_text(json.dumps(out, indent=2))
    print("\nx-floor > 1 means the predictor beats the identity map;")
    print("step vs true shows whether the learned step size tracks real motion.")
    print("wrote results/cpu_rotation_bench.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {time.time()-t0:.1f}s")
