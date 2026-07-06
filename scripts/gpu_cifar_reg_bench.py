#!/usr/bin/env python3
"""Real-scale static-SSL regularizer comparison (papers 1+2, GPU).

SimSiam-style JEPA on CIFAR-10: ResNet-18 -> projector -> S^127, cosine
prediction with stop-grad, plus one uniformity regularizer per run. Evaluates
kNN / ridge linear probe on the sphere embedding, clumping / mean-norm,
CLEAN-vs-NOISY kNN (robustness gap) and an empirical Lipschitz probe -- so one
sweep serves both the finite-beats-continuous ranking and the
uniformity-robustness tradeoff at real scale.

Self-contained: torch + torchvision only (no LeWM stack). Idempotent: one JSON
per method in --out; finished methods are skipped.

Typical timing: ~17 min/method/40ep on L4 (bs 512, AMP), ~7 min on A100.
  python scripts/gpu_cifar_reg_bench.py --out /content/drive/MyDrive/spinangle_benchmark/cifar_reg
  python scripts/gpu_cifar_reg_bench.py --smoke   # CPU wiring check
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# regularizers on unit-norm z (B, D)
# --------------------------------------------------------------------------- #
def reg_sigreg(z, sigma=1.0, R=64):
    B, d = z.shape
    u = F.normalize(torch.randn(R, d, device=z.device), dim=-1)
    t = (z @ u.t()).sort(dim=0)[0]
    q = torch.erfinv(2 * (torch.arange(B, device=z.device) + 0.5) / B - 1)
    q = q * math.sqrt(2) * sigma
    return (t - q.unsqueeze(1)).pow(2).mean()


def reg_mmd_energy(z, t=2.0):
    sq = torch.cdist(z, z).pow(2)
    B = z.size(0)
    off = ~torch.eye(B, dtype=torch.bool, device=z.device)
    return torch.logsumexp(-t * sq[off], dim=0) - math.log(B * (B - 1))


_SIMPLEX = {}


def simplex_protos(d, device):
    if d not in _SIMPLEX:
        K = d + 1
        M = torch.eye(K) - torch.full((K, K), 1.0 / K)
        _, _, Vt = torch.linalg.svd(M)
        _SIMPLEX[d] = F.normalize(M @ Vt[:d].t(), dim=-1)
    return _SIMPLEX[d].to(device)


def reg_simplex(z, tau=0.1):
    C = simplex_protos(z.size(-1), z.device)
    sim = z @ C.t()
    with torch.no_grad():
        Q = torch.exp(sim.float() / tau)
        for _ in range(3):
            Q = Q / Q.sum(0, keepdim=True).clamp_min(1e-12)
            Q = Q / Q.sum(1, keepdim=True).clamp_min(1e-12)
        Q = Q.to(sim.dtype)
    return -(Q * sim).sum(1).mean()


def loss_nce(z1, z2, tau=0.2):
    logits = z1 @ z2.t() / tau
    return F.cross_entropy(logits, torch.arange(z1.size(0), device=z1.device))


METHODS = {  # name -> (reg_fn or None, lambda, add_weak_nce)
    "none":          (None, 0.0, False),
    "sigreg":        (lambda z: reg_sigreg(z, 1.0), 4.0, False),
    "sigreg_1overd": (lambda z: reg_sigreg(z, None), 4.0, False),  # sigma set at runtime
    "mmd_energy":    (reg_mmd_energy, 0.5, False),
    "simplex":       (reg_simplex, 1.0, False),
    "simplex_nce":   (reg_simplex, 1.0, True),
}


# --------------------------------------------------------------------------- #
def build_model(dz=128, smoke=False):
    if smoke:
        backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU())
        feat = 256
    else:
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem
        backbone.maxpool = nn.Identity()
        feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
    proj = nn.Sequential(nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
                         nn.Linear(512, dz))
    pred = nn.Sequential(nn.Linear(dz, 512), nn.BatchNorm1d(512), nn.ReLU(),
                         nn.Linear(512, dz))
    return backbone, proj, pred


def get_data(root, smoke=False):
    if smoke:
        x = torch.rand(512, 3, 32, 32)
        y = torch.randint(0, 10, (512,))
        return (x, y), (x[:256], y[:256])
    import torchvision
    import torchvision.transforms as T
    tt = T.ToTensor()
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tt)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tt)
    xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor([te[i][1] for i in range(len(te))])
    return (xtr, ytr), (xte, yte)


def augment(x, g):
    B = x.size(0)
    out = x
    if torch.rand((), generator=g) < 2.0:      # always: random crop via padding
        pad = F.pad(out, (4, 4, 4, 4), mode="reflect")
        i = torch.randint(0, 9, (2,), generator=g)
        out = pad[:, :, i[0]:i[0] + 32, i[1]:i[1] + 32]
    flip = torch.rand(B, generator=g, device=x.device) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(-1), out)
    # brightness/contrast jitter (cheap, on-GPU)
    b = 1 + 0.4 * (torch.rand(B, 1, 1, 1, generator=g, device=x.device) - 0.5)
    m = out.mean(dim=(1, 2, 3), keepdim=True)
    out = ((out - m) * b + m).clamp(0, 1)
    return out


@torch.no_grad()
def embed(backbone, proj, x, device, bs=1024):
    zs = []
    for i in range(0, x.size(0), bs):
        z = proj(backbone(x[i:i + bs].to(device)))
        zs.append(F.normalize(z, dim=-1).cpu())
    return torch.cat(zs)


@torch.no_grad()
def evaluate(backbone, proj, tr, te, device):
    (xtr, ytr), (xte, yte) = tr, te
    backbone.eval(); proj.eval()
    ztr, zte = embed(backbone, proj, xtr, device), embed(backbone, proj, xte, device)

    def knn(q):
        acc = 0
        for i in range(0, q.size(0), 512):
            nbr = (q[i:i + 512] @ ztr.t()).topk(20, dim=1).indices
            acc += (torch.mode(ytr[nbr], 1).values == yte[i:i + 512]).sum().item()
        return acc / q.size(0)

    clean = knn(zte)
    znoisy = embed(backbone, proj, (xte + 0.10 * torch.randn_like(xte)).clamp(0, 1),
                   device)
    noisy = knn(znoisy)
    # Lipschitz probe: angular displacement per unit input perturbation
    delta = 0.02 * F.normalize(torch.randn_like(xte[:2000]).flatten(1), dim=-1)
    delta = delta.reshape(-1, 3, 32, 32)
    zp = embed(backbone, proj, (xte[:2000] + delta).clamp(0, 1), device)
    ang = (zte[:2000] * zp).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
    lip = (ang / delta.flatten(1).norm(dim=-1)).mean().item()

    Y = F.one_hot(ytr, 10).float()
    A = ztr.t() @ ztr + 1e-2 * torch.eye(ztr.size(1))
    W = torch.linalg.solve(A, ztr.t() @ Y)
    lin = ((zte @ W).argmax(1) == yte).float().mean().item()
    sub = zte[:4000]
    sim = sub @ sub.t()
    clump = ((sim.sum() - sub.size(0)) / (sub.size(0) * (sub.size(0) - 1))).item()
    return dict(knn=clean, knn_noisy=noisy, gap=clean - noisy, lip=lip,
                linear=lin, clump=clump, mean_norm=zte.mean(0).norm().item())


def run_method(name, tr, te, args, device):
    torch.manual_seed(args.seed)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    backbone, proj, pred = build_model(args.dz, args.smoke)
    backbone, proj, pred = backbone.to(device), proj.to(device), pred.to(device)
    params = list(backbone.parameters()) + list(proj.parameters()) + list(pred.parameters())
    opt = torch.optim.SGD(params, lr=0.06, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, tr[0].size(0) // args.bs))
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    reg_fn, lam, add_nce = METHODS[name]
    if name == "sigreg_1overd":
        reg_fn = lambda z: reg_sigreg(z, 1.0 / math.sqrt(args.dz))

    xtr, _ = tr
    n = xtr.size(0)
    t0 = time.time()
    for ep in range(args.epochs):
        backbone.train(); proj.train(); pred.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - args.bs + 1, args.bs):
            xb = xtr[perm[i:i + args.bs]].to(device, non_blocking=True)
            v1, v2 = augment(xb, g), augment(xb, g)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                z1 = F.normalize(proj(backbone(v1)), dim=-1)
                z2 = F.normalize(proj(backbone(v2)), dim=-1)
                p1 = F.normalize(pred(z1), dim=-1)
                p2 = F.normalize(pred(z2), dim=-1)
                loss = 1 - 0.5 * ((p1 * z2.detach()).sum(-1).mean()
                                  + (p2 * z1.detach()).sum(-1).mean())
                if reg_fn is not None and lam > 0:
                    loss = loss + lam * reg_fn(z1.float())
                if add_nce:
                    loss = loss + 0.25 * loss_nce(z1.float(), z2.float().detach())
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        print(f"[{name}] epoch {ep + 1}/{args.epochs} loss={loss.item():.4f} "
              f"({(time.time() - t0) / (ep + 1):.1f}s/ep)", flush=True)
    m = evaluate(backbone, proj, tr, te, device)
    m.update(method=name, lam=lam, epochs=args.epochs, seed=args.seed,
             wall_s=round(time.time() - t0, 1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/cifar_reg")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--methods", default="none,sigreg,sigreg_1overd,mmd_energy,simplex,simplex_nce")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--dz", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.bs = 1, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr, te = get_data(args.data_root, args.smoke)
    print(f"device={device} train={tr[0].shape} methods={args.methods}")

    rows = []
    for name in args.methods.split(","):
        sink = out / f"{name}_s{args.seed}.json"
        if sink.exists():
            rows.append(json.loads(sink.read_text()))
            print(f"[skip] {sink.name} exists"); continue
        m = run_method(name, tr, te, args, device)
        sink.write_text(json.dumps(m, indent=2))
        rows.append(m)
        print(f"[done] {name}: {m}")

    print(f"\n{'method':<14}{'knn':>7}{'noisy':>7}{'gap':>7}{'lip':>7}"
          f"{'linear':>8}{'clump':>8}{'|mean|':>8}{'min':>6}")
    for m in rows:
        print(f"{m['method']:<14}{m['knn']:>7.3f}{m['knn_noisy']:>7.3f}"
              f"{m['gap']:>7.3f}{m['lip']:>7.2f}{m['linear']:>8.3f}"
              f"{m['clump']:>8.3f}{m['mean_norm']:>8.3f}{m['wall_s']/60:>6.1f}")


if __name__ == "__main__":
    main()
