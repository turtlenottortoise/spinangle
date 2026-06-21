"""Evaluation metrics for LeWM / nGPT-JEPA latent world models.

Pure tensor functions (no training-stack deps) so they can be unit-tested on CPU
and reused by the offline eval driver (scripts/eval_latent_metrics.py). Covers the
brief's latent-rollout, retrieval, and representation metric families. Control /
planning metrics come from LeWM's own evaluation (eval.py) and are not re-derived
here.

Run ``python metrics.py`` for a self-check.
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Latent rollout
# --------------------------------------------------------------------------- #
def rollout_errors(pred, tgt, horizons=(1, 5, 10, 20), spherical=False):
    """Per-horizon prediction error between a rolled-out trajectory and the true
    (encoded) trajectory.

    pred, tgt: (B, L, D) aligned so that pred[:, k] is the model's estimate of the
    latent at rollout step k+1 and tgt[:, k] is the true encoded latent there.
    For spherical models both are unit-norm and we report cosine distance
    1 - cos; otherwise we report MSE (summed over D, averaged over B).
    Returns {k: error} for each horizon k that fits within L.
    """
    out = {}
    L = pred.size(1)
    for k in horizons:
        if k > L:
            continue
        p, t = pred[:, k - 1], tgt[:, k - 1]
        if spherical:
            p = F.normalize(p, dim=-1)
            t = F.normalize(t, dim=-1)
            out[k] = (1.0 - (p * t).sum(-1)).mean().item()
        else:
            out[k] = (p - t).pow(2).sum(-1).mean().item()
    return out


def rollout_drift(pred, tgt, spherical=False):
    """Total accumulated error along the rollout (sum over steps of per-step error)."""
    errs = rollout_errors(pred, tgt, horizons=range(1, pred.size(1) + 1),
                          spherical=spherical)
    return float(sum(errs.values()))


def norm_drift(traj):
    """Std of latent norm across the rollout horizon (0 for perfectly spherical
    models; grows when Euclidean latents explode/shrink). traj: (B, L, D)."""
    norms = traj.norm(dim=-1)              # (B, L)
    return norms.std(dim=1).mean().item()


def angular_drift(traj):
    """Mean step-to-step angle (radians) along a (unit-norm) latent rollout.
    traj: (B, L, D). Large values => the trajectory swings erratically."""
    a = F.normalize(traj[:, :-1], dim=-1)
    b = F.normalize(traj[:, 1:], dim=-1)
    cos = (a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    return cos.arccos().mean().item()


# --------------------------------------------------------------------------- #
# Retrieval (future-state / goal-state)
# --------------------------------------------------------------------------- #
def retrieval_metrics(query, keys, num_candidates=None, seed=0):
    """Query i must retrieve key i among a candidate pool.

    query, keys: (M, D). For each query we rank candidate keys by cosine
    similarity. If ``num_candidates`` (e.g. 32/128/256) is given, we sample that
    many candidates per query (always including the true key) as hard distractors;
    None uses all M keys. Returns retrieval@1, @5 and MRR.
    """
    q = F.normalize(query, dim=-1)
    k = F.normalize(keys, dim=-1)
    M = q.size(0)
    g = torch.Generator().manual_seed(seed)

    if num_candidates is None or num_candidates >= M:
        sims = q @ k.t()                                  # (M, M)
        ranks = _ranks_of_true(sims, torch.arange(M))
    else:
        ranks = torch.empty(M, dtype=torch.long)
        for i in range(M):
            pool = torch.randperm(M, generator=g)[:num_candidates]
            if (pool == i).any():
                idx = pool
            else:
                idx = torch.cat([pool[:-1], torch.tensor([i])])
            true_pos = (idx == i).nonzero(as_tuple=True)[0].item()
            sims = (q[i:i + 1] @ k[idx].t()).squeeze(0)   # (num_candidates,)
            ranks[i] = _ranks_of_true(sims.unsqueeze(0), torch.tensor([true_pos]))[0]

    r = ranks.float() + 1.0
    return {
        "r@1": (ranks == 0).float().mean().item(),
        "r@5": (ranks < 5).float().mean().item(),
        "mrr": (1.0 / r).mean().item(),
    }


def _ranks_of_true(sims, true_idx):
    """Rank (0-based) of the true item per row, by descending similarity."""
    order = sims.argsort(dim=1, descending=True)
    eq = order == true_idx.unsqueeze(1)
    return eq.float().argmax(dim=1)


# --------------------------------------------------------------------------- #
# Representation geometry
# --------------------------------------------------------------------------- #
def effective_rank(emb, center=True, eps=1e-12):
    """Effective rank (Roy & Vetterli 2007): exp(entropy of normalized singular
    value spectrum). emb: (N, D). Higher => representation uses more dimensions."""
    x = emb.float()
    if center:
        x = x - x.mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


def mean_pairwise_cosine(emb):
    """Mean off-diagonal pairwise cosine similarity (clumping). emb: (N, D).
    ~0 for a well-spread representation; ->1 indicates collapse to a point."""
    h = F.normalize(emb.float(), dim=-1)
    sim = h @ h.t()
    n = h.size(0)
    return ((sim.sum() - sim.diagonal().sum()) / (n * (n - 1) + 1e-9)).item()


def knn_probe(emb, labels, k=5):
    """Leave-one-out kNN accuracy of a (frozen) representation. emb: (N, D),
    labels: (N,). Returns top-1 accuracy with cosine similarity."""
    h = F.normalize(emb.float(), dim=-1)
    sim = h @ h.t()
    sim.fill_diagonal_(-1e9)
    nbr = sim.topk(k, dim=1).indices               # (N, k)
    votes = labels[nbr]                            # (N, k)
    pred = torch.mode(votes, dim=1).values
    return (pred == labels).float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 256, 32

    # rollout: a perfect spherical predictor has ~0 error at every horizon
    tgt = F.normalize(torch.randn(8, 20, D), dim=-1)
    assert max(rollout_errors(tgt, tgt, spherical=True).values()) < 1e-5
    drifted = F.normalize(tgt + 0.3 * torch.randn_like(tgt), dim=-1)
    re = rollout_errors(drifted, tgt, spherical=True)
    assert re[1] < re[20], "rollout error should grow with horizon under drift"
    print("rollout_errors (drift):", {k: round(v, 4) for k, v in re.items()})
    print("angular_drift:", round(angular_drift(tgt), 4),
          "| norm_drift(euclid):", round(norm_drift(torch.randn(8, 20, D)), 4))

    # retrieval: identical keys -> perfect; random -> chance-ish
    emb = F.normalize(torch.randn(N, D), dim=-1)
    perfect = retrieval_metrics(emb, emb.clone())
    assert perfect["r@1"] > 0.99 and perfect["mrr"] > 0.99
    for nc in (32, 128, None):
        m = retrieval_metrics(emb, emb + 0.05 * torch.randn_like(emb), num_candidates=nc)
        print(f"retrieval (cand={nc}):", {k: round(v, 3) for k, v in m.items()})

    # representation geometry
    print("effective_rank(iso):", round(effective_rank(torch.randn(N, D)), 2),
          "| clumping(iso):", round(mean_pairwise_cosine(torch.randn(N, D)), 4))
    collapsed = torch.randn(1, D).repeat(N, 1) + 1e-3 * torch.randn(N, D)
    assert mean_pairwise_cosine(collapsed) > 0.9, "clumping should flag collapse"
    # collapse shows up in the *uncentered* spectrum (rank ~1); centered effrank
    # instead measures deviations from the mean (full-rank noise here), so use both.
    assert effective_rank(collapsed, center=False) < 3.0, "uncentered effrank should flag collapse"
    print("collapse: clumping=%.3f effrank_uncentered=%.2f effrank_centered=%.2f" % (
        mean_pairwise_cosine(collapsed),
        effective_rank(collapsed, center=False),
        effective_rank(collapsed, center=True)))

    labels = torch.randint(0, 4, (N,))
    centers = F.normalize(torch.randn(4, D), dim=-1)[labels]
    clustered = F.normalize(centers + 0.1 * torch.randn(N, D), dim=-1)
    print("knn_probe(clustered):", round(knn_probe(clustered, labels), 3),
          "| knn_probe(random):", round(knn_probe(emb, labels), 3))
    print("\nmetrics.py self-check PASSED")
