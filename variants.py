"""nGPT-JEPA variants for LeWorldModel.

This module is *additive*: it does not change the official LeWM model, loss, or
planner. It provides spherical-transition predictor classes (built on top of the
official ``ARPredictor`` backbone so capacity stays matched), a projector used to
apply SIGReg on a Gaussian latent ``z = Projector(h)`` instead of directly on the
unit-sphere state ``h``, and a small in-batch memory/NCE helper.

Predictor variants implemented (see project brief, items 1-6):

  1. ARPredictor                  -> official LeWM (imported from module.py, unchanged)
  2. PlainMLPPredictor            -> minimal Euclidean per-step MLP (no transformer)
  3. SphericalARPredictor(simple) -> h_pred = normalize(UpdateNet(h_t, a_t))
  4. SphericalARPredictor(residual) -> h_pred = normalize(h_t + alpha * normalize(U))
  5. SphericalARPredictor(gated)  -> u = normalize(U); g = sigmoid(Gate);
                                     h_pred = normalize((1 - g) * h_t + g * u)
  6. SphericalARPredictor(ssm)    -> keep = sigmoid(K); write = sigmoid(W);
                                     h_pred = normalize(keep * h_t + write * normalize(C))

All spherical predictors are called with the exact same signature as the official
predictor, ``forward(x, c)`` where ``x`` is the (unit-norm) latent sequence
``(B, T, D)`` and ``c`` is the action-embedding sequence ``(B, T, A)``. They return
a unit-norm prediction sequence ``(B, T, D)``, so ``JEPA.predict`` and ``JEPA.rollout``
work unchanged.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from module import MLP, ARPredictor


def l2norm(x, dim=-1, eps=1e-6):
    """Project onto the unit sphere along ``dim``."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


class PlainMLPPredictor(nn.Module):
    """Minimal Euclidean per-step predictor (no transformer, no normalization).

    A deliberately simple, Markov ``h_{t+1} = MLP(h_t, a_t)`` baseline. The
    primary "non-normalized MSE JEPA" baseline (variant C) is the official
    ARPredictor with SIGReg disabled; this class is provided for completeness so
    the contribution of the AR transformer backbone itself can be isolated.
    """

    def __init__(self, *, input_dim, hidden_dim, output_dim=None, act_dim=None,
                 num_frames=None, **_ignored):
        super().__init__()
        act_dim = act_dim if act_dim is not None else input_dim
        self.net = MLP(
            input_dim=input_dim + act_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            norm_fn=nn.LayerNorm,
        )

    def forward(self, x, c):
        # x: (B, T, D), c: (B, T, A)
        return self.net(torch.cat([x, c], dim=-1))


class SphericalARPredictor(nn.Module):
    """Spherical-transition predictor wrapping the official ``ARPredictor``.

    The wrapped ``ARPredictor`` is used verbatim as the *update network*
    ``U(h_t, a_t)`` (same depth/heads/mlp_dim as official LeWM, so the backbone
    capacity is matched). To match the parameter budget of the official
    ``pred_proj`` MLP, an optional ``update_mlp`` (LayerNorm variant, identical
    shape to ``pred_proj``) is appended to the update network; spherical configs
    set the external ``pred_proj`` to Identity so geometry is not destroyed after
    the predictor.

    Modes:
      - ``simple``   : h_pred = normalize(U)
      - ``residual`` : h_pred = normalize(h_t + alpha * normalize(U))
      - ``gated``    : u = normalize(U); g = sigmoid(Gate(h_t, a_t));
                       h_pred = normalize((1 - g) * h_t + g * u)
      - ``ssm``      : keep = sigmoid(K(h_t, a_t)); write = sigmoid(W(h_t, a_t));
                       h_pred = normalize(keep * h_t + write * normalize(U))
      - ``tangent``  : d = U - (U.h_t)h_t (tangent projection); d = normalize(d);
                       alpha = sigmoid(Step(h_t, a_t)) (scalar gate);
                       h_pred = normalize(h_t + alpha * d)  -- Riemannian step; the
                       direction is a pure unit tangent vector and alpha is the gate,
                       so no capacity is spent on the radial DOF normalize() deletes.

    The gate / keep / write heads are functions of (h_t, a_t) as specified in the
    brief (NOT of the candidate u), so the model decides *how far* to move along
    the sphere from the current state before seeing the proposed endpoint.
    """

    VALID_MODES = ("simple", "residual", "gated", "ssm", "tangent")

    def __init__(
        self,
        *,
        mode="gated",
        alpha=1.0,
        gate_dim="channel",          # "channel" (per-dim) or "scalar"
        gate_bias_init=0.0,          # init bias of gate head (0 -> g~0.5)
        anchor_beta=0.0,             # recurrent-anchor residual weight (0 = off)
        update_mlp=True,             # append a pred_proj-shaped MLP to U for capacity match
        update_mlp_hidden=2048,
        # ---- ARPredictor backbone kwargs (same names as module.ARPredictor) ----
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        act_dim=None,
    ):
        super().__init__()
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.mode = mode
        self.alpha = float(alpha)
        self.gate_dim = gate_dim
        self.anchor_beta = float(anchor_beta)
        # probe buffer (scalar diagnostics, filled each forward; read by the
        # training objective / offline eval to test the mechanism predictions).
        self.probe = {}
        D = output_dim or input_dim
        self.dim = D
        # action-embedding dimension feeding the gate heads (defaults to D, which
        # matches the official Embedder whose emb_dim == embed_dim).
        A = act_dim if act_dim is not None else input_dim

        self.update_net = ARPredictor(
            num_frames=num_frames,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.update_mlp = (
            MLP(input_dim=D, hidden_dim=update_mlp_hidden, output_dim=D, norm_fn=nn.LayerNorm)
            if update_mlp
            else nn.Identity()
        )

        gate_out = D if gate_dim == "channel" else 1
        if mode == "gated":
            self.gate = nn.Linear(D + A, gate_out)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias_init)
        elif mode == "tangent":
            # Step size alpha MUST be a scalar per token: a per-channel alpha would
            # warp the tangent vector so alpha*delta is no longer orthogonal to h,
            # reintroducing the wasted radial component normalize() deletes.
            self.gate = nn.Linear(D + A, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias_init)
        elif mode == "ssm":
            self.keep = nn.Linear(D + A, gate_out)
            self.write = nn.Linear(D + A, gate_out)
            # init so keep~1 (retain state) and write~0 (write little) at start:
            nn.init.zeros_(self.keep.weight)
            nn.init.constant_(self.keep.bias, 2.0)   # sigmoid(2) ~ 0.88
            nn.init.zeros_(self.write.weight)
            nn.init.constant_(self.write.bias, -2.0)  # sigmoid(-2) ~ 0.12

    def _anchor(self, x):
        # Recurrent-anchor residual: pull each step toward the oldest state in the
        # current window (x[:, :1]). Over an autoregressive rollout this anchors to
        # the trailing reference HS steps back -- a horizon-axis value-residual. The
        # planner-level GOAL anchor (anchor = goal_emb) is the deeper variant; see
        # RESEARCH_NOTES.md ("recurrent anchor").
        return x[:, :1].expand_as(x) if self.anchor_beta > 0 else 0.0

    def forward(self, x, c):
        # x: (B, T, D) current unit state h_t ; c: (B, T, A) action embedding a_t
        u_raw = self.update_net(x, c)
        u_raw = self.update_mlp(u_raw)
        u = l2norm(u_raw)
        anchor = self._anchor(x)
        b = self.anchor_beta

        if self.mode == "simple":
            h_pred = u if b == 0 else l2norm(u + b * anchor)
        elif self.mode == "residual":
            h_pred = l2norm(x + self.alpha * u + b * anchor)
        elif self.mode == "gated":
            g = torch.sigmoid(self.gate(torch.cat([x, c], dim=-1)))
            h_pred = l2norm((1.0 - g) * x + g * u + b * anchor)
            self._stash_gate(g)
        elif self.mode == "tangent":
            # Learned tangent-space step (Riemannian retraction): project the update
            # into the tangent plane at h, take a unit direction, step by a learned
            # (gated) SCALAR size alpha. Unlike the LERP, this never spends capacity on
            # the radial direction that normalize() removes; alpha is the step gate.
            xn = l2norm(x)
            delta = u_raw - (u_raw * xn).sum(-1, keepdim=True) * xn   # remove radial part
            delta = l2norm(delta)                                     # unit tangent dir
            alpha = torch.sigmoid(self.gate(torch.cat([x, c], dim=-1)))   # (B, T, 1)
            h_pred = l2norm(xn + alpha * delta + b * anchor)
            self._stash_gate(alpha)
        else:  # ssm
            keep = torch.sigmoid(self.keep(torch.cat([x, c], dim=-1)))
            write = torch.sigmoid(self.write(torch.cat([x, c], dim=-1)))
            h_pred = l2norm(keep * x + write * u + b * anchor)
            self._stash_gate(write, keep=keep)

        self._stash_step(x, h_pred)
        return h_pred

    @torch.no_grad()
    def _stash_gate(self, g, keep=None):
        self.probe["gate_mean"] = g.mean().item()
        self.probe["gate_std"] = g.std().item()
        # fraction of "active" updates -- tests the sparse-event prediction
        self.probe["gate_frac_active"] = (g > 0.5).float().mean().item()
        if keep is not None:
            self.probe["keep_mean"] = keep.mean().item()

    @torch.no_grad()
    def _stash_step(self, x, h_pred):
        # realized per-step angle psi_t = arccos(<h_t, h_pred>) in radians
        cos = (l2norm(x) * h_pred).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        self.probe["step_angle_mean"] = cos.arccos().mean().item()


class SIGRegProjector(nn.Module):
    """Maps a unit-sphere state h to a Gaussian latent z for SIGReg.

    Used by variant G so SIGReg / isotropic-Gaussian pressure is applied to
    ``z = Projector(h)`` rather than directly to the unit-sphere world state
    ``h`` (the two geometries are otherwise in conflict).
    """

    def __init__(self, input_dim, hidden_dim=2048, output_dim=None):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            norm_fn=nn.LayerNorm,
        )

    def forward(self, x):
        return self.net(x)


def memory_nce_loss(pred, target, temperature=0.1):
    """In-batch InfoNCE: predicted future state should retrieve its own target.

    pred, target: (B, T, D). Targets are unit-normalized and detached (used as a
    lookup memory of future/goal keys). Returns a scalar cross-entropy loss.
    """
    p = l2norm(rearrange(pred, "b t d -> (b t) d"))
    k = l2norm(rearrange(target, "b t d -> (b t) d")).detach()
    logits = (p @ k.t()) / temperature
    labels = torch.arange(p.size(0), device=p.device)
    return F.cross_entropy(logits, labels)


def pairwise_cosine_penalty(emb):
    """Spherical anti-collapse: mean off-diagonal cosine similarity of unit states.

    emb: (B, T, D). Minimizing this discourages all states from clumping to one
    point on the sphere (the spherical analogue of SIGReg's anti-collapse role).
    """
    h = l2norm(rearrange(emb, "b t d -> (b t) d"))
    sim = h @ h.t()
    n = h.size(0)
    off = (sim.sum() - sim.diagonal().sum()) / (n * (n - 1) + 1e-9)
    return off
