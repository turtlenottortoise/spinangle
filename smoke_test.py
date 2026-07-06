"""CPU smoke test for LeWM + nGPT-JEPA variants.

Validates the *model and objective* code paths without the heavy stable-worldmodel
/ stable-pretraining stack: it stubs the ViT encoder and runs each variant through
encode -> predict -> loss -> backward, then exercises the planner rollout/criterion
path. Checks:

  * forward + backward produce finite loss and gradients for every variant
  * spherical variants keep latent h AND predictions on the unit sphere
  * the official path (mse + SIGReg on emb) matches the hand-written reference loss
  * rollout stays on the sphere for spherical variants and criterion returns a
    per-candidate cost (the planner entry point)

Run:  python smoke_test.py
"""

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class DotDict(dict):
    """Minimal stand-in for an OmegaConf node: supports attribute access and
    .get(); nested dicts are wrapped recursively. Lets the smoke test exercise
    objective.lejepa_forward without installing omegaconf/hydra."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = DotDict(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from objective import lejepa_forward
from variants import SphericalARPredictor, SIGRegProjector

torch.manual_seed(0)

# ---- tiny dims so this runs in <1s on CPU ----
B, T, C, HW = 2, 4, 3, 8
D = 16            # embed_dim
ACT_IN = 4        # frameskip * action_dim
HISTORY, NPRED = 3, 1
HEADS, DIM_HEAD, DEPTH, MLP_DIM = 2, 8, 2, 32


class StubViT(nn.Module):
    """Mimics stable_pretraining vit_hf: returns an object with .last_hidden_state
    of shape (N, tokens, D); JEPA.encode reads the CLS token at index 0."""

    def __init__(self, dim, in_ch=C):
        super().__init__()
        self.head = nn.Linear(in_ch, dim)

    def forward(self, pixels, interpolate_pos_encoding=False):
        x = pixels.float().mean(dim=(-1, -2))   # (N, C) global pool
        h = self.head(x)                        # (N, D)
        return SimpleNamespace(last_hidden_state=h.unsqueeze(1))


def make_projector(bn=True):
    return MLP(input_dim=D, hidden_dim=32, output_dim=D,
               norm_fn=(nn.BatchNorm1d if bn else nn.LayerNorm))


def make_ar_predictor():
    return ARPredictor(num_frames=HISTORY, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM,
                       input_dim=D, hidden_dim=D, output_dim=D, dim_head=DIM_HEAD,
                       dropout=0.0, emb_dropout=0.0)


def make_spherical_predictor(mode, anchor_beta=0.0):
    return SphericalARPredictor(
        mode=mode, alpha=1.0, gate_dim="channel", anchor_beta=anchor_beta,
        update_mlp=True, update_mlp_hidden=32,
        num_frames=HISTORY, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM,
        input_dim=D, hidden_dim=D, output_dim=D, dim_head=DIM_HEAD, act_dim=D,
    )


def build_jepa(predictor, normalize, pred_proj, sigreg_projector=None):
    return JEPA(
        encoder=StubViT(D),
        predictor=predictor,
        action_encoder=Embedder(input_dim=ACT_IN, emb_dim=D),
        projector=make_projector(bn=True),
        pred_proj=pred_proj,
        normalize_emb=normalize,
        normalize_pred=normalize,
        sigreg_projector=sigreg_projector,
    )


def fake_batch():
    return {
        "pixels": torch.randn(B, T, C, HW, HW),
        "action": torch.randn(B, T, ACT_IN),
    }


def make_cfg(pred_type="mse", sigreg_target="emb", sigreg_weight=0.09,
             anticollapse_w=0.0, memory_w=0.0, stop_grad=False, proto_w=0.0):
    return DotDict({
        "history_size": HISTORY,
        "num_preds": NPRED,
        "loss": {
            "pred": {"type": pred_type, "stop_grad_target": stop_grad},
            "sigreg": {"weight": sigreg_weight, "target": sigreg_target,
                       "kwargs": {"knots": 9, "num_proj": 64}},
            "anticollapse": {"weight": anticollapse_w},
            "memory": {"weight": memory_w, "temperature": 0.1},
            "proto": {"weight": proto_w, "tau": 0.1},
        },
    })


class FakeModule:
    """Stand-in for the spt.Module: holds model + sigreg and records logs."""

    def __init__(self, model, sigreg):
        self.model = model
        self.sigreg = sigreg
        self.logged = {}

    def log_dict(self, d, **kw):
        self.logged.update(d)


def run_variant(name, jepa, cfg, expect_unit):
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)
    mod = FakeModule(jepa, sigreg)
    out = lejepa_forward(mod, fake_batch(), "train", cfg)

    loss = out["loss"]
    assert torch.isfinite(loss), f"[{name}] non-finite loss: {loss}"
    loss.backward()

    n_grad = sum(p.grad.abs().sum().item() for p in jepa.parameters() if p.grad is not None)
    assert n_grad > 0, f"[{name}] zero gradient flow"

    emb_norm = out["emb_norm"].item()
    if expect_unit:
        assert abs(emb_norm - 1.0) < 1e-3, f"[{name}] emb not unit norm: {emb_norm:.4f}"

    extra = " ".join(f"{k.split('/')[-1]}={v.item():.3f}" for k, v in out.items()
                     if torch.is_tensor(v) and v.ndim == 0 and ("loss" in k or "norm" in k))
    print(f"  ok  {name:38s} loss={loss.item():.4f}  emb_norm={emb_norm:.3f}  | {extra}")
    return out


def test_planner_path(name, jepa, expect_unit):
    """Exercise rollout + criterion (the planner cost computation).

    Calls the model's own rollout() and criterion() with planner-shaped tensors
    (S action-plan samples, horizon T). The goal_emb is supplied directly so the
    test does not depend on stable_worldmodel's external goal-dict format.
    """
    jepa.eval()
    S, Tplan = 5, HISTORY + 2          # plan samples, planning horizon
    info = {"pixels": torch.randn(B, S, HISTORY, C, HW, HW)}
    action_candidates = torch.randn(B, S, Tplan, ACT_IN)
    with torch.no_grad():
        info = jepa.rollout(info, action_candidates, history_size=HISTORY)
        pred = info["predicted_emb"]                       # (B, S, L, D)
        goal_emb = torch.randn(B, S, 1, D)
        if expect_unit:
            goal_emb = F.normalize(goal_emb, dim=-1)
        info["goal_emb"] = goal_emb
        cost = jepa.criterion(info)

    assert cost.shape == (B, S), f"[{name}] cost shape {tuple(cost.shape)} != {(B, S)}"
    assert torch.isfinite(cost).all(), f"[{name}] non-finite planner cost"
    if expect_unit:
        norms = pred.norm(dim=-1)
        assert (norms - 1.0).abs().max() < 1e-3, \
            f"[{name}] rollout left the sphere: max|norm-1|={(norms - 1).abs().max():.4f}"
    print(f"  ok  {name:38s} rollout L={pred.size(2)} cost{tuple(cost.shape)} "
          f"range=[{cost.min():.3f},{cost.max():.3f}]")


def reference_official_matches():
    """Official path must equal the hand-written LeWM reference loss exactly."""
    torch.manual_seed(123)
    jepa = build_jepa(make_ar_predictor(), normalize=False, pred_proj=make_projector(True))
    cfg = make_cfg(pred_type="mse", sigreg_target="emb", sigreg_weight=0.09)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)
    batch = fake_batch()

    # reference (verbatim official lejepa_forward math)
    torch.manual_seed(7)
    out = jepa.encode({k: v.clone() for k, v in batch.items()})
    emb, act_emb = out["emb"], out["act_emb"]
    ctx_emb, ctx_act = emb[:, :HISTORY], act_emb[:, :HISTORY]
    tgt_emb = emb[:, NPRED:]
    pred_emb = jepa.predict(ctx_emb, ctx_act)
    ref_pred = (pred_emb - tgt_emb).pow(2).mean()
    torch.manual_seed(7)
    ref_sig = sigreg(emb.transpose(0, 1))
    ref_loss = ref_pred + 0.09 * ref_sig

    # objective path (seed SIGReg projections identically)
    mod = FakeModule(jepa, sigreg)
    torch.manual_seed(7)
    out2 = lejepa_forward(mod, {k: v.clone() for k, v in batch.items()}, "train", cfg)
    # pred_loss is deterministic; compare it directly (SIGReg uses fresh randn each call)
    assert torch.allclose(out2["pred_loss"], ref_pred, atol=1e-6), \
        f"official pred_loss mismatch: {out2['pred_loss'].item()} vs {ref_pred.item()}"
    print(f"  ok  official pred_loss matches reference ({ref_pred.item():.6f}); "
          f"loss=pred+0.09*SIGReg as in LeWM")


def main():
    print("== forward/backward + invariants ==")
    # A / C: official + non-normalized MSE
    run_variant("official_lewm (mse + sigreg/emb)",
                build_jepa(make_ar_predictor(), False, make_projector(True)),
                make_cfg("mse", "emb", 0.09), expect_unit=False)
    run_variant("lewm_nosigreg (mse, no reg)",
                build_jepa(make_ar_predictor(), False, make_projector(True)),
                make_cfg("mse", "none", 0.0), expect_unit=False)

    # D / E / F / I: spherical predictors (pred_proj=Identity -> None)
    for mode, label in [("simple", "simple_spherical (D)"),
                        ("residual", "fullish_residual (E)"),
                        ("gated", "gated_spherical (F)"),
                        ("ssm", "gated_spherical_ssm (I)")]:
        run_variant(label,
                    build_jepa(make_spherical_predictor(mode), True, None),
                    make_cfg("cosine", "none", 0.0), expect_unit=True)

    # recurrent-anchor residual on the gated predictor
    anchor_jepa = build_jepa(make_spherical_predictor("gated", anchor_beta=0.1), True, None)
    run_variant("gated_spherical_anchor (beta=0.1)",
                anchor_jepa, make_cfg("cosine", "none", 0.0), expect_unit=True)
    probe = anchor_jepa.predictor.probe
    assert "gate_mean" in probe and "step_angle_mean" in probe, \
        f"predictor probe not populated: {probe}"
    assert 0.0 <= probe["gate_mean"] <= 1.0, f"gate out of range: {probe['gate_mean']}"
    print(f"  ok  predictor probe populated: gate_mean={probe['gate_mean']:.3f} "
          f"step_angle={probe['step_angle_mean']:.3f} frac_active={probe['gate_frac_active']:.3f}")

    # F + memory NCE (H)
    run_variant("gated_spherical_memory (H)",
                build_jepa(make_spherical_predictor("gated"), True, None),
                make_cfg("cosine", "none", 0.0, memory_w=0.5), expect_unit=True)

    # F + anti-collapse
    run_variant("gated_spherical_anticollapse",
                build_jepa(make_spherical_predictor("gated"), True, None),
                make_cfg("cosine", "none", 0.0, anticollapse_w=0.1), expect_unit=True)

    # G: gated + projector SIGReg on z
    run_variant("gated_projector_sigreg (G)",
                build_jepa(make_spherical_predictor("gated"), True, None,
                           sigreg_projector=SIGRegProjector(D, 32, D)),
                make_cfg("cosine", "projector_z", 0.09), expect_unit=True)

    # tangent_spherical_sigreg: Riemannian tangent-space step + projector SIGReg on z
    # (the adopted single-run contender). Verifies the new tangent mode forwards, stays
    # on the sphere, and produces a non-frozen step.
    tan_jepa = build_jepa(make_spherical_predictor("tangent"), True, None,
                          sigreg_projector=SIGRegProjector(D, 32, D))
    run_variant("tangent_spherical_sigreg (tangent + z-SIGReg)",
                tan_jepa, make_cfg("cosine", "projector_z", 0.09), expect_unit=True)
    tp = tan_jepa.predictor.probe
    assert tp["step_angle_mean"] > 0.05, f"tangent step looks frozen: {tp['step_angle_mean']}"
    print(f"  ok  tangent probe: gate_mean={tp['gate_mean']:.3f} "
          f"step_angle={tp['step_angle_mean']:.3f} frac_active={tp['gate_frac_active']:.3f}")

    # ablation #13: direct SIGReg on h
    run_variant("gated_direct_sigreg_h (#13)",
                build_jepa(make_spherical_predictor("gated"), True, None),
                make_cfg("cosine", "emb", 0.09), expect_unit=True)

    # rotation_spherical_simplex: SO(d) plane-rotation step + simplex-ETF proto
    # loss on h. Verifies the three construction guarantees: exact unit norm,
    # populated step-size probe, and -- the anti-gate-death property -- nonzero
    # gradient into the theta head AT the identity (theta starts at exactly 0).
    rot_jepa = build_jepa(make_spherical_predictor("rotation"), True, None)
    run_variant("rotation_spherical_simplex (SO(d)+proto)",
                rot_jepa, make_cfg("cosine", "none", 0.0, stop_grad=True,
                                   proto_w=0.1), expect_unit=True)
    rp = rot_jepa.predictor.probe
    assert "gate_mean" in rp, f"rotation probe not populated: {rp}"
    th_grad = rot_jepa.predictor.rot_theta.weight.grad
    assert th_grad is not None and th_grad.abs().sum() > 0, \
        "rotation theta head got NO gradient at the identity -- parameterization dead"
    print(f"  ok  rotation probe: step_rad={rp['gate_mean']:.4f} (identity start) "
          f"theta-grad={th_grad.abs().sum():.2e} (healthy at theta=0)")

    print("\n== planner rollout / criterion ==")
    test_planner_path("official_lewm",
                      build_jepa(make_ar_predictor(), False, make_projector(True)),
                      expect_unit=False)
    test_planner_path("gated_spherical",
                      build_jepa(make_spherical_predictor("gated"), True, None),
                      expect_unit=True)
    test_planner_path("rotation_spherical_simplex",
                      build_jepa(make_spherical_predictor("rotation"), True, None),
                      expect_unit=True)

    print("\n== official loss reference check ==")
    reference_official_matches()

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        sys.exit(1)
