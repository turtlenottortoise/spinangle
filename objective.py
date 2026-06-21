"""Variant-aware LeWM / nGPT-JEPA training objective.

Extracted into its own torch-only module (no stable_pretraining /
stable_worldmodel imports) so it can be unit/smoke-tested on CPU without the full
training stack. ``train.py`` imports ``lejepa_forward`` from here.

The official LeWM objective is recovered exactly with the default config
(loss.pred.type=mse, loss.sigreg.target=emb, anti-collapse/memory weights 0):

    loss = MSE(pred, target) + 0.09 * SIGReg(emb)
"""

import torch
from einops import rearrange

from variants import memory_nce_loss, pairwise_cosine_penalty


def _get(cfg_node, key, default):
    """Safe getter for OmegaConf / dict nodes."""
    try:
        val = cfg_node.get(key, default)
    except AttributeError:
        val = getattr(cfg_node, key, default)
    return default if val is None else val


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses.

    ``self`` is the training module exposing ``self.model`` (a ``jepa.JEPA``),
    ``self.sigreg`` (a ``module.SIGReg``), and ``self.log_dict``.
    """
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds

    # ---- variant flags (defaults reproduce official LeWM) ----
    lcfg = cfg.loss
    pcfg = _get(lcfg, "pred", {})
    pred_type = _get(pcfg, "type", "mse")              # mse | cosine
    stop_grad_tgt = _get(pcfg, "stop_grad_target", False)
    reg_target = _get(lcfg.sigreg, "target", "emb")    # emb | projector_z | none
    lambd = lcfg.sigreg.weight
    ac_w = _get(_get(lcfg, "anticollapse", {}), "weight", 0.0)
    mcfg = _get(lcfg, "memory", {})
    mem_w = _get(mcfg, "weight", 0.0)
    mem_temp = _get(mcfg, "temperature", 0.1)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb = output["emb"]            # (B, T, D) -- unit-norm if model.normalize_emb
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]                       # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # align lengths (no-op for official ctx_len=3, n_preds=1)
    L = min(pred_emb.size(1), tgt_emb.size(1))
    pred_emb, tgt_emb = pred_emb[:, :L], tgt_emb[:, :L]

    # ---- prediction loss ----
    tgt = tgt_emb.detach() if stop_grad_tgt else tgt_emb
    if pred_type == "cosine":
        output["pred_loss"] = (1.0 - (pred_emb * tgt).sum(dim=-1)).mean()
    else:
        output["pred_loss"] = (pred_emb - tgt).pow(2).mean()

    output["loss"] = output["pred_loss"]

    # ---- isotropic-Gaussian regularizer (SIGReg) ----
    if lambd > 0 and reg_target != "none":
        if reg_target == "projector_z":
            assert self.model.sigreg_projector is not None, \
                "loss.sigreg.target=projector_z requires model.sigreg_projector"
            z = self.model.sigreg_projector(rearrange(emb, "b t d -> (b t) d"))
            z = rearrange(z, "(b t) d -> b t d", b=emb.size(0))
            output["sigreg_loss"] = self.sigreg(z.transpose(0, 1))
        else:  # "emb" -- official: SIGReg directly on encoder embeddings
            output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["loss"] + lambd * output["sigreg_loss"]

    # ---- optional spherical anti-collapse ----
    if ac_w > 0:
        output["anticollapse_loss"] = pairwise_cosine_penalty(emb)
        output["loss"] = output["loss"] + ac_w * output["anticollapse_loss"]

    # ---- optional future/goal memory NCE ----
    if mem_w > 0:
        output["memory_loss"] = memory_nce_loss(pred_emb, tgt_emb, temperature=mem_temp)
        output["loss"] = output["loss"] + mem_w * output["memory_loss"]

    # ---- latent-norm diagnostic (tracks collapse / norm drift) ----
    with torch.no_grad():
        output["emb_norm"] = emb.norm(p=2, dim=-1).mean()

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items()
                   if ("loss" in k or "norm" in k)}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output
