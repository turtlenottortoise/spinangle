#!/usr/bin/env python3
"""Promote a per-epoch training export (``weights_epoch_N.pt``) to a canonical,
unambiguously-loadable checkpoint -- without re-training.

Why this exists
---------------
``SaveCkptCallback`` (utils.py) writes one *state_dict* per epoch via
``save_pretrained`` into ``<cache>/checkpoints/<run_name>/weights_epoch_N.pt``
(+ ``config.json``). If a run is interrupted on its last epoch, the best surviving
artifact is the last ``weights_epoch_N.pt`` on Drive. But:

  * it is a state_dict, not a model object, and
  * a folder with several ``weights_epoch_*.pt`` is ambiguous for ``load_pretrained``.

This script rebuilds the model from ``config.json`` + the chosen epoch's state_dict
(falling back to the experiment config if needed) and re-saves it as a single pickled
object at ``<cache>/checkpoints/<run_name>.pt`` -- i.e. ``load_pretrained``'s first
resolution target, exactly the format ``download_assets.convert_ckpt`` produces for the
official checkpoint. It then verifies the round-trip through ``load_pretrained``.

Example
-------
python scripts/finalize_from_epoch.py \
    --run-name tworoom/official_lewm_l4_single --epoch 4 \
    --experiment official_lewm --data tworoom
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

import hydra
import stable_pretraining as spt  # noqa: F401  (registers backbones referenced by config)
import stable_worldmodel as swm

import jepa  # noqa: F401  (hydra _target_ + unpickle)
import module  # noqa: F401
try:
    import variants  # noqa: F401  (spherical/gated variants; harmless for official)
except Exception:
    pass


def _build_from_config_json(cfg_path: Path):
    """Primary path: config.json is the resolved cfg.model (has _target_s + input_dim)."""
    cfg_model = OmegaConf.create(json.loads(cfg_path.read_text()))
    # Some serializations nest the model under a 'model' key; unwrap if so.
    if "_target_" not in cfg_model and "model" in cfg_model:
        cfg_model = cfg_model.model
    model = hydra.utils.instantiate(cfg_model)
    return model, cfg_model


def _build_from_experiment(experiment: str, data: str, input_dim):
    """Rebuild via the full experiment config exactly like train.py. Composing the
    *full* config (not just cfg.model) is what lets interpolations such as
    ``${embed_dim}`` / ``${img_size}`` resolve. ``input_dim`` (the action-encoder
    Conv1d in-channels) is normally recovered from the weights by the caller; if it
    is None we fall back to reading it from the dataset (train.py's logic)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Something we imported may already have initialized Hydra's global singleton;
    # clear it so init doesn't raise "GlobalHydra is already initialized". Use an
    # absolute config dir (initialize_config_dir) to avoid relative-path ambiguity.
    cfg_dir = str(Path(__file__).resolve().parent.parent / "config" / "train")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        c = compose(config_name="lewm",
                    overrides=[f"+experiment={experiment}", f"data={data}"])
    if input_dim is None:
        dcfg = OmegaConf.to_container(c.data.dataset, resolve=True)
        name = dcfg.pop("name")
        ds = swm.data.load_dataset(name, transform=None, **dcfg)
        input_dim = int(c.data.dataset.frameskip * ds.get_dim("action"))
    OmegaConf.set_struct(c, False)
    c.model.action_encoder.input_dim = int(input_dim)
    return hydra.utils.instantiate(c.model), c.model


def _build_manual(experiment: str, input_dim):
    """Hydra-compose-free rebuild: load the base + experiment + model YAMLs with
    OmegaConf, supply the top-level interpolation vars (${embed_dim} etc.), and
    instantiate. ``hydra.utils.instantiate`` does not need Hydra's global init, so
    this path avoids initialize()/compose entirely."""
    root = Path(__file__).resolve().parent.parent / "config" / "train"
    base = OmegaConf.load(root / "lewm.yaml")
    model_name = "lewm"
    exp_path = root / "experiment" / f"{experiment}.yaml"
    exp = OmegaConf.load(exp_path) if exp_path.exists() else OmegaConf.create({})
    for d in (exp.get("defaults", []) or []):
        if isinstance(d, (dict,)) or hasattr(d, "items"):
            for k, v in dict(d).items():
                if "model" in str(k):
                    model_name = str(v)
    model_cfg = OmegaConf.load(root / "model" / f"{model_name}.yaml")
    if "model" in exp:                       # experiment-level model overrides, if any
        model_cfg = OmegaConf.merge(model_cfg, exp.model)
    ctx = OmegaConf.create({
        "embed_dim": base.get("embed_dim", 192),
        "img_size": base.get("img_size", 224),
        "history_size": base.get("history_size", 3),
        "num_preds": base.get("num_preds", 1),
        "model": model_cfg,
    })
    OmegaConf.set_struct(ctx, False)
    if input_dim is not None:
        ctx.model.action_encoder.input_dim = int(input_dim)
    return hydra.utils.instantiate(ctx.model), ctx.model


def _action_input_dim(state):
    """Recover the action-encoder input dim straight from the weights: Embedder's
    first layer is Conv1d(input_dim, ...) so its weight is [out, input_dim, 1]."""
    for k in ("action_encoder.patch_embed.weight", "model.action_encoder.patch_embed.weight"):
        if k in state:
            return int(state[k].shape[1])
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                    help="output_model_name used at train time, e.g. tworoom/official_lewm_l4_single_run")
    ap.add_argument("--epoch", default="auto",
                    help="which weights_epoch_N.pt to finalize; 'auto' = newest by mtime")
    ap.add_argument("--out-name", default=None,
                    help="canonical name to write (default: --run-name), "
                         "e.g. tworoom/official_lewm_l4_single")
    ap.add_argument("--experiment", default=None, help="experiment name (fallback rebuild)")
    ap.add_argument("--data", default=None, help="data config name (fallback rebuild)")
    args = ap.parse_args()

    ckpt_root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    run_dir = ckpt_root / args.run_name
    out_name = args.out_name or args.run_name
    cfg_path = run_dir / "config.json"

    if str(args.epoch).lower() == "auto":
        cands = sorted(run_dir.glob("weights_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        if not cands:
            sys.exit(f"[finalize] FATAL: no weights_epoch_*.pt in {run_dir}")
        w_path = cands[-1]
        print(f"[finalize] --epoch auto -> newest: {w_path.name}")
    else:
        w_path = run_dir / f"weights_epoch_{int(args.epoch)}.pt"

    if not w_path.exists():
        sys.exit(f"[finalize] FATAL: weights not found: {w_path}\n"
                 f"           (restore the weights from Drive first)")
    print(f"[finalize] weights : {w_path}  ({w_path.stat().st_size/1e6:.1f} MB)")
    print(f"[finalize] config  : {cfg_path}  ({'present' if cfg_path.exists() else 'MISSING'})")

    blob = torch.load(w_path, map_location="cpu", weights_only=False)

    if isinstance(blob, torch.nn.Module):
        model = blob
        print("[finalize] file is a full nn.Module object -> using directly")
    else:
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        n_keys = len(state) if hasattr(state, "__len__") else "?"
        print(f"[finalize] file is a state_dict ({n_keys} tensors) -> rebuilding architecture")
        input_dim = _action_input_dim(state)
        print(f"[finalize] action input_dim from weights: {input_dim}")
        model = None
        errs = []
        # Primary: compose the full experiment config (resolves ${...} interpolations)
        # and patch input_dim from the weights -- no dataset / config.json needed.
        if args.experiment and args.data:
            try:
                model, _ = _build_from_experiment(args.experiment, args.data, input_dim)
                print(f"[finalize] rebuilt via experiment config "
                      f"(+experiment={args.experiment} data={args.data})")
            except Exception as e:
                import traceback
                errs.append(f"experiment: {e!r}")
                print(f"[finalize] experiment rebuild failed: {e!r}")
                traceback.print_exc()
        # Fallback A: hydra-free build straight from the YAML configs.
        if model is None and args.experiment:
            try:
                model, _ = _build_manual(args.experiment, input_dim)
                print(f"[finalize] rebuilt via manual YAML build (experiment={args.experiment})")
            except Exception as e:
                import traceback
                errs.append(f"manual: {e!r}")
                print(f"[finalize] manual rebuild failed: {e!r}")
                traceback.print_exc()
        # Fallback B: the saved config.json (only if it carries _target_s + resolved values).
        if model is None and cfg_path.exists():
            try:
                model, _ = _build_from_config_json(cfg_path)
                print("[finalize] rebuilt via config.json")
            except Exception as e:
                errs.append(f"config.json: {e!r}")
                print(f"[finalize] config.json rebuild failed: {e!r}")
        # Architecture rebuild is only a SANITY check now (the canonical folder below
        # uses the original config.json, and load_pretrained is the real gate), so a
        # rebuild failure is a warning, not fatal.
        if model is None:
            print("[finalize] WARN: architecture sanity-rebuild unavailable:\n  " + "\n  ".join(errs))
        else:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[finalize] sanity load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if missing or unexpected:
                print("           (mismatch -> WARN only; load_pretrained below is the real gate)")

    # load_pretrained resolves a FOLDER <ckpt_root>/<out_name>/ containing one .pt +
    # config.json (the native save_pretrained layout), NOT a bare <out_name>.pt file.
    canon_dir = ckpt_root / out_name
    canon_dir.mkdir(parents=True, exist_ok=True)
    for old in canon_dir.glob("*.pt"):          # keep exactly one .pt -> unambiguous
        old.unlink()
    shutil.copy2(w_path, canon_dir / w_path.name)
    if cfg_path.exists():
        shutil.copy2(cfg_path, canon_dir / "config.json")
    print(f"[finalize] wrote canonical folder: {canon_dir}  "
          f"({w_path.name} + {'config.json' if cfg_path.exists() else 'NO config.json!'})")
    (canon_dir / "FINALIZED.json").write_text(json.dumps(
        {"finalized_from": str(w_path), "run_name": args.run_name, "out_name": out_name}, indent=2))

    # verify round-trip through the package's own loader (exactly what eval uses)
    m2 = swm.wm.utils.load_pretrained(out_name)
    n_params = sum(p.numel() for p in m2.parameters())
    print(f"[finalize] load_pretrained('{out_name}') OK -> {type(m2).__name__}, "
          f"{n_params/1e6:.2f}M params")
    print("[finalize] DONE")


if __name__ == "__main__":
    main()
