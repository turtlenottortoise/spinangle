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


def _build_from_experiment(experiment: str, data: str, input_dim, cache_dir):
    """Fallback: rebuild via the experiment config like train.py. Needs the action
    input_dim (taken from config.json if present, else computed from the dataset)."""
    from hydra import compose, initialize

    repo = Path(__file__).resolve().parent.parent
    rel = Path("..") / repo.name / "config" / "train"  # initialize wants a relative path
    # initialize() is finicky about relative paths; use config_path relative to this file.
    with initialize(version_base=None, config_path="../config/train"):
        c = compose(config_name="lewm",
                    overrides=[f"+experiment={experiment}", f"data={data}"])
    if input_dim is None:
        # last resort: load the dataset to read the action dimension (train.py logic)
        dcfg = OmegaConf.to_container(c.data.dataset, resolve=True)
        name = dcfg.pop("name")
        ds = swm.data.load_dataset(name, transform=None, cache_dir=cache_dir, **dcfg)
        input_dim = int(c.data.dataset.frameskip * ds.get_dim("action"))
    OmegaConf.set_struct(c, False)
    c.model.action_encoder.input_dim = int(input_dim)
    return hydra.utils.instantiate(c.model), c.model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                    help="output_model_name used at train time, e.g. tworoom/official_lewm_l4_single")
    ap.add_argument("--epoch", type=int, required=True, help="which weights_epoch_N.pt to finalize")
    ap.add_argument("--experiment", default=None, help="experiment name (fallback rebuild)")
    ap.add_argument("--data", default=None, help="data config name (fallback rebuild)")
    args = ap.parse_args()

    ckpt_root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    run_dir = ckpt_root / args.run_name
    w_path = run_dir / f"weights_epoch_{args.epoch}.pt"
    cfg_path = run_dir / "config.json"

    if not w_path.exists():
        sys.exit(f"[finalize] FATAL: weights not found: {w_path}\n"
                 f"           (restore weights_epoch_{args.epoch}.pt from Drive first)")
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
        cache_dir = None
        input_dim = None
        model = None
        if cfg_path.exists():
            try:
                model, cfg_used = _build_from_config_json(cfg_path)
                input_dim = OmegaConf.select(cfg_used, "action_encoder.input_dim")
                print("[finalize] rebuilt via hydra.instantiate(config.json)")
            except Exception as e:
                print(f"[finalize] instantiate(config.json) failed ({e!r}); trying experiment config")
                try:
                    input_dim = int(OmegaConf.select(
                        OmegaConf.create(json.loads(cfg_path.read_text())),
                        "action_encoder.input_dim"))
                except Exception:
                    input_dim = None
        if model is None:
            if not args.experiment or not args.data:
                sys.exit("[finalize] FATAL: cannot rebuild -- pass --experiment and --data, "
                         "or restore config.json next to the weights.")
            model, _ = _build_from_experiment(args.experiment, args.data, input_dim, cache_dir)
            print("[finalize] rebuilt via composed experiment config")

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[finalize] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("           missing[:8]   :", list(missing)[:8])
        if unexpected:
            print("           unexpected[:8]:", list(unexpected)[:8])
        if missing or unexpected:
            sys.exit("[finalize] FATAL: state_dict does not match architecture (config drift). "
                     "Aborting rather than saving a corrupt checkpoint.")

    model.eval()
    canon = ckpt_root / f"{args.run_name}.pt"
    canon.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, canon)
    print(f"[finalize] wrote canonical object: {canon}  ({canon.stat().st_size/1e6:.1f} MB)")

    # marker so it's obvious this run was finalized from an earlier epoch (honest, no faked epoch file)
    (run_dir / "FINALIZED.json").write_text(json.dumps(
        {"finalized_from_epoch": args.epoch, "canonical_ckpt": str(canon),
         "run_name": args.run_name}, indent=2))

    # verify round-trip through the package's own loader (what eval_latent_metrics uses)
    m2 = swm.wm.utils.load_pretrained(args.run_name)
    n_params = sum(p.numel() for p in m2.parameters())
    print(f"[finalize] load_pretrained('{args.run_name}') OK -> {type(m2).__name__}, "
          f"{n_params/1e6:.2f}M params")
    print("[finalize] DONE")


if __name__ == "__main__":
    main()
