#!/usr/bin/env python3
"""Download official LeWM datasets + checkpoints and convert checkpoints to the
``_object.ckpt`` that ``eval.py`` expects.

Repos / archive names (verified from the HuggingFace `quentinll/lewm` collection):

  benchmark | dataset repo (type=dataset)   | dataset file                    | ckpt repo (type=model)
  ----------|-------------------------------|---------------------------------|------------------------
  pusht     | quentinll/lewm-pusht          | pusht_expert_train.h5.zst (13G) | quentinll/lewm-pusht
  reacher   | quentinll/lewm-reacher        | reacher.tar.zst          (24G)  | quentinll/lewm-reacher
  tworoom   | quentinll/lewm-tworooms       | tworoom.tar.zst          (3.4G) | quentinll/lewm-tworooms
  cube      | quentinll/lewm-cube           | cube_single_expert.tar.zst(46G) | quentinll/lewm-cube

`.h5.zst` -> `zstd -d`; `.tar.zst` -> `tar --zstd -x` into $STABLEWM_HOME. The
converted checkpoint lands at $STABLEWM_HOME/<benchmark>/lewm_object.ckpt so that
`python eval.py --config-name=<benchmark>.yaml policy=<benchmark>/lewm` resolves.

Examples
--------
python scripts/download_assets.py --benchmark tworoom --data --ckpt   # lightest, do this first
python scripts/download_assets.py --benchmark pusht   --ckpt          # ckpt only (data is 13G)
"""
import argparse
import os
import subprocess
from pathlib import Path

ASSETS = {
    "pusht":   dict(ds_repo="quentinll/lewm-pusht",    ds_file="pusht_expert_train.h5.zst",  kind="zst", ckpt_repo="quentinll/lewm-pusht"),
    "reacher": dict(ds_repo="quentinll/lewm-reacher",  ds_file="reacher.tar.zst",            kind="tar", ckpt_repo="quentinll/lewm-reacher"),
    "tworoom": dict(ds_repo="quentinll/lewm-tworooms", ds_file="tworoom.tar.zst",            kind="tar", ckpt_repo="quentinll/lewm-tworooms"),
    "cube":    dict(ds_repo="quentinll/lewm-cube",     ds_file="cube_single_expert.tar.zst", kind="tar", ckpt_repo="quentinll/lewm-cube"),
}


def home():
    p = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_data(bench):
    from huggingface_hub import hf_hub_download
    a = ASSETS[bench]
    dst = home()
    print(f"[data] downloading {a['ds_repo']}/{a['ds_file']} -> {dst}")
    local = hf_hub_download(repo_id=a["ds_repo"], filename=a["ds_file"],
                            repo_type="dataset", local_dir=dst)
    local = Path(local)
    subprocess.run("which zstd || (apt-get -qq update && apt-get -qq install -y zstd)",
                   shell=True)
    if a["kind"] == "zst":
        out = local.with_suffix("")            # strip .zst -> .h5
        print(f"[data] zstd -d -> {out}")
        subprocess.run(f"zstd -d -f -o '{out}' '{local}'", shell=True, check=True)
    else:
        print(f"[data] tar -I zstd -x -> {dst}")   # -I zstd is more portable than --zstd
        subprocess.run(f"tar -I zstd -xf '{local}' -C '{dst}'", shell=True, check=True)
    print(f"[data] done. contents of {dst}:")
    subprocess.run(["ls", "-la", str(dst)])


def convert_ckpt(bench):
    """Rebuild the official LeWM weights into the local jepa.JEPA and pickle it as
    the object checkpoint eval.py loads (mirrors UPSTREAM_README.md)."""
    import json
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    from huggingface_hub import hf_hub_download
    a = ASSETS[bench]
    cfg_path = hf_hub_download(a["ckpt_repo"], "config.json")
    w_path = hf_hub_download(a["ckpt_repo"], "weights.pt")
    cfg = json.loads(Path(cfg_path).read_text())

    def mlp(key):
        return MLP(input_dim=cfg[key]["input_dim"], output_dim=cfg[key]["output_dim"],
                   hidden_dim=cfg[key]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"], patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"], pretrained=False, use_mask_token=False,
    )
    # config.json predictor kwargs match module.ARPredictor exactly
    pred_kwargs = {k: v for k, v in cfg["predictor"].items() if k != "_target_"}
    act_kwargs = {k: v for k, v in cfg["action_encoder"].items() if k != "_target_"}
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**pred_kwargs),
        action_encoder=Embedder(**act_kwargs),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    sd = torch.load(w_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)   # official path adds no new params

    # eval.py calls swm.wm.utils.load_pretrained(policy). Its resolution order is
    # <cache_dir>/checkpoints/<name>.pt (pickled object) OR a folder with a .pt +
    # config.json; the upstream README also documents <cache_dir>/<name>_object.ckpt.
    # Write all three so `policy=<bench>/lewm` resolves regardless of swm version.
    root = Path(swm.data.utils.get_cache_dir())
    ckpt_root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    targets = [
        ckpt_root / bench / "lewm.pt",              # load_pretrained resolution #1
        root / bench / "lewm_object.ckpt",          # upstream README convention
    ]
    for out in targets:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, out)
    print(f"[ckpt] {a['ckpt_repo']} -> {targets[0]} (+ README path)  "
          f"(eval with policy={bench}/lewm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=list(ASSETS))
    ap.add_argument("--data", action="store_true", help="download + extract the dataset")
    ap.add_argument("--ckpt", action="store_true", help="download + convert the checkpoint")
    args = ap.parse_args()
    if not (args.data or args.ckpt):
        ap.error("pass --data and/or --ckpt")
    print(f"STABLEWM_HOME={home()}", flush=True)
    import sys
    import traceback
    if args.data:
        try:
            fetch_data(args.benchmark)
        except Exception:
            print("\n[FAILED] DATA stage:", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)
    if args.ckpt:
        try:
            convert_ckpt(args.benchmark)
        except Exception:
            print("\n[FAILED] CKPT stage:", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)


if __name__ == "__main__":
    main()
