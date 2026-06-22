#!/usr/bin/env bash
# Real setup for a rented GPU box (Ubuntu 22.04, root, fresh) OR Colab.
# Uses the repo's official install: uv + an isolated Python 3.10 venv + the FULL
# stable-worldmodel[train,env]. Run from the repo root:  bash bootstrap.sh
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
VENV="${VENV:-$PWD/.venv-lewm}"
PY="$VENV/bin/python"
mkdir -p "$STABLEWM_HOME"
echo "STABLEWM_HOME=$STABLEWM_HOME  VENV=$VENV"

# --- system libs for headless env rendering during eval (pygame + MuJoCo) ---
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get -qq update
  sudo apt-get -qq install -y xvfb zstd ffmpeg patchelf git curl \
    libegl1 libgl1-mesa-glx libosmesa6 libglfw3 libglew2.2 >/dev/null
fi

# --- uv + isolated Python 3.10 venv + full official stack ---
command -v uv >/dev/null 2>&1 || pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv --python 3.10 "$VENV"
# [train] is required (hydra, stable-pretraining, transformers, wandb).
uv pip install --python "$PY" "stable-worldmodel[train]" matplotlib huggingface_hub
# Try the full [env] extra; if it can't resolve (it bundles Crafter/craftax + Atari/
# ale-py, which aren't LeWM benchmarks and pin conflicting jax/gymnasium), install the
# env deps that the LeWM benchmarks (pusht/tworoom/reacher/cube) actually use.
uv pip install --python "$PY" "stable-worldmodel[env]" \
  || uv pip install --python "$PY" pygame pymunk shapely dm_control mujoco ogbench

# ensure CUDA torch in the venv
if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  uv pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi
"$PY" -c "import torch,hydra,stable_worldmodel,stable_pretraining; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| stack OK')"

echo; echo "== CPU smoke test + metrics (harness sanity) =="
"$PY" smoke_test.py
"$PY" metrics.py

cat <<EOF

Stack ready. Use the venv python ($PY) for everything, e.g.:

  # data + checkpoints (tworoom is lightest; pusht/reacher/cube are large)
  $PY scripts/download_assets.py --benchmark tworoom --data --ckpt

  # Phase 1 reproduce official LeWM (eval renders -> wrap with xvfb-run if headless)
  xvfb-run -a $PY eval.py --config-name=tworoom.yaml policy=tworoom/lewm

  # train + eval a variant (see RUN_MATRIX.md for the full matrix)
  $PY train.py +experiment=gated_spherical data=tworoom output_model_name=tworoom/gated_spherical \\
      trainer.max_epochs=100 wandb.enabled=false
  xvfb-run -a $PY eval.py --config-name=tworoom.yaml policy=tworoom/gated_spherical
EOF
