#!/usr/bin/env bash
# Environment bootstrap for LeWM + nGPT-JEPA (Colab / Linux GPU box).
# Mirrors the official LeWM install, then verifies the model code with the CPU
# smoke test. Heavy deps (stable-worldmodel/-pretraining, mujoco) are only needed
# for actual training/eval, not for smoke_test.py / metrics.py.
set -euo pipefail

STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export STABLEWM_HOME
mkdir -p "$STABLEWM_HOME"
echo "STABLEWM_HOME=$STABLEWM_HOME"

# --- core training/eval stack (official LeWM instructions) ---
if command -v uv >/dev/null 2>&1; then
  uv pip install --system "stable-worldmodel[train,env]"
else
  pip install "stable-worldmodel[train,env]"
fi

# --- analysis-only deps used by our scripts ---
pip install matplotlib

echo
echo "== CPU smoke test (model variants) =="
python smoke_test.py
echo
echo "== metrics self-check =="
python metrics.py

cat <<'EOF'

Next:
  1) Download official data/checkpoints into $STABLEWM_HOME
     (see official_lewm_reproduction.md / HuggingFace collection quentinll/lewm).
  2) Phase 1 reproduce:  python eval.py --config-name=pusht.yaml policy=pusht/lewm
  3) Variants:           see RUN_MATRIX.md
EOF
