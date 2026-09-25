#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/yetbye/h3-interactive"
CONDA_BIN="/usr/local/miniconda3/bin/conda"
ENV_PREFIX="/mnt/data/yetbye/envs/h3-interactive"

if [[ "$(pwd -P)" != "$ROOT" ]]; then
  echo "ERROR: run from $ROOT" >&2
  exit 2
fi

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found at $CONDA_BIN" >&2
  exit 3
fi

mkdir -p "$(dirname "$ENV_PREFIX")" "$ROOT/.conda/pkgs" "$ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$ROOT/.cache/pip"

if [[ -e "$ENV_PREFIX" ]]; then
  if [[ -d "$ENV_PREFIX/conda-meta" ]]; then
    echo "Environment already exists: $ENV_PREFIX"
  else
    echo "ERROR: refusing to overwrite non-Conda path: $ENV_PREFIX" >&2
    exit 4
  fi
else
  "$CONDA_BIN" create --prefix "$ENV_PREFIX" python=3.10 pip -y
fi

"$CONDA_BIN" run --prefix "$ENV_PREFIX" python --version
"$CONDA_BIN" run --prefix "$ENV_PREFIX" python -m pip --version
"$CONDA_BIN" list --prefix "$ENV_PREFIX" --explicit \
  > "$ROOT/configs/environments/base-conda-explicit.lock"
"$CONDA_BIN" run --prefix "$ENV_PREFIX" python -m pip freeze \
  > "$ROOT/configs/environments/base-pip-freeze.txt"

echo "environment=$ENV_PREFIX"
echo "status=PASS"
