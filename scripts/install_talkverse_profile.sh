#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/yetbye/h3-interactive"
CONDA_BIN="/usr/local/miniconda3/bin/conda"
ENV_PREFIX="/mnt/data/yetbye/envs/h3-interactive"
PYTHON="$ENV_PREFIX/bin/python"

if [[ "$(pwd -P)" != "$ROOT" ]]; then
  echo "ERROR: run from $ROOT" >&2
  exit 2
fi

if [[ ! -x "$PYTHON" || ! -d "$ENV_PREFIX/conda-meta" ]]; then
  echo "ERROR: base Conda environment is missing; run scripts/create_conda_env.sh" >&2
  exit 3
fi

mkdir -p "$ROOT/.conda/pkgs" "$ROOT/.cache/pip" "$ROOT/.cache/huggingface" "$ROOT/tmp/pip-tmp"
export CONDA_PKGS_DIRS="$ROOT/.conda/pkgs"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$ROOT/tmp/pip-tmp"

"$PYTHON" -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

grep -vE '^flash_attn([<>=~!].*)?$' \
  "$ROOT/third_party/TalkVerse/requirements.txt" \
  > "$ROOT/tmp/talkverse-requirements-no-flash-attn.txt"

"$PYTHON" -m pip install \
  -r "$ROOT/tmp/talkverse-requirements-no-flash-attn.txt" \
  -r "$ROOT/third_party/TalkVerse/requirements_s2v.txt"

"$PYTHON" -m pip install ninja
"$PYTHON" -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

set +e
PIP_CHECK_OUTPUT=$("$PYTHON" -m pip check 2>&1)
PIP_CHECK_RC=$?
set -e
printf '%s\n' "$PIP_CHECK_OUTPUT"
if [[ $PIP_CHECK_RC -ne 0 ]]; then
  UNEXPECTED_CHECK_OUTPUT=$(printf '%s\n' "$PIP_CHECK_OUTPUT" \
    | grep -vFx 'decord 0.6.0 is not supported on this platform' || true)
  if [[ -n "$UNEXPECTED_CHECK_OUTPUT" ]]; then
    echo "ERROR: unexpected pip dependency error" >&2
    exit 4
  fi
  echo "WARNING: decord's published wheel has a CPython 3.6 tag; runtime import is checked below" >&2
fi
"$PYTHON" - <<'PY'
import torch
import torchvision
import torchaudio
import transformers
import diffusers
import flash_attn
import librosa
import decord

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"transformers={transformers.__version__}")
print(f"diffusers={diffusers.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"decord={decord.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

"$CONDA_BIN" list --prefix "$ENV_PREFIX" --explicit \
  > "$ROOT/configs/environments/talkverse-conda-explicit.lock"
"$PYTHON" -m pip freeze \
  > "$ROOT/configs/environments/talkverse-pip-freeze.txt"

echo "profile=talkverse"
echo "status=PASS"
