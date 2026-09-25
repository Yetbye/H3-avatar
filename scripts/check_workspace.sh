#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ROOT="/mnt/data/yetbye/h3-interactive"
ACTUAL_ROOT="$(pwd -P)"

case "$ACTUAL_ROOT" in
  "$EXPECTED_ROOT"|"$EXPECTED_ROOT"/*) ;;
  *)
    echo "ERROR: run inside $EXPECTED_ROOT, current directory is $ACTUAL_ROOT" >&2
    exit 2
    ;;
esac

echo "workspace=$ACTUAL_ROOT"
echo "user=$(id -un)"
echo "host=$(hostname)"
echo "disk:"
df -h "$EXPECTED_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
  echo "gpu=nvidia-smi unavailable"
fi

if command -v git >/dev/null 2>&1 && git -C "$EXPECTED_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git -C "$EXPECTED_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "git_head=$(git -C "$EXPECTED_ROOT" rev-parse --short HEAD)"
  else
    echo "git_head=unborn"
  fi
  git -C "$EXPECTED_ROOT" status --short
else
  echo "git=not initialized"
fi

echo "boundary_check=PASS"
