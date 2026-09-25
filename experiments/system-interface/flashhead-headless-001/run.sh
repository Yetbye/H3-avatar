#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="/mnt/data/yetbye/envs/h3-interactive/bin/python"
experiment_dir="${project_root}/experiments/system-interface/flashhead-headless-001"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing project Python: ${python_bin}" >&2
  exit 1
fi

# Refuse to interfere with other GPU jobs (same guard as run_talkverse_smoke.sh).
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "GPU is already in use; refusing to interfere. Active compute PIDs: ${gpu_pids[*]}" >&2
  exit 2
fi

mkdir -p "${experiment_dir}/outputs" "${experiment_dir}/logs"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  > "${experiment_dir}/logs/gpu-before.txt" || true

: > "${experiment_dir}/logs/run.log"
set +e
"${python_bin}" "${experiment_dir}/run_headless.py" \
  --ckpt-dir "${project_root}/checkpoints/SoulX-FlashHead-1_3B" \
  --wav2vec-dir "${project_root}/checkpoints/wav2vec2-base-960h" \
  --repo-dir "${project_root}/third_party/SoulX-FlashHead" \
  --cond-image "${project_root}/data/raw/talkverse-smoke/reference.png" \
  --audio-wav "${project_root}/data/raw/talkverse-smoke/drive_6s.wav" \
  --model-type lite \
  --duration 60 \
  --post-reset-seconds 5 \
  --out-dir "${experiment_dir}" \
  2>&1 | tee -a "${experiment_dir}/logs/run.log"
exit_code=${PIPESTATUS[0]}
set -e

echo "${exit_code}" > "${experiment_dir}/logs/exit-code.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  > "${experiment_dir}/logs/gpu-after.txt" || true
pgrep -af "${python_bin}" > "${experiment_dir}/logs/python-processes-after.txt" || true

cd "${experiment_dir}"
find . -type f ! -name "SHA256SUMS" -print0 | sort -z | xargs -0 sha256sum > "${experiment_dir}/logs/SHA256SUMS"
