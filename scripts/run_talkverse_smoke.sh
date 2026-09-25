#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/mnt/data/yetbye/envs/h3-interactive/bin/python"
batch_file="${project_root}/data/raw/talkverse-smoke/batch.json"
experiment_dir="${project_root}/experiments/talkverse-smoke-001"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing project Python: ${python_bin}" >&2
  exit 1
fi

mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} > 0 )); then
  echo "GPU is already in use; refusing to interfere. Active compute PIDs: ${gpu_pids[*]}" >&2
  exit 2
fi

if [[ ! -f "${batch_file}" ]]; then
  echo "Missing smoke batch file: ${batch_file}" >&2
  exit 1
fi

"${python_bin}" "${project_root}/scripts/preflight_talkverse.py" \
  --ckpt-dir "${project_root}/checkpoints/Wan2.2-TI2V-5B" \
  --lora-ckpt "${project_root}/checkpoints/talkverse-s2v-5b/talkverse_s2v_5b.safetensors" \
  --wav2vec-dir "${project_root}/checkpoints/Wan2.2-TI2V-5B/wav2vec2-large-xlsr-53-english" \
  --generate-py "${project_root}/third_party/TalkVerse/generate.py"

mkdir -p "${experiment_dir}/outputs" "${experiment_dir}/logs" "${project_root}/tmp/bin"
ffmpeg_exe="$(${python_bin} -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sfn "${ffmpeg_exe}" "${project_root}/tmp/bin/ffmpeg"

cd "${project_root}/third_party/TalkVerse"
PATH="${project_root}/tmp/bin:${PATH}" "${python_bin}" generate.py \
  --task s2v-5B \
  --ckpt_dir "${project_root}/checkpoints/Wan2.2-TI2V-5B" \
  --lora_ckpt "${project_root}/checkpoints/talkverse-s2v-5b/talkverse_s2v_5b.safetensors" \
  --lora_rank 128 \
  --lora_alpha 128 \
  --lora_merge True \
  --size_bucket fasttalk-480 \
  --batch_file "${batch_file}" \
  --output_dir "${experiment_dir}/outputs" \
  --num_clip 1 \
  --infer_frames 120 \
  --sample_shift 3.0 \
  --offload_model True \
  --t5_cpu \
  2>&1 | tee "${experiment_dir}/logs/run.log"
