# Experiment: livetalking-bridge-001

Status: `PASS`

## Question and hypothesis

Can the stable asynchronous session contract translate audio, control, video, reset, status, and close operations to a LiveTalking-like object without importing the legacy project or a model?

The hypothesis passes if a fake legacy runtime receives correctly normalized float32 audio and control calls, returns timestamped video, and observes complete reset/close calls.

This experiment does not validate the real LiveTalking process, aiortc integration, wav2lip, FlashHead, H3, visual quality, or realtime performance.

## Fixed boundary

- The legacy source directory remains read-only and is not imported.
- PCM input remains 16 kHz mono s16le, 320 samples / 20 ms.
- Conversion is little-endian int16 to float32 divided by 32768.
- Supported controls are `text` and `custom_state` only.
- Reset must flush legacy audio/TTS state, drain video output, clear control state and sequences, and increment epoch.
- The bridge owns protocol translation; a later runtime wrapper will own aiortc/private-queue access.

## Environment and resources

- Project environment: `/mnt/data/yetbye/h3-interactive/.conda/envs/h3-interactive`.
- CPU only with `CUDA_VISIBLE_DEVICES=''`.
- Standard-library unittest plus NumPy already present in the project profile.

## PASS gates

1. Existing mock lifecycle regression tests remain green.
2. Bridge tests verify PCM normalization, metadata, controls, video PTS, timeout translation, reset, status, and idempotent close.
3. `compileall` and all tests exit 0.
4. GPU memory does not increase during validation.

## Result

- Python 3.10.21.
- `compileall`: exit code 0.
- Combined regression and bridge suite: 6/6 tests passed in 0.090 seconds, exit code 0.
- GPU memory before/after: 2806 MiB / 2806 MiB. Utilization belonged to the pre-existing workload; validation ran with `CUDA_VISIBLE_DEVICES=''`.
- Evidence: `logs/unittest.log`, `logs/exit-code.txt`, `logs/gpu-before.txt`, `logs/gpu-after.txt`, and `logs/SHA256SUMS`.
