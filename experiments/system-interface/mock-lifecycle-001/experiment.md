# Experiment: mock-lifecycle-001

Status: `PASS`

## Question and hypothesis

Can a model-independent asynchronous contract represent the existing 20 ms audio / 40 ms video lifecycle without importing a model or transport implementation?

The hypothesis passes if a deterministic CPU mock supports start, audio input, control update, timed video output, reset, backpressure, and idempotent close with explicit state errors.

This experiment does not validate LiveTalking, FlashHead, H3, WebRTC, visual quality, realtime performance, or GPU compatibility.

## Fixed contract

- Audio: 16 kHz, mono, PCM s16le, exactly 320 samples / 640 bytes / 20 ms per chunk.
- Video: one 40 ms chunk for each pair of accepted audio chunks.
- Input sequence and PTS start at zero and increase monotonically within an epoch.
- Reset clears pending input/output, controls, sequence counters, and increments the epoch.
- Output capacity is bounded; producers receive an explicit retryable backpressure error.
- Close is idempotent at the manager boundary.

## Environment and resources

- Project environment: `/mnt/data/yetbye/h3-interactive/.conda/envs/h3-interactive`.
- CPU only; no CUDA calls, models, weights, network listeners, or external data.
- Test runner: Python standard-library `unittest`.

## PASS gates

1. All contract tests pass with exit code 0.
2. GPU memory and utilization are not increased by the test process.
3. Source compiles with `compileall`.
4. Test covers full lifecycle, timing, control revision, reset epoch, timeout, capacity, backpressure retry, unknown session, and closed-state rejection.

## Result

- Python 3.10.21.
- `compileall`: exit code 0.
- `unittest`: 3/3 tests passed in 0.089 seconds, exit code 0.
- GPU snapshot before/after: 2806 MiB and 100% utilization in both snapshots; the test ran with `CUDA_VISIBLE_DEVICES=''` and did not change GPU use.
- Evidence: `logs/unittest.log`, `logs/exit-code.txt`, `logs/gpu-before.txt`, `logs/gpu-after.txt`, and `logs/SHA256SUMS`.
