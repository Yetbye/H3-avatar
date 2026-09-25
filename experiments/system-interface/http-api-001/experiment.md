# Experiment: http-api-001

Status: `PASS`

## Question and hypothesis

Can a private HTTP API expose the stable session contract so a local legacy livestream client can use model backends hosted in `/mnt/data/yetbye/h3-interactive`?

The hypothesis passes if an in-process MockBackend supports session creation, raw PCM upload, control update, timed video pull, status, reset, close, and stable HTTP error mapping.

This experiment does not import or modify the local legacy project, expose a public port, authenticate users, load a model, or validate WebRTC/realtime performance.

## Fixed API

- `POST /v1/sessions`
- `POST /v1/sessions/{id}/audio` with raw PCM and sequence/PTS headers
- `POST /v1/sessions/{id}/control`
- `GET /v1/sessions/{id}/video?timeout=...`
- `POST /v1/sessions/{id}/reset`
- `GET /v1/sessions/{id}/status`
- `DELETE /v1/sessions/{id}`

The app has a 1 MiB request limit. Video pull timeout is limited to 0-30 seconds. The future deployment default must bind to loopback or a controlled private interface; public exposure requires authentication, TLS, rate limits, privacy review, and content-safety controls.

## Environment and resources

- Project environment: `/mnt/data/yetbye/h3-interactive/.conda/envs/h3-interactive`.
- CPU-only MockBackend with `CUDA_VISIBLE_DEVICES=''`.
- aiohttp test server on an ephemeral loopback port; no persistent listener.

## PASS gates

1. All previous regression tests remain green.
2. HTTP lifecycle and error tests pass with exit code 0.
3. Source compiles successfully.
4. No persistent process or listening port remains after the test.
5. GPU memory does not increase.

## Result

- Final suite: 8/8 tests passed in 0.073 seconds, exit code 0.
- `compileall`: exit code 0.
- GPU memory remained 2806 MiB across the initial validation; tests ran with `CUDA_VISIBLE_DEVICES=''`.
- No project Python process remained after the HTTP test server closed.
- Evidence: `logs/unittest-final.log`, `logs/exit-code-final.txt`, `logs/gpu-before.txt`, `logs/gpu-after.txt`, and `logs/SHA256SUMS.final`.
