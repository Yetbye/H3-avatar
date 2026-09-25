# Experiment: client-sdk-001

Status: `PASS`

## Question and hypothesis

Can a centrally maintained async Python SDK let a local livestream application consume the private server API without duplicating HTTP, session, sequence, PTS, control-revision, and error-handling logic?

The hypothesis passes if `RemoteSession` automatically maintains audio/control counters across normal operation and reset, reconstructs typed video/status objects, closes idempotently, and maps stable server errors to typed exceptions.

This experiment does not modify the local legacy project, start a persistent service, access a public network, load a model, or validate realtime performance.

## Fixed behavior

- Async aiohttp client matching the legacy application's asyncio stack.
- SDK automatically assigns audio sequence, audio PTS, and control revision.
- Audio upload is never automatically retried because it is non-idempotent.
- Session reset resets client-side counters only after the server confirms reset.
- Session close is idempotent locally.
- Tests use an ephemeral loopback TestServer with MockBackend.

## PASS gates

1. Full regression suite passes.
2. SDK lifecycle covers create, control, audio, video, status, reset, and close.
3. Capacity, timeout, invalid local audio, unknown session, and closed-state errors are typed.
4. No persistent process remains and GPU memory does not increase.

## Result

- Full suite: 10/10 tests passed in 0.106 seconds, exit code 0.
- `compileall`: exit code 0.
- GPU memory before/after: 2806 MiB / 2806 MiB; validation ran with `CUDA_VISIBLE_DEVICES=''`.
- No project Python process remained after the SDK and ephemeral server closed.
- Evidence: `logs/unittest.log`, `logs/exit-code.txt`, `logs/gpu-before.txt`, `logs/gpu-after.txt`, `logs/python-processes-after.txt`, and `logs/SHA256SUMS`.
