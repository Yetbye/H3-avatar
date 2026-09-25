# FlashHead Adapter Contract 001

- Date: 2026-09-25
- Scope: CPU-only contract adapter; no model weights, CUDA initialization, or third-party edits.
- Upstream revision: SoulX-FlashHead `9bc03de`
- Environment: `/mnt/data/yetbye/envs/h3-interactive`

## Hypothesis

The stable 16 kHz PCM session contract can be separated from the upstream GPU runtime so sequence, PTS, timeout, reset, close, and queue status semantics are testable before weights are available.

## Success criteria

- A `FlashHeadBackend` accepts an injected runtime and validates the stable audio/video timing contract.
- Fake-runtime tests cover PCM conversion, frame mapping, timeout, reset, idempotent close, and unsupported controls.
- `--backend flashhead` refuses to listen when the real runtime is not configured.
- Existing CPU regressions remain green.

## Result

`PARTIAL_PASS`: the adapter contract and safe CLI boundary pass. This is not evidence of FlashHead model loading, video generation, realtime speed, or GPU compatibility.
