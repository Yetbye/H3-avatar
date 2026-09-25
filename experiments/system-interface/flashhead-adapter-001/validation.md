# Validation

- Python compile check: `PASS`
- Full CPU suite: `15/15 PASS` in 0.124 seconds
- FlashHead-specific adapter tests: `2/2 PASS`
- Unconfigured CLI guard: exit code `1`, with an explicit runtime/weights error
- Listener residue on port 8765 after guard: none
- GPU use: none (`CUDA_VISIBLE_DEVICES=''` for checks)
- Third-party source changes: none

## Remaining gates

- Weight provenance and license review
- FlashHead profile dependency switch and import smoke
- Real model load and fixed-sample generation
- Chunk continuity, latency, peak VRAM, reset, and long-session validation
