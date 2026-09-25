# Validation

Result: `PASS`

The bridge correctly converts 640-byte little-endian PCM s16le chunks into 320-sample float32 arrays, preserves sequence/PTS/epoch metadata, maps text and custom-state controls, translates legacy frame timeouts, returns timestamped video chunks, and performs explicit flush/drain on reset and close.

The original LiveTalking directory was not imported or modified. This is bridge-contract evidence only; it is not evidence that the real legacy runtime, aiortc, wav2lip, FlashHead, or H3 can run.

The next gate is a runtime wrapper around the real LiveTalking object and queues. It should first be checked statically and with injected fake queues before any model load.
