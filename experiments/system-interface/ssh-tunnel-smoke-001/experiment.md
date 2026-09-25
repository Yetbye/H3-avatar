# Experiment: ssh-tunnel-smoke-001

Status: `PASS`

## Question and hypothesis

Can the local SDK complete a session against the server MockBackend through an SSH local-forward tunnel while the service remains bound only to server loopback?

The hypothesis passes if the local client creates a session, sends control and two PCM chunks, receives the expected video chunk, resets, closes, and leaves no server/tunnel process or listener after cleanup.

This experiment does not expose a public port, authenticate end users, import the legacy project, load a model, or validate realtime performance.

## Fixed topology

```text
local SDK → 127.0.0.1:18765
          → SSH -L tunnel
          → server 127.0.0.1:18765
          → MockBackend
```

- Server bind guard rejects non-loopback addresses unless an explicit unsafe-development flag is supplied.
- The server process runs with `CUDA_VISIBLE_DEVICES=''`.
- The service and tunnel are temporary and must be terminated after the test.
- No credentials are stored in commands, scripts, logs, or manifests.

## PASS gates

1. Full unit regression suite passes.
2. Local cross-host smoke exits 0 and returns `mock:0:0`, sequence 0, PTS 0, control revision 1, and reset epoch 1.
3. Server is reachable only through the SSH tunnel during the test.
4. No service/tunnel process or port 18765 listener remains after cleanup.
5. GPU memory does not increase.

## Result

- Unit regression: 12/12 tests passed in 0.113 seconds.
- Local Python 3.10 client connected through `ssh -L` and exited 0.
- Received `mock:0:0`, video sequence 0, PTS 0, control revision 1; reset returned epoch 1.
- Server listened only on `127.0.0.1:18765`.
- Cleanup confirmed no local or remote listener on port 18765 and no remaining server process.
- GPU memory changed from 2806 MiB to 2794 MiB while the unrelated existing workload remained active; server ran with `CUDA_VISIBLE_DEVICES=''`.
- Evidence: `logs/server.log`, `logs/client-output.json`, `logs/cleanup.txt`, `logs/unittest.log`, `logs/unit-exit-code.txt`, `logs/gpu-before.txt`, `logs/gpu-after.txt`, and `logs/SHA256SUMS`.
