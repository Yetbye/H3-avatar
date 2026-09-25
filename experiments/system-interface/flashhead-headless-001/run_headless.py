"""Headless FlashHead system-baseline run (experiments/system-interface/flashhead-headless-001).

Exercises the real SoulXFlashHeadRuntime through the FlashHeadBackend adapter:
realtime-paced audio push (20 ms PCM chunks), continuous video pull with a
playback-schedule emulation, a mid-session reset, and a clean close. Writes
metrics.json, an MP4 (muxed with the source audio), inspection stills, and a
machine-readable PASS/FAIL verdict.

PASS gates (see experiment.md):
  G1  continuous >=60 s session at 25 FPS with no frame-PTS gaps, plus a
      clean 5 s epoch-1 session after reset
  G2  playback-side frame interval p95 <= 40 ms (zero starvation of a
      24-frame jitter buffer) and steady-state realtime factor >= 1.0
  G3  session reset < 2 s
  G4  no worker crash, clean close, evidence artifacts present

Exit codes: 0 = all gates passed, 1 = run completed but a gate failed,
2 = GPU busy (refusing to interfere), 3 = setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import time
import wave
from dataclasses import asdict

import imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from h3_interactive.backends.flashhead import FlashHeadBackend
from h3_interactive.backends.flashhead_runtime import make_runtime_factory
from h3_interactive.protocol import AudioChunk, SessionConfig


def gpu_busy_pids() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        return ["<nvidia-smi-failed>"]
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def load_wav_extended(path: str, duration_s: float) -> np.ndarray:
    """Load a 16 kHz mono s16le WAV and loop-extend it to exactly duration_s."""
    with wave.open(path, "rb") as wf:
        if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (1, 2, 16_000):
            raise ValueError(f"{path}: expected 16 kHz mono s16le WAV")
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    source = frames.astype(np.float32) / 32768.0
    if len(source) == 0:
        raise ValueError(f"{path}: empty WAV")
    target = int(duration_s * 16_000)
    repeats, tail = divmod(target, len(source))
    return np.concatenate([np.tile(source, repeats), source[:tail]])


class PhaseResult:
    def __init__(self) -> None:
        self.frames = 0
        self.first_arrival_s: float | None = None
        self.last_arrival_s: float | None = None
        self.raw_intervals_ms: list[float] = []
        self.playback_intervals_ms: list[float] = []
        self.starve_count = 0
        self.pts_gaps: list[int] = []
        self.producer_lag_ms = 0.0


async def run_phase(
    backend: FlashHeadBackend,
    audio: np.ndarray,
    epoch: int,
    expected_frames: int,
    producer_start: float,
    first_frame_timeout: float,
    frame_timeout: float,
    writer,
    stills: dict[int, np.ndarray | None],
    global_offset: int,
) -> PhaseResult:
    result = PhaseResult()
    chunk_samples = 320  # 20 ms at 16 kHz

    async def producer() -> None:
        start = time.monotonic()
        n_chunks = len(audio) // chunk_samples
        for i in range(n_chunks):
            target = producer_start + i * 0.020
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            payload = (audio[i * chunk_samples:(i + 1) * chunk_samples] * 32767).astype("<i2").tobytes()
            await backend.push_audio(AudioChunk(
                payload=payload, sequence=i, pts_ms=i * 20, end_of_stream=(i == n_chunks - 1),
            ))
        result.producer_lag_ms = max(0.0, (time.monotonic() - start) - n_chunks * 0.020) * 1000.0

    async def consumer() -> None:
        prev_playback_time: float | None = None
        while result.frames < expected_frames:
            timeout = first_frame_timeout if result.frames == 0 else frame_timeout
            chunk = await backend.pull_video_chunk(timeout_seconds=timeout)
            if chunk.epoch != epoch:
                raise RuntimeError(f"epoch mismatch: got {chunk.epoch}, expected {epoch}")
            now = time.monotonic()
            expected_pts = result.frames * 40
            if chunk.pts_ms != expected_pts:
                result.pts_gaps.append(expected_pts)
            if result.first_arrival_s is None:
                result.first_arrival_s = now - producer_start
            if prev_playback_time is not None:
                result.raw_intervals_ms.append((now - result.last_arrival_s) * 1000.0)
            # playback emulation: frame i is due at first_arrival + i * 40 ms
            due = producer_start + result.first_arrival_s + result.frames * 0.040
            if now > due + 0.040:
                result.starve_count += 1
            playback_time = max(due, now)
            if prev_playback_time is not None:
                result.playback_intervals_ms.append((playback_time - prev_playback_time) * 1000.0)
            prev_playback_time = playback_time
            result.frames += 1
            result.last_arrival_s = now

            frame = _decode_jpeg(chunk.payload)
            writer.append_data(frame)
            still_index = global_offset + result.frames - 1
            if still_index in stills:
                stills[still_index] = frame

    producer_task = asyncio.create_task(producer())
    try:
        await consumer()
    finally:
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
    return result


def _decode_jpeg(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as img:
        return np.asarray(img.convert("RGB"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round(p / 100.0 * len(sorted_values))) - 1))
    return sorted_values[index]


async def main(args) -> int:
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    busy = gpu_busy_pids()
    if busy:
        print(f"GPU is already in use; refusing to interfere. Active compute PIDs: {busy}", file=sys.stderr)
        return 2

    if args.model_type not in ("lite", "pro"):
        print(f"unsupported model type: {args.model_type}", file=sys.stderr)
        return 3

    session_audio = load_wav_extended(args.audio_wav, args.duration + args.post_reset_seconds)
    phase1_audio = session_audio[: int(args.duration * 16_000)]
    phase2_audio = session_audio[int(args.duration * 16_000):]
    expected_phase1 = int(args.duration * 25)
    expected_phase2 = int(args.post_reset_seconds * 25)

    factory = make_runtime_factory(
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        repo_dir=args.repo_dir,
        cond_image=args.cond_image,
        model_type=args.model_type,
        seed=args.seed,
        use_face_crop=args.use_face_crop,
    )
    backend = FlashHeadBackend(factory)

    mp4_path = os.path.join(out_dir, "outputs", "session.mp4")
    wav_path = os.path.join(out_dir, "outputs", "session_audio.wav")
    temp_mp4 = mp4_path.replace(".mp4", "_video_only.mp4")
    stills_dir = os.path.join(out_dir, "outputs", "stills")
    os.makedirs(stills_dir, exist_ok=True)
    stills: dict[int, np.ndarray | None] = {int(t * 25): None for t in (10, 20, 30, 40, 50, 60)}

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes((np.clip(session_audio, -1, 1) * 32767).astype("<i2").tobytes())

    status_log: list[dict] = []

    async def monitor() -> None:
        while True:
            status_log.append(asdict(backend.status()))
            await asyncio.sleep(5.0)

    monitor_task = asyncio.create_task(monitor())
    try:
        await backend.start(SessionConfig())
        print(f"session started; model_type={args.model_type}; expected {expected_phase1} + {expected_phase2} frames")

        with imageio.get_writer(
            temp_mp4, format="mp4", mode="I", fps=25, codec="h264", ffmpeg_params=["-bf", "0"],
        ) as writer:
            phase1_start = time.monotonic()
            print(f"phase 1: pushing {args.duration:.0f} s of paced audio (epoch 0)")
            phase1 = await run_phase(
                backend, phase1_audio, epoch=0, expected_frames=expected_phase1,
                producer_start=phase1_start, first_frame_timeout=args.first_frame_timeout,
                frame_timeout=args.frame_timeout, writer=writer, stills=stills, global_offset=0,
            )
            print(f"phase 1 done: {phase1.frames} frames, first frame +{phase1.first_arrival_s:.1f} s, "
                  f"starve {phase1.starve_count}, pts gaps {len(phase1.pts_gaps)}")

            reset_t0 = time.monotonic()
            await backend.reset()
            reset_ms = (time.monotonic() - reset_t0) * 1000.0
            print(f"reset: {reset_ms:.1f} ms; epoch now {backend.status().epoch}")

            phase2_start = time.monotonic()
            print(f"phase 2: pushing {args.post_reset_seconds:.0f} s of paced audio (epoch 1)")
            phase2 = await run_phase(
                backend, phase2_audio, epoch=1, expected_frames=expected_phase2,
                producer_start=phase2_start, first_frame_timeout=args.frame_timeout,
                frame_timeout=args.frame_timeout, writer=writer, stills=stills,
                global_offset=expected_phase1,
            )
            print(f"phase 2 done: {phase2.frames} frames, starve {phase2.starve_count}, "
                  f"pts gaps {len(phase2.pts_gaps)}")
    finally:
        close_t0 = time.monotonic()
        await backend.close()
        close_ms = (time.monotonic() - close_t0) * 1000.0
        monitor_task.cancel()
    print(f"closed in {close_ms:.1f} ms; final state {backend.status().state.value}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", temp_mp4, "-i", wav_path, "-c:v", "copy", "-c:a", "aac", "-shortest", mp4_path],
        check=True, capture_output=True,
    )
    os.remove(temp_mp4)
    for t in (10, 20, 30, 40, 50, 60):
        frame = stills.get(int(t * 25))
        if frame is not None:
            Image.fromarray(frame).save(os.path.join(stills_dir, f"t{t:02d}s.png"))

    try:
        import torch

        peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20
    except Exception:
        peak_vram_mb = -1.0

    # ----- metrics -----------------------------------------------------
    def phase_metrics(phase: PhaseResult) -> dict:
        wall = None
        if phase.first_arrival_s is not None and phase.last_arrival_s is not None:
            wall = phase.last_arrival_s - phase.first_arrival_s
        return {
            "frames": phase.frames,
            "first_frame_latency_s": phase.first_arrival_s,
            "generation_wall_s": wall,
            "realtime_factor_steady": (
                0.0 if wall in (None, 0.0) else phase.frames * 0.040 / wall
            ),
            "raw_arrival_interval_ms": {
                "p50": percentile(phase.raw_intervals_ms, 50),
                "p95": percentile(phase.raw_intervals_ms, 95),
                "p99": percentile(phase.raw_intervals_ms, 99),
                "max": max(phase.raw_intervals_ms, default=0.0),
            },
            "playback_interval_ms": {
                "p50": percentile(phase.playback_intervals_ms, 50),
                "p95": percentile(phase.playback_intervals_ms, 95),
                "p99": percentile(phase.playback_intervals_ms, 99),
                "max": max(phase.playback_intervals_ms, default=0.0),
            },
            "starve_count": phase.starve_count,
            "pts_gap_count": len(phase.pts_gaps),
            "producer_lag_ms": phase.producer_lag_ms,
        }

    gates = {
        "G1_continuous_60s_no_pts_gaps": (
            phase1.frames == expected_phase1 and not phase1.pts_gaps
            and phase2.frames == expected_phase2 and not phase2.pts_gaps
        ),
        "G2_playback_p95_le_40ms_and_rtf": (
            percentile(phase1.playback_intervals_ms, 95) <= 40.0
            and phase1.starve_count == 0
            and (
                phase1.realtime_factor_steady >= 1.0
                if args.model_type == "lite"
                else True  # pro is known non-realtime; RTF reported, not gated
            )
        ),
        "G3_reset_under_2s": reset_ms < 2000.0,
        "G4_no_crash_clean_close_artifacts": (
            backend.status().state.value == "closed"
            and os.path.exists(mp4_path)
            and len(os.listdir(stills_dir)) == 6
        ),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"

    metrics = {
        "model_type": args.model_type,
        "duration_s": args.duration,
        "post_reset_s": args.post_reset_seconds,
        "reset_ms": reset_ms,
        "close_ms": close_ms,
        "peak_vram_mb": peak_vram_mb,
        "phase1": phase_metrics(phase1),
        "phase2": phase_metrics(phase2),
        "status_samples": len(status_log),
        "gates": gates,
        "verdict": verdict,
    }
    metrics_path = os.path.join(out_dir, "logs", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics["gates"], indent=2))
    print(f"verdict: {verdict}; artifacts under {out_dir}")
    return 0 if verdict == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--wav2vec-dir", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--cond-image", required=True)
    parser.add_argument("--audio-wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-type", default="lite", choices=["lite", "pro"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-face-crop", action="store_true")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--post-reset-seconds", type=float, default=5.0)
    parser.add_argument("--first-frame-timeout", type=float, default=600.0)
    parser.add_argument("--frame-timeout", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
