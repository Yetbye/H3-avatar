"""Contract adapter for a future SoulX-FlashHead streaming runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from ..protocol import (
    AudioChunk,
    ControlUpdate,
    FrameTimeoutError,
    InteractiveBackend,
    InvalidStateError,
    SessionConfig,
    SessionState,
    SessionStatus,
    VideoChunk,
)


@dataclass(frozen=True)
class FlashHeadFrame:
    """One encoded frame produced by a FlashHead runtime implementation."""

    payload: bytes
    pts_ms: int
    duration_ms: int = 40


class FlashHeadRuntime(Protocol):
    """Narrow boundary around the upstream GPU pipeline.

    A concrete runtime will own reference-image preparation, audio buffering,
    Wav2Vec encoding, model inference, and encoded-frame queues.
    """

    def push_audio(self, samples: np.ndarray, metadata: Mapping[str, Any]) -> None: ...

    async def pull_video_frame(self, timeout_seconds: float | None) -> FlashHeadFrame: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def pending_audio_chunks(self) -> int: ...

    def pending_video_frames(self) -> int: ...


class FlashHeadBackend(InteractiveBackend):
    """Map the stable service contract to an injected FlashHead runtime."""

    def __init__(self, runtime_factory: Callable[[SessionConfig], FlashHeadRuntime]) -> None:
        self._runtime_factory = runtime_factory
        self._runtime: FlashHeadRuntime | None = None
        self._config: SessionConfig | None = None
        self._state = SessionState.CREATED
        self._epoch = 0
        self._next_audio_sequence = 0
        self._next_video_sequence = 0

    async def start(self, config: SessionConfig) -> None:
        if self._state is not SessionState.CREATED:
            raise InvalidStateError(f"cannot start FlashHead backend in state {self._state.value}")
        config.validate()
        self._runtime = self._runtime_factory(config)
        self._config = config
        self._state = SessionState.ACTIVE

    async def push_audio(self, chunk: AudioChunk) -> None:
        runtime, config = self._require_active()
        if chunk.sequence != self._next_audio_sequence:
            raise ValueError(
                f"expected audio sequence {self._next_audio_sequence}, got {chunk.sequence}"
            )
        expected_pts = chunk.sequence * config.audio_chunk_ms
        if chunk.pts_ms != expected_pts:
            raise ValueError(f"expected audio pts_ms {expected_pts}, got {chunk.pts_ms}")
        if len(chunk.payload) != config.bytes_per_audio_chunk:
            raise ValueError(
                f"expected {config.bytes_per_audio_chunk} PCM bytes, got {len(chunk.payload)}"
            )

        samples = np.frombuffer(chunk.payload, dtype="<i2").astype(np.float32) / 32768.0
        runtime.push_audio(
            samples,
            {
                "sequence": chunk.sequence,
                "pts_ms": chunk.pts_ms,
                "end_of_stream": chunk.end_of_stream,
                "epoch": self._epoch,
            },
        )
        self._next_audio_sequence += 1

    async def update_control(self, update: ControlUpdate) -> None:
        self._require_active()
        raise ValueError(
            f"FlashHead control kind {update.kind!r} is not mapped by the upstream runtime"
        )

    async def pull_video_chunk(self, timeout_seconds: float | None = None) -> VideoChunk:
        runtime, config = self._require_active()
        try:
            frame = await runtime.pull_video_frame(timeout_seconds)
        except asyncio.TimeoutError as error:
            raise FrameTimeoutError("timed out waiting for a FlashHead video frame") from error
        if frame.duration_ms != config.video_frame_ms:
            raise ValueError(
                f"expected {config.video_frame_ms} ms video frame, got {frame.duration_ms}"
            )
        result = VideoChunk(
            payload=frame.payload,
            sequence=self._next_video_sequence,
            pts_ms=frame.pts_ms,
            duration_ms=frame.duration_ms,
            epoch=self._epoch,
        )
        self._next_video_sequence += 1
        return result

    async def reset(self) -> None:
        runtime, _config = self._require_active()
        runtime.reset()
        self._epoch += 1
        self._next_audio_sequence = 0
        self._next_video_sequence = 0

    async def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        if self._runtime is not None:
            self._runtime.close()
        self._state = SessionState.CLOSED

    def status(self) -> SessionStatus:
        runtime = self._runtime
        return SessionStatus(
            state=self._state,
            epoch=self._epoch,
            pending_audio_chunks=0 if runtime is None else runtime.pending_audio_chunks(),
            pending_video_chunks=0 if runtime is None else runtime.pending_video_frames(),
            metadata={"backend": "flashhead", "runtime": "injected"},
        )

    def _require_active(self) -> tuple[FlashHeadRuntime, SessionConfig]:
        if (
            self._state is not SessionState.ACTIVE
            or self._runtime is None
            or self._config is None
        ):
            raise InvalidStateError(f"FlashHead backend is {self._state.value}, expected active")
        return self._runtime, self._config
