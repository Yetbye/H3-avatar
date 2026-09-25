"""Bridge from the stable session contract to a LiveTalking-like runtime."""

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
class LegacyVideoFrame:
    payload: bytes
    pts_ms: int
    duration_ms: int = 40


class LegacyRuntime(Protocol):
    def put_audio_frame(self, samples: np.ndarray, datainfo: Mapping[str, Any]) -> None: ...

    def put_msg_txt(self, text: str, datainfo: Mapping[str, Any]) -> None: ...

    def set_custom_state(self, audiotype: int, reinit: bool = True) -> None: ...

    async def pull_video_frame(self, timeout_seconds: float | None) -> LegacyVideoFrame: ...

    def flush_talk(self) -> None: ...

    def drain_video(self) -> None: ...

    def is_speaking(self) -> bool: ...

    def pending_audio_chunks(self) -> int: ...

    def pending_video_chunks(self) -> int: ...

    def close(self) -> None: ...


class LiveTalkingBridgeBackend(InteractiveBackend):
    def __init__(self, runtime_factory: Callable[[SessionConfig], LegacyRuntime]) -> None:
        self._runtime_factory = runtime_factory
        self._runtime: LegacyRuntime | None = None
        self._config: SessionConfig | None = None
        self._state = SessionState.CREATED
        self._epoch = 0
        self._next_audio_sequence = 0
        self._next_video_sequence = 0
        self._control_revision: int | None = None

    async def start(self, config: SessionConfig) -> None:
        if self._state is not SessionState.CREATED:
            raise InvalidStateError(f"cannot start bridge in state {self._state.value}")
        config.validate()
        self._config = config
        self._runtime = self._runtime_factory(config)
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
        runtime.put_audio_frame(
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
        runtime, _config = self._require_active()
        if self._control_revision is not None and update.revision <= self._control_revision:
            raise ValueError("control revision must increase monotonically")
        if update.kind == "text":
            text = update.payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("text control requires a non-empty 'text' string")
            runtime.put_msg_txt(text, {"revision": update.revision, "epoch": self._epoch})
        elif update.kind == "custom_state":
            audiotype = update.payload.get("audiotype")
            if not isinstance(audiotype, int):
                raise ValueError("custom_state requires integer 'audiotype'")
            runtime.set_custom_state(audiotype, bool(update.payload.get("reinit", True)))
        else:
            raise ValueError(f"unsupported LiveTalking control kind: {update.kind}")
        self._control_revision = update.revision

    async def pull_video_chunk(self, timeout_seconds: float | None = None) -> VideoChunk:
        runtime, config = self._require_active()
        try:
            legacy_frame = await runtime.pull_video_frame(timeout_seconds)
        except asyncio.TimeoutError as error:
            raise FrameTimeoutError("timed out waiting for a LiveTalking video frame") from error
        if legacy_frame.duration_ms != config.video_frame_ms:
            raise ValueError(
                f"expected {config.video_frame_ms} ms video frame, got {legacy_frame.duration_ms}"
            )
        frame = VideoChunk(
            payload=legacy_frame.payload,
            sequence=self._next_video_sequence,
            pts_ms=legacy_frame.pts_ms,
            duration_ms=legacy_frame.duration_ms,
            epoch=self._epoch,
            control_revision=self._control_revision,
        )
        self._next_video_sequence += 1
        return frame

    async def reset(self) -> None:
        runtime, _config = self._require_active()
        runtime.flush_talk()
        runtime.drain_video()
        self._epoch += 1
        self._next_audio_sequence = 0
        self._next_video_sequence = 0
        self._control_revision = None

    async def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        if self._runtime is not None:
            self._runtime.flush_talk()
            self._runtime.drain_video()
            self._runtime.close()
        self._state = SessionState.CLOSED

    def status(self) -> SessionStatus:
        runtime = self._runtime
        return SessionStatus(
            state=self._state,
            epoch=self._epoch,
            pending_audio_chunks=0 if runtime is None else runtime.pending_audio_chunks(),
            pending_video_chunks=0 if runtime is None else runtime.pending_video_chunks(),
            metadata={
                "backend": "livetalking_bridge",
                "speaking": False if runtime is None else runtime.is_speaking(),
            },
        )

    def _require_active(self) -> tuple[LegacyRuntime, SessionConfig]:
        if (
            self._state is not SessionState.ACTIVE
            or self._runtime is None
            or self._config is None
        ):
            raise InvalidStateError(f"bridge is {self._state.value}, expected active")
        return self._runtime, self._config
