"""Deterministic CPU-only backend used to validate the service contract."""

from __future__ import annotations

import asyncio

from ..protocol import (
    AudioChunk,
    BackpressureError,
    ControlUpdate,
    FrameTimeoutError,
    InteractiveBackend,
    InvalidStateError,
    SessionConfig,
    SessionState,
    SessionStatus,
    VideoChunk,
)


class MockBackend(InteractiveBackend):
    def __init__(self) -> None:
        self._state = SessionState.CREATED
        self._config: SessionConfig | None = None
        self._audio: list[AudioChunk] = []
        self._video: asyncio.Queue[VideoChunk] | None = None
        self._epoch = 0
        self._next_audio_sequence = 0
        self._next_video_sequence = 0
        self._control_revision: int | None = None

    async def start(self, config: SessionConfig) -> None:
        if self._state is not SessionState.CREATED:
            raise InvalidStateError(f"cannot start session in state {self._state.value}")
        config.validate()
        self._config = config
        self._video = asyncio.Queue(maxsize=config.max_output_chunks)
        self._state = SessionState.ACTIVE

    async def push_audio(self, chunk: AudioChunk) -> None:
        self._require_active()
        assert self._config is not None and self._video is not None
        if chunk.sequence != self._next_audio_sequence:
            raise ValueError(
                f"expected audio sequence {self._next_audio_sequence}, got {chunk.sequence}"
            )
        expected_pts = chunk.sequence * self._config.audio_chunk_ms
        if chunk.pts_ms != expected_pts:
            raise ValueError(f"expected audio pts_ms {expected_pts}, got {chunk.pts_ms}")
        if len(chunk.payload) != self._config.bytes_per_audio_chunk:
            raise ValueError(
                f"expected {self._config.bytes_per_audio_chunk} PCM bytes, got {len(chunk.payload)}"
            )
        if len(self._audio) == 1 and self._video.full():
            raise BackpressureError("video output queue is full; retry the audio chunk after pulling output")

        self._audio.append(chunk)
        self._next_audio_sequence += 1
        if len(self._audio) == 2:
            first, _second = self._audio
            frame = VideoChunk(
                payload=f"mock:{self._epoch}:{self._next_video_sequence}".encode("ascii"),
                sequence=self._next_video_sequence,
                pts_ms=first.pts_ms,
                duration_ms=self._config.video_frame_ms,
                epoch=self._epoch,
                control_revision=self._control_revision,
            )
            self._video.put_nowait(frame)
            self._audio.clear()
            self._next_video_sequence += 1

    async def update_control(self, update: ControlUpdate) -> None:
        self._require_active()
        if not update.kind:
            raise ValueError("control kind must be non-empty")
        if self._control_revision is not None and update.revision <= self._control_revision:
            raise ValueError("control revision must increase monotonically")
        self._control_revision = update.revision

    async def pull_video_chunk(self, timeout_seconds: float | None = None) -> VideoChunk:
        self._require_active()
        assert self._video is not None
        try:
            if timeout_seconds is None:
                return await self._video.get()
            return await asyncio.wait_for(self._video.get(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise FrameTimeoutError("timed out waiting for a video chunk") from error

    async def reset(self) -> None:
        self._require_active()
        assert self._video is not None
        self._audio.clear()
        while not self._video.empty():
            self._video.get_nowait()
        self._epoch += 1
        self._next_audio_sequence = 0
        self._next_video_sequence = 0
        self._control_revision = None

    async def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        self._audio.clear()
        if self._video is not None:
            while not self._video.empty():
                self._video.get_nowait()
        self._state = SessionState.CLOSED

    def status(self) -> SessionStatus:
        return SessionStatus(
            state=self._state,
            epoch=self._epoch,
            pending_audio_chunks=len(self._audio),
            pending_video_chunks=0 if self._video is None else self._video.qsize(),
            metadata={"backend": "mock"},
        )

    def _require_active(self) -> None:
        if self._state is not SessionState.ACTIVE:
            raise InvalidStateError(f"session is {self._state.value}, expected active")
