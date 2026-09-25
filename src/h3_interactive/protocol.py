"""Stable protocol shared by mock, LiveTalking, FlashHead, and H3 backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class SessionState(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"
    FAILED = "failed"


class ProtocolError(RuntimeError):
    """Base exception for stable service-contract failures."""


class InvalidStateError(ProtocolError):
    pass


class BackpressureError(ProtocolError):
    pass


class CapacityError(ProtocolError):
    pass


class FrameTimeoutError(ProtocolError):
    pass


class UnknownSessionError(ProtocolError):
    pass


@dataclass(frozen=True)
class SessionConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    audio_chunk_ms: int = 20
    video_frame_ms: int = 40
    max_output_chunks: int = 8

    @property
    def samples_per_audio_chunk(self) -> int:
        return self.sample_rate * self.audio_chunk_ms // 1_000

    @property
    def bytes_per_audio_chunk(self) -> int:
        return self.samples_per_audio_chunk * self.channels * self.sample_width_bytes

    def validate(self) -> None:
        if (self.sample_rate, self.channels, self.sample_width_bytes) != (16_000, 1, 2):
            raise ValueError("only 16 kHz mono PCM s16le is supported in the baseline contract")
        if (self.audio_chunk_ms, self.video_frame_ms) != (20, 40):
            raise ValueError("baseline timing must remain 20 ms audio / 40 ms video")
        if self.max_output_chunks <= 0:
            raise ValueError("max_output_chunks must be positive")


@dataclass(frozen=True)
class AudioChunk:
    payload: bytes
    sequence: int
    pts_ms: int
    end_of_stream: bool = False


@dataclass(frozen=True)
class ControlUpdate:
    kind: str
    payload: Mapping[str, Any]
    revision: int


@dataclass(frozen=True)
class VideoChunk:
    payload: bytes
    sequence: int
    pts_ms: int
    duration_ms: int
    epoch: int
    control_revision: int | None = None


@dataclass(frozen=True)
class SessionStatus:
    state: SessionState
    epoch: int
    pending_audio_chunks: int
    pending_video_chunks: int
    last_error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InteractiveBackend(ABC):
    @abstractmethod
    async def start(self, config: SessionConfig) -> None: ...

    @abstractmethod
    async def push_audio(self, chunk: AudioChunk) -> None: ...

    @abstractmethod
    async def update_control(self, update: ControlUpdate) -> None: ...

    @abstractmethod
    async def pull_video_chunk(self, timeout_seconds: float | None = None) -> VideoChunk: ...

    @abstractmethod
    async def reset(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def status(self) -> SessionStatus: ...
