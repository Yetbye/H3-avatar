"""Session lifecycle independent of HTTP, WebRTC, and model implementations."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from .protocol import (
    AudioChunk,
    CapacityError,
    ControlUpdate,
    InteractiveBackend,
    SessionConfig,
    SessionStatus,
    UnknownSessionError,
    VideoChunk,
)


class SessionManager:
    def __init__(self, backend_factory: Callable[[], InteractiveBackend], max_sessions: int = 1) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._backend_factory = backend_factory
        self._max_sessions = max_sessions
        self._sessions: dict[str, InteractiveBackend] = {}

    async def start_session(self, config: SessionConfig, session_id: str | None = None) -> str:
        if len(self._sessions) >= self._max_sessions:
            raise CapacityError("session capacity reached")
        resolved_id = session_id or uuid4().hex
        if resolved_id in self._sessions:
            raise ValueError(f"duplicate session id: {resolved_id}")
        backend = self._backend_factory()
        await backend.start(config)
        self._sessions[resolved_id] = backend
        return resolved_id

    async def push_audio(self, session_id: str, chunk: AudioChunk) -> None:
        await self._get(session_id).push_audio(chunk)

    async def update_control(self, session_id: str, update: ControlUpdate) -> None:
        await self._get(session_id).update_control(update)

    async def pull_video_chunk(
        self, session_id: str, timeout_seconds: float | None = None
    ) -> VideoChunk:
        return await self._get(session_id).pull_video_chunk(timeout_seconds)

    async def reset(self, session_id: str) -> None:
        await self._get(session_id).reset()

    async def close_session(self, session_id: str) -> bool:
        backend = self._sessions.pop(session_id, None)
        if backend is None:
            return False
        await backend.close()
        return True

    def status(self, session_id: str) -> SessionStatus:
        return self._get(session_id).status()

    def _get(self, session_id: str) -> InteractiveBackend:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise UnknownSessionError(f"unknown session: {session_id}") from error
