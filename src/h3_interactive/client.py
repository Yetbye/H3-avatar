"""Async Python SDK for the private H3 interactive HTTP API."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import aiohttp

from .protocol import (
    BackpressureError,
    CapacityError,
    FrameTimeoutError,
    InvalidStateError,
    SessionConfig,
    SessionState,
    SessionStatus,
    UnknownSessionError,
    VideoChunk,
)


class RemoteApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class AsyncInteractiveClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AsyncInteractiveClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def create_session(
        self, config: SessionConfig | None = None, session_id: str | None = None
    ) -> "RemoteSession":
        resolved_config = config or SessionConfig()
        body: dict[str, Any] = {"config": asdict(resolved_config)}
        if session_id is not None:
            body["session_id"] = session_id
        response = await self._json_request("POST", "/v1/sessions", json=body)
        return RemoteSession(self, response["session_id"], resolved_config)

    async def push_audio(
        self,
        session_id: str,
        payload: bytes,
        sequence: int,
        pts_ms: int,
        end_of_stream: bool = False,
    ) -> None:
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Audio-Sequence": str(sequence),
            "X-Audio-PTS-Ms": str(pts_ms),
            "X-End-Of-Stream": str(end_of_stream).lower(),
        }
        await self._empty_request(
            "POST", f"/v1/sessions/{session_id}/audio", data=payload, headers=headers
        )

    async def update_control(
        self, session_id: str, kind: str, payload: Mapping[str, Any], revision: int
    ) -> None:
        await self._empty_request(
            "POST",
            f"/v1/sessions/{session_id}/control",
            json={"kind": kind, "payload": dict(payload), "revision": revision},
        )

    async def pull_video_chunk(
        self, session_id: str, timeout_seconds: float = 5.0
    ) -> VideoChunk:
        session = await self._ensure_session()
        async with session.get(
            self._url(f"/v1/sessions/{session_id}/video"),
            params={"timeout": str(timeout_seconds)},
        ) as response:
            await self._raise_for_error(response)
            payload = await response.read()
            control_revision = response.headers.get("X-Control-Revision")
            return VideoChunk(
                payload=payload,
                sequence=int(response.headers["X-Video-Sequence"]),
                pts_ms=int(response.headers["X-Video-PTS-Ms"]),
                duration_ms=int(response.headers["X-Video-Duration-Ms"]),
                epoch=int(response.headers["X-Session-Epoch"]),
                control_revision=None if control_revision is None else int(control_revision),
            )

    async def reset(self, session_id: str) -> None:
        await self._empty_request("POST", f"/v1/sessions/{session_id}/reset")

    async def get_status(self, session_id: str) -> SessionStatus:
        body = await self._json_request("GET", f"/v1/sessions/{session_id}/status")
        return SessionStatus(
            state=SessionState(body["state"]),
            epoch=int(body["epoch"]),
            pending_audio_chunks=int(body["pending_audio_chunks"]),
            pending_video_chunks=int(body["pending_video_chunks"]),
            last_error=body.get("last_error"),
            metadata=body.get("metadata", {}),
        )

    async def close_session(self, session_id: str) -> None:
        await self._empty_request("DELETE", f"/v1/sessions/{session_id}")

    async def _json_request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        session = await self._ensure_session()
        async with session.request(method, self._url(path), **kwargs) as response:
            await self._raise_for_error(response)
            body = await response.json()
            if not isinstance(body, dict):
                raise RemoteApiError(response.status, "invalid_response", "expected JSON object")
            return body

    async def _empty_request(self, method: str, path: str, **kwargs) -> None:
        session = await self._ensure_session()
        async with session.request(method, self._url(path), **kwargs) as response:
            await self._raise_for_error(response)
            await response.read()

    async def _raise_for_error(self, response: aiohttp.ClientResponse) -> None:
        if response.status < 400:
            return
        try:
            body = await response.json()
            error = body["error"]
            code = str(error["code"])
            message = str(error["message"])
        except (aiohttp.ContentTypeError, KeyError, TypeError, ValueError):
            raise RemoteApiError(response.status, "invalid_error_response", await response.text())

        exception_types = {
            "unknown_session": UnknownSessionError,
            "backpressure": BackpressureError,
            "capacity_reached": CapacityError,
            "frame_timeout": FrameTimeoutError,
            "invalid_state": InvalidStateError,
            "invalid_request": ValueError,
        }
        exception_type = exception_types.get(code)
        if exception_type is not None:
            raise exception_type(message)
        raise RemoteApiError(response.status, code, message)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"


class RemoteSession:
    def __init__(
        self, client: AsyncInteractiveClient, session_id: str, config: SessionConfig
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.config = config
        self._audio_sequence = 0
        self._control_revision = 0
        self._closed = False

    async def push_audio(self, payload: bytes, end_of_stream: bool = False) -> None:
        self._require_open()
        if len(payload) != self.config.bytes_per_audio_chunk:
            raise ValueError(
                f"expected {self.config.bytes_per_audio_chunk} PCM bytes, got {len(payload)}"
            )
        sequence = self._audio_sequence
        await self.client.push_audio(
            self.session_id,
            payload,
            sequence=sequence,
            pts_ms=sequence * self.config.audio_chunk_ms,
            end_of_stream=end_of_stream,
        )
        self._audio_sequence += 1

    async def update_control(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._require_open()
        revision = self._control_revision + 1
        await self.client.update_control(self.session_id, kind, payload, revision)
        self._control_revision = revision

    async def pull_video_chunk(self, timeout_seconds: float = 5.0) -> VideoChunk:
        self._require_open()
        return await self.client.pull_video_chunk(self.session_id, timeout_seconds)

    async def status(self) -> SessionStatus:
        self._require_open()
        return await self.client.get_status(self.session_id)

    async def reset(self) -> None:
        self._require_open()
        await self.client.reset(self.session_id)
        self._audio_sequence = 0
        self._control_revision = 0

    async def close(self) -> None:
        if self._closed:
            return
        await self.client.close_session(self.session_id)
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise InvalidStateError("remote session is closed")
