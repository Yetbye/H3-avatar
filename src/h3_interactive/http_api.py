"""Private HTTP control/data plane for model backends."""

from __future__ import annotations

from typing import Any

from aiohttp import web

from .protocol import (
    AudioChunk,
    BackpressureError,
    CapacityError,
    ControlUpdate,
    FrameTimeoutError,
    InvalidStateError,
    SessionConfig,
    UnknownSessionError,
)
from .session import SessionManager


MANAGER_KEY: web.AppKey[SessionManager] = web.AppKey("session_manager", SessionManager)


@web.middleware
async def protocol_errors(request: web.Request, handler):
    try:
        return await handler(request)
    except UnknownSessionError as error:
        return _error(404, "unknown_session", str(error))
    except BackpressureError as error:
        return _error(429, "backpressure", str(error))
    except FrameTimeoutError as error:
        return _error(504, "frame_timeout", str(error))
    except InvalidStateError as error:
        return _error(409, "invalid_state", str(error))
    except CapacityError as error:
        return _error(429, "capacity_reached", str(error))
    except (KeyError, TypeError, ValueError) as error:
        return _error(400, "invalid_request", str(error))


def create_app(manager: SessionManager) -> web.Application:
    app = web.Application(client_max_size=1024 * 1024, middlewares=[protocol_errors])
    app[MANAGER_KEY] = manager
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/v1/sessions", create_session)
    app.router.add_post("/v1/sessions/{session_id}/audio", push_audio)
    app.router.add_post("/v1/sessions/{session_id}/control", update_control)
    app.router.add_get("/v1/sessions/{session_id}/video", pull_video)
    app.router.add_post("/v1/sessions/{session_id}/reset", reset_session)
    app.router.add_get("/v1/sessions/{session_id}/status", get_status)
    app.router.add_delete("/v1/sessions/{session_id}", close_session)
    return app


async def healthz(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def create_session(request: web.Request) -> web.Response:
    body = await _json_object(request)
    config_data = body.get("config", {})
    if not isinstance(config_data, dict):
        raise ValueError("config must be an object")
    config = SessionConfig(**config_data)
    session_id = await _manager(request).start_session(config, body.get("session_id"))
    return web.json_response({"session_id": session_id}, status=201)


async def push_audio(request: web.Request) -> web.Response:
    sequence = _required_int_header(request, "X-Audio-Sequence")
    pts_ms = _required_int_header(request, "X-Audio-PTS-Ms")
    end_of_stream = request.headers.get("X-End-Of-Stream", "false").lower() == "true"
    payload = await request.read()
    await _manager(request).push_audio(
        request.match_info["session_id"],
        AudioChunk(
            payload=payload,
            sequence=sequence,
            pts_ms=pts_ms,
            end_of_stream=end_of_stream,
        ),
    )
    return web.Response(status=204)


async def update_control(request: web.Request) -> web.Response:
    body = await _json_object(request)
    payload = body.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    await _manager(request).update_control(
        request.match_info["session_id"],
        ControlUpdate(kind=str(body["kind"]), payload=payload, revision=int(body["revision"])),
    )
    return web.Response(status=204)


async def pull_video(request: web.Request) -> web.Response:
    timeout = float(request.query.get("timeout", "5"))
    if not 0 <= timeout <= 30:
        raise ValueError("timeout must be between 0 and 30 seconds")
    frame = await _manager(request).pull_video_chunk(
        request.match_info["session_id"], timeout_seconds=timeout
    )
    headers = {
        "X-Video-Sequence": str(frame.sequence),
        "X-Video-PTS-Ms": str(frame.pts_ms),
        "X-Video-Duration-Ms": str(frame.duration_ms),
        "X-Session-Epoch": str(frame.epoch),
    }
    if frame.control_revision is not None:
        headers["X-Control-Revision"] = str(frame.control_revision)
    return web.Response(body=frame.payload, content_type="application/octet-stream", headers=headers)


async def reset_session(request: web.Request) -> web.Response:
    await _manager(request).reset(request.match_info["session_id"])
    return web.Response(status=204)


async def get_status(request: web.Request) -> web.Response:
    status = _manager(request).status(request.match_info["session_id"])
    return web.json_response(
        {
            "state": status.state.value,
            "epoch": status.epoch,
            "pending_audio_chunks": status.pending_audio_chunks,
            "pending_video_chunks": status.pending_video_chunks,
            "last_error": status.last_error,
            "metadata": dict(status.metadata),
        }
    )


async def close_session(request: web.Request) -> web.Response:
    closed = await _manager(request).close_session(request.match_info["session_id"])
    if not closed:
        raise UnknownSessionError(f"unknown session: {request.match_info['session_id']}")
    return web.Response(status=204)


async def _json_object(request: web.Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _required_int_header(request: web.Request, name: str) -> int:
    value = request.headers.get(name)
    if value is None:
        raise ValueError(f"missing header: {name}")
    return int(value)


def _manager(request: web.Request) -> SessionManager:
    return request.app[MANAGER_KEY]


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status)
