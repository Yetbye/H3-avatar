"""Safe command-line entry point for the private interactive model service."""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Callable

from aiohttp import web

from .backends import FlashHeadBackend, FlashHeadRuntime, MockBackend
from .http_api import create_app
from .protocol import SessionConfig
from .session import SessionManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "flashhead"], default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Explicitly acknowledge binding an unauthenticated development API off loopback.",
    )
    return parser.parse_args()


def validate_bind(host: str, port: int, allow_non_loopback: bool) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.lower() == "localhost"
    if not is_loopback and not allow_non_loopback:
        raise ValueError(
            "refusing non-loopback bind without --allow-non-loopback; use an SSH tunnel"
        )


def build_app(
    backend: str,
    max_sessions: int,
    flashhead_runtime_factory: Callable[[SessionConfig], FlashHeadRuntime] | None = None,
) -> web.Application:
    if backend == "mock":
        backend_factory = MockBackend
    elif backend == "flashhead":
        if flashhead_runtime_factory is None:
            raise ValueError(
                "FlashHead adapter contract is available, but the real upstream runtime "
                "and weights are not configured (set FLASHHEAD_CKPT_DIR, "
                "FLASHHEAD_WAV2VEC_DIR, FLASHHEAD_REPO_DIR, FLASHHEAD_COND_IMAGE)"
            )
        backend_factory = lambda: FlashHeadBackend(flashhead_runtime_factory)
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return create_app(SessionManager(backend_factory, max_sessions=max_sessions))


def main() -> None:
    args = parse_args()
    validate_bind(args.host, args.port, args.allow_non_loopback)
    runtime_factory = None
    if args.backend == "flashhead":
        from .backends.flashhead_runtime import runtime_factory_from_env

        runtime_factory = runtime_factory_from_env()
    try:
        app = build_app(args.backend, args.max_sessions, flashhead_runtime_factory=runtime_factory)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from None
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        access_log=None,
        print=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
