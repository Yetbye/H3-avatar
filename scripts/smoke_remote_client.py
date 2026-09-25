#!/usr/bin/env python3
"""Run one complete SDK lifecycle against a remote MockBackend service."""

from __future__ import annotations

import argparse
import asyncio
import json

from h3_interactive import AsyncInteractiveClient


async def run(base_url: str) -> None:
    async with AsyncInteractiveClient(base_url) as client:
        session = await client.create_session(session_id="cross-host-smoke")
        pcm = b"\0" * session.config.bytes_per_audio_chunk
        await session.update_control("prompt", {"text": "cross-host smoke"})
        await session.push_audio(pcm)
        await session.push_audio(pcm, end_of_stream=True)
        frame = await session.pull_video_chunk(timeout_seconds=2.0)
        status = await session.status()
        await session.reset()
        reset_status = await session.status()
        await session.close()
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "frame_payload": frame.payload.decode("ascii"),
                    "frame_sequence": frame.sequence,
                    "frame_pts_ms": frame.pts_ms,
                    "control_revision": frame.control_revision,
                    "state_before_reset": status.state.value,
                    "epoch_after_reset": reset_status.epoch,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    asyncio.run(run(args.base_url))


if __name__ == "__main__":
    main()
