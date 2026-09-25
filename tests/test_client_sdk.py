import unittest

from aiohttp.test_utils import TestServer

from h3_interactive import (
    AsyncInteractiveClient,
    CapacityError,
    FrameTimeoutError,
    InvalidStateError,
    SessionManager,
    SessionState,
    UnknownSessionError,
)
from h3_interactive.backends import MockBackend
from h3_interactive.http_api import create_app


class ClientSdkTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = TestServer(create_app(SessionManager(MockBackend, max_sessions=1)))
        await self.server.start_server()
        self.client = AsyncInteractiveClient(str(self.server.make_url("/")).rstrip("/"))

    async def asyncTearDown(self):
        await self.client.close()
        await self.server.close()

    async def test_remote_session_manages_sequences_controls_and_reset(self):
        session = await self.client.create_session(session_id="sdk-test")
        pcm = b"\0" * session.config.bytes_per_audio_chunk

        await session.update_control("prompt", {"text": "wave"})
        await session.push_audio(pcm)
        await session.push_audio(pcm)
        frame = await session.pull_video_chunk(timeout_seconds=0.1)
        self.assertEqual((frame.payload, frame.sequence, frame.pts_ms), (b"mock:0:0", 0, 0))
        self.assertEqual(frame.control_revision, 1)

        status = await session.status()
        self.assertEqual(status.state, SessionState.ACTIVE)
        await session.reset()
        await session.push_audio(pcm)
        await session.push_audio(pcm)
        frame = await session.pull_video_chunk(timeout_seconds=0.1)
        self.assertEqual((frame.sequence, frame.pts_ms, frame.epoch), (0, 0, 1))
        self.assertIsNone(frame.control_revision)

        await session.close()
        await session.close()
        with self.assertRaises(InvalidStateError):
            await session.status()

    async def test_typed_server_errors_and_client_validation(self):
        session = await self.client.create_session(session_id="only-session")
        with self.assertRaises(CapacityError):
            await self.client.create_session(session_id="second")
        with self.assertRaises(FrameTimeoutError):
            await session.pull_video_chunk(timeout_seconds=0.001)
        with self.assertRaises(ValueError):
            await session.push_audio(b"bad")
        with self.assertRaises(UnknownSessionError):
            await self.client.get_status("missing")
        await session.close()


if __name__ == "__main__":
    unittest.main()
