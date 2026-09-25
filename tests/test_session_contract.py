import unittest

from h3_interactive import (
    AudioChunk,
    BackpressureError,
    CapacityError,
    ControlUpdate,
    FrameTimeoutError,
    InvalidStateError,
    SessionConfig,
    SessionManager,
    SessionState,
    UnknownSessionError,
)
from h3_interactive.backends import MockBackend


def audio(sequence: int, config: SessionConfig) -> AudioChunk:
    return AudioChunk(
        payload=b"\0" * config.bytes_per_audio_chunk,
        sequence=sequence,
        pts_ms=sequence * config.audio_chunk_ms,
    )


class SessionContractTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = SessionConfig(max_output_chunks=2)
        self.manager = SessionManager(MockBackend, max_sessions=1)
        self.session_id = await self.manager.start_session(self.config, "test-session")

    async def asyncTearDown(self) -> None:
        await self.manager.close_session(self.session_id)

    async def test_full_lifecycle_and_timing(self) -> None:
        await self.manager.update_control(
            self.session_id, ControlUpdate(kind="prompt", payload={"text": "wave"}, revision=1)
        )
        await self.manager.push_audio(self.session_id, audio(0, self.config))
        await self.manager.push_audio(self.session_id, audio(1, self.config))

        frame = await self.manager.pull_video_chunk(self.session_id, timeout_seconds=0.1)
        self.assertEqual((frame.sequence, frame.pts_ms, frame.duration_ms), (0, 0, 40))
        self.assertEqual(frame.control_revision, 1)
        self.assertEqual(frame.epoch, 0)

        await self.manager.reset(self.session_id)
        status = self.manager.status(self.session_id)
        self.assertEqual(status.epoch, 1)
        self.assertEqual(status.pending_audio_chunks, 0)
        self.assertEqual(status.pending_video_chunks, 0)

        await self.manager.push_audio(self.session_id, audio(0, self.config))
        await self.manager.push_audio(self.session_id, audio(1, self.config))
        frame = await self.manager.pull_video_chunk(self.session_id, timeout_seconds=0.1)
        self.assertEqual((frame.sequence, frame.pts_ms, frame.epoch), (0, 0, 1))
        self.assertIsNone(frame.control_revision)

        self.assertTrue(await self.manager.close_session(self.session_id))
        self.assertFalse(await self.manager.close_session(self.session_id))
        with self.assertRaises(UnknownSessionError):
            self.manager.status(self.session_id)

    async def test_backpressure_preserves_retryable_audio_sequence(self) -> None:
        manager = SessionManager(MockBackend)
        config = SessionConfig(max_output_chunks=1)
        session_id = await manager.start_session(config)
        await manager.push_audio(session_id, audio(0, config))
        await manager.push_audio(session_id, audio(1, config))
        await manager.push_audio(session_id, audio(2, config))
        with self.assertRaises(BackpressureError):
            await manager.push_audio(session_id, audio(3, config))
        await manager.pull_video_chunk(session_id, timeout_seconds=0.1)
        await manager.push_audio(session_id, audio(3, config))
        frame = await manager.pull_video_chunk(session_id, timeout_seconds=0.1)
        self.assertEqual((frame.sequence, frame.pts_ms), (1, 40))
        await manager.close_session(session_id)

    async def test_validation_timeout_capacity_and_closed_state(self) -> None:
        with self.assertRaises(CapacityError):
            await self.manager.start_session(self.config, "second")
        with self.assertRaises(FrameTimeoutError):
            await self.manager.pull_video_chunk(self.session_id, timeout_seconds=0.001)
        with self.assertRaises(ValueError):
            await self.manager.push_audio(
                self.session_id, AudioChunk(payload=b"bad", sequence=0, pts_ms=0)
            )

        backend = MockBackend()
        await backend.close()
        self.assertEqual(backend.status().state, SessionState.CLOSED)
        with self.assertRaises(InvalidStateError):
            await backend.push_audio(audio(0, self.config))


if __name__ == "__main__":
    unittest.main()
