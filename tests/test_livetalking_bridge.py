import asyncio
import struct
import unittest

import numpy as np

from h3_interactive import AudioChunk, ControlUpdate, FrameTimeoutError, SessionConfig
from h3_interactive.backends import LegacyVideoFrame, LiveTalkingBridgeBackend


class FakeLegacyRuntime:
    def __init__(self) -> None:
        self.audio = []
        self.text = []
        self.custom_states = []
        self.video = asyncio.Queue()
        self.flush_count = 0
        self.drain_count = 0
        self.closed = False

    def put_audio_frame(self, samples, datainfo):
        self.audio.append((samples.copy(), dict(datainfo)))

    def put_msg_txt(self, text, datainfo):
        self.text.append((text, dict(datainfo)))

    def set_custom_state(self, audiotype, reinit=True):
        self.custom_states.append((audiotype, reinit))

    async def pull_video_frame(self, timeout_seconds):
        if timeout_seconds is None:
            return await self.video.get()
        return await asyncio.wait_for(self.video.get(), timeout_seconds)

    def flush_talk(self):
        self.flush_count += 1
        self.audio.clear()

    def drain_video(self):
        self.drain_count += 1
        while not self.video.empty():
            self.video.get_nowait()

    def is_speaking(self):
        return bool(self.audio)

    def pending_audio_chunks(self):
        return len(self.audio)

    def pending_video_chunks(self):
        return self.video.qsize()

    def close(self):
        self.closed = True


class LiveTalkingBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FakeLegacyRuntime()
        self.config = SessionConfig()
        self.bridge = LiveTalkingBridgeBackend(lambda _config: self.runtime)
        await self.bridge.start(self.config)

    async def asyncTearDown(self):
        await self.bridge.close()

    async def test_audio_conversion_control_and_video(self):
        values = [-32768, 0, 32767] + [0] * (self.config.samples_per_audio_chunk - 3)
        payload = struct.pack(f"<{len(values)}h", *values)
        await self.bridge.push_audio(AudioChunk(payload=payload, sequence=0, pts_ms=0))

        samples, metadata = self.runtime.audio[0]
        np.testing.assert_allclose(samples[:3], [-1.0, 0.0, 32767 / 32768])
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(metadata, {"sequence": 0, "pts_ms": 0, "end_of_stream": False, "epoch": 0})

        await self.bridge.update_control(
            ControlUpdate(kind="text", payload={"text": "hello"}, revision=1)
        )
        await self.bridge.update_control(
            ControlUpdate(
                kind="custom_state", payload={"audiotype": 2, "reinit": False}, revision=2
            )
        )
        self.assertEqual(self.runtime.text, [("hello", {"revision": 1, "epoch": 0})])
        self.assertEqual(self.runtime.custom_states, [(2, False)])

        await self.runtime.video.put(LegacyVideoFrame(payload=b"frame", pts_ms=0))
        frame = await self.bridge.pull_video_chunk(timeout_seconds=0.1)
        self.assertEqual((frame.payload, frame.sequence, frame.pts_ms), (b"frame", 0, 0))
        self.assertEqual(frame.control_revision, 2)
        self.assertTrue(self.bridge.status().metadata["speaking"])

    async def test_reset_clears_legacy_queues_and_restarts_epoch(self):
        await self.bridge.push_audio(
            AudioChunk(payload=b"\0" * self.config.bytes_per_audio_chunk, sequence=0, pts_ms=0)
        )
        await self.runtime.video.put(LegacyVideoFrame(payload=b"stale", pts_ms=0))
        await self.bridge.reset()

        status = self.bridge.status()
        self.assertEqual((status.epoch, status.pending_audio_chunks, status.pending_video_chunks), (1, 0, 0))
        self.assertEqual((self.runtime.flush_count, self.runtime.drain_count), (1, 1))
        await self.bridge.push_audio(
            AudioChunk(payload=b"\0" * self.config.bytes_per_audio_chunk, sequence=0, pts_ms=0)
        )
        self.assertEqual(self.runtime.audio[0][1]["epoch"], 1)

    async def test_timeout_validation_and_idempotent_close(self):
        with self.assertRaises(FrameTimeoutError):
            await self.bridge.pull_video_chunk(timeout_seconds=0.001)
        with self.assertRaises(ValueError):
            await self.bridge.update_control(
                ControlUpdate(kind="unsupported", payload={}, revision=1)
            )
        await self.bridge.close()
        await self.bridge.close()
        self.assertTrue(self.runtime.closed)
        self.assertEqual((self.runtime.flush_count, self.runtime.drain_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
