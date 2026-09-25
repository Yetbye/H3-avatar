import asyncio
import struct
import unittest

import numpy as np

from h3_interactive import AudioChunk, ControlUpdate, FrameTimeoutError, SessionConfig
from h3_interactive.backends import FlashHeadBackend, FlashHeadFrame


class FakeFlashHeadRuntime:
    def __init__(self) -> None:
        self.audio = []
        self.video = asyncio.Queue()
        self.reset_count = 0
        self.closed = False

    def push_audio(self, samples, metadata):
        self.audio.append((samples.copy(), dict(metadata)))

    async def pull_video_frame(self, timeout_seconds):
        if timeout_seconds is None:
            return await self.video.get()
        return await asyncio.wait_for(self.video.get(), timeout_seconds)

    def reset(self):
        self.reset_count += 1
        self.audio.clear()
        while not self.video.empty():
            self.video.get_nowait()

    def close(self):
        self.closed = True

    def pending_audio_chunks(self):
        return len(self.audio)

    def pending_video_frames(self):
        return self.video.qsize()


class FlashHeadBackendTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = FakeFlashHeadRuntime()
        self.config = SessionConfig()
        self.backend = FlashHeadBackend(lambda _config: self.runtime)
        await self.backend.start(self.config)

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_audio_frame_and_status_mapping(self):
        values = [-32768, 0, 32767] + [0] * (self.config.samples_per_audio_chunk - 3)
        payload = struct.pack(f"<{len(values)}h", *values)
        await self.backend.push_audio(AudioChunk(payload=payload, sequence=0, pts_ms=0))

        samples, metadata = self.runtime.audio[0]
        np.testing.assert_allclose(samples[:3], [-1.0, 0.0, 32767 / 32768])
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(metadata, {"sequence": 0, "pts_ms": 0, "end_of_stream": False, "epoch": 0})

        await self.runtime.video.put(FlashHeadFrame(payload=b"jpeg", pts_ms=0))
        frame = await self.backend.pull_video_chunk(timeout_seconds=0.1)
        self.assertEqual((frame.payload, frame.sequence, frame.pts_ms), (b"jpeg", 0, 0))
        self.assertEqual(self.backend.status().metadata["backend"], "flashhead")

    async def test_reset_timeout_control_and_close(self):
        await self.backend.push_audio(
            AudioChunk(payload=b"\0" * self.config.bytes_per_audio_chunk, sequence=0, pts_ms=0)
        )
        await self.runtime.video.put(FlashHeadFrame(payload=b"stale", pts_ms=0))
        await self.backend.reset()
        self.assertEqual(self.backend.status().epoch, 1)
        self.assertEqual(self.runtime.reset_count, 1)

        with self.assertRaises(FrameTimeoutError):
            await self.backend.pull_video_chunk(timeout_seconds=0.001)
        with self.assertRaisesRegex(ValueError, "not mapped"):
            await self.backend.update_control(
                ControlUpdate(kind="prompt", payload={"text": "wave"}, revision=1)
            )

        await self.backend.close()
        await self.backend.close()
        self.assertTrue(self.runtime.closed)


if __name__ == "__main__":
    unittest.main()
