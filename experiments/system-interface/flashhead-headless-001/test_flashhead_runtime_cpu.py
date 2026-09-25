"""CPU-only verification for the FlashHead runtime plumbing.

Run with CUDA_VISIBLE_DEVICES='' so the worker's pipeline load fails fast
instead of touching a shared GPU. Validates chunk arithmetic, audio-chunk
accumulation with EOS padding, the clean no-GPU failure path, and close
semantics. No model is loaded and no GPU memory is used.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

import numpy as np

from h3_interactive.backends.flashhead_runtime import (
    SoulXFlashHeadRuntime,
    chunk_layout,
    make_runtime_factory,
)
from h3_interactive.protocol import SessionConfig


class ChunkLayoutTest(unittest.TestCase):
    def test_lite_layout(self):
        self.assertEqual(chunk_layout("lite"), (24, 15360))

    def test_pro_layout(self):
        self.assertEqual(chunk_layout("pro"), (28, 17920))

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            chunk_layout("teacher")


class RuntimePlumbingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for name in ("ckpt", "wav2vec", "repo", "cond"):
            os.makedirs(os.path.join(self.tmp.name, name), exist_ok=True)
        self.paths = {
            "ckpt_dir": os.path.join(self.tmp.name, "ckpt"),
            "wav2vec_dir": os.path.join(self.tmp.name, "wav2vec"),
            "repo_dir": os.path.join(self.tmp.name, "repo"),
            "cond_image": os.path.join(self.tmp.name, "cond"),
        }
        self.addCleanup(self.tmp.cleanup)

    def make_runtime(self, model_type="lite", **overrides):
        paths = dict(self.paths)
        paths.update(overrides)
        return SoulXFlashHeadRuntime(SessionConfig(), model_type=model_type, **paths)

    def test_factory_binds_assets(self):
        factory = make_runtime_factory(**self.paths, model_type="pro", seed=7)
        runtime = factory(SessionConfig())
        self.assertEqual(runtime._model_type, "pro")
        self.assertEqual(runtime._seed, 7)

    def test_missing_asset_rejected(self):
        with self.assertRaises(ValueError):
            self.make_runtime(cond_image="/nonexistent/path.png")

    def test_push_chunking_and_eos_padding(self):
        runtime = self.make_runtime()
        runtime._worker = object()  # white-box: keep the worker thread from starting
        chunk = np.arange(15360, dtype=np.float32) / 15360.0
        for i in range(48):  # 48 x 320 samples = exactly one lite generation chunk
            runtime.push_audio(chunk[i * 320:(i + 1) * 320], {"sequence": i, "pts_ms": i * 20})
        self.assertEqual(runtime.pending_audio_chunks(), 1)
        queued = runtime._worker_q.get_nowait()
        self.assertEqual(queued[0], "audio")
        np.testing.assert_array_equal(queued[1], chunk)

        tail = np.ones(100, dtype=np.float32) * 0.5
        runtime.push_audio(tail, {"sequence": 48, "pts_ms": 960, "end_of_stream": True})
        self.assertEqual(runtime.pending_audio_chunks(), 1)
        queued = runtime._worker_q.get_nowait()
        np.testing.assert_array_equal(queued[1][:100], tail)
        np.testing.assert_array_equal(queued[1][100:], np.zeros(15260, dtype=np.float32))

        runtime._worker = None  # nothing was ever started
        runtime.close()
        runtime.close()  # idempotent

    def test_reset_before_start_is_noop(self):
        runtime = self.make_runtime()
        runtime.reset()  # must not raise before any push
        runtime.close()

    def test_worker_fails_cleanly_without_gpu(self):
        runtime = self.make_runtime()
        runtime.push_audio(np.zeros(15360, dtype=np.float32), {"sequence": 0, "pts_ms": 0})
        deadline = time.monotonic() + 180.0
        while runtime._worker_error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(runtime._worker_error, "worker should fail without a GPU")
        with self.assertRaises(RuntimeError):
            asyncio.run(runtime.pull_video_frame(timeout_seconds=30.0))
        runtime.close()
        self.assertEqual(runtime.pending_audio_chunks(), 0)
        self.assertEqual(runtime.pending_video_frames(), 0)


if __name__ == "__main__":
    unittest.main()
