"""Concrete SoulX-FlashHead GPU runtime implementing the FlashHeadRuntime protocol.

The upstream pipeline (``third_party/SoulX-FlashHead``) is chunk-stateful:
``pipeline.generate()`` carries motion latents (``latent_motion_frames``)
between calls, and the streaming path requires a rolling 8 s audio cache.
All GPU work therefore lives on a single worker thread; the event loop only
ever touches CPU-side queues via ``push_audio`` / ``pull_video_frame``.

Layout constants mirror upstream ``flash_head/configs/infer_params.yaml``
(frame_num=33, tgt_fps=25, sample_rate=16000, cached_audio_duration=8) plus
the per-model VAE stride (LTX=8 for "lite", Wan=4 for "pro"), which fixes
motion_frames_num = (motion_frames_latent_num - 1) * stride + 1.

Known behavior carried over from the upstream streaming path:

- The first ~8 s of frames are generated from a mostly-silent audio cache
  (the cache is pre-filled with zeros, as in ``generate_video.py``).
- Frames arrive in bursts of one chunk (24 frames for lite, 28 for pro);
  smooth 40 ms playback is the consumer's job (jitter buffer).
"""

from __future__ import annotations

import asyncio
import io
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Mapping

import numpy as np

from ..protocol import SessionConfig
from .flashhead import FlashHeadFrame

# Upstream constants (flash_head/configs/infer_params.yaml)
FRAME_NUM = 33
TGT_FPS = 25
SAMPLE_RATE = 16_000
MOTION_FRAMES_LATENT_NUM = 2
VAE_STRIDE = {"lite": 8, "pro": 4}  # lite uses the LTX VAE, pro uses the Wan VAE


def chunk_layout(model_type: str) -> tuple[int, int]:
    """Return (output_frames_per_chunk, audio_samples_per_chunk).

    Mirrors the upstream calculation: motion_frames_num =
    (motion_frames_latent_num - 1) * vae_stride + 1, and each chunk carries
    slice_len = frame_num - motion_frames_num frames' worth of audio.
    """
    if model_type not in VAE_STRIDE:
        raise ValueError(f"unsupported FlashHead model type: {model_type!r}")
    motion_frames = (MOTION_FRAMES_LATENT_NUM - 1) * VAE_STRIDE[model_type] + 1
    slice_len = FRAME_NUM - motion_frames
    return slice_len, slice_len * SAMPLE_RATE // TGT_FPS


class SoulXFlashHeadRuntime:
    """Streaming FlashHead runtime bound to one SessionConfig."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        ckpt_dir: str,
        wav2vec_dir: str,
        repo_dir: str,
        cond_image: str,
        model_type: str = "lite",
        seed: int = 42,
        use_face_crop: bool = False,
    ) -> None:
        for name, value in (
            ("ckpt_dir", ckpt_dir),
            ("wav2vec_dir", wav2vec_dir),
            ("repo_dir", repo_dir),
            ("cond_image", cond_image),
        ):
            if not value or not os.path.exists(value):
                raise ValueError(f"FlashHead {name} does not exist: {value!r}")
        self._config = config
        self._ckpt_dir = ckpt_dir
        self._wav2vec_dir = wav2vec_dir
        self._repo_dir = repo_dir
        self._cond_image = cond_image
        self._model_type = model_type
        self._seed = seed
        self._use_face_crop = use_face_crop

        self._slice_len, self._chunk_samples = chunk_layout(model_type)
        self._frame_ms = config.video_frame_ms

        self._partial: list[np.ndarray] = []
        self._partial_samples = 0
        self._worker_q: queue.Queue = queue.Queue()
        self._frames_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._closed = False

    # -- FlashHeadRuntime protocol -------------------------------------

    def push_audio(self, samples: np.ndarray, metadata: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("FlashHead runtime is closed")
        self._ensure_worker()
        self._partial.append(np.asarray(samples, dtype=np.float32))
        self._partial_samples += len(samples)
        while self._partial_samples >= self._chunk_samples:
            self._worker_q.put(("audio", self._take_chunk()))
        if metadata.get("end_of_stream") and self._partial_samples > 0:
            # Pad the tail with silence like upstream generate_video.py so the
            # last partial chunk is not truncated.
            pad = np.zeros(self._chunk_samples - self._partial_samples, dtype=np.float32)
            self._worker_q.put(("audio", self._take_chunk(pad)))

    async def pull_video_frame(self, timeout_seconds: float | None) -> FlashHeadFrame:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            if self._worker_error is not None:
                raise RuntimeError("FlashHead worker crashed") from self._worker_error
            try:
                return self._frames_q.get_nowait()
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise asyncio.TimeoutError
                await asyncio.sleep(0.005)

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError("FlashHead runtime is closed")
        self._drain(self._worker_q)
        self._partial = []
        self._partial_samples = 0
        if self._worker is None:
            return  # never started; the first session begins from a clean state
        done = threading.Event()
        self._worker_q.put(("reset", done))
        if not done.wait(timeout=60.0):
            raise RuntimeError("FlashHead worker did not process reset within 60 s")
        # Drain only after the worker finished resetting: an in-flight
        # generation may have pushed old-epoch frames while reset was queued.
        self._drain(self._frames_q)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker_q.put(("close", None))
            self._worker.join(timeout=120.0)
        self._drain(self._worker_q)
        self._drain(self._frames_q)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def pending_audio_chunks(self) -> int:
        return self._worker_q.qsize()

    def pending_video_frames(self) -> int:
        return self._frames_q.qsize()

    # -- internals -----------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="flashhead-worker", daemon=True
            )
            self._worker.start()

    def _take_chunk(self, tail: np.ndarray | None = None) -> np.ndarray:
        parts = list(self._partial)
        if tail is not None:
            parts.append(tail)
        data = np.concatenate(parts) if len(parts) > 1 else parts[0]
        chunk, rest = data[: self._chunk_samples], data[self._chunk_samples :]
        self._partial = [rest] if len(rest) else []
        self._partial_samples = len(rest)
        return chunk

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _load_upstream(self):
        """Import upstream inference with its expected cwd and load the GPU pipeline."""
        if self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)
        # Upstream reads flash_head/configs/infer_params.yaml relative to cwd,
        # so the worker must run with the repo root as cwd (process-global).
        os.chdir(self._repo_dir)
        from flash_head.inference import (  # noqa: PLC0415
            get_audio_embedding,
            get_base_data,
            get_infer_params,
            get_pipeline,
            run_pipeline,
        )

        pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=self._ckpt_dir,
            model_type=self._model_type,
            wav2vec_dir=self._wav2vec_dir,
        )
        params = get_infer_params()
        expected_slice = params["frame_num"] - params["motion_frames_num"]
        if expected_slice != self._slice_len:
            raise RuntimeError(
                f"pipeline layout mismatch for {self._model_type}: "
                f"expected {self._slice_len} output frames/chunk, got {expected_slice}"
            )
        get_base_data(
            pipeline,
            cond_image_path_or_dir=self._cond_image,
            base_seed=self._seed,
            use_face_crop=self._use_face_crop,
        )
        return pipeline, params, get_audio_embedding, run_pipeline

    def _worker_loop(self) -> None:
        try:
            pipeline, params, get_audio_embedding, run_pipeline = self._load_upstream()
        except BaseException as error:
            self._worker_error = error
            return

        cached_samples = params["cached_audio_duration"] * params["sample_rate"]
        motion_frames_num = params["motion_frames_num"]
        audio_end_idx = params["cached_audio_duration"] * params["tgt_fps"]
        audio_start_idx = audio_end_idx - params["frame_num"]
        audio_window = deque([0.0] * cached_samples, maxlen=cached_samples)
        chunk_idx = 0
        try:
            while True:
                item = self._worker_q.get()
                kind = item[0]
                if kind == "close":
                    return
                if kind == "reset":
                    pipeline.reset_person_name()
                    audio_window.clear()
                    audio_window.extend([0.0] * cached_samples)
                    chunk_idx = 0
                    item[1].set()
                    continue

                samples = item[1]
                audio_window.extend(samples.tolist())
                audio_array = np.array(audio_window, dtype=np.float32)
                audio_embedding = get_audio_embedding(
                    pipeline, audio_array, audio_start_idx, audio_end_idx
                )
                video = run_pipeline(pipeline, audio_embedding)
                video = video[motion_frames_num:]  # drop the motion carryover frames
                frames_np = video.cpu().numpy().astype(np.uint8)  # (slice_len, H, W, C)
                base_pts = chunk_idx * self._slice_len * self._frame_ms
                for offset in range(frames_np.shape[0]):
                    self._frames_q.put(
                        FlashHeadFrame(
                            payload=self._encode_jpeg(frames_np[offset]),
                            pts_ms=base_pts + offset * self._frame_ms,
                            duration_ms=self._frame_ms,
                        )
                    )
                chunk_idx += 1
        except BaseException as error:
            self._worker_error = error

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes:
        from PIL import Image  # noqa: PLC0415

        buffer = io.BytesIO()
        Image.fromarray(frame, mode="RGB").save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()


def make_runtime_factory(
    ckpt_dir: str,
    wav2vec_dir: str,
    repo_dir: str,
    cond_image: str,
    model_type: str = "lite",
    seed: int = 42,
    use_face_crop: bool = False,
) -> Callable[[SessionConfig], SoulXFlashHeadRuntime]:
    """Return a FlashHeadBackend runtime factory bound to fixed model assets."""

    def factory(config: SessionConfig) -> SoulXFlashHeadRuntime:
        return SoulXFlashHeadRuntime(
            config,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            repo_dir=repo_dir,
            cond_image=cond_image,
            model_type=model_type,
            seed=seed,
            use_face_crop=use_face_crop,
        )

    return factory


def runtime_factory_from_env(environ: Mapping[str, str] | None = None) -> Callable | None:
    """Build a factory from FLASHHEAD_* env vars, or return None if unset."""
    env = os.environ if environ is None else environ
    required = (
        "FLASHHEAD_CKPT_DIR",
        "FLASHHEAD_WAV2VEC_DIR",
        "FLASHHEAD_REPO_DIR",
        "FLASHHEAD_COND_IMAGE",
    )
    if not all(env.get(key) for key in required):
        return None
    return make_runtime_factory(
        ckpt_dir=env["FLASHHEAD_CKPT_DIR"],
        wav2vec_dir=env["FLASHHEAD_WAV2VEC_DIR"],
        repo_dir=env["FLASHHEAD_REPO_DIR"],
        cond_image=env["FLASHHEAD_COND_IMAGE"],
        model_type=env.get("FLASHHEAD_MODEL_TYPE", "lite"),
        seed=int(env.get("FLASHHEAD_SEED", "42")),
        use_face_crop=env.get("FLASHHEAD_USE_FACE_CROP", "0") == "1",
    )
