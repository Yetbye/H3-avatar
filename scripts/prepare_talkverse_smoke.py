#!/usr/bin/env python3
"""Prepare one small, project-local TalkVerse smoke-test fixture."""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    if not args.image.is_file() or not args.audio.is_file():
        raise FileNotFoundError("input image or audio is missing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_out = args.output_dir / f"reference{args.image.suffix.lower()}"
    audio_out = args.output_dir / "drive_6s.wav"
    batch_out = args.output_dir / "batch.json"
    shutil.copy2(args.image, image_out)

    with wave.open(str(args.audio), "rb") as source:
        frame_count = min(source.getnframes(), round(args.seconds * source.getframerate()))
        params = source.getparams()
        frames = source.readframes(frame_count)
    with wave.open(str(audio_out), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)

    batch = [
        {
            "name": "smoke_001",
            "image": str(image_out.resolve()),
            "audio": str(audio_out.resolve()),
            "prompt": "A woman speaks naturally to the camera with subtle facial motion.",
        }
    ]
    batch_out.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    print(batch_out.resolve())


if __name__ == "__main__":
    main()
