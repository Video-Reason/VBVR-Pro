"""Create a tiny local raw-I2V dataset for end-to-end training smoke tests.

The generated dataset follows the same Parquet contract as production raw
training data, but uses deterministic synthetic MP4s and first-frame images.

Example:
    .venv/bin/python scripts/dev/create_i2v_smoke_dataset.py \
        --output-dir storage/smoke/i2v_512x512x81 \
        --samples 4 --frames 81 --height 512 --width 512 --fps 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diffusers.utils import export_to_video
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("samples", "frames", "height", "width", "fps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16")


def _make_video(sample_index: int, *, frames: int, height: int, width: int) -> list[np.ndarray]:
    x_gradient = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y_gradient = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[..., 0] = (x_gradient.astype(np.uint16) + sample_index * 37) % 256
    background[..., 1] = (y_gradient.astype(np.uint16) + sample_index * 59) % 256
    background[..., 2] = 48 + sample_index * 31

    square_size = max(16, min(height, width) // 8)
    max_x = max(1, width - square_size)
    max_y = max(1, height - square_size)
    video: list[np.ndarray] = []
    for frame_index in range(frames):
        frame = background.copy()
        progress = frame_index / max(1, frames - 1)
        x0 = int(progress * max_x)
        y0 = int(((sample_index % 2) * (1.0 - progress) + ((sample_index + 1) % 2) * progress) * max_y)
        color = (
            255,
            int((sample_index * 67 + frame_index * 3) % 256),
            int((192 + sample_index * 29) % 256),
        )
        frame[y0 : y0 + square_size, x0 : x0 + square_size] = color
        video.append(frame)
    return video


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    media_dir = output_dir / "media"
    parquet_path = output_dir / "samples.parquet"
    dataset_path = output_dir / "dataset.json"

    planned = [parquet_path, dataset_path]
    planned.extend(
        media_dir / f"sample-{index:03d}{suffix}" for index in range(args.samples) for suffix in (".mp4", ".png")
    )
    existing = [path for path in planned if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"{existing[0]} already exists; pass --force to replace generated smoke files")

    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for sample_index in range(args.samples):
        frames = _make_video(
            sample_index,
            frames=args.frames,
            height=args.height,
            width=args.width,
        )
        video_name = f"sample-{sample_index:03d}.mp4"
        image_name = f"sample-{sample_index:03d}.png"
        export_to_video(frames, str(media_dir / video_name), fps=args.fps)
        Image.fromarray(frames[0]).save(media_dir / image_name)
        rows.append(
            {
                "video": f"media/{video_name}",
                "image": f"media/{image_name}",
                "prompt": (
                    f"A brightly colored square moves smoothly across a synthetic gradient scene, "
                    f"variation {sample_index}."
                ),
                "sample_id": sample_index,
            }
        )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path, compression="zstd")
    descriptor = [
        {
            "data_path": "samples.parquet",
            "root": ".",
            "num_frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
        }
    ]
    dataset_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "samples": args.samples,
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
                "fps": args.fps,
                "dataset_json": str(dataset_path),
                "parquet": str(parquet_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
