"""Prepare generated VBVR videos for rule evaluation.

The command recursively mirrors ``input_dir`` into ``output_dir``. Each video
is resized without cropping, padded to the requested canvas, and retimed by
raising its frame rate while preserving every source frame.

Example:
    .venv/bin/python -m src.cli.prepare_vbvr_eval_videos \
        --input-dir storage/eval_out/vbvr_pro/generated_256 \
        --output-dir storage/eval_out/vbvr_pro/eval_1024 \
        --width 1024 --height 1024 --max-duration 5 \
        --workers 16 --expected-videos 750 --crf 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class VideoPreparationError(RuntimeError):
    """Raised when one or more videos cannot be prepared or validated."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    average_fps: Fraction
    nominal_fps: Fraction
    duration: float

    @property
    def source_fps(self) -> Fraction:
        return self.average_fps if self.average_fps > 0 else self.nominal_fps


@dataclass(frozen=True)
class ProcessResult:
    relative_path: Path
    status: str
    frame_count: int
    output_fps: Fraction
    duration: float


@dataclass(frozen=True)
class PreparationSummary:
    discovered: int
    processed: int
    skipped: int
    outputs: tuple[Path, ...]


def _parse_fraction(value: object) -> Fraction:
    if value in (None, "", "N/A", "0/0"):
        return Fraction(0, 1)
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoPreparationError(f"Invalid frame rate from ffprobe: {value!r}") from exc
    return result if result > 0 else Fraction(0, 1)


def _parse_duration(stream: dict, payload: dict, frame_count: int, fps: Fraction) -> float:
    for value in (stream.get("duration"), payload.get("format", {}).get("duration")):
        if value not in (None, "", "N/A"):
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                return duration
    if fps > 0:
        return float(Fraction(frame_count, 1) / fps)
    raise VideoPreparationError("ffprobe did not report a usable video duration")


def probe_video(path: Path, *, ffprobe: str = "ffprobe") -> VideoInfo:
    """Read dimensions, exact decoded-frame count, frame rate, and duration."""
    ffprobe = _require_executable(ffprobe)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffprobe error"
        raise VideoPreparationError(f"ffprobe failed for {path}: {detail}")

    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
        frame_count = int(frame_value)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoPreparationError(f"Incomplete ffprobe metadata for {path}") from exc

    if width <= 0 or height <= 0 or frame_count <= 0:
        raise VideoPreparationError(f"Invalid video metadata for {path}: size={width}x{height}, frames={frame_count}")
    average_fps = _parse_fraction(stream.get("avg_frame_rate"))
    nominal_fps = _parse_fraction(stream.get("r_frame_rate"))
    source_fps = average_fps if average_fps > 0 else nominal_fps
    if source_fps <= 0:
        raise VideoPreparationError(f"ffprobe did not report a usable frame rate for {path}")

    return VideoInfo(
        width=width,
        height=height,
        frame_count=frame_count,
        average_fps=average_fps,
        nominal_fps=nominal_fps,
        duration=_parse_duration(stream, payload, frame_count, source_fps),
    )


def compute_output_fps(source_fps: Fraction, frame_count: int, max_duration: float | Fraction) -> Fraction:
    """Return ``max(source_fps, ceil(frame_count / max_duration))``."""
    duration = max_duration if isinstance(max_duration, Fraction) else Fraction(str(max_duration))
    if source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")
    required_fps = Fraction(math.ceil(Fraction(frame_count, 1) / duration), 1)
    return max(source_fps, required_fps)


def _validate_output(
    path: Path,
    *,
    source: VideoInfo,
    width: int,
    height: int,
    output_fps: Fraction,
    max_duration: float,
    ffprobe: str,
) -> tuple[bool, str, VideoInfo | None]:
    try:
        info = probe_video(path, ffprobe=ffprobe)
    except VideoPreparationError as exc:
        return False, str(exc), None

    errors: list[str] = []
    if (info.width, info.height) != (width, height):
        errors.append(f"size={info.width}x{info.height}, expected={width}x{height}")
    if info.frame_count != source.frame_count:
        errors.append(f"frames={info.frame_count}, expected={source.frame_count}")
    if info.nominal_fps != output_fps or info.average_fps != output_fps:
        errors.append(f"fps={info.average_fps} (nominal {info.nominal_fps}), expected={output_fps}")
    if info.duration > max_duration + 1e-3:
        errors.append(f"duration={info.duration:.6f}s, limit={max_duration:.6f}s")
    return not errors, "; ".join(errors), info


def _ffmpeg_filter(width: int, height: int, output_fps: Fraction) -> str:
    fps = f"{output_fps.numerator}/{output_fps.denominator}"
    time_base = f"{output_fps.denominator}/{output_fps.numerator}"
    return ",".join(
        [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
            f"settb=expr={time_base}",
            "setpts=N",
            f"fps=fps={fps}:eof_action=pass",
        ]
    )


def _prepare_one(
    source_path: Path,
    output_path: Path,
    relative_path: Path,
    *,
    width: int,
    height: int,
    max_duration: float,
    crf: int,
    ffmpeg: str,
    ffprobe: str,
    force: bool,
) -> ProcessResult:
    source = probe_video(source_path, ffprobe=ffprobe)
    output_fps = compute_output_fps(source.source_fps, source.frame_count, max_duration)
    if Fraction(source.frame_count, 1) / output_fps > Fraction(str(max_duration)):
        raise AssertionError("computed output frame rate does not satisfy max_duration")

    if output_path.is_file() and not force and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        valid, _, existing = _validate_output(
            output_path,
            source=source,
            width=width,
            height=height,
            output_fps=output_fps,
            max_duration=max_duration,
            ffprobe=ffprobe,
        )
        if valid and existing is not None:
            return ProcessResult(relative_path, "skipped", existing.frame_count, output_fps, existing.duration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-vf",
        _ffmpeg_filter(width, height, output_fps),
        "-an",
        "-map_metadata",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        f"{output_fps.denominator}:{output_fps.numerator}",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(temp_path),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
            raise VideoPreparationError(f"ffmpeg failed for {relative_path}: {detail[-4000:]}")
        valid, reason, prepared = _validate_output(
            temp_path,
            source=source,
            width=width,
            height=height,
            output_fps=output_fps,
            max_duration=max_duration,
            ffprobe=ffprobe,
        )
        if not valid or prepared is None:
            raise VideoPreparationError(f"Prepared video failed validation for {relative_path}: {reason}")
        os.replace(temp_path, output_path)
        return ProcessResult(relative_path, "processed", prepared.frame_count, output_fps, prepared.duration)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare_video(
    source_path: Path,
    output_path: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    max_duration: float = 5.0,
    crf: int = 12,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    force: bool = True,
) -> ProcessResult:
    """Prepare one video with the exact contract used by VBVR-Pro evaluation.

    Training rewards call this entry point so their generated videos go through
    the same scale/pad/retime/encode/validation path as the final batch
    evaluation. Every source frame is retained.
    """
    source = source_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("source_path and output_path must be different files")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"width and height must be positive even integers, got {width}x{height}")
    if max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")
    if not 0 <= crf <= 51:
        raise ValueError(f"crf must be in [0, 51], got {crf}")
    ffmpeg = _require_executable(ffmpeg)
    ffprobe = _require_executable(ffprobe)
    return _prepare_one(
        source,
        output,
        Path(source.name),
        width=width,
        height=height,
        max_duration=max_duration,
        crf=crf,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        force=force,
    )


def _discover_mp4s(root: Path, *, exclude: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        if exclude is not None and path.resolve().is_relative_to(exclude):
            continue
        paths.append(path)
    return sorted(paths)


def _require_executable(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is not None:
        return resolved

    if command in {"ffmpeg", "ffprobe"}:
        try:
            import ffmpeg_binaries

            bundled = ffmpeg_binaries.FFMPEG_PATH if command == "ffmpeg" else ffmpeg_binaries.FFPROBE_PATH
            if bundled is not None and bundled.is_file():
                return str(bundled)
        except ImportError:
            pass

    raise FileNotFoundError(f"Required executable not found: {command}")


def prepare_videos(
    input_dir: Path,
    output_dir: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    max_duration: float = 5.0,
    crf: int = 12,
    workers: int = 8,
    expected_videos: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    force: bool = False,
    progress: Callable[[ProcessResult], None] | None = None,
) -> PreparationSummary:
    input_root = input_dir.expanduser().resolve()
    output_root = output_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("input_dir and output_dir must be different directories")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"width and height must be positive even integers, got {width}x{height}")
    if max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")
    if not 0 <= crf <= 51:
        raise ValueError(f"crf must be in [0, 51], got {crf}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if expected_videos is not None and expected_videos < 0:
        raise ValueError(f"expected_videos must be non-negative, got {expected_videos}")
    ffmpeg = _require_executable(ffmpeg)
    ffprobe = _require_executable(ffprobe)

    exclude = output_root if output_root.is_relative_to(input_root) else None
    source_paths = _discover_mp4s(input_root, exclude=exclude)
    if expected_videos is not None and len(source_paths) != expected_videos:
        raise VideoPreparationError(
            f"Discovered {len(source_paths)} input videos, expected exactly {expected_videos}: {input_root}"
        )
    if not source_paths and expected_videos != 0:
        raise VideoPreparationError(f"No MP4 videos found under {input_root}")

    relative_paths = [path.relative_to(input_root) for path in source_paths]
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[ProcessResult] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vbvr-video") as executor:
        futures = {
            executor.submit(
                _prepare_one,
                source_path,
                output_root / relative_path,
                relative_path,
                width=width,
                height=height,
                max_duration=max_duration,
                crf=crf,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                force=force,
            ): relative_path
            for source_path, relative_path in zip(source_paths, relative_paths, strict=True)
        }
        for future in as_completed(futures):
            relative_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{relative_path}: {exc}")
                continue
            results.append(result)
            if progress is not None:
                progress(result)

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise VideoPreparationError(f"Failed to prepare {len(errors)} video(s):\n{preview}{more}")

    expected_relatives = set(relative_paths)
    actual_relatives = {path.relative_to(output_root) for path in _discover_mp4s(output_root)}
    if actual_relatives != expected_relatives:
        missing = sorted(str(path) for path in expected_relatives - actual_relatives)
        extra = sorted(str(path) for path in actual_relatives - expected_relatives)
        raise VideoPreparationError(
            f"Output video set does not exactly match input video set: missing={missing[:10]}, extra={extra[:10]}"
        )
    if expected_videos is not None and len(actual_relatives) != expected_videos:
        raise VideoPreparationError(
            f"Prepared {len(actual_relatives)} output videos, expected exactly {expected_videos}: {output_root}"
        )

    return PreparationSummary(
        discovered=len(source_paths),
        processed=sum(result.status == "processed" for result in results),
        skipped=sum(result.status == "skipped" for result in results),
        outputs=tuple(sorted(output_root / path for path in actual_relatives)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-duration", type=float, default=5.0, help="Maximum output duration in seconds")
    parser.add_argument("--crf", type=int, default=12, help="libx264 CRF; lower preserves more scorer-relevant detail")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--expected-videos",
        type=int,
        default=None,
        help="Require exactly this many input and output MP4s",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even when their validation passes")
    return parser.parse_args(argv)


def _print_progress(result: ProcessResult) -> None:
    print(
        f"[{result.status}] {result.relative_path} frames={result.frame_count} "
        f"fps={float(result.output_fps):.6g} duration={result.duration:.3f}s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_videos(
            args.input_dir,
            args.output_dir,
            width=args.width,
            height=args.height,
            max_duration=args.max_duration,
            crf=args.crf,
            workers=args.workers,
            expected_videos=args.expected_videos,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            force=args.force,
            progress=_print_progress,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"Done. discovered={summary.discovered} processed={summary.processed} "
        f"skipped={summary.skipped} output_dir={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
