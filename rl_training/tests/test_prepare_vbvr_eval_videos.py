import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from src.cli.prepare_vbvr_eval_videos import (
    VideoPreparationError,
    compute_output_fps,
    prepare_video,
    prepare_videos,
    probe_video,
)

try:
    import ffmpeg_binaries
except ImportError:
    ffmpeg_binaries = None


def _bundled_executable(name: str) -> str | None:
    if ffmpeg_binaries is None:
        return None
    path = getattr(ffmpeg_binaries, name)
    return str(path) if path is not None else None


FFMPEG = shutil.which("ffmpeg") or _bundled_executable("FFMPEG_PATH")
FFPROBE = shutil.which("ffprobe") or _bundled_executable("FFPROBE_PATH")


def _make_video(path: Path, *, frames: int = 161, fps: int = 16, size: str = "80x40") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={size}:r={fps}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _read_first_rgb_frame(path: Path, width: int, height: int) -> bytes:
    completed = subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    expected_bytes = width * height * 3
    if len(completed.stdout) != expected_bytes:
        raise AssertionError(f"decoded {len(completed.stdout)} bytes, expected {expected_bytes}")
    return completed.stdout


def _pixel(frame: bytes, width: int, x: int, y: int) -> tuple[int, int, int]:
    offset = (y * width + x) * 3
    return tuple(frame[offset : offset + 3])


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
class TestPrepareVBVREvalVideos(unittest.TestCase):
    def test_compute_output_fps(self):
        self.assertEqual(compute_output_fps(Fraction(16, 1), 161, 5.0), Fraction(33, 1))
        self.assertEqual(compute_output_fps(Fraction(60, 1), 161, 5.0), Fraction(60, 1))
        self.assertEqual(compute_output_fps(Fraction(16, 1), 165, 5.0), Fraction(33, 1))

    def test_prepare_one_video_uses_the_batch_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.mp4"
            output = root / "prepared.mp4"
            _make_video(source, frames=17, fps=8, size="80x40")

            result = prepare_video(
                source,
                output,
                width=64,
                height=64,
                max_duration=2.0,
            )
            info = probe_video(output)

            self.assertEqual(result.status, "processed")
            self.assertEqual((info.width, info.height), (64, 64))
            self.assertEqual(info.frame_count, 17)
            self.assertEqual(info.average_fps, Fraction(9, 1))
            self.assertLessEqual(info.duration, 2.0)

    def test_recursive_resize_pad_retime_and_valid_skip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            source = input_dir / "split" / "task" / "sample.mp4"
            _make_video(source)

            first = prepare_videos(
                input_dir,
                output_dir,
                width=96,
                height=96,
                max_duration=5.0,
                workers=2,
                expected_videos=1,
            )
            output = output_dir / "split" / "task" / "sample.mp4"
            info = probe_video(output)

            self.assertEqual(first.processed, 1)
            self.assertEqual(first.skipped, 0)
            self.assertEqual(first.outputs, (output,))
            self.assertEqual((info.width, info.height), (96, 96))
            self.assertEqual(info.frame_count, 161)
            self.assertEqual(info.average_fps, Fraction(33, 1))
            self.assertEqual(info.nominal_fps, Fraction(33, 1))
            self.assertLessEqual(info.duration, 5.0)

            frame = _read_first_rgb_frame(output, 96, 96)
            top = _pixel(frame, 96, 48, 4)
            center = _pixel(frame, 96, 48, 48)
            self.assertLess(max(top), 24)
            self.assertGreater(center[0], 180)
            self.assertLess(center[1], 60)
            self.assertLess(center[2], 60)

            output_mtime = output.stat().st_mtime_ns
            second = prepare_videos(
                input_dir,
                output_dir,
                width=96,
                height=96,
                max_duration=5.0,
                workers=1,
                expected_videos=1,
            )
            self.assertEqual(second.processed, 0)
            self.assertEqual(second.skipped, 1)
            self.assertEqual(output.stat().st_mtime_ns, output_mtime)

    def test_invalid_existing_output_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            source = input_dir / "sample.mp4"
            output = output_dir / "sample.mp4"
            _make_video(source, frames=17, fps=8)
            prepare_videos(input_dir, output_dir, width=64, height=64, max_duration=2.0, expected_videos=1)

            output.write_bytes(b"not an mp4")
            rebuilt = prepare_videos(
                input_dir,
                output_dir,
                width=64,
                height=64,
                max_duration=2.0,
                expected_videos=1,
            )

            self.assertEqual(rebuilt.processed, 1)
            self.assertEqual(probe_video(output).frame_count, 17)

    def test_expected_count_and_output_set_are_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            _make_video(input_dir / "a.mp4", frames=9, fps=10)
            _make_video(input_dir / "nested" / "b.mp4", frames=9, fps=10)

            with self.assertRaisesRegex(VideoPreparationError, "expected exactly 1"):
                prepare_videos(input_dir, output_dir, width=64, height=64, expected_videos=1)
            self.assertFalse(output_dir.exists())

            summary = prepare_videos(
                input_dir,
                output_dir,
                width=64,
                height=64,
                workers=2,
                expected_videos=2,
            )
            self.assertEqual(summary.processed, 2)
            shutil.copyfile(output_dir / "a.mp4", output_dir / "stale.mp4")

            with self.assertRaisesRegex(VideoPreparationError, "does not exactly match"):
                prepare_videos(
                    input_dir,
                    output_dir,
                    width=64,
                    height=64,
                    workers=2,
                    expected_videos=2,
                )


if __name__ == "__main__":
    unittest.main()
