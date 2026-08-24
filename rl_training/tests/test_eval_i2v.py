import json
from pathlib import Path

import pytest

from src.cli import eval_i2v
from src.cli.eval_i2v import (
    _can_resume_video,
    _expected_output_paths,
    _item_num_frames,
    _output_path,
    _temporary_video_path,
    _validate_output_set,
)


class _FakeFrame:
    def __init__(self, height: int, width: int):
        self.shape = (height, width, 3)


class _FakeVideoReader:
    def __init__(self, *, width: int, height: int, frames: int, fps: float):
        self._width = width
        self._height = height
        self._frames = frames
        self._fps = fps

    def __len__(self):
        return self._frames

    def __getitem__(self, index: int):
        if not 0 <= index < self._frames:
            raise IndexError(index)
        return _FakeFrame(self._height, self._width)

    def get_avg_fps(self):
        return self._fps


def _install_fake_readers(monkeypatch: pytest.MonkeyPatch, metadata: dict[str, tuple[int, int, int, float]]):
    def fake_reader(path: str, *, num_threads: int):
        assert num_threads == 1
        width, height, frames, fps = metadata[str(Path(path))]
        return _FakeVideoReader(width=width, height=height, frames=frames, fps=fps)

    monkeypatch.setattr(eval_i2v.decord, "VideoReader", fake_reader)


def _write_placeholder(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")


def test_output_path_keeps_nested_eval_layout(tmp_path: Path):
    output = _output_path(tmp_path, "In-Domain_50/task/00000")

    assert output == tmp_path / "In-Domain_50/task/00000.mp4"


@pytest.mark.parametrize("name", ["../escape", "task/../../escape", "/absolute/path"])
def test_output_path_rejects_paths_outside_output_dir(tmp_path: Path, name: str):
    with pytest.raises(ValueError, match="Unsafe output name"):
        _output_path(tmp_path, name)


def test_temporary_video_path_keeps_mp4_extension(tmp_path: Path):
    output = tmp_path / "task/00000.mp4"

    temporary = _temporary_video_path(output, rank=3)

    assert temporary.parent == output.parent
    assert temporary.suffix == ".mp4"
    assert "rank3" in temporary.name


def test_expected_output_paths_use_names_ids_and_indices_and_reject_duplicates(tmp_path: Path):
    data = [{"name": "split/task/named"}, {"id": "split/task/identified"}, {}, {"id": 0}]

    expected = _expected_output_paths(data, tmp_path)

    assert expected == (
        tmp_path / "0.mp4",
        tmp_path / "2.mp4",
        tmp_path / "split/task/identified.mp4",
        tmp_path / "split/task/named.mp4",
    )
    with pytest.raises(ValueError, match="Duplicate eval output path"):
        _expected_output_paths([{"name": "task/sample"}, {"id": "task/sample.mp4"}], tmp_path)


def test_strict_output_validation_rejects_missing_extra_and_wrong_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "output"
    data = [{"name": "split/task/a"}, {"name": "split/task/b"}]
    first = output_dir / "split/task/a.mp4"
    second = output_dir / "split/task/b.mp4"
    _write_placeholder(first)
    _write_placeholder(second)
    metadata = {
        str(first): (64, 48, 7, 12.0),
        str(second): (64, 48, 7, 12.0),
    }
    _install_fake_readers(monkeypatch, metadata)

    assert _validate_output_set(data, output_dir, width=64, height=48, num_frames=7, fps=12) == 2

    extra = output_dir / "stale.mp4"
    _write_placeholder(extra)
    with pytest.raises(RuntimeError, match="extra=1"):
        _validate_output_set(data, output_dir, width=64, height=48, num_frames=7, fps=12)
    extra.unlink()

    second.unlink()
    with pytest.raises(RuntimeError, match="missing=1"):
        _validate_output_set(data, output_dir, width=64, height=48, num_frames=7, fps=12)
    _write_placeholder(second)

    metadata[str(second)] = (64, 48, 7, 11.0)
    with pytest.raises(RuntimeError, match=r"invalid split/task/b\.mp4: fps=11\.0, expected=12"):
        _validate_output_set(data, output_dir, width=64, height=48, num_frames=7, fps=12)


def test_output_validation_cleans_only_stale_export_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "output"
    output = output_dir / "sample.mp4"
    stale_temporary = output_dir / ".sample.tmp-rank3-pid101.mp4"
    live_temporary = output_dir / ".sample.tmp-rank4-pid202.mp4"
    _write_placeholder(output)
    _write_placeholder(stale_temporary)
    _write_placeholder(live_temporary)
    _install_fake_readers(monkeypatch, {str(output): (64, 48, 7, 12.0)})
    monkeypatch.setattr(eval_i2v, "_pid_is_alive", lambda pid: pid == 202)

    assert _validate_output_set([{"name": "sample"}], output_dir, width=64, height=48, num_frames=7, fps=12) == 1
    assert not stale_temporary.exists()
    assert live_temporary.exists()

    lookalike = output_dir / ".sample.tmp-rankx-pid303.mp4"
    _write_placeholder(lookalike)
    with pytest.raises(RuntimeError, match="extra=1"):
        _validate_output_set([{"name": "sample"}], output_dir, width=64, height=48, num_frames=7, fps=12)
    assert lookalike.exists()


def test_resume_requires_matching_fps_and_force_disables_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / "sample.mp4"
    _write_placeholder(output)
    metadata = {str(output): (64, 48, 7, 12.0)}
    _install_fake_readers(monkeypatch, metadata)

    assert _can_resume_video(output, width=64, height=48, num_frames=7, fps=12, force=False)
    assert not _can_resume_video(output, width=64, height=48, num_frames=7, fps=11, force=False)
    assert not _can_resume_video(output, width=64, height=48, num_frames=7, fps=12, force=True)


def test_item_num_frames_is_opt_in_and_requires_a_positive_integer():
    item = {"num_frames": 13}

    assert _item_num_frames(item, 0, default=7, enabled=False) == 7
    assert _item_num_frames(item, 0, default=7, enabled=True) == 13
    assert _item_num_frames({}, 0, default=7, enabled=True) == 7
    with pytest.raises(ValueError, match="positive integer"):
        _item_num_frames({"num_frames": True}, 0, default=7, enabled=True)
    with pytest.raises(ValueError, match="positive integer"):
        _item_num_frames({"num_frames": 0}, 0, default=7, enabled=True)


def test_output_validation_can_use_per_item_frame_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "output"
    first = output_dir / "split/task/a.mp4"
    second = output_dir / "split/task/b.mp4"
    _write_placeholder(first)
    _write_placeholder(second)
    _install_fake_readers(
        monkeypatch,
        {
            str(first): (64, 48, 5, 16.0),
            str(second): (64, 48, 13, 16.0),
        },
    )
    data = [
        {"name": "split/task/a", "num_frames": 5},
        {"name": "split/task/b", "num_frames": 13},
    ]

    assert (
        _validate_output_set(
            data,
            output_dir,
            width=64,
            height=48,
            num_frames=7,
            fps=16,
            use_item_num_frames=True,
        )
        == 2
    )
    with pytest.raises(RuntimeError, match=r"invalid split/task/a\.mp4: frames=5, expected=7"):
        _validate_output_set(data, output_dir, width=64, height=48, num_frames=7, fps=16)


def test_validate_only_finishes_before_distributed_cuda_or_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    eval_json = tmp_path / "eval.json"
    output_dir = tmp_path / "output"
    output = output_dir / "split/task/sample.mp4"
    eval_json.write_text(json.dumps([{"name": "split/task/sample"}]))
    _write_placeholder(output)
    _install_fake_readers(monkeypatch, {str(output): (64, 48, 7, 12.0)})

    def forbidden(*args, **kwargs):
        raise AssertionError("distributed/CUDA/model setup must not run in validation-only mode")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(eval_i2v.dist, "init_process_group", forbidden)
    monkeypatch.setattr(eval_i2v.torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(eval_i2v, "_load_pipeline_rank_serialized", forbidden)

    assert (
        eval_i2v.main(
            [
                "--eval_json",
                str(eval_json),
                "--output_dir",
                str(output_dir),
                "--validate_only",
                "--height",
                "48",
                "--width",
                "64",
                "--num_frames",
                "7",
                "--fps",
                "12",
            ]
        )
        == 0
    )


def test_validate_only_requires_fixed_dimensions(tmp_path: Path):
    eval_json = tmp_path / "eval.json"
    eval_json.write_text("[]")

    with pytest.raises(ValueError, match="requires fixed --height and --width"):
        eval_i2v.main(
            [
                "--eval_json",
                str(eval_json),
                "--output_dir",
                str(tmp_path / "output"),
                "--validate_only",
            ]
        )
