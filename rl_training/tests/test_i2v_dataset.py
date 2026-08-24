import json
from pathlib import Path

import pyarrow as pa
import pytest

from src.data.i2v_dataset import I2VDataset


def test_legacy_videos_column_uses_only_final_target() -> None:
    table = pa.table(
        {
            "videos": [["intermediate.mp4", "target.mp4"]],
            "prompt": ["move the object"],
        }
    )

    video, prompt, image = I2VDataset._read_row(table, 0)

    assert video == "target.mp4"
    assert prompt == "move the object"
    assert image is None


def test_video_column_takes_precedence_over_legacy_videos() -> None:
    table = pa.table(
        {
            "video": ["target.mp4"],
            "videos": [["old-intermediate.mp4", "old-target.mp4"]],
        }
    )

    video, _prompt, _image = I2VDataset._read_row(table, 0)

    assert video == "target.mp4"


def test_legacy_videos_column_rejects_empty_list() -> None:
    table = pa.table({"videos": pa.array([[]], type=pa.list_(pa.string()))})

    with pytest.raises(ValueError, match="empty list"):
        I2VDataset._read_row(table, 0)


def _record(task: str, split: str, source: Path) -> dict:
    return {
        "task": task,
        "split": split,
        "source": str(source),
        "train": ["train-only", "shared-with-bench"],
        "bench": ["shared-with-bench", "bench-only"],
    }


def test_vbvr_pro_filters_task_split_and_excludes_bench_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "VBVR-Pro"
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                _record("in-task", "In-Domain_50", data_root / "in-task"),
                _record("out-task", "Out-of-Domain_50", data_root / "out-task"),
            ]
        )
    )
    entry = {
        "split_manifest": str(manifest),
        "data_roots": [str(data_root)],
        "split": "train",
        "allowed_task_splits": ["In-Domain_50"],
        "exclude_sample_ids_from_splits": ["bench"],
        "check_files": False,
    }

    samples = I2VDataset._load_vbvr_pro_manifest(entry, tmp_path)

    assert [(sample.task_name, sample.sample_id) for sample in samples] == [("in-task", "train-only")]


def test_vbvr_pro_rejects_missing_exclusion_split(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    record = _record("in-task", "In-Domain_50", tmp_path / "VBVR-Pro" / "in-task")
    record.pop("bench")
    manifest.write_text(json.dumps([record]))
    entry = {
        "split_manifest": str(manifest),
        "split": "train",
        "allowed_task_splits": "In-Domain_50",
        "exclude_sample_ids_from_splits": "bench",
        "check_files": False,
    }

    with pytest.raises(ValueError, match="no exclusion split 'bench'"):
        I2VDataset._load_vbvr_pro_manifest(entry, tmp_path)
