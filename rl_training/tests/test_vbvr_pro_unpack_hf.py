import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.data.vbvr_pro_unpack_hf import (
    _FIELD_TARGETS,
    PINNED_SOURCE_REVISION,
    SOURCE_REPO_ID,
    materialize,
)
from src.data.i2v_dataset import I2VDataset


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def _write_official_archive(dataset_root: Path, *, omit_field: str | None = None) -> tuple[str, dict[str, bytes]]:
    task_name = "G-18_grid_shortest_path_data-generator"
    task_dir = "grid_shortest_path_task"
    archive_dir = dataset_root / "VBVR-Pro-RL-Video"
    archive_dir.mkdir(parents=True)
    values = {
        "first_frame.png": b"first",
        "metadata.json": b'{"meta": true}',
        "video/final_frame.png": b"final",
        "video/ground_truth.mp4": b"video",
        "video/prompt.txt": b"move the object",
    }
    with tarfile.open(archive_dir / f"{task_name}.tar.gz", "w:gz") as archive:
        for sample_id in ("grid_shortest_path_00005004", "grid_shortest_path_00005005"):
            prefix = f"{task_name}/{task_dir}/{sample_id}"
            for field, value in values.items():
                if field != omit_field:
                    _add_bytes(archive, f"{prefix}/{field}", value)
            _add_bytes(archive, f"{prefix}/video/unused.json", b"{}")
    return task_name, values


def test_materialize_official_vbvr_pro_rl_archives(tmp_path: Path) -> None:
    dataset_root = tmp_path / "VBVR-Pro-RL"
    task_name, values = _write_official_archive(dataset_root)
    output_dir = dataset_root / "materialized"

    results = materialize(
        dataset_root,
        output_dir,
        expected_tasks=1,
        expected_samples=2,
        workers=2,
    )

    assert len(results) == 1
    assert results[0].task_name == task_name
    assert results[0].written_files == 2 * len(_FIELD_TARGETS)
    sample_dir = output_dir / "raw" / task_name / "grid_shortest_path_00005004"
    for field, target in _FIELD_TARGETS.items():
        assert (sample_dir / target).read_bytes() == values[field]

    descriptor = json.loads((output_dir / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor[0]["format"] == "vbvr_pro"
    assert descriptor[0]["data_roots"] == ["raw"]
    manifest = json.loads((output_dir / "split_manifest_rl.json").read_text(encoding="utf-8"))
    assert manifest == [
        {
            "task": task_name,
            "source": task_name,
            "split": "In-Domain_50",
            "rl": ["grid_shortest_path_00005004", "grid_shortest_path_00005005"],
        }
    ]
    provenance = json.loads((output_dir / "materialization.json").read_text(encoding="utf-8"))
    assert provenance["source_repo_id"] == SOURCE_REPO_ID
    assert provenance["source_revision"] == PINNED_SOURCE_REVISION
    assert provenance["samples"] == 2
    assert provenance["tasks"] == 1
    assert len(I2VDataset(str(output_dir / "dataset.json"))) == 2

    resumed = materialize(
        dataset_root,
        output_dir,
        expected_tasks=1,
        expected_samples=2,
        workers=1,
        verify_existing=True,
    )
    assert resumed[0].reused_files == 2 * len(_FIELD_TARGETS)

    corrupted = sample_dir / _FIELD_TARGETS["first_frame.png"]
    corrupted.write_bytes(b"wrong")
    repaired = materialize(
        dataset_root,
        output_dir,
        expected_tasks=1,
        expected_samples=2,
        workers=1,
        verify_existing=True,
    )
    assert repaired[0].written_files == 1
    assert repaired[0].reused_files == 2 * len(_FIELD_TARGETS) - 1
    assert corrupted.read_bytes() == values["first_frame.png"]


def test_materialize_rejects_incomplete_official_sample(tmp_path: Path) -> None:
    dataset_root = tmp_path / "VBVR-Pro-RL"
    _write_official_archive(dataset_root, omit_field="video/prompt.txt")

    with pytest.raises(ValueError, match="incomplete samples"):
        materialize(dataset_root, dataset_root / "materialized", expected_tasks=1, workers=1)
