import json
from fractions import Fraction
from pathlib import Path

import pytest

from src.cli.prepare_vbvr_eval_videos import VideoInfo
from src.eval import build_vbvr_eval_json
from src.eval.build_vbvr_eval_json import build_entries, derive_generation_num_frames


def _write_sample(root: Path, domain: str, task: str, index: str, prompt: str) -> None:
    sample = root / domain / task / index
    sample.mkdir(parents=True)
    (sample / "first_frame.png").write_bytes(b"image")
    (sample / "prompt.txt").write_text(prompt, encoding="utf-8")


def test_build_entries_uses_evalkit_domain_layout(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task-in", "00000", "in prompt")
    _write_sample(tmp_path, "Out-of-Domain_50", "task-out", "00004", "out prompt")
    _write_sample(tmp_path, "Extra_50", "task-extra", "00000", "extra prompt")

    entries = build_entries(tmp_path)

    assert [entry["name"] for entry in entries] == [
        "In-Domain_50/task-in/00000",
        "Out-of-Domain_50/task-out/00004",
    ]
    assert [entry["prompt"] for entry in entries] == ["in prompt", "out prompt"]


def test_build_entries_can_keep_legacy_fixed_split_layout(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task", "00000", "prompt")

    entries = build_entries(tmp_path, layout="split", split="Hidden_40")

    assert entries[0]["name"] == "Hidden_40/task/00000"


def test_build_entries_validates_flattened_sample_against_manifest(tmp_path: Path):
    _write_sample(tmp_path, "In-Domain_50", "task", "00000", "prompt")
    sample = tmp_path / "In-Domain_50/task/00000"
    (sample / "metadata.json").write_text(json.dumps({"task_id": "sample-id"}), encoding="utf-8")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps([{"task": "task", "split": "In-Domain_50", "bench": ["sample-id"]}]),
        encoding="utf-8",
    )

    entries = build_entries(tmp_path, split_manifest=manifest)

    assert len(entries) == 1

    (sample / "metadata.json").write_text(json.dumps({"task_id": "stale-id"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        build_entries(tmp_path, split_manifest=manifest)


@pytest.mark.parametrize(
    ("ground_truth_frames", "expected"),
    [
        (10, 9),
        (60, 61),
        (62, 61),
        (64, 65),
        (282, 281),
    ],
)
def test_derive_generation_num_frames_matches_diffusers_wan_alignment(ground_truth_frames: int, expected: int):
    assert (
        derive_generation_num_frames(
            ground_truth_frame_count=ground_truth_frames,
            ground_truth_fps=Fraction(16, 1),
            generation_fps=16,
        )
        == expected
    )


def test_build_entries_can_embed_gt_derived_generation_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_sample(tmp_path, "In-Domain_50", "task", "00000", "prompt")
    ground_truth = tmp_path / "In-Domain_50/task/00000/ground_truth.mp4"
    ground_truth.write_bytes(b"video")
    monkeypatch.setattr(
        build_vbvr_eval_json,
        "probe_video",
        lambda path: VideoInfo(
            width=1024,
            height=1024,
            frame_count=60,
            average_fps=Fraction(16, 1),
            nominal_fps=Fraction(16, 1),
            duration=3.75,
        ),
    )

    entries = build_entries(tmp_path, generation_fps=16)

    assert entries[0]["ground_truth_video"] == str(ground_truth.resolve())
    assert entries[0]["ground_truth_width"] == 1024
    assert entries[0]["ground_truth_height"] == 1024
    assert entries[0]["ground_truth_frame_count"] == 60
    assert entries[0]["ground_truth_fps"] == 16.0
    assert entries[0]["generation_fps"] == 16
    assert entries[0]["num_frames"] == 61
