from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.cli.vbvr_rl_demo as rl_demo
from src.cli.vbvr_rl_demo import (
    _build_parser,
    _hardlink_or_copy,
    _selection_records,
    _write_final_gallery,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_selection_records_rejects_duplicates_and_boolean_indices(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    _write_json(selection, [{"checkpoint": 300, "sample_index": 4}])
    assert _selection_records(selection) == [{"checkpoint": 300, "sample_index": 4}]

    _write_json(
        selection,
        [
            {"checkpoint": 300, "sample_index": 4},
            {"checkpoint": 300, "sample_index": 4},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate selection"):
        _selection_records(selection)

    _write_json(selection, [{"checkpoint": 300, "sample_index": False}])
    with pytest.raises(ValueError, match="invalid sample_index"):
        _selection_records(selection)


def test_hardlink_or_copy_is_idempotent_and_refuses_different_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"rollout-video")
    destination = tmp_path / "nested" / "destination.bin"

    assert _hardlink_or_copy(source, destination) in {"hardlink", "copy"}
    assert destination.read_bytes() == source.read_bytes()
    assert _hardlink_or_copy(source, destination) == "existing"

    different = tmp_path / "different.bin"
    different.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="differs from source"):
        _hardlink_or_copy(source, different)


def test_final_gallery_includes_aggregate_and_all_rollout_scores(tmp_path: Path) -> None:
    cases = [
        {
            "rank": 1,
            "display_id": "demo_01",
            "checkpoint": 300,
            "task_name": "maze",
            "canonical_name": "In-Domain_50/maze/00000",
            "score_range": 0.75,
            "prompt": "Solve the maze.",
            "manual_review": {"note": "four distinct paths"},
            "input_image": "cases/demo_01/input.png",
            "ground_truth_video": "cases/demo_01/ground_truth.mp4",
            "audit_sheet": "cases/demo_01/manual_audit.jpg",
            "rollouts": [
                {
                    "rollout_index": index,
                    "seed": 10 + index,
                    "score": score,
                    "native_video": f"cases/demo_01/rollout_{index:02d}_native.mp4",
                }
                for index, score in enumerate((0.0, 0.25, 0.5, 0.75))
            ],
        }
    ]
    contract = {"cps_noise_level": 0.7, "num_inference_steps": 30, "guidance_scale": 1.0}
    aggregate = {
        "plot_png": "evidence/trend.png",
        "samples_per_cell": 500,
        "complete_cells": 144,
        "cps_0p7": {
            "baseline_overall": 0.47,
            "best_step": 2200,
            "best_overall": 0.55,
            "best_delta": 0.08,
        },
    }

    gallery = _write_final_gallery(tmp_path, cases, contract, aggregate)
    document = gallery.read_text(encoding="utf-8")
    assert "500 样本 checkpoint 曲线" in document
    assert "evidence/trend.png" in document
    assert "0.000000" in document
    assert "0.750000" in document
    assert "four distinct paths" in document


def test_package_parser_requires_trend_root() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "package",
                "--selection",
                "selection.json",
                "--output-root",
                "output",
            ]
        )


def test_compose_selected_case_applies_one_rollout_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = {
        "sampler": "flow_cps",
        "cps_noise_level": 0.7,
        "height": 512,
        "width": 512,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "rollouts_per_case": 4,
        "seed_base": 100,
    }
    base_root = tmp_path / "base"
    override_root = tmp_path / "override"
    base_root.mkdir()
    override_root.mkdir()

    def case(case_id: str, scores: tuple[float, ...], seed_base: int) -> dict[str, object]:
        return {
            "case_id": case_id,
            "checkpoint": 300,
            "canonical_name": "In-Domain_50/maze/00000",
            "rollouts": [
                {
                    "rollout_index": index,
                    "seed": seed_base + index,
                    "score": score,
                    "generated_path": str(tmp_path / f"{case_id}_{index}_native.mp4"),
                    "prepared_path": str(tmp_path / f"{case_id}_{index}_scored.mp4"),
                }
                for index, score in enumerate(scores)
            ],
        }

    _write_json(base_root / "candidate_manifest.json", {"contract": contract})
    _write_json(base_root / "candidate_scores.json", {"cases": [case("base_case", (0.0, 0.1, 0.2, 0.3), 10)]})
    override_contract = {**contract, "rollouts_per_case": 8, "seed_base": 200}
    _write_json(override_root / "candidate_manifest.json", {"contract": override_contract})
    _write_json(
        override_root / "candidate_scores.json",
        {"cases": [case("override_case", (0.4, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1), 20)]},
    )
    monkeypatch.setattr(
        rl_demo,
        "_video_diversity",
        lambda paths: {"unique_video_sha256": len(paths)},
    )

    selected, manifest, roots = rl_demo._compose_selected_case(
        {
            "source_root": str(base_root),
            "case_id": "base_case",
            "rollout_overrides": [
                {
                    "rollout_index": 0,
                    "source_root": str(override_root),
                    "case_id": "override_case",
                    "source_rollout_index": 1,
                }
            ],
        },
        {},
    )

    assert manifest["contract"] == contract
    assert roots == {base_root.resolve(), override_root.resolve()}
    assert selected["scores"] == [0.8, 0.1, 0.2, 0.3]
    assert selected["score_range"] == pytest.approx(0.7)
    assert selected["rollouts"][0]["rollout_index"] == 0
    assert selected["rollouts"][0]["seed"] == 21
    assert selected["rollouts"][0]["selection_source_rollout_index"] == 1
