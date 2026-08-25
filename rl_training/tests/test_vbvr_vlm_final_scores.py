from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vbvr_vlm_final_scores import render_final_scores, write_cell_final_scores


def _summary() -> dict:
    return {
        "state": "complete",
        "summary": {
            "overall": {"mean_score": 0.5, "num_samples": 4, "by_task": {"G-1": 0.75, "O-1": 0.25}},
            "In_Domain": {"mean_score": 0.75, "num_samples": 2},
            "Out_of_Domain": {"mean_score": 0.25, "num_samples": 2},
        },
    }


def test_render_final_scores_matches_evalkit_layout() -> None:
    assert render_final_scores(_summary()) == (
        "Overall:        0.500000\n"
        "In-Domain:      0.750000\n"
        "Out-of-Domain:  0.250000\n"
        "\n"
        "Samples: 4 (2 in-domain + 2 out-of-domain)\n"
        "Tasks:   2\n"
    )


def test_write_cell_final_scores_reads_summary_json(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps(_summary()))
    output = write_cell_final_scores(tmp_path)
    assert output == tmp_path / "final_scores.txt"
    assert output.read_text() == render_final_scores(_summary())


def test_render_final_scores_rejects_incomplete_or_inconsistent_summary() -> None:
    incomplete = _summary()
    incomplete["state"] = "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        render_final_scores(incomplete)

    inconsistent = _summary()
    inconsistent["summary"]["overall"]["num_samples"] = 5
    with pytest.raises(ValueError, match="do not add up"):
        render_final_scores(inconsistent)
