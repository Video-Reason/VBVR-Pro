import json

from src.cli.watch_vbvr_trajectory_progress import read_cell_progress


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_progress_aggregates_complete_sample_shards(tmp_path):
    cell = tmp_path / "baseline-cps0p1"
    common = {
        "global_sample_count": 5,
        "sample_shard_count": 2,
        "started_at_unix": 10.0,
        "updated_at_unix": 20.0,
    }
    _write(
        cell / "cell_manifest.shard-000-of-002.json",
        {
            **common,
            "sample_shard_index": 0,
            "selected_sample_count": 3,
            "completed_selected_count": 2,
            "initial_completed_selected_count": 1,
            "state": "in_progress",
        },
    )
    _write(
        cell / "cell_manifest.shard-001-of-002.json",
        {
            **common,
            "sample_shard_index": 1,
            "selected_sample_count": 2,
            "completed_selected_count": 2,
            "initial_completed_selected_count": 1,
            "state": "complete",
        },
    )

    progress = read_cell_progress(tmp_path, "baseline", "cps0p1", shard_count=2, samples_per_cell=5)
    assert progress.completed == 4
    assert progress.total == 5
    assert progress.initial_completed == 2
    assert progress.state == "in_progress"


def test_progress_prefers_complete_canonical_manifest(tmp_path):
    cell = tmp_path / "baseline-cps0p1"
    _write(
        cell / "cell_manifest.json",
        {"state": "complete", "sample_count": 5, "completed_count": 5, "updated_at_unix": 30.0},
    )

    progress = read_cell_progress(tmp_path, "baseline", "cps0p1", shard_count=2, samples_per_cell=5)
    assert progress.completed == progress.total == 5
    assert progress.state == "complete"
