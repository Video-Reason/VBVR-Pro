import json

from src.cli.summarize_vbvr_sampler_matrix import CATEGORIES, SAMPLERS, _eval_root, _load_rows


def _write_cell(eval_base, trajectory_root, model_id, sampler_id, label):
    eval_root = _eval_root(eval_base, model_id, sampler_id, label)
    eval_root.mkdir(parents=True)
    name = "domain/task/00000"
    (eval_root / "eval_samples.json").write_text(json.dumps([{"name": name}]), encoding="utf-8")
    result = eval_root / "scores/eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
    result.parent.mkdir()
    result.write_text(
        json.dumps(
            {
                "summary": {
                    "overall": {
                        "mean_score": 0.5,
                        "by_category": {category: 0.5 for category in CATEGORIES},
                    },
                    "In_Domain": {"mean_score": 0.6},
                    "Out_of_Domain": {"mean_score": 0.4},
                }
            }
        ),
        encoding="utf-8",
    )
    cell = trajectory_root / f"{model_id}-{sampler_id}"
    sample = cell / name
    sample.mkdir(parents=True)
    (cell / "cell_manifest.json").write_text(
        json.dumps({"state": "complete", "completed_count": 1}),
        encoding="utf-8",
    )
    (sample / "manifest.json").write_text(
        json.dumps({"formal_final_binding": {"source": "formal.mp4", "sha256": "abc"}}),
        encoding="utf-8",
    )
    (sample / "steps_grid.mp4").write_bytes(b"grid")
    (sample / "step_contact_sheet.jpg").write_bytes(b"sheet")


def test_load_rows_supports_all_sample_trajectory_layout(tmp_path):
    eval_base = tmp_path / "eval"
    trajectory_root = tmp_path / "trajectories"
    for model_id in ("baseline", "100"):
        for sampler_id, _, label in SAMPLERS:
            _write_cell(eval_base, trajectory_root, model_id, sampler_id, label)

    rows = _load_rows(
        eval_base,
        trajectory_root,
        layout="all-samples",
        model_ids=["baseline", "100"],
    )

    assert len(rows) == 12
    assert {row["trajectory_layout"] for row in rows} == {"all-samples"}
    assert {row["trajectory_sample_count"] for row in rows} == {1}
    assert all(row["trajectory"].endswith("domain/task/00000") for row in rows)
