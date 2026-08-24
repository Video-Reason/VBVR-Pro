import json

from src.cli.build_vbvr_trajectory_gallery import (
    _cell_document,
    _renderer_sampler,
    _root_document,
    _trajectory_model_ids,
)
from src.cli.summarize_vbvr_sampler_matrix import _trajectory_paths


def _cell(tmp_path):
    sample_root = tmp_path / "trajectories/baseline-cps0p7/domain/task/00000"
    return {
        "model_id": "baseline",
        "model": "Baseline <model>",
        "sampler_id": "cps0p7",
        "sampler": "CPS 0.7",
        "score": {"overall": 0.5, "in_domain": 0.6, "out_of_domain": 0.4},
        "samples": [
            {
                "index": 0,
                "name": "domain/task/<00000>",
                "task_name": "task",
                "domain": "domain",
                "root": sample_root,
                "grid": sample_root / "steps_grid.mp4",
                "contact_sheet": sample_root / "step_contact_sheet.jpg",
                "manifest": sample_root / "manifest.json",
                "formal_final": tmp_path / "formal/domain/task/00000.mp4",
            }
        ],
    }


def test_renderer_sampler_mapping():
    assert _renderer_sampler("cps0p1") == ("cps", 0.1)
    assert _renderer_sampler("cps0p9") == ("cps", 0.9)
    assert _renderer_sampler("euler") == ("euler", None)
    assert _renderer_sampler("unipc") == ("unipc", None)


def test_trajectory_model_ids_come_from_completed_matrix_cells(tmp_path):
    root = tmp_path / "trajectories"
    for model_id in ("baseline", "300", "100"):
        cell = root / f"{model_id}-cps0p7"
        cell.mkdir(parents=True)
        (cell / "cell_manifest.json").write_text("{}", encoding="utf-8")

    assert _trajectory_model_ids(root) == ["baseline", "100", "300"]


def test_score_summarizer_resolves_sample_zero_in_all_sample_layout(tmp_path):
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    (eval_root / "eval_samples.json").write_text(
        json.dumps([{"name": "domain/task/00000"}]),
        encoding="utf-8",
    )
    trajectory_root = tmp_path / "trajectories"
    cell = trajectory_root / "baseline-cps0p7"
    cell.mkdir(parents=True)
    cell_manifest = cell / "cell_manifest.json"
    cell_manifest.write_text("{}", encoding="utf-8")

    sample, resolved_cell_manifest, layout = _trajectory_paths(
        eval_root=eval_root,
        trajectory_root=trajectory_root,
        model_id="baseline",
        sampler_id="cps0p7",
        layout="auto",
    )

    assert sample == cell / "domain/task/00000"
    assert resolved_cell_manifest == cell_manifest
    assert layout == "all-samples"


def test_cell_gallery_lazy_loads_grid_and_links_every_step(tmp_path):
    page = tmp_path / "gallery/cells/baseline-cps0p7.html"
    document = _cell_document(_cell(tmp_path), page_path=page, root_index=tmp_path / "gallery/index.html")

    assert "data-src=" in document
    assert 'preload="none"' in document
    assert "exact scored final" in document
    assert "Baseline &lt;model&gt;" in document
    assert "domain/task/&lt;00000&gt;" in document
    assert sum(f"step_{index:02d}.mp4" in document for index in range(30)) == 30


def test_root_gallery_links_cell_and_reports_output_count(tmp_path):
    cell = _cell(tmp_path)
    index = tmp_path / "gallery/index.html"
    cell_page = tmp_path / "gallery/cells/baseline-cps0p7.html"
    document = _root_document(
        [cell],
        root_index=index,
        cell_pages={("baseline", "cps0p7"): cell_page},
    )

    assert "cells/baseline-cps0p7.html" in document
    assert "1 model outputs" in document
    assert "500 × 30" in document
