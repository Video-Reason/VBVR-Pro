from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.cli.plot_vbvr_checkpoint_trends import (
    EXPECTED_SAMPLES,
    SAMPLERS,
    load_evalkit_series,
    load_vlm_judge_series,
    write_outputs,
)
from src.cli.plot_vbvr_vlm_checkpoint_trends import main as plot_vlm_main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _vlm_cell_name(step: int, suffix: str) -> str:
    if step:
        return f"dancegrpo_vbvr_pro_5b_checkpoint-{step}-{suffix}"
    if suffix == "cps-noise-0.7":
        suffix = "cps0p7-30steps-cfg1"
    return f"diffsynth_step35500-baseline-{suffix}"


def _build_vlm_fixture(root: Path, *, steps: tuple[int, ...] = (0, 100)) -> None:
    contract = {
        "model": "fixture-qwen",
        "model_revision": "a" * 40,
        "prompt_source_sha256": "b" * 64,
        "sampled_video_frames": 32,
    }
    contract_sha = "c" * 64
    rows: list[dict[str, object]] = []
    for step in steps:
        for index, sampler in enumerate(SAMPLERS):
            cell_name = _vlm_cell_name(step, sampler.suffix)
            overall = 0.4 + step / 1000 + index / 1000
            in_domain = overall + 0.1
            out_of_domain = overall - 0.1
            cell = root / cell_name
            _write_json(
                cell / "metadata.json",
                {
                    "judge_contract": contract,
                    "judge_contract_sha256": contract_sha,
                },
            )
            _write_json(
                cell / "summary.json",
                {
                    "state": "complete",
                    "judge_contract_sha256": contract_sha,
                    "expected_samples": EXPECTED_SAMPLES,
                    "completed_samples": EXPECTED_SAMPLES,
                    "error_samples": 0,
                    "summary": {
                        "overall": {"mean_score": overall, "num_samples": 500},
                        "In_Domain": {"mean_score": in_domain, "num_samples": 250},
                        "Out_of_Domain": {"mean_score": out_of_domain, "num_samples": 250},
                    },
                },
            )
            rows.append(
                {
                    "step": step,
                    "model": "baseline" if step == 0 else f"checkpoint-{step}",
                    "sampler": sampler.label.replace(" noise", "") if sampler.key.startswith("cps") else sampler.label,
                    "overall": overall,
                    "in_domain": in_domain,
                    "out_of_domain": out_of_domain,
                    "num_samples": EXPECTED_SAMPLES,
                    "errors": 0,
                    "cell_name": cell_name,
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        root / "summary.json",
        {
            "state": "complete",
            "judge_contract": contract,
            "judge_contract_sha256": contract_sha,
            "num_cells": len(rows),
            "num_sample_judgments": len(rows) * EXPECTED_SAMPLES,
        },
    )


def _formal_cell_name(step: int, suffix: str) -> str:
    if step:
        return f"dancegrpo_vbvr_pro_5b_checkpoint-{step}-{suffix}"
    if suffix == "cps-noise-0.7":
        suffix = "cps0p7-30steps-cfg1"
    return f"diffsynth_step35500-baseline-{suffix}"


def _write_formal_cell(root: Path, step: int, sampler_index: int, *, complete: bool = True) -> None:
    sampler = SAMPLERS[sampler_index]
    cell = root / _formal_cell_name(step, sampler.suffix)
    result = cell / "scores/eval_fixture_vbvr_results.json"
    overall = 0.3 + step / 2000 + sampler_index / 1000
    payload = {
        "samples": [{"error": None} for _ in range(EXPECTED_SAMPLES)],
        "summary": {
            "overall": {"mean_score": overall, "num_samples": 500},
            "In_Domain": {"mean_score": overall + 0.1, "num_samples": 250},
            "Out_of_Domain": {"mean_score": overall - 0.1, "num_samples": 250},
        },
    }
    _write_json(result, payload)
    result_bytes = result.read_bytes()
    _write_json(
        cell / "score-provenance.json",
        {
            "stage": "vbvr-pro-score",
            "values": {
                "state": "complete" if complete else "in_progress_rewrite",
                "evalkit_revision": "e140038f",
                "evalkit_source_sha256": "d" * 64,
                "scorer_dependencies": json.dumps({"contract": "fixture-v1", "sha256": "e" * 64}),
            },
            "output_files": (
                {
                    "result": {
                        "path": str(result.resolve()),
                        "sha256": hashlib.sha256(result_bytes).hexdigest(),
                        "size": len(result_bytes),
                    }
                }
                if complete
                else {}
            ),
        },
    )


def _build_formal_fixture(root: Path, baseline_root: Path) -> None:
    for sampler_index in range(len(SAMPLERS)):
        _write_formal_cell(baseline_root, 0, sampler_index)
        _write_formal_cell(root, 100, sampler_index)
        _write_formal_cell(root, 200, sampler_index, complete=sampler_index != len(SAMPLERS) - 1)


def test_loaders_audit_complete_cells_and_preserve_incremental_gap(tmp_path: Path) -> None:
    vlm_root = tmp_path / "vlm"
    formal_root = tmp_path / "formal"
    baseline_root = tmp_path / "baseline"
    _build_vlm_fixture(vlm_root)
    _build_formal_fixture(formal_root, baseline_root)

    vlm = load_vlm_judge_series(vlm_root, epoch_one_end=None)
    evalkit = load_evalkit_series(formal_root, baseline_root=baseline_root)

    assert len(vlm.rows) == 12
    assert vlm.expected_steps == (100,)
    assert not vlm.missing_cells
    assert len(evalkit.rows) == 17
    assert evalkit.expected_steps == (100, 200)
    assert evalkit.missing_cells == ((200, "unipc"),)
    assert all(row.baseline_source == baseline_root.resolve() for row in evalkit.baseline_rows)


def test_write_outputs_creates_individual_and_comparison_artifacts(tmp_path: Path) -> None:
    vlm_root = tmp_path / "vlm"
    formal_root = tmp_path / "formal"
    baseline_root = tmp_path / "baseline"
    _build_vlm_fixture(vlm_root)
    _build_formal_fixture(formal_root, baseline_root)
    vlm = load_vlm_judge_series(vlm_root, epoch_one_end=None)
    evalkit = load_evalkit_series(formal_root, baseline_root=baseline_root)

    outputs = write_outputs(vlm, evalkit, tmp_path / "plots", dpi=20)

    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    summary = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert summary["series"][1]["missing_cells"] == [{"sampler": "UniPC ODE", "sampler_id": "unipc", "step": 200}]


def test_vlm_checkpoint_only_series_reuses_contract_matched_external_baseline(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint-vlm"
    baseline_root = tmp_path / "baseline-vlm"
    _build_vlm_fixture(checkpoint_root, steps=(100,))
    _build_vlm_fixture(baseline_root, steps=(0,))

    series = load_vlm_judge_series(
        checkpoint_root,
        baseline_root=baseline_root,
        epoch_one_end=None,
        key="qwen_judge_rl_qwen_judge",
        label="Qwen-judge-RL model · Qwen3.6-27B task judge",
    )

    assert len(series.rows) == 12
    assert series.expected_steps == (100,)
    assert series.baseline_root == baseline_root.resolve()
    assert all(row.baseline_source == baseline_root.resolve() for row in series.baseline_rows)


def test_vlm_only_plot_cli_writes_curve_csv_and_audit_summary(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint-vlm"
    baseline_root = tmp_path / "baseline-vlm"
    output_dir = tmp_path / "plots"
    _build_vlm_fixture(checkpoint_root, steps=(100,))
    _build_vlm_fixture(baseline_root, steps=(0,))

    status = plot_vlm_main(
        [
            "--vlm-judge-root",
            str(checkpoint_root),
            "--vlm-baseline-root",
            str(baseline_root),
            "--output-dir",
            str(output_dir),
            "--dpi",
            "20",
        ]
    )

    assert status == 0
    expected = (
        output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_trends.png",
        output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_trends.svg",
        output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_scores.csv",
        output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_trend_summary.json",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
