"""Plot audited VBVR-Pro sampler trends for rule-RL and VLM-judge-RL runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.switch_backend("Agg")


EXPECTED_SAMPLES = 500
RESULT_GLOB = "scores/*_vbvr_results.json"
METRICS = (
    ("overall", "Overall"),
    ("in_domain", "In-Domain"),
    ("out_of_domain", "Out-of-Domain"),
)


@dataclass(frozen=True)
class SamplerSpec:
    key: str
    label: str
    suffix: str
    color: str
    marker: str
    linestyle: str = "-"


SAMPLERS = (
    SamplerSpec("cps0p1", "CPS noise 0.1", "cps-noise-0.1", "#4C78A8", "o"),
    SamplerSpec("cps0p3", "CPS noise 0.3", "cps-noise-0.3", "#72B7B2", "s"),
    SamplerSpec("cps0p7", "CPS noise 0.7", "cps-noise-0.7", "#54A24B", "^"),
    SamplerSpec("cps0p9", "CPS noise 0.9", "cps-noise-0.9", "#F2A927", "D"),
    SamplerSpec("euler", "Euler ODE", "euler-ode-30steps-cfg1", "#E45756", "P", "--"),
    SamplerSpec("unipc", "UniPC ODE", "unipc-ode-30steps-cfg1", "#B279A2", "X", "--"),
)
SAMPLER_BY_KEY = {sampler.key: sampler for sampler in SAMPLERS}
SAMPLER_KEY_BY_LABEL = {
    **{sampler.label: sampler.key for sampler in SAMPLERS},
    "CPS 0.1": "cps0p1",
    "CPS 0.3": "cps0p3",
    "CPS 0.7": "cps0p7",
    "CPS 0.9": "cps0p9",
}
SAMPLER_KEY_BY_SUFFIX = {sampler.suffix: sampler.key for sampler in SAMPLERS}
# The CPS 0.7 baseline predates the uniform cell naming scheme.
BASELINE_KEY_BY_SUFFIX = {
    **SAMPLER_KEY_BY_SUFFIX,
    "cps0p7-30steps-cfg1": "cps0p7",
}
SAMPLER_ORDER = {sampler.key: index for index, sampler in enumerate(SAMPLERS)}


@dataclass(frozen=True)
class ScoreRow:
    step: int
    sampler: str
    overall: float
    in_domain: float
    out_of_domain: float
    num_samples: int
    errors: int
    cell_name: str
    result_path: Path
    baseline_source: Path | None = None


@dataclass(frozen=True)
class TrendSeries:
    key: str
    label: str
    evaluator: str
    root: Path
    rows: tuple[ScoreRow, ...]
    contract: dict[str, Any]
    expected_steps: tuple[int, ...]
    missing_cells: tuple[tuple[int, str], ...]
    epoch_one_end: int | None
    subtitle: str
    baseline_root: Path | None = None

    @property
    def trained_rows(self) -> tuple[ScoreRow, ...]:
        return tuple(row for row in self.rows if row.step > 0)

    @property
    def baseline_rows(self) -> tuple[ScoreRow, ...]:
        return tuple(row for row in self.rows if row.step == 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-judge-root", required=True, type=Path)
    parser.add_argument("--evalkit-root", required=True, type=Path)
    parser.add_argument(
        "--evalkit-baseline-root",
        required=True,
        type=Path,
        help="Formal EvalKit result root containing the six matched DiffSynth baselines.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--vlm-epoch-one-end",
        type=int,
        default=1546,
        help="Epoch-one boundary for the model whose videos are rescored by the VLM judge.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _score(value: Any, *, label: str, path: Path) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label} score in {path}: {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"{label} score must be finite and in [0, 1] in {path}, got {score!r}")
    return score


def _int(value: Any, *, label: str, path: Path) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label} in {path}: {value!r}") from exc
    if number < 0 or number != value:
        raise RuntimeError(f"{label} must be a nonnegative integer in {path}, got {value!r}")
    return number


def _require_close(actual: float, expected: float, *, label: str, path: Path) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label} mismatch in {path}: summary={actual!r}, index={expected!r}")


def _cell_scores(summary: dict[str, Any], path: Path) -> tuple[float, float, float]:
    try:
        overall = summary["overall"]
        in_domain = summary["In_Domain"]
        out_of_domain = summary["Out_of_Domain"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Missing overall/domain score sections in {path}") from exc
    if not all(isinstance(section, dict) for section in (overall, in_domain, out_of_domain)):
        raise RuntimeError(f"Score sections must be objects in {path}")
    return (
        _score(overall.get("mean_score"), label="overall", path=path),
        _score(in_domain.get("mean_score"), label="in-domain", path=path),
        _score(out_of_domain.get("mean_score"), label="out-of-domain", path=path),
    )


def _validate_unique_rows(rows: Iterable[ScoreRow], *, source: Path) -> tuple[ScoreRow, ...]:
    ordered = tuple(sorted(rows, key=lambda row: (row.step, SAMPLER_ORDER[row.sampler])))
    seen: set[tuple[int, str]] = set()
    for row in ordered:
        key = (row.step, row.sampler)
        if key in seen:
            raise RuntimeError(f"Duplicate score cell {key} under {source}")
        seen.add(key)
    return ordered


def _expected_missing(rows: Iterable[ScoreRow], steps: Iterable[int]) -> tuple[tuple[int, str], ...]:
    present = {(row.step, row.sampler) for row in rows}
    return tuple(
        (step, sampler.key) for step in sorted(steps) for sampler in SAMPLERS if (step, sampler.key) not in present
    )


def _parse_vlm_csv_row(raw: dict[str, str], index_path: Path) -> tuple[int, str, float, float, float, int, int]:
    try:
        step = int(raw["step"])
        sampler = SAMPLER_KEY_BY_LABEL[raw["sampler"]]
        overall = _score(raw["overall"], label="overall", path=index_path)
        in_domain = _score(raw["in_domain"], label="in-domain", path=index_path)
        out_of_domain = _score(raw["out_of_domain"], label="out-of-domain", path=index_path)
        num_samples = int(raw["num_samples"])
        errors = int(raw["errors"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed row in {index_path}: {raw!r}") from exc
    if step < 0:
        raise RuntimeError(f"Negative checkpoint step in {index_path}: {step}")
    return step, sampler, overall, in_domain, out_of_domain, num_samples, errors


def _load_vlm_judge_root(root: Path) -> tuple[Path, tuple[ScoreRow, ...], str, dict[str, Any]]:
    """Load one complete offline judge root without imposing matrix shape."""

    root = root.expanduser().resolve()
    root_summary_path = root / "summary.json"
    index_path = root / "summary.csv"
    root_summary = _read_json(root_summary_path)
    if root_summary.get("state") != "complete":
        raise RuntimeError(f"VLM judge root is not complete: {root_summary_path}")
    contract_sha = root_summary.get("judge_contract_sha256")
    contract = root_summary.get("judge_contract")
    if not isinstance(contract_sha, str) or len(contract_sha) != 64 or not isinstance(contract, dict):
        raise RuntimeError(f"Incomplete VLM judge contract in {root_summary_path}")

    try:
        handle = index_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read {index_path}: {exc}") from exc
    rows: list[ScoreRow] = []
    with handle:
        for raw in csv.DictReader(handle):
            step, sampler, overall, in_domain, out_of_domain, num_samples, errors = _parse_vlm_csv_row(raw, index_path)
            cell_name = raw.get("cell_name") or ""
            if not cell_name:
                raise RuntimeError(f"Missing cell_name in {index_path}: {raw!r}")
            cell = root / cell_name
            cell_summary_path = cell / "summary.json"
            cell_summary = _read_json(cell_summary_path)
            metadata = _read_json(cell / "metadata.json")
            if cell_summary.get("state") != "complete":
                raise RuntimeError(f"VLM judge cell is not complete: {cell_summary_path}")
            if cell_summary.get("judge_contract_sha256") != contract_sha:
                raise RuntimeError(f"VLM judge contract mismatch in {cell_summary_path}")
            if metadata.get("judge_contract_sha256") != contract_sha or metadata.get("judge_contract") != contract:
                raise RuntimeError(f"VLM judge metadata contract mismatch in {cell / 'metadata.json'}")
            expected = _int(cell_summary.get("expected_samples"), label="expected_samples", path=cell_summary_path)
            completed = _int(cell_summary.get("completed_samples"), label="completed_samples", path=cell_summary_path)
            cell_errors = _int(cell_summary.get("error_samples"), label="error_samples", path=cell_summary_path)
            if (expected, completed, cell_errors, num_samples, errors) != (
                EXPECTED_SAMPLES,
                EXPECTED_SAMPLES,
                0,
                EXPECTED_SAMPLES,
                0,
            ):
                raise RuntimeError(
                    f"Expected a strict 500-sample, zero-error VLM cell in {cell_summary_path}; "
                    f"got expected={expected}, completed={completed}, cell_errors={cell_errors}, "
                    f"index_samples={num_samples}, index_errors={errors}"
                )
            actual_scores = _cell_scores(cell_summary.get("summary", {}), cell_summary_path)
            for label, actual, indexed in zip(
                ("overall", "in-domain", "out-of-domain"),
                actual_scores,
                (overall, in_domain, out_of_domain),
                strict=True,
            ):
                _require_close(actual, indexed, label=label, path=cell_summary_path)
            rows.append(
                ScoreRow(
                    step=step,
                    sampler=sampler,
                    overall=overall,
                    in_domain=in_domain,
                    out_of_domain=out_of_domain,
                    num_samples=num_samples,
                    errors=errors,
                    cell_name=cell_name,
                    result_path=cell_summary_path,
                )
            )

    ordered = _validate_unique_rows(rows, source=root)
    if root_summary.get("num_cells") != len(ordered):
        raise RuntimeError(
            f"VLM root cell count mismatch in {root_summary_path}: "
            f"summary={root_summary.get('num_cells')!r}, index={len(ordered)}"
        )
    if root_summary.get("num_sample_judgments") != len(ordered) * EXPECTED_SAMPLES:
        raise RuntimeError(f"VLM root judgment count mismatch in {root_summary_path}")
    return root, ordered, contract_sha, contract


def load_vlm_judge_series(
    root: Path,
    *,
    epoch_one_end: int | None = 1546,
    baseline_root: Path | None = None,
    key: str = "rule_rl_qwen_judge",
    label: str = "Rule-RL model · Qwen3.6-27B task judge",
) -> TrendSeries:
    """Load and cross-check a complete six-sampler offline VLM-judge series.

    A checkpoint-only judge run may reuse the six baseline rows from another
    complete judge root. The full judge contract must match exactly, so this
    cannot silently mix model revisions, prompt contracts, or media settings.
    """

    root, root_rows, contract_sha, contract = _load_vlm_judge_root(root)
    rows = list(root_rows)
    resolved_baseline_root: Path | None = None
    if baseline_root is not None:
        if any(row.step == 0 for row in rows):
            raise RuntimeError(f"VLM judge root already contains baseline cells: {root}")
        resolved_baseline_root, baseline_rows, baseline_contract_sha, baseline_contract = _load_vlm_judge_root(
            baseline_root
        )
        if baseline_contract_sha != contract_sha or baseline_contract != contract:
            raise RuntimeError(
                "VLM baseline judge contract does not exactly match the checkpoint judge contract: "
                f"checkpoint={contract_sha} baseline={baseline_contract_sha}"
            )
        selected_baselines = tuple(row for row in baseline_rows if row.step == 0)
        baseline_missing = _expected_missing(selected_baselines, (0,))
        if baseline_missing or len(selected_baselines) != len(SAMPLERS):
            raise RuntimeError(
                f"VLM baseline root does not contain one complete six-sampler baseline: {resolved_baseline_root}"
            )
        rows.extend(
            ScoreRow(
                step=row.step,
                sampler=row.sampler,
                overall=row.overall,
                in_domain=row.in_domain,
                out_of_domain=row.out_of_domain,
                num_samples=row.num_samples,
                errors=row.errors,
                cell_name=row.cell_name,
                result_path=row.result_path,
                baseline_source=resolved_baseline_root,
            )
            for row in selected_baselines
        )

    ordered = _validate_unique_rows(rows, source=root)
    steps = tuple(sorted({row.step for row in ordered if row.step > 0}))
    missing = _expected_missing(ordered, (0, *steps))
    if missing:
        raise RuntimeError(f"VLM judge matrix is incomplete under {root}: {missing[:12]}")

    selected_contract = {
        "kind": "qwen-task-specific-direct-video-judge",
        "judge_contract_sha256": contract_sha,
        "model": contract.get("model"),
        "model_revision": contract.get("model_revision"),
        "prompt_source_sha256": contract.get("prompt_source_sha256"),
        "sampled_video_frames": contract.get("sampled_video_frames"),
    }
    return TrendSeries(
        key=key,
        label=label,
        evaluator="Qwen3.6-27B task-specific direct-video judge",
        root=root,
        rows=ordered,
        contract=selected_contract,
        expected_steps=steps,
        missing_cells=(),
        epoch_one_end=epoch_one_end,
        subtitle=("Task-specific direct-video judge · 512×512×81 · 500 samples per cell · matched 30-step samplers"),
        baseline_root=resolved_baseline_root,
    )


def _parse_formal_cell_name(name: str) -> tuple[int, str] | None:
    checkpoint = re.fullmatch(r"dancegrpo_vbvr_pro_5b_checkpoint-(\d+)-(.+)", name)
    if checkpoint:
        sampler = SAMPLER_KEY_BY_SUFFIX.get(checkpoint.group(2))
        return (int(checkpoint.group(1)), sampler) if sampler else None
    baseline = re.fullmatch(r"diffsynth_step35500-baseline-(.+)", name)
    if baseline:
        sampler = BASELINE_KEY_BY_SUFFIX.get(baseline.group(1))
        return (0, sampler) if sampler else None
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_contract(provenance: dict[str, Any], path: Path) -> dict[str, Any]:
    if provenance.get("stage") != "vbvr-pro-score":
        raise RuntimeError(f"Unexpected provenance stage in {path}: {provenance.get('stage')!r}")
    values = provenance.get("values")
    if not isinstance(values, dict) or values.get("state") != "complete":
        raise RuntimeError(f"Incomplete score provenance in {path}")
    try:
        dependencies = json.loads(values["scorer_dependencies"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid scorer dependency contract in {path}") from exc
    if not isinstance(dependencies, dict) or not dependencies.get("contract") or not dependencies.get("sha256"):
        raise RuntimeError(f"Incomplete scorer dependency contract in {path}")
    contract = {
        "kind": "vbvr-main-v2-evalkit",
        "evalkit_revision": values.get("evalkit_revision"),
        "evalkit_source_sha256": values.get("evalkit_source_sha256"),
        "scorer_dependency_contract": dependencies.get("contract"),
        "scorer_dependency_sha256": dependencies.get("sha256"),
    }
    if not all(contract.values()):
        raise RuntimeError(f"Incomplete EvalKit contract in {path}: {contract!r}")
    return contract


def _load_formal_cell(cell: Path, *, step: int, sampler: str) -> tuple[ScoreRow, dict[str, Any]]:
    results = tuple(sorted(cell.glob(RESULT_GLOB)))
    if len(results) != 1:
        raise RuntimeError(f"Expected exactly one formal result under {cell}, found {len(results)}")
    result_path = results[0]
    payload = _read_json(result_path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SAMPLES} scored samples in {result_path}")
    errors = sum(bool(sample.get("error")) for sample in samples if isinstance(sample, dict))
    if len([sample for sample in samples if isinstance(sample, dict)]) != EXPECTED_SAMPLES or errors:
        raise RuntimeError(f"Formal result is not a strict zero-error 500-sample result: {result_path}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"Missing score summary in {result_path}")
    overall, in_domain, out_of_domain = _cell_scores(summary, result_path)
    for split, expected in (("overall", 500), ("In_Domain", 250), ("Out_of_Domain", 250)):
        section = summary.get(split)
        if not isinstance(section, dict) or section.get("num_samples") != expected:
            raise RuntimeError(f"Expected {expected} {split} samples in {result_path}")

    provenance_path = cell / "score-provenance.json"
    provenance = _read_json(provenance_path)
    contract = _formal_contract(provenance, provenance_path)
    try:
        recorded = provenance["output_files"]["result"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Missing result binding in {provenance_path}") from exc
    if recorded.get("path") != str(result_path.resolve()):
        raise RuntimeError(f"Result path binding mismatch in {provenance_path}")
    if recorded.get("size") != result_path.stat().st_size or recorded.get("sha256") != _sha256(result_path):
        raise RuntimeError(f"Result fingerprint mismatch in {provenance_path}")
    return (
        ScoreRow(
            step=step,
            sampler=sampler,
            overall=overall,
            in_domain=in_domain,
            out_of_domain=out_of_domain,
            num_samples=EXPECTED_SAMPLES,
            errors=0,
            cell_name=cell.name,
            result_path=result_path,
        ),
        contract,
    )


def _discover_formal_rows(
    root: Path,
    *,
    include_checkpoints: bool,
    include_baselines: bool,
) -> tuple[tuple[ScoreRow, ...], tuple[int, ...], tuple[tuple[int, str], ...], dict[str, Any]]:
    root = root.expanduser().resolve()
    discovered: dict[tuple[int, str], Path] = {}
    for cell in sorted(root.iterdir() if root.is_dir() else ()):
        if not cell.is_dir():
            continue
        parsed = _parse_formal_cell_name(cell.name)
        if parsed is None:
            continue
        step, sampler = parsed
        if (step == 0 and not include_baselines) or (step > 0 and not include_checkpoints):
            continue
        key = (step, sampler)
        if key in discovered:
            raise RuntimeError(f"Duplicate formal result cell {key} under {root}")
        discovered[key] = cell
    if not discovered:
        raise RuntimeError(f"No matching formal result cells found under {root}")

    rows: list[ScoreRow] = []
    contracts: list[dict[str, Any]] = []
    for (step, sampler), cell in discovered.items():
        provenance_path = cell / "score-provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
        values = provenance.get("values")
        if (
            provenance.get("stage") != "vbvr-pro-score"
            or not isinstance(values, dict)
            or values.get("state") != "complete"
        ):
            # Incremental evaluators create the cell directory before the strict
            # score result is committed. Keep that point as an explicit gap.
            continue
        results = tuple(cell.glob(RESULT_GLOB))
        if not results:
            continue
        row, contract = _load_formal_cell(cell, step=step, sampler=sampler)
        rows.append(row)
        contracts.append(contract)
    if not rows:
        raise RuntimeError(f"No complete formal result cells found under {root}")
    contract = contracts[0]
    if any(item != contract for item in contracts[1:]):
        raise RuntimeError(f"Mixed EvalKit contracts found under {root}")
    ordered = _validate_unique_rows(rows, source=root)
    steps = tuple(sorted({step for step, _ in discovered if step > 0}))
    expected_steps: tuple[int, ...] = steps
    if include_baselines and not include_checkpoints:
        missing = _expected_missing(ordered, (0,))
    else:
        missing = _expected_missing(ordered, expected_steps)
    return ordered, expected_steps, missing, contract


def load_evalkit_series(root: Path, *, baseline_root: Path) -> TrendSeries:
    """Load strict formal scores, retaining genuine gaps for incomplete cells."""

    root = root.expanduser().resolve()
    baseline_root = baseline_root.expanduser().resolve()
    trained, steps, missing, contract = _discover_formal_rows(
        root,
        include_checkpoints=True,
        include_baselines=False,
    )
    baselines, _, baseline_missing, baseline_contract = _discover_formal_rows(
        baseline_root,
        include_checkpoints=False,
        include_baselines=True,
    )
    if baseline_missing:
        raise RuntimeError(f"Matched EvalKit baseline matrix is incomplete: {baseline_missing}")
    if contract != baseline_contract:
        raise RuntimeError(
            "Cannot combine checkpoint and baseline scores from different EvalKit contracts: "
            f"checkpoint={contract!r}, baseline={baseline_contract!r}"
        )
    baselines = tuple(
        ScoreRow(
            **{
                **row.__dict__,
                "baseline_source": baseline_root,
            }
        )
        for row in baselines
    )
    rows = _validate_unique_rows((*baselines, *trained), source=root)
    return TrendSeries(
        key="qwen_judge_rl_evalkit",
        label="Qwen-judge-RL model · VBVR EvalKit",
        evaluator="VBVR-Pro main-v2 rule EvalKit",
        root=root,
        rows=rows,
        contract=contract,
        expected_steps=steps,
        missing_cells=missing,
        epoch_one_end=None,
        subtitle=("Main-v2 rule evaluator · 512×512×81 · 500 samples per complete cell · matched 30-step samplers"),
        baseline_root=baseline_root,
    )


def _best(series: TrendSeries, metric: str) -> ScoreRow:
    return max(
        series.trained_rows,
        key=lambda row: (getattr(row, metric), row.step, -SAMPLER_ORDER[row.sampler]),
    )


def _x_ticks(max_step: int) -> list[int]:
    interval = 100 if max_step <= 1000 else 200
    ticks = [0, *range(interval, max_step + 1, interval)]
    if ticks[-1] != max_step:
        ticks.append(max_step)
    return ticks


def _missing_note(series: TrendSeries) -> str:
    if not series.missing_cells:
        return ""
    grouped: dict[str, list[int]] = {}
    for step, sampler in series.missing_cells:
        grouped.setdefault(sampler, []).append(step)
    items = []
    for sampler in SAMPLERS:
        steps = grouped.get(sampler.key)
        if steps:
            items.append(f"{sampler.label} @ {','.join(map(str, steps))}")
    return "Incomplete and omitted: " + "; ".join(items)


def _plot_panel(ax: plt.Axes, series: TrendSeries, metric: str, title: str, *, compact: bool) -> None:
    by_key = {(row.step, row.sampler): row for row in series.rows}
    x_values = [0, *series.expected_steps]
    for sampler in SAMPLERS:
        values = [
            getattr(by_key[(step, sampler.key)], metric) if (step, sampler.key) in by_key else math.nan
            for step in x_values
        ]
        ax.plot(
            x_values,
            values,
            color=sampler.color,
            marker=sampler.marker,
            markersize=4.8 if compact else 5.2,
            linewidth=1.8,
            linestyle=sampler.linestyle,
            label=sampler.label,
            zorder=3,
        )
        baseline = by_key.get((0, sampler.key))
        if baseline is not None:
            ax.scatter(
                [0],
                [getattr(baseline, metric)],
                marker="*",
                s=82 if compact else 92,
                facecolor=sampler.color,
                edgecolor="#3F3F46",
                linewidth=0.7,
                zorder=5,
            )

    best = _best(series, metric)
    best_sampler = SAMPLER_BY_KEY[best.sampler]
    best_value = getattr(best, metric)
    ax.scatter(
        [best.step],
        [best_value],
        s=150 if compact else 175,
        facecolors="none",
        edgecolors=best_sampler.color,
        linewidths=2.1,
        zorder=6,
    )
    ax.text(
        0.025,
        0.965,
        f"Best: {best_sampler.label} @ {best.step}\n{best_value:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0 if compact else 10.0,
        color="#3F3F46",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D4D4D8", "alpha": 0.92},
        zorder=8,
    )

    max_step = max(series.expected_steps)
    if series.epoch_one_end is not None and 0 < series.epoch_one_end <= max_step:
        ax.axvline(series.epoch_one_end, color="#737373", linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    scores = [getattr(row, metric) for row in series.rows]
    low, high = min(scores), max(scores)
    span = max(high - low, 0.01)
    padding = max(0.004, span * 0.12)
    ax.set_ylim(max(0.0, low - padding), min(1.0, high + padding))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlim(-max_step * 0.025, max_step * 1.025)
    ticks = _x_ticks(max_step)
    ax.set_xticks(ticks, ["Base" if tick == 0 else str(tick) for tick in ticks])
    ax.tick_params(axis="x", labelrotation=45, labelsize=8.5 if compact else 9.5)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.tick_params(axis="y", labelsize=8.5 if compact else 9.5)
    ax.grid(True, color="#E4E4E7", alpha=0.55, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A1A1AA")
    ax.spines["bottom"].set_color("#A1A1AA")
    ax.set_title(title, fontsize=15 if compact else 17, fontweight="bold", pad=9, color="#27272A")
    ax.set_xlabel("Training checkpoint step", fontsize=9.5 if compact else 10.5, labelpad=6)


def _legend_handles(*, include_epoch: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=sampler.color,
            marker=sampler.marker,
            linestyle=sampler.linestyle,
            linewidth=1.8,
            markersize=5.5,
            label=sampler.label,
        )
        for sampler in SAMPLERS
    ]
    if include_epoch:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#737373",
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                label="Rule-RL epoch 1 end (step 1546)",
            )
        )
    return handles


def _apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#3F3F46",
            "text.color": "#27272A",
            "xtick.color": "#52525B",
            "ytick.color": "#52525B",
            "figure.facecolor": "#FAFAF8",
            "axes.facecolor": "#FAFAF8",
            "savefig.facecolor": "#FAFAF8",
        }
    )


def plot_series(series: TrendSeries, output_stem: Path, *, dpi: int = 200) -> tuple[Path, Path]:
    _apply_figure_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.92))
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        _plot_panel(axis, series, metric, title, compact=False)
    axes[0].set_ylabel("VBVR score", fontsize=11)
    fig.suptitle(series.label, fontsize=22, fontweight="bold", y=0.965)
    fig.text(0.5, 0.912, series.subtitle, ha="center", fontsize=11, color="#52525B")
    fig.legend(
        handles=_legend_handles(include_epoch=series.epoch_one_end is not None),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=7 if series.epoch_one_end is not None else 6,
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.6,
    )
    missing = _missing_note(series)
    footer = "Base = sampler-matched DiffSynth step-35500 baseline. Ring = best trained checkpoint in each panel."
    if missing:
        footer += f" {missing}."
    fig.text(0.5, 0.038, footer, ha="center", fontsize=9.5, color="#71717A")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.78, wspace=0.23)
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_comparison(
    top: TrendSeries,
    bottom: TrendSeries,
    output_stem: Path,
    *,
    dpi: int = 200,
) -> tuple[Path, Path]:
    _apply_figure_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(22, 13.0))
    for row_index, series in enumerate((top, bottom)):
        for axis, (metric, title) in zip(axes[row_index], METRICS, strict=True):
            _plot_panel(axis, series, metric, title, compact=True)
        axes[row_index, 0].set_ylabel("VBVR score", fontsize=10)

    fig.suptitle("VBVR-Pro checkpoint trends under matched evaluators", fontsize=22, fontweight="bold", y=0.982)
    fig.text(
        0.5,
        0.949,
        "Top: rule-RL videos scored by Qwen3.6-27B · Bottom: Qwen-judge-RL videos scored by VBVR EvalKit "
        "· absolute scores across rows are not directly comparable",
        ha="center",
        fontsize=10.5,
        color="#52525B",
    )
    fig.legend(
        handles=_legend_handles(include_epoch=top.epoch_one_end is not None),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.917),
        ncol=7,
        frameon=False,
        fontsize=9.0,
        handlelength=2.2,
        columnspacing=1.45,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.09, top=0.83, wspace=0.23, hspace=0.52)
    for row_index, series in enumerate((top, bottom)):
        position = axes[row_index, 0].get_position()
        expected = len(series.expected_steps) * len(SAMPLERS)
        complete = len(series.trained_rows)
        status = f"{complete}/{expected} trained cells complete"
        missing = _missing_note(series)
        if missing:
            status += f" · {missing}"
        fig.text(
            0.055,
            position.y1 + 0.027,
            f"{'A' if row_index == 0 else 'B'} · {series.label}  |  {status}",
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#3F3F46",
        )
    fig.text(
        0.5,
        0.028,
        "Base = sampler-matched DiffSynth step-35500 baseline. Ring = best trained checkpoint in each panel. "
        "Every plotted cell has 500 samples and zero scorer errors.",
        ha="center",
        fontsize=9.2,
        color="#71717A",
    )
    png = output_stem.with_suffix(".png")
    svg = output_stem.with_suffix(".svg")
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_scores_csv(series: tuple[TrendSeries, ...], path: Path) -> None:
    columns = (
        "series_id",
        "series",
        "evaluator",
        "step",
        "model",
        "sampler_id",
        "sampler",
        "overall",
        "in_domain",
        "out_of_domain",
        "num_samples",
        "errors",
        "cell_name",
        "result_path",
        "baseline_source",
    )
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for item in series:
        for row in item.rows:
            writer.writerow(
                {
                    "series_id": item.key,
                    "series": item.label,
                    "evaluator": item.evaluator,
                    "step": row.step,
                    "model": "baseline" if row.step == 0 else f"checkpoint-{row.step}",
                    "sampler_id": row.sampler,
                    "sampler": SAMPLER_BY_KEY[row.sampler].label,
                    "overall": f"{row.overall:.9f}",
                    "in_domain": f"{row.in_domain:.9f}",
                    "out_of_domain": f"{row.out_of_domain:.9f}",
                    "num_samples": row.num_samples,
                    "errors": row.errors,
                    "cell_name": row.cell_name,
                    "result_path": str(row.result_path),
                    "baseline_source": str(row.baseline_source or ""),
                }
            )
    lines.append(buffer.getvalue())
    _atomic_write_text(path, "".join(lines))


def _best_payload(series: TrendSeries, metric: str) -> dict[str, Any]:
    row = _best(series, metric)
    return {
        "step": row.step,
        "sampler_id": row.sampler,
        "sampler": SAMPLER_BY_KEY[row.sampler].label,
        "score": getattr(row, metric),
        "cell_name": row.cell_name,
    }


def _series_payload(series: TrendSeries) -> dict[str, Any]:
    return {
        "id": series.key,
        "label": series.label,
        "evaluator": series.evaluator,
        "root": str(series.root),
        "baseline_root": str(series.baseline_root) if series.baseline_root else None,
        "contract": series.contract,
        "checkpoint_steps": list(series.expected_steps),
        "complete_trained_cells": len(series.trained_rows),
        "expected_trained_cells": len(series.expected_steps) * len(SAMPLERS),
        "baseline_cells": len(series.baseline_rows),
        "missing_cells": [
            {"step": step, "sampler_id": sampler, "sampler": SAMPLER_BY_KEY[sampler].label}
            for step, sampler in series.missing_cells
        ],
        "best_trained": {metric: _best_payload(series, metric) for metric, _ in METRICS},
    }


def write_outputs(
    vlm: TrendSeries,
    evalkit: TrendSeries,
    output_dir: Path,
    *,
    dpi: int = 200,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vlm_png, vlm_svg = plot_series(
        vlm,
        output_dir / "rule_rl_qwen_judge_sampler_checkpoint_trends",
        dpi=dpi,
    )
    evalkit_png, evalkit_svg = plot_series(
        evalkit,
        output_dir / "qwen_judge_rl_evalkit_sampler_checkpoint_trends",
        dpi=dpi,
    )
    comparison_png, comparison_svg = plot_comparison(
        vlm,
        evalkit,
        output_dir / "sampler_checkpoint_trends_comparison",
        dpi=dpi,
    )
    csv_path = output_dir / "sampler_checkpoint_scores.csv"
    _write_scores_csv((vlm, evalkit), csv_path)
    summary_path = output_dir / "sampler_checkpoint_trend_summary.json"
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_note": (
            "The two series use different evaluators. Compare sampler/checkpoint trends within a series; "
            "do not interpret absolute scores across series as directly comparable."
        ),
        "series": [_series_payload(vlm), _series_payload(evalkit)],
        "artifacts": {
            "comparison_png": str(comparison_png),
            "comparison_svg": str(comparison_svg),
            "vlm_png": str(vlm_png),
            "vlm_svg": str(vlm_svg),
            "evalkit_png": str(evalkit_png),
            "evalkit_svg": str(evalkit_svg),
            "scores_csv": str(csv_path),
        },
    }
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "comparison_png": comparison_png,
        "comparison_svg": comparison_svg,
        "vlm_png": vlm_png,
        "vlm_svg": vlm_svg,
        "evalkit_png": evalkit_png,
        "evalkit_svg": evalkit_svg,
        "scores_csv": csv_path,
        "summary_json": summary_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    vlm = load_vlm_judge_series(args.vlm_judge_root, epoch_one_end=args.vlm_epoch_one_end)
    evalkit = load_evalkit_series(args.evalkit_root, baseline_root=args.evalkit_baseline_root)
    outputs = write_outputs(vlm, evalkit, args.output_dir, dpi=args.dpi)
    report = {
        "vlm": _series_payload(vlm),
        "evalkit": _series_payload(evalkit),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
