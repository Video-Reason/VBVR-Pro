#!/usr/bin/env python3
"""Recompute the checkpoint-600 CPS/ODE report metrics from scored JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluation_provenance import verify_recorded_manifest  # noqa: E402

DEFAULT_ROOT = Path("storage/eval_out/vbvr_pro_main_v2_indomain_strict_manifest_326f7bda")
DEFAULT_OUTPUT = Path("storage/reports/cps_ode_evalkit_report/checkpoint600_metrics.json")
RESULT_RELATIVE_PATH = Path("scores_evalkit_eb977da6/eval_1024x1024_161f_5s_vbvr_results.json")
PROVENANCE_NAME = "score-provenance-evalkit-eb977da6.json"
EXPECTED_EVALKIT_REVISION = "6fedd9d9edb8daafa56aca8e53885aa8ad6f6037"
EXPECTED_EVALKIT_SOURCE_SHA256 = "eb977da60e95456734063ba018b14d805680179fdf0e3e3b2ba6f603f27a935c"

RUNS = {
    "unipc_ode": {
        "label": "UniPC ODE",
        "directory": "dancegrpo_vbvr_pro_5b_checkpoint-600",
        "steps": 50,
        "cfg": 5.0,
    },
    "euler_ode": {
        "label": "FlowMatch Euler ODE",
        "directory": "dancegrpo_vbvr_pro_5b_checkpoint-600-euler",
        "steps": 50,
        "cfg": 5.0,
    },
    "flowcps_eta0": {
        "label": "Flow-CPS eta=0",
        "directory": "dancegrpo_vbvr_pro_5b_checkpoint-600-cps-noise-0",
        "steps": 30,
        "cfg": 1.0,
    },
    "flowcps_eta07": {
        "label": "Flow-CPS eta=0.7",
        "directory": "dancegrpo_vbvr_pro_5b_checkpoint-600-cps-noise-0.7",
        "steps": 30,
        "cfg": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_724)
    return parser.parse_args()


def _load_scored_run(root: Path, spec: dict[str, object]) -> dict[str, object]:
    run_root = root / str(spec["directory"])
    result_path = run_root / RESULT_RELATIVE_PATH
    provenance_path = run_root / PROVENANCE_NAME

    matches, detail = verify_recorded_manifest(
        provenance_path,
        expected_stage="vbvr-pro-score",
        require_complete=True,
        sections=("output_files",),
    )
    if not matches:
        raise RuntimeError(f"{spec['label']}: invalid score provenance: {detail}")

    provenance = json.loads(provenance_path.read_text())
    values = provenance["values"]
    if values.get("evalkit_revision_actual") != EXPECTED_EVALKIT_REVISION:
        raise RuntimeError(f"{spec['label']}: unexpected EvalKit revision {values.get('evalkit_revision_actual')}")
    if values.get("evalkit_source_sha256") != EXPECTED_EVALKIT_SOURCE_SHA256:
        raise RuntimeError(f"{spec['label']}: unexpected EvalKit contract hash {values.get('evalkit_source_sha256')}")
    recorded_result = provenance["output_files"]["result"]["path"]
    if recorded_result != str(result_path.resolve()):
        raise RuntimeError(f"{spec['label']}: provenance is bound to another result file")

    result = json.loads(result_path.read_text())
    samples = result.get("samples", [])
    if len(samples) != 500 or any(sample.get("error") for sample in samples):
        raise RuntimeError(f"{spec['label']}: expected 500 error-free samples")
    summary = result["summary"]
    task_scores = summary["overall"]["by_task"]
    if len(task_scores) != 100:
        raise RuntimeError(f"{spec['label']}: expected 100 task scores")

    historical_path = run_root / "scores/eval_1024x1024_161f_5s_vbvr_results.json"
    historical = json.loads(historical_path.read_text())["summary"]
    return {
        "label": spec["label"],
        "directory": spec["directory"],
        "steps": spec["steps"],
        "cfg": spec["cfg"],
        "result_path": str(result_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "scores": {
            "overall": float(summary["overall"]["mean_score"]),
            "in_domain": float(summary["In_Domain"]["mean_score"]),
            "out_of_domain": float(summary["Out_of_Domain"]["mean_score"]),
        },
        "historical_scores": {
            "overall": float(historical["overall"]["mean_score"]),
            "in_domain": float(historical["In_Domain"]["mean_score"]),
            "out_of_domain": float(historical["Out_of_Domain"]["mean_score"]),
        },
        "task_scores": {name: float(score) for name, score in task_scores.items()},
        "sample_count": len(samples),
        "task_count": len(task_scores),
        "error_count": sum(bool(sample.get("error")) for sample in samples),
    }


def _bootstrap_task_delta(
    cps_task_scores: dict[str, float],
    baseline_task_scores: dict[str, float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, object]:
    tasks = sorted(cps_task_scores)
    if tasks != sorted(baseline_task_scores):
        raise RuntimeError("Task sets differ across sampler results")
    deltas = np.asarray(
        [cps_task_scores[task] - baseline_task_scores[task] for task in tasks],
        dtype=np.float64,
    )
    indices = rng.integers(0, len(deltas), size=(resamples, len(deltas)))
    bootstrap_means = deltas[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_task_delta": float(deltas.mean()),
        "median_task_delta": float(np.median(deltas)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int((deltas > 0).sum()),
        "ties": int((deltas == 0).sum()),
        "losses": int((deltas < 0).sum()),
    }


def main() -> int:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")

    runs = {name: _load_scored_run(args.root, spec) for name, spec in RUNS.items()}
    cps = runs["flowcps_eta07"]
    rng = np.random.default_rng(args.seed)
    comparisons: dict[str, object] = {}
    for baseline_name in ("unipc_ode", "euler_ode", "flowcps_eta0"):
        baseline = runs[baseline_name]
        comparisons[baseline_name] = {
            "overall_delta": cps["scores"]["overall"] - baseline["scores"]["overall"],
            "in_domain_delta": cps["scores"]["in_domain"] - baseline["scores"]["in_domain"],
            "out_of_domain_delta": (cps["scores"]["out_of_domain"] - baseline["scores"]["out_of_domain"]),
            "task_bootstrap": _bootstrap_task_delta(
                cps["task_scores"],
                baseline["task_scores"],
                rng=rng,
                resamples=args.bootstrap_resamples,
            ),
        }

    for run in runs.values():
        scores = run["scores"]
        historical = run["historical_scores"]
        run["scorer_migration_delta"] = {key: scores[key] - historical[key] for key in scores}

    payload = {
        "schema_version": 1,
        "checkpoint": 600,
        "evalkit_revision": EXPECTED_EVALKIT_REVISION,
        "evalkit_source_sha256": EXPECTED_EVALKIT_SOURCE_SHA256,
        "bootstrap": {
            "unit": "task",
            "task_count": 100,
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "interval": "percentile_95",
        },
        "runs": runs,
        "cps_eta07_comparisons": comparisons,
        "limitations": [
            "Cross-sampler videos do not prove identical initial latent tensors.",
            "UniPC/Euler use T=50, CFG=5; Flow-CPS paths use T=30, CFG=1.",
            "This is a scorer-only re-evaluation of existing generated/prepared videos.",
            (
                "Historical/latest prepared-video provenance does not provide "
                "content-only SHA-256 equality for all four modes."
            ),
            (
                "Only a post-RL checkpoint is included, so the sampler gap cannot "
                "be attributed solely to RL without a pre/post difference-in-differences."
            ),
            "Only checkpoint 600 is included.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(f"Wrote report metrics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
