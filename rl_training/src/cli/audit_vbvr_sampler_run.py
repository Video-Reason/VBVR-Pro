"""Strictly audit one matched 30-step VBVR-Pro sampler result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.evaluation_provenance import verify_recorded_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--converted-model", required=True, type=Path)
    parser.add_argument("--gt-base", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--evalkit-revision", required=True)
    parser.add_argument("--evalkit-source-sha256", required=True)
    parser.add_argument("--generation-mode", required=True, choices=("cps", "ode"))
    parser.add_argument("--cps-noise-level", default=None)
    parser.add_argument("--ode-solver", choices=("euler", "unipc"), default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Validate recorded contracts and artifacts without re-fingerprinting large trees",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.generation_mode == "cps" and args.cps_noise_level is None:
        parser.error("--cps-noise-level is required for CPS")
    if args.generation_mode == "cps" and args.ode_solver is not None:
        parser.error("--ode-solver is invalid for CPS")
    if args.generation_mode == "ode" and args.ode_solver is None:
        parser.error("--ode-solver is required for ODE")
    if args.generation_mode == "ode" and args.cps_noise_level is not None:
        parser.error("--cps-noise-level is invalid for ODE")
    return args


def _recorded_path(manifest: dict, section: str, name: str) -> str | None:
    return manifest.get(section, {}).get(name, {}).get("path")


def audit(args: argparse.Namespace) -> dict:
    root = args.output_root.resolve()
    converted_model = args.converted_model.resolve()
    gt_base = args.gt_base.resolve()
    prepared_name = "eval_1024x1024_81f_fps16_5p0625s"
    generated = root / f"generated_{args.height}x{args.width}x81"
    prepared = root / prepared_name
    result = root / "scores" / f"{prepared_name}_vbvr_results.json"
    workbook = root / "scores" / f"{prepared_name}_task_scores.xlsx"
    summary_path = root / "final_scores.txt"
    provenance_paths = {
        "generation": (root / "generation-provenance.json", "vbvr-pro-generation"),
        "preparation": (root / "preparation-provenance.json", "vbvr-pro-preparation"),
        "score": (root / "score-provenance.json", "vbvr-pro-score"),
    }
    required = (result, workbook, summary_path, *(path for path, _ in provenance_paths.values()))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    score_data = json.loads(result.read_text())
    samples = score_data.get("samples")
    if not isinstance(samples, list) or len(samples) != args.expected_samples:
        raise RuntimeError(f"expected {args.expected_samples} scored samples, got {type(samples).__name__}")
    if any(sample.get("error") for sample in samples):
        raise RuntimeError("one or more scored samples contains an error")
    if len({sample.get("task_name") for sample in samples}) != 100:
        raise RuntimeError("expected exactly 100 tasks")
    summary = score_data.get("summary", {})
    for split, count in (("In_Domain", 250), ("Out_of_Domain", 250), ("overall", args.expected_samples)):
        if summary.get(split, {}).get("num_samples") != count:
            raise RuntimeError(f"unexpected {split} sample count")

    manifests: dict[str, dict] = {}
    for name, (path, stage) in provenance_paths.items():
        manifest = json.loads(path.read_text())
        if args.fast:
            if manifest.get("stage") != stage or manifest.get("values", {}).get("state") != "complete":
                raise RuntimeError(f"{path} is not a recorded complete {stage} manifest")
        else:
            matches, detail = verify_recorded_manifest(path, expected_stage=stage, require_complete=True)
            if not matches:
                raise RuntimeError(detail)
        manifests[name] = manifest

    generation = manifests["generation"]
    generation_values = generation.get("values", {})
    expected_values = {
        "state": "complete",
        "height": str(args.height),
        "width": str(args.width),
        "num_frames": "81",
        "fps": "16",
        "num_inference_steps": "30",
        "guidance_scale": "1.0",
        "seed": "0",
        "generation_mode": args.generation_mode,
    }
    if args.generation_mode == "cps":
        expected_values["cps_noise_level"] = str(args.cps_noise_level)
    else:
        expected_values["ode_solver"] = "flowmatch_euler" if args.ode_solver == "euler" else "unipc"
    mismatches = {
        key: (generation_values.get(key), expected)
        for key, expected in expected_values.items()
        if str(generation_values.get(key)) != expected
    }
    if mismatches:
        raise RuntimeError(f"generation provenance mismatch: {mismatches}")
    if generation.get("files", {}).get("split_manifest", {}).get("sha256") != args.manifest_sha256:
        raise RuntimeError("split manifest fingerprint mismatch")

    preparation_values = manifests["preparation"].get("values", {})
    if preparation_values.get("state") != "complete" or str(preparation_values.get("max_duration")) != "5.0625":
        raise RuntimeError("preparation provenance mismatch")
    score_values = manifests["score"].get("values", {})
    if score_values.get("state") != "complete":
        raise RuntimeError("score provenance is incomplete")
    if score_values.get("evalkit_revision") != args.evalkit_revision:
        raise RuntimeError("EvalKit revision mismatch")
    if score_values.get("evalkit_source_sha256") != args.evalkit_source_sha256:
        raise RuntimeError("EvalKit source fingerprint mismatch")
    dependencies = json.loads(score_values["scorer_dependencies"])
    if not args.fast:
        from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime

        runtime = validate_vbvr_scorer_runtime()
        if dependencies.get("contract") != runtime["contract"] or dependencies.get("sha256") != runtime["sha256"]:
            raise RuntimeError("scorer dependency contract mismatch")
    elif not dependencies.get("contract") or not dependencies.get("sha256"):
        raise RuntimeError("recorded scorer dependency contract is incomplete")

    bindings = (
        (generation, "media_trees", "generated_videos", generated),
        (generation, "trees", "converted_model", converted_model),
        (generation, "trees", "eval_source", gt_base),
        (manifests["preparation"], "media_trees", "prepared_videos", prepared),
        (manifests["score"], "trees", "prepared_videos", prepared),
        (manifests["score"], "trees", "ground_truth", gt_base),
        (manifests["score"], "output_files", "result", result),
    )
    for manifest, section, name, expected_path in bindings:
        actual = _recorded_path(manifest, section, name)
        if actual != str(expected_path.resolve()):
            raise RuntimeError(f"{section}.{name} path mismatch: expected={expected_path.resolve()} actual={actual}")

    generated_count = sum(1 for _ in generated.rglob("*.mp4")) if generated.is_dir() else 0
    prepared_count = sum(1 for _ in prepared.rglob("*.mp4")) if prepared.is_dir() else 0
    if generated_count != args.expected_samples or prepared_count != args.expected_samples:
        raise RuntimeError(
            f"media count mismatch: generated={generated_count}, prepared={prepared_count}, "
            f"expected={args.expected_samples}"
        )

    report = {
        "output_root": str(root),
        "generation_mode": args.generation_mode,
        "cps_noise_level": args.cps_noise_level,
        "ode_solver": args.ode_solver,
        "overall": summary["overall"]["mean_score"],
        "in_domain": summary["In_Domain"]["mean_score"],
        "out_of_domain": summary["Out_of_Domain"]["mean_score"],
        "samples": args.expected_samples,
        "tasks": 100,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit(args)
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
