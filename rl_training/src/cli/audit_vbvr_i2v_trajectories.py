"""Lightweight strict audit for a full cell of VBVR 30-step trajectories.

This module intentionally imports no Torch, Diffusers, Decord, or PIL code so
that launchers can scan resumable million-file evaluations without paying model
runtime import costs for every model/sampler cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _noise_level(value: str) -> float:
    level = float(value)
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError(f"CPS noise level must be in [0, 1], got {value}")
    return level


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_json", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--formal_final_root", required=True, type=Path)
    parser.add_argument("--sampler", required=True, choices=("cps", "euler", "unipc"))
    parser.add_argument("--noise_level", type=_noise_level, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--write-cell-manifest",
        action="store_true",
        help="Atomically publish the canonical complete cell manifest after every sample passes",
    )
    args = parser.parse_args(argv)
    if args.sampler == "cps" and args.noise_level is None:
        parser.error("--noise_level is required when --sampler=cps")
    if args.sampler != "cps" and args.noise_level is not None:
        parser.error("--noise_level is only valid when --sampler=cps")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _sample_name(item: dict[str, Any], index: int) -> str:
    if not isinstance(item, dict):
        raise ValueError(f"Eval item {index} must be an object, got {type(item).__name__}")
    for key in ("name", "id"):
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def _relative_video_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe output name {name!r}: expected a relative path without '..'")
    if not relative.parts or str(relative) in {"", "."}:
        raise ValueError("Output name must not be empty")
    return relative.with_suffix(".mp4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sampler_name(sampler: str) -> str:
    return {"cps": "flow_cps", "euler": "flowmatch_euler", "unipc": "unipc"}[sampler]


def _sample_error(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    formal_final: Path,
    sample_index: int,
    sample_name: str,
) -> str | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return f"missing or empty manifest: {manifest_path}"
    required = [output_dir / f"step_{index:02d}.mp4" for index in range(args.num_inference_steps)]
    required += [
        output_dir / "final_00.mp4",
        output_dir / "steps_grid.mp4",
        output_dir / "step_contact_sheet.jpg",
        formal_final,
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return f"missing or empty artifacts: {missing[:5]}"
    expected_mp4_names = {
        *(f"step_{index:02d}.mp4" for index in range(args.num_inference_steps)),
        "final_00.mp4",
        "steps_grid.mp4",
    }
    actual_mp4_names = {path.name for path in output_dir.glob("*.mp4") if path.is_file()}
    if actual_mp4_names != expected_mp4_names:
        return f"unexpected MP4 artifacts: {sorted(actual_mp4_names - expected_mp4_names)[:5]}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid manifest: {exc}"

    expected: dict[str, Any] = {
        "sample_index": sample_index,
        "sample_name": sample_name,
        "sampler": _expected_sampler_name(args.sampler),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed + sample_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == "num_inference_steps" and actual is None:
            actual = manifest.get("num_sampling_steps")
        if actual != value:
            return f"manifest {key}={actual!r}, expected {value!r}"
    if args.sampler == "cps":
        actual_noise_level = manifest.get("noise_level", manifest.get("noise_scale"))
        if actual_noise_level != args.noise_level:
            return f"manifest CPS noise level={actual_noise_level!r}, expected {args.noise_level!r}"
    try:
        recorded_model = Path(manifest["model_path"]).resolve()
    except Exception as exc:
        return f"invalid manifest model_path: {exc}"
    if recorded_model != args.model_path.resolve():
        return f"manifest model_path={recorded_model}, expected {args.model_path.resolve()}"
    previews = manifest.get("step_previews")
    if not isinstance(previews, list) or len(previews) != args.num_inference_steps:
        return f"manifest has {len(previews) if isinstance(previews, list) else 'invalid'} step previews"
    for index, preview in enumerate(previews):
        expected_kind = "final_latent" if index == args.num_inference_steps - 1 else "predicted_clean_x0"
        if not isinstance(preview, dict):
            return f"step preview {index} is not an object"
        if preview.get("display_step") != index + 1 or preview.get("file_index") != index:
            return f"step preview {index} has invalid one-based/file indices"
        if preview.get("kind") != expected_kind or preview.get("output_sigma") != 0.0:
            return f"step preview {index} has invalid kind/output sigma"

    binding = manifest.get("formal_final_binding")
    if not isinstance(binding, dict):
        return "formal_final_binding is missing"
    source = str(formal_final.resolve())
    if binding.get("source") != source:
        return f"formal binding source={binding.get('source')!r}, expected {source!r}"
    digest = _sha256(formal_final)
    if binding.get("sha256") != digest:
        return "formal binding digest does not match the quantitative MP4"
    for bound in (output_dir / "final_00.mp4", output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4"):
        if _sha256(bound) != digest:
            return f"bound output digest mismatch: {bound}"
    return None


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output_dir.is_dir():
        raise RuntimeError(f"Trajectory output directory does not exist: {args.output_dir}")
    data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {args.eval_json}")
    if args.limit is not None:
        data = data[: args.limit]

    expected_manifest_paths: set[Path] = set()
    errors: list[str] = []
    for index, item in enumerate(data):
        name = _sample_name(item, index)
        relative_video = _relative_video_path(name)
        sample_dir = (args.output_dir / relative_video).with_suffix("")
        formal_final = args.formal_final_root / relative_video
        expected_manifest_paths.add((sample_dir / "manifest.json").resolve())
        error = _sample_error(
            args,
            output_dir=sample_dir,
            formal_final=formal_final,
            sample_index=index,
            sample_name=name,
        )
        if error is not None:
            errors.append(f"{index}:{name}: {error}")

    actual_manifest_paths = (
        {path.resolve() for path in args.output_dir.rglob("manifest.json") if path.is_file()}
        if args.output_dir.is_dir()
        else set()
    )
    extra_manifests = sorted(actual_manifest_paths - expected_manifest_paths)
    if extra_manifests:
        errors.append(f"unexpected sample manifests: {[str(path) for path in extra_manifests[:5]]}")

    expected_cell = {
        "state": "complete",
        "model_path": str(args.model_path.resolve()),
        "eval_json": str(args.eval_json.resolve()),
        "formal_final_root": str(args.formal_final_root.resolve()),
        "sampler": args.sampler,
        "noise_level": args.noise_level,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "sample_count": len(data),
        "completed_count": len(data),
    }
    cell_manifest_path = args.output_dir / "cell_manifest.json"
    if args.write_cell_manifest and not errors:
        payload = {
            **expected_cell,
            "files_per_sample": args.num_inference_steps + 4,
            "updated_at_unix": time.time(),
        }
        temporary = args.output_dir / f".cell_manifest.tmp-{time.time_ns()}.json"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(cell_manifest_path)
    try:
        cell = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid cell manifest: {exc}")
    else:
        mismatches = {key: (cell.get(key), value) for key, value in expected_cell.items() if cell.get(key) != value}
        if mismatches:
            errors.append(f"cell manifest mismatch: {mismatches}")

    if errors:
        detail = "\n  - ".join(errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise RuntimeError(f"Trajectory audit failed with {len(errors)} error(s):\n  - {detail}{suffix}")
    return {
        "output_dir": str(args.output_dir.resolve()),
        "model_path": str(args.model_path.resolve()),
        "sampler": args.sampler,
        "noise_level": args.noise_level,
        "samples": len(data),
        "steps_per_sample": args.num_inference_steps,
        "state": "complete",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit(args)
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
