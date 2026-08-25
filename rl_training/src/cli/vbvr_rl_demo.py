"""Build, score, and summarize same-input Flow-CPS rollout groups for RL demos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import statistics
import textwrap
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_result_name(sample: dict[str, Any]) -> str:
    return f"{sample['folder']}/{sample['task_name']}/{Path(sample['video_file']).stem}"


def _find_formal_result(formal_root: Path, checkpoint: int, cps_noise_level: float) -> Path:
    cell = formal_root / f"dancegrpo_vbvr_pro_5b_checkpoint-{checkpoint}-cps-noise-{cps_noise_level:g}"
    matches = sorted((cell / "scores").glob("*_vbvr_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one formal score JSON for checkpoint {checkpoint}, found {matches}")
    return matches[0]


def _model_tree_is_readable(path: Path) -> bool:
    return all(not item.is_file() or os.access(item, os.R_OK) for item in path.rglob("*"))


def _load_formal_scores(formal_root: Path, checkpoint: int, cps_noise_level: float) -> list[dict[str, Any]]:
    result_path = _find_formal_result(formal_root, checkpoint, cps_noise_level)
    result = _json(result_path)
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Formal score JSON has no samples: {result_path}")
    errors = [sample for sample in samples if sample.get("error")]
    if errors:
        raise ValueError(f"Formal score JSON contains {len(errors)} errors: {result_path}")
    return samples


def _selection_records(path: Path) -> list[dict[str, int]]:
    raw = _json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Selection must be a non-empty JSON list: {path}")
    records: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Selection item {index} must be an object")
        checkpoint = item.get("checkpoint")
        sample_index = item.get("sample_index")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint <= 0:
            raise ValueError(f"Selection item {index} has invalid checkpoint: {checkpoint!r}")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError(f"Selection item {index} has invalid sample_index: {sample_index!r}")
        key = (checkpoint, sample_index)
        if key in seen:
            raise ValueError(f"Duplicate selection: checkpoint={checkpoint}, sample_index={sample_index}")
        seen.add(key)
        records.append({"checkpoint": checkpoint, "sample_index": sample_index})
    return records


def _build(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    eval_json_path = Path(args.eval_json).resolve()
    formal_root = Path(args.formal_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    converted_root = Path(args.converted_root).resolve()
    base_model = Path(args.base_model).resolve()
    output_root = Path(args.output_root).resolve()
    eval_samples = _json(eval_json_path)
    if not isinstance(eval_samples, list) or not eval_samples:
        raise ValueError(f"Expected a non-empty eval JSON list: {eval_json_path}")

    selection = _selection_records(selection_path)
    formal_by_checkpoint: dict[int, list[dict[str, Any]]] = {}
    jobs_by_checkpoint: dict[int, list[dict[str, Any]]] = {}
    cases: list[dict[str, Any]] = []
    canonical_names: set[str] = set()

    for case_number, record in enumerate(selection, start=1):
        checkpoint = record["checkpoint"]
        sample_index = record["sample_index"]
        if sample_index >= len(eval_samples):
            raise IndexError(f"sample_index={sample_index} is outside the {len(eval_samples)}-sample eval JSON")
        sample = eval_samples[sample_index]
        if not isinstance(sample, dict):
            raise ValueError(f"Eval sample {sample_index} is not an object")
        canonical_name = str(sample.get("name", ""))
        if not canonical_name:
            raise ValueError(f"Eval sample {sample_index} has no canonical name")
        if canonical_name in canonical_names:
            raise ValueError(f"A canonical sample may appear only once in the candidate pool: {canonical_name}")
        canonical_names.add(canonical_name)

        checkpoint_dir = checkpoint_root / f"checkpoint-{checkpoint}"
        if not (checkpoint_dir / "high" / ".metadata").is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint_dir}")
        converted_model = converted_root / f"{args.converted_prefix}_checkpoint-{checkpoint}"
        if not (converted_model / "model_index.json").is_file():
            raise FileNotFoundError(f"Incomplete converted model: {converted_model}")
        if _model_tree_is_readable(converted_model):
            inference_model_path = converted_model
            inference_checkpoint = None
            inference_load_mode = "converted"
        else:
            if not (base_model / "model_index.json").is_file() or not _model_tree_is_readable(base_model):
                raise PermissionError(
                    f"Converted model is not fully readable and no readable base-model fallback is available: "
                    f"{converted_model}"
                )
            inference_model_path = base_model
            inference_checkpoint = checkpoint_dir
            inference_load_mode = "base_plus_dcp"

        formal_samples = formal_by_checkpoint.setdefault(
            checkpoint,
            _load_formal_scores(formal_root, checkpoint, args.cps_noise_level),
        )
        if sample_index >= len(formal_samples):
            raise IndexError(f"Formal result for checkpoint {checkpoint} has no sample {sample_index}")
        formal_sample = formal_samples[sample_index]
        if _canonical_result_name(formal_sample) != canonical_name:
            raise ValueError(
                f"Formal result order mismatch at checkpoint {checkpoint}, sample {sample_index}: "
                f"{_canonical_result_name(formal_sample)!r} != {canonical_name!r}"
            )

        case_id = f"case_{case_number:03d}"
        generated_dir = output_root / "generated" / f"checkpoint-{checkpoint}"
        prepared_dir = output_root / "prepared" / f"checkpoint-{checkpoint}"
        rollouts: list[dict[str, Any]] = []
        for rollout_index in range(args.rollouts):
            seed = args.seed_base + (case_number - 1) * args.rollouts + rollout_index
            generation_name = f"{case_id}/rollout_{rollout_index:02d}"
            generated_path = generated_dir / f"{generation_name}.mp4"
            prepared_path = prepared_dir / f"{generation_name}.mp4"
            rollout = {
                "rollout_index": rollout_index,
                "seed": seed,
                "generation_name": generation_name,
                "generated_path": str(generated_path),
                "prepared_path": str(prepared_path),
            }
            rollouts.append(rollout)
            job = dict(sample)
            job.update(
                {
                    "name": generation_name,
                    "seed": seed,
                    "demo_case_id": case_id,
                    "source_name": canonical_name,
                    "rollout_index": rollout_index,
                    "checkpoint": checkpoint,
                }
            )
            jobs_by_checkpoint.setdefault(checkpoint, []).append(job)

        sample_dir = Path(str(sample["image"])).resolve().parent
        ground_truth = sample_dir / "ground_truth.mp4"
        cases.append(
            {
                "case_id": case_id,
                "checkpoint": checkpoint,
                "sample_index": sample_index,
                "canonical_name": canonical_name,
                "domain": sample.get("domain"),
                "task_name": sample.get("task_name"),
                "video_idx": sample.get("video_idx"),
                "prompt": sample.get("prompt"),
                "first_frame": str(Path(str(sample["image"])).resolve()),
                "ground_truth": str(ground_truth.resolve()),
                "formal_seed": sample_index,
                "formal_score": float(formal_sample["score"]),
                "checkpoint_dir": str(checkpoint_dir),
                "converted_model": str(converted_model),
                "inference_model_path": str(inference_model_path),
                "inference_checkpoint": str(inference_checkpoint) if inference_checkpoint is not None else None,
                "inference_load_mode": inference_load_mode,
                "rollouts": rollouts,
            }
        )

    for checkpoint, jobs in sorted(jobs_by_checkpoint.items()):
        path = output_root / "generation_jobs" / f"checkpoint-{checkpoint}.json"
        _write_json(path, jobs)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "contract": {
            "sampler": "flow_cps",
            "cps_noise_level": args.cps_noise_level,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "rollouts_per_case": args.rollouts,
            "seed_base": args.seed_base,
        },
        "sources": {
            "selection": str(selection_path),
            "selection_sha256": _sha256(selection_path),
            "eval_json": str(eval_json_path),
            "eval_json_sha256": _sha256(eval_json_path),
            "formal_root": str(formal_root),
            "checkpoint_root": str(checkpoint_root),
            "converted_root": str(converted_root),
            "converted_prefix": args.converted_prefix,
            "base_model": str(base_model),
        },
        "case_count": len(cases),
        "checkpoints": sorted(jobs_by_checkpoint),
        "cases": cases,
    }
    _write_json(output_root / "candidate_manifest.json", manifest)
    print(f"Wrote {len(cases)} cases and {len(cases) * args.rollouts} rollout jobs under {output_root}")


def _hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size and _sha256(destination) == _sha256(source):
            return "existing"
        raise FileExistsError(f"Destination differs from source: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _stage(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = _json(manifest_path)
    output_root = manifest_path.parent
    rollout_count = int(manifest["contract"]["rollouts_per_case"])
    staged: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        canonical_name = str(case["canonical_name"])
        for rollout in case["rollouts"]:
            rollout_index = int(rollout["rollout_index"])
            source = Path(rollout["prepared_path"])
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing prepared rollout: {source}")
            destination = output_root / "score_inputs" / f"rollout_{rollout_index:02d}" / f"{canonical_name}.mp4"
            mode = _hardlink_or_copy(source, destination)
            staged.append(
                {
                    "case_id": case["case_id"],
                    "rollout_index": rollout_index,
                    "canonical_name": canonical_name,
                    "source": str(source),
                    "destination": str(destination),
                    "size": source.stat().st_size,
                    "sha256": _sha256(source),
                    "mode": mode,
                }
            )
    expected = int(manifest["case_count"]) * rollout_count
    if len(staged) != expected:
        raise RuntimeError(f"Staged {len(staged)} videos, expected {expected}")
    _write_json(output_root / "score_inputs_manifest.json", {"schema_version": 1, "videos": staged})
    print(f"Staged {len(staged)} score inputs under {output_root / 'score_inputs'}")


def _video_diversity(paths: list[Path]) -> dict[str, float | int]:
    import decord
    import numpy as np

    batches = []
    hashes = []
    for path in paths:
        reader = decord.VideoReader(str(path), num_threads=1)
        if len(reader) < 2:
            raise ValueError(f"Rollout has fewer than two frames: {path}")
        indices = sorted({round((len(reader) - 1) * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)})
        frames = reader.get_batch(indices).asnumpy()[:, ::4, ::4].astype(np.int16)
        batches.append(frames)
        hashes.append(_sha256(path))

    temporal_mae: list[float] = []
    changed_fraction: list[float] = []
    final_changed_fraction: list[float] = []
    for left, right in combinations(batches, 2):
        difference = np.abs(left - right)
        temporal_mae.append(float(difference.mean() / 255.0))
        changed_fraction.append(float((difference.max(axis=-1) > 24).mean()))
        final_changed_fraction.append(float((difference[-1].max(axis=-1) > 24).mean()))
    return {
        "unique_video_sha256": len(set(hashes)),
        "pairwise_temporal_mae_mean": statistics.mean(temporal_mae),
        "pairwise_changed_fraction_mean": statistics.mean(changed_fraction),
        "pairwise_changed_fraction_min": min(changed_fraction),
        "pairwise_final_changed_fraction_mean": statistics.mean(final_changed_fraction),
    }


_ROLLOUT_OVERRIDE_CONTRACT_KEYS = (
    "sampler",
    "cps_noise_level",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "guidance_scale",
)


def _load_candidate_source(
    source_root: Path,
    source_cache: dict[Path, tuple[dict[str, Any], dict[str, dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source_root = source_root.resolve()
    cached = source_cache.get(source_root)
    if cached is not None:
        return cached
    candidate_manifest = _json(source_root / "candidate_manifest.json")
    score_data = _json(source_root / "candidate_scores.json")
    cases = {str(case["case_id"]): case for case in score_data["cases"]}
    cached = (candidate_manifest, cases)
    source_cache[source_root] = cached
    return cached


def _compose_selected_case(
    item: dict[str, Any],
    source_cache: dict[Path, tuple[dict[str, Any], dict[str, dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, Any], set[Path]]:
    """Load one selected case and apply optional per-rollout candidate overrides."""
    source_root = Path(str(item["source_root"])).resolve()
    case_id = str(item["case_id"])
    candidate_manifest, cases = _load_candidate_source(source_root, source_cache)
    base_case = cases.get(case_id)
    if base_case is None:
        raise KeyError(f"Unknown case {case_id!r} under {source_root}")

    case = {**base_case, "rollouts": [dict(rollout) for rollout in base_case["rollouts"]]}
    base_contract = candidate_manifest["contract"]
    referenced_roots = {source_root}
    for rollout in case["rollouts"]:
        rollout["selection_source_root"] = str(source_root)
        rollout["selection_source_case_id"] = case_id
        rollout["selection_source_rollout_index"] = int(rollout["rollout_index"])

    overrides = item.get("rollout_overrides", [])
    if not isinstance(overrides, list):
        raise ValueError(f"rollout_overrides must be a list for {source_root} / {case_id}")
    seen_targets: set[int] = set()
    by_target = {int(rollout["rollout_index"]): index for index, rollout in enumerate(case["rollouts"])}
    for override in overrides:
        if not isinstance(override, dict):
            raise ValueError(f"Rollout override must be an object for {source_root} / {case_id}")
        target_index = override.get("rollout_index")
        source_index = override.get("source_rollout_index")
        if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index not in by_target:
            raise ValueError(f"Invalid rollout override target {target_index!r} for {source_root} / {case_id}")
        if target_index in seen_targets:
            raise ValueError(f"Duplicate rollout override target {target_index} for {source_root} / {case_id}")
        seen_targets.add(target_index)
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"Invalid source_rollout_index {source_index!r} for rollout {target_index}")

        override_root = Path(str(override["source_root"])).resolve()
        override_case_id = str(override["case_id"])
        override_manifest, override_cases = _load_candidate_source(override_root, source_cache)
        override_case = override_cases.get(override_case_id)
        if override_case is None:
            raise KeyError(f"Unknown override case {override_case_id!r} under {override_root}")
        if str(override_case["canonical_name"]) != str(case["canonical_name"]):
            raise ValueError(
                "Rollout override canonical sample mismatch: "
                f"{override_case['canonical_name']} != {case['canonical_name']}"
            )
        if int(override_case["checkpoint"]) != int(case["checkpoint"]):
            raise ValueError(
                f"Rollout override checkpoint mismatch: {override_case['checkpoint']} != {case['checkpoint']}"
            )
        override_contract = override_manifest["contract"]
        for key in _ROLLOUT_OVERRIDE_CONTRACT_KEYS:
            if override_contract.get(key) != base_contract.get(key):
                raise ValueError(
                    f"Rollout override contract mismatch for {key}: "
                    f"{override_contract.get(key)!r} != {base_contract.get(key)!r}"
                )
        source_rollout = next(
            (rollout for rollout in override_case["rollouts"] if int(rollout["rollout_index"]) == source_index),
            None,
        )
        if source_rollout is None:
            raise KeyError(
                f"Missing source rollout {source_index} in override case {override_root} / {override_case_id}"
            )
        replacement = {
            **source_rollout,
            "rollout_index": target_index,
            "selection_source_root": str(override_root),
            "selection_source_case_id": override_case_id,
            "selection_source_rollout_index": source_index,
        }
        case["rollouts"][by_target[target_index]] = replacement
        referenced_roots.add(override_root)

    case["rollouts"].sort(key=lambda rollout: int(rollout["rollout_index"]))
    scores = [float(rollout["score"]) for rollout in case["rollouts"]]
    case["scores"] = scores
    case["score_min"] = min(scores)
    case["score_max"] = max(scores)
    case["score_mean"] = statistics.mean(scores)
    case["score_std"] = statistics.pstdev(scores)
    case["score_range"] = max(scores) - min(scores)
    case["diversity"] = _video_diversity([Path(rollout["generated_path"]) for rollout in case["rollouts"]])
    return case, candidate_manifest, referenced_roots


def _score_result_for_rollout(output_root: Path, rollout_index: int) -> Path:
    result_dir = output_root / "score_results" / f"rollout_{rollout_index:02d}"
    matches = sorted(result_dir.glob("*_vbvr_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one result JSON under {result_dir}, found {matches}")
    return matches[0]


def _relative_url(path: Path, *, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _write_candidate_gallery(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    gallery_path = output_root / "candidate_gallery.html"
    base = gallery_path.parent
    cards = []
    for row in rows:
        videos = []
        for rollout in row["rollouts"]:
            source = _relative_url(Path(rollout["generated_path"]), base=base)
            videos.append(
                "<figure>"
                f'<video controls loop muted preload="metadata" src="{html.escape(source)}"></video>'
                f"<figcaption>rollout {rollout['rollout_index']} · seed {rollout['seed']} · "
                f"score {rollout['score']:.6f}</figcaption></figure>"
            )
        first_frame = _relative_url(Path(row["first_frame"]), base=base)
        ground_truth = _relative_url(Path(row["ground_truth"]), base=base)
        cards.append(
            f'<section id="{row["case_id"]}">'
            f"<h2>{row['case_id']} · checkpoint {row['checkpoint']} · {html.escape(row['task_name'])}</h2>"
            f"<p><b>score range:</b> {row['score_range']:.6f} · <b>mean:</b> {row['score_mean']:.6f} · "
            f"<b>visual changed fraction:</b> {row['diversity']['pairwise_changed_fraction_mean']:.6f}</p>"
            f"<p>{html.escape(row['prompt'])}</p>"
            '<div class="references">'
            f'<figure><img src="{html.escape(first_frame)}"><figcaption>input</figcaption></figure>'
            f'<figure><video controls loop muted preload="metadata" src="{html.escape(ground_truth)}"></video>'
            "<figcaption>ground truth</figcaption></figure></div>"
            f'<div class="rollouts">{"".join(videos)}</div></section>'
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>VBVR RL rollout candidates</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; margin: 24px; background: #f6f7f9; color: #17191d; }}
section {{ background: white; border-radius: 12px; padding: 18px; margin: 0 0 24px; box-shadow: 0 1px 5px #0002; }}
h2 {{ font-size: 18px; margin: 0 0 8px; }} p {{ max-width: 1200px; }}
.rollouts, .references {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
figure {{ margin: 0; }} video, img {{ display: block; width: 100%; background: #111; border-radius: 7px; }}
figcaption {{ margin-top: 5px; font-size: 13px; }}
</style></head><body><h1>VBVR RL rollout candidates</h1>
<p>Sorted by within-group EvalKit score range. All rollouts use exact Flow-CPS 0.7, 30 steps, CFG 1.0.</p>
{"".join(cards)}</body></html>
"""
    gallery_path.write_text(document, encoding="utf-8")
    return gallery_path


def _aggregate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = _json(manifest_path)
    output_root = manifest_path.parent
    rollout_count = int(manifest["contract"]["rollouts_per_case"])
    score_maps: list[dict[str, dict[str, Any]]] = []
    for rollout_index in range(rollout_count):
        result_path = _score_result_for_rollout(output_root, rollout_index)
        result = _json(result_path)
        samples = result.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Score result has no sample list: {result_path}")
        errors = [sample for sample in samples if sample.get("error")]
        if errors:
            raise ValueError(f"Score result contains {len(errors)} errors: {result_path}")
        mapping = {_canonical_result_name(sample): sample for sample in samples}
        if len(mapping) != int(manifest["case_count"]):
            raise ValueError(f"Score result has {len(mapping)} unique samples, expected {manifest['case_count']}")
        score_maps.append(mapping)

    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        canonical_name = str(case["canonical_name"])
        rollout_rows = []
        scores = []
        generated_paths = []
        for rollout in case["rollouts"]:
            rollout_index = int(rollout["rollout_index"])
            score_sample = score_maps[rollout_index].get(canonical_name)
            if score_sample is None:
                raise KeyError(f"Missing score for rollout {rollout_index}: {canonical_name}")
            score = float(score_sample["score"])
            scores.append(score)
            generated_path = Path(rollout["generated_path"])
            generated_paths.append(generated_path)
            rollout_rows.append({**rollout, "score": score, "score_dimensions": score_sample.get("dimensions", {})})
        diversity = _video_diversity(generated_paths)
        rows.append(
            {
                **{key: value for key, value in case.items() if key != "rollouts"},
                "rollouts": rollout_rows,
                "scores": scores,
                "score_min": min(scores),
                "score_max": max(scores),
                "score_mean": statistics.mean(scores),
                "score_std": statistics.pstdev(scores),
                "score_range": max(scores) - min(scores),
                "diversity": diversity,
            }
        )
    rows.sort(
        key=lambda row: (
            row["score_range"],
            row["diversity"]["pairwise_changed_fraction_mean"],
        ),
        reverse=True,
    )
    _write_json(output_root / "candidate_scores.json", {"schema_version": 1, "cases": rows})
    csv_path = output_root / "candidate_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "case_id",
                "checkpoint",
                "task_name",
                "canonical_name",
                *[f"score_{index}" for index in range(rollout_count)],
                "score_range",
                "score_mean",
                "score_std",
                "unique_video_sha256",
                "pairwise_changed_fraction_mean",
                "pairwise_changed_fraction_min",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    row["case_id"],
                    row["checkpoint"],
                    row["task_name"],
                    row["canonical_name"],
                    *row["scores"],
                    row["score_range"],
                    row["score_mean"],
                    row["score_std"],
                    row["diversity"]["unique_video_sha256"],
                    row["diversity"]["pairwise_changed_fraction_mean"],
                    row["diversity"]["pairwise_changed_fraction_min"],
                ]
            )
    gallery = _write_candidate_gallery(output_root, rows)
    print(f"Aggregated {len(rows)} cases: {csv_path}")
    print(f"Candidate gallery: {gallery}")


def _video_snapshots(path: Path, *, count: int = 5):
    import decord
    from PIL import Image

    reader = decord.VideoReader(str(path), num_threads=1)
    if not reader:
        raise ValueError(f"Video has no frames: {path}")
    indices = [round((len(reader) - 1) * index / (count - 1)) for index in range(count)]
    return [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]


def _thumbnail(image, *, width: int, height: int):
    from PIL import Image

    converted = image.convert("RGB")
    converted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#111111")
    canvas.paste(converted, ((width - converted.width) // 2, (height - converted.height) // 2))
    return canvas


def _audit(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    selection_path = Path(args.selection).resolve()
    output_dir = Path(args.output_dir).resolve()
    selection = _json(selection_path)
    if not isinstance(selection, list) or not selection:
        raise ValueError(f"Audit selection must be a non-empty list: {selection_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    regular = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    title = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    source_cache: dict[Path, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    index_cards = []
    audit_manifest = []

    for rank, item in enumerate(selection, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Audit selection item {rank} must be an object")
        source_root = Path(str(item["source_root"])).resolve()
        case_id = str(item["case_id"])
        case, _, _ = _compose_selected_case(item, source_cache)

        cell_width = 224
        cell_height = 224
        label_width = 250
        header_height = 180
        row_height = cell_height + 38
        columns = 5
        rows = 1 + len(case["rollouts"])
        canvas = Image.new("RGB", (label_width + columns * cell_width, header_height + rows * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        display_id = str(item.get("display_id", f"review_{rank:02d}"))
        draw.text(
            (18, 14),
            f"{display_id} | {case_id} | checkpoint {case['checkpoint']} | score range {case['score_range']:.6f}",
            fill="black",
            font=title,
        )
        draw.multiline_text(
            (18, 50),
            "\n".join(textwrap.wrap(str(case["task_name"]), width=100)),
            fill="#222222",
            font=regular,
            spacing=3,
        )
        prompt = "\n".join(textwrap.wrap(str(case["prompt"]), width=155)[:3])
        draw.multiline_text((18, 92), prompt, fill="#444444", font=small, spacing=2)
        for column, fraction in enumerate((0, 25, 50, 75, 100)):
            draw.text(
                (label_width + column * cell_width + 8, header_height - 25),
                f"t={fraction}%",
                fill="#555555",
                font=small,
            )

        rows_to_render = [("ground truth", Path(case["ground_truth"]), None)]
        rows_to_render.extend(
            (
                f"rollout {rollout['rollout_index']}\nseed {rollout['seed']}\nscore {rollout['score']:.6f}",
                Path(rollout["generated_path"]),
                rollout,
            )
            for rollout in case["rollouts"]
        )
        for row_index, (label, video_path, _) in enumerate(rows_to_render):
            top = header_height + row_index * row_height
            draw.multiline_text((16, top + 12), label, fill="black", font=regular, spacing=4)
            snapshots = _video_snapshots(video_path, count=columns)
            for column, snapshot in enumerate(snapshots):
                thumb = _thumbnail(snapshot, width=cell_width, height=cell_height)
                canvas.paste(thumb, (label_width + column * cell_width, top))

        sheet_path = output_dir / f"{rank:02d}_{display_id}_{case_id}.jpg"
        canvas.save(sheet_path, quality=92, subsampling=0)
        audit_manifest.append(
            {
                "rank": rank,
                "display_id": display_id,
                "source_root": str(source_root),
                "case_id": case_id,
                "checkpoint": case["checkpoint"],
                "task_name": case["task_name"],
                "scores": case["scores"],
                "score_range": case["score_range"],
                "rollout_overrides": item.get("rollout_overrides", []),
                "sheet": str(sheet_path),
            }
        )
        index_cards.append(
            f"<article><h2>{rank:02d}. {html.escape(display_id)} · {html.escape(case_id)} · "
            f"checkpoint {case['checkpoint']} · range {case['score_range']:.6f}</h2>"
            f'<a href="{sheet_path.name}"><img loading="lazy" src="{sheet_path.name}"></a></article>'
        )
        print(f"[{rank}/{len(selection)}] {sheet_path}")

    _write_json(output_dir / "audit_manifest.json", {"schema_version": 1, "cases": audit_manifest})
    index = f"""<!doctype html><html><head><meta charset="utf-8"><title>VBVR RL audit sheets</title>
<style>body{{font:14px system-ui;margin:24px;background:#f4f5f7}}article{{background:white;padding:14px;margin:18px 0}}
h2{{font-size:16px}}img{{width:min(100%,1100px);height:auto}}</style></head><body>
<h1>VBVR RL manual audit sheets</h1>{"".join(index_cards)}</body></html>"""
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"Audit index: {output_dir / 'index.html'}")


def _write_final_gallery(
    output_root: Path,
    cases: list[dict[str, Any]],
    contract: dict[str, Any],
    aggregate_evidence: dict[str, Any],
) -> Path:
    gallery_path = output_root / "index.html"
    cards = []
    navigation = []
    for case in cases:
        display_id = str(case["display_id"])
        navigation.append(
            f'<a href="#{html.escape(display_id)}">{case["rank"]:02d}. {html.escape(str(case["task_name"]))}</a>'
        )
        rollout_cards = []
        for rollout in case["rollouts"]:
            score = float(rollout["score"])
            hue = max(0.0, min(120.0, score * 120.0))
            rollout_cards.append(
                '<figure class="rollout">'
                f'<video controls loop muted preload="metadata" src="{html.escape(rollout["native_video"])}"></video>'
                f'<figcaption><span class="score" style="--hue:{hue:.1f}">{score:.6f}</span>'
                f" rollout {rollout['rollout_index']} · seed {rollout['seed']}</figcaption></figure>"
            )
        note = html.escape(str(case["manual_review"]["note"]))
        cards.append(
            f'<section id="{html.escape(display_id)}">'
            f'<div class="heading"><h2>{case["rank"]:02d}. checkpoint {case["checkpoint"]} · '
            f'{html.escape(str(case["task_name"]))}</h2><span class="range">Δ {case["score_range"]:.6f}</span></div>'
            f'<p class="meta">{html.escape(str(case["canonical_name"]))}</p>'
            f"<p>{html.escape(str(case['prompt']))}</p>"
            f'<p class="audit"><b>人工复核：</b>{note}</p>'
            '<div class="references">'
            f'<figure><img loading="lazy" src="{html.escape(case["input_image"])}">'
            "<figcaption>输入首帧</figcaption></figure>"
            '<figure><video controls loop muted preload="metadata" '
            f'src="{html.escape(case["ground_truth_video"])}"></video>'
            "<figcaption>标准答案</figcaption></figure></div>"
            f'<div class="rollouts">{"".join(rollout_cards)}</div>'
            "<details><summary>查看五时刻人工审核接触表</summary>"
            f'<img loading="lazy" src="{html.escape(case["audit_sheet"])}"></details>'
            "</section>"
        )
    cps = aggregate_evidence["cps_0p7"]
    aggregate_card = (
        '<section class="aggregate"><h2>总体证据：500 样本 checkpoint 曲线</h2>'
        f'<img src="{html.escape(aggregate_evidence["plot_png"])}">'
        f"<p>CPS 0.7 matched baseline {cps['baseline_overall']:.6f}；"
        f"最佳 checkpoint {cps['best_step']} 为 {cps['best_overall']:.6f}；"
        f"绝对提升 +{cps['best_delta']:.6f}。"
        f"每个 cell {aggregate_evidence['samples_per_cell']} 个样本，"
        f"完整 cell {aggregate_evidence['complete_cells']} 个，scorer error 为 0。</p></section>"
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VBVR-RL · RL 同输入四次 rollout demo</title><style>
:root {{ color-scheme: light; }}
body {{ font:15px/1.5 system-ui,sans-serif; margin:0; background:#f3f5f8; color:#18202a; }}
header {{ padding:32px max(24px,calc((100vw - 1500px)/2)); background:#172033; color:white; }}
header h1 {{ margin:0 0 8px; font-size:27px; }} header p {{ margin:5px 0; max-width:1100px; color:#d9e0ee; }}
nav {{ display:flex; gap:7px; flex-wrap:wrap; margin-top:16px; }}
nav a {{ color:#dbe9ff; background:#ffffff18; padding:5px 9px; border-radius:6px; text-decoration:none; }}
main {{ max-width:1500px; margin:24px auto; padding:0 20px; }}
section {{ background:white; border-radius:12px; padding:19px; margin:0 0 24px; box-shadow:0 1px 6px #15233b18; }}
.heading {{ display:flex; justify-content:space-between; gap:16px; align-items:start; }}
h2 {{ margin:0; font-size:18px; }}
.range {{ white-space:nowrap; font-weight:700; background:#eef3ff; padding:4px 8px; border-radius:6px; }}
.meta {{ color:#667085; font-family:ui-monospace,monospace; font-size:12px; }}
.audit {{ background:#f4fbf5; border-left:4px solid #44a15d; padding:8px 10px; }}
.aggregate {{ border:2px solid #85a8e8; }} .aggregate>img {{ max-width:100%; background:white; }}
.references {{ display:grid; grid-template-columns:repeat(2,minmax(220px,1fr)); gap:12px; max-width:750px; }}
.rollouts {{ display:grid; grid-template-columns:repeat(4,minmax(220px,1fr)); gap:12px; }}
figure {{ margin:0; }} video,img {{ display:block; width:100%; background:#10141b; border-radius:7px; }}
figcaption {{ margin-top:5px; font-size:13px; }}
.score {{ display:inline-block; background:hsl(var(--hue) 65% 42%); color:white;
padding:2px 7px; border-radius:999px; }}
details {{ margin-top:15px; }} details>img {{ margin-top:10px; max-width:1370px; }}
@media (max-width:1000px) {{ .rollouts {{ grid-template-columns:repeat(2,minmax(200px,1fr)); }} }}
@media (max-width:560px) {{ .rollouts,.references {{ grid-template-columns:1fr; }} }}
</style></head><body><header><h1>RL 同输入四次 Flow-CPS rollout：20 组演示</h1>
<p>checkpoint 300–2200；Flow-CPS={contract["cps_noise_level"]},
{contract["num_inference_steps"]} steps, CFG={contract["guidance_scale"]}；每组四个显式随机种子。</p>
<p>分数由固定 e140 EvalKit 在 1024×1024、81 帧、16 FPS 输入上计算；
页面视频为对应的原生 512×512 生成结果。Δ 是组内最高分减最低分。</p>
<nav>{"".join(navigation)}</nav></header><main>{aggregate_card}{"".join(cards)}</main></body></html>"""
    gallery_path.write_text(document, encoding="utf-8")
    return gallery_path


def _package(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    output_root = Path(args.output_root).resolve()
    selection = _json(selection_path)
    if not isinstance(selection, list) or len(selection) != args.expected_cases:
        raise ValueError(
            f"Final selection must contain exactly {args.expected_cases} items, got "
            f"{len(selection) if isinstance(selection, list) else type(selection).__name__}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    trend_root = Path(args.trend_root).resolve()
    trend_summary_path = trend_root / "sampler_checkpoint_trend_summary.json"
    trend_plot_path = trend_root / "sampler_checkpoint_trends.png"
    trend_svg_path = trend_root / "sampler_checkpoint_trends.svg"
    trend_data_path = trend_root / "sampler_checkpoint_scores.csv"
    for path in (trend_summary_path, trend_plot_path, trend_svg_path, trend_data_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing aggregate checkpoint evidence: {path}")
    trend_summary = _json(trend_summary_path)
    score_contract = trend_summary["score_contract"]
    if int(score_contract["samples_per_cell"]) != 500 or int(score_contract["scorer_errors"]) != 0:
        raise ValueError(f"Aggregate evidence is incomplete or contains scorer errors: {trend_summary_path}")
    cps_summary = trend_summary["samplers"]["cps_0.7"]
    aggregate_destinations = {}
    for source in (trend_summary_path, trend_plot_path, trend_svg_path, trend_data_path):
        destination = output_root / "evidence" / source.name
        _hardlink_or_copy(source, destination)
        aggregate_destinations[source.name] = destination
    aggregate_evidence = {
        "source_root": str(trend_root),
        "evalkit_source_sha256": str(score_contract["evalkit_source_sha256"]),
        "samples_per_cell": int(score_contract["samples_per_cell"]),
        "complete_cells": int(score_contract["complete_cells"]),
        "scorer_errors": int(score_contract["scorer_errors"]),
        "plot_png": _relative_url(aggregate_destinations[trend_plot_path.name], base=output_root),
        "plot_png_sha256": _sha256(trend_plot_path),
        "plot_svg": _relative_url(aggregate_destinations[trend_svg_path.name], base=output_root),
        "plot_svg_sha256": _sha256(trend_svg_path),
        "score_csv": _relative_url(aggregate_destinations[trend_data_path.name], base=output_root),
        "score_csv_sha256": _sha256(trend_data_path),
        "summary_json": _relative_url(aggregate_destinations[trend_summary_path.name], base=output_root),
        "summary_json_sha256": _sha256(trend_summary_path),
        "cps_0p7": {
            "baseline_overall": float(cps_summary["baseline_overall"]),
            "best_overall": float(cps_summary["best_overall"]),
            "best_step": int(cps_summary["best_step"]),
            "best_delta": float(cps_summary["best_delta_vs_matched_baseline"]),
            "latest_step": int(cps_summary["latest_step"]),
            "latest_overall": float(cps_summary["latest_overall"]),
            "latest_delta": float(cps_summary["latest_delta_vs_matched_baseline"]),
        },
    }
    source_cache: dict[Path, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    contracts: list[dict[str, Any]] = []
    seed_bases: set[int] = set()
    final_cases: list[dict[str, Any]] = []
    seen_cases: set[tuple[Path, str]] = set()
    seen_canonical: set[str] = set()
    source_records: dict[Path, dict[str, Any]] = {}

    for rank, item in enumerate(selection, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Final selection item {rank} must be an object")
        source_root = Path(str(item["source_root"])).resolve()
        case_id = str(item["case_id"])
        key = (source_root, case_id)
        if key in seen_cases:
            raise ValueError(f"Duplicate final case: {source_root} / {case_id}")
        seen_cases.add(key)
        if str(item.get("manual_review")) != "pass":
            raise ValueError(f"Final case {rank} has not passed manual review: {item.get('manual_review')!r}")
        audit_sheet = Path(str(item["audit_sheet"])).resolve()
        if not audit_sheet.is_file():
            raise FileNotFoundError(f"Missing manual audit sheet: {audit_sheet}")

        case, candidate_manifest, referenced_roots = _compose_selected_case(item, source_cache)
        contracts.append(candidate_manifest["contract"])
        for referenced_root in referenced_roots:
            referenced_manifest, _ = _load_candidate_source(referenced_root, source_cache)
            seed_bases.add(int(referenced_manifest["contract"]["seed_base"]))
            if referenced_root not in source_records:
                candidate_manifest_path = referenced_root / "candidate_manifest.json"
                candidate_scores_path = referenced_root / "candidate_scores.json"
                source_records[referenced_root] = {
                    "candidate_root": str(referenced_root),
                    "candidate_manifest": str(candidate_manifest_path),
                    "candidate_manifest_sha256": _sha256(candidate_manifest_path),
                    "candidate_scores": str(candidate_scores_path),
                    "candidate_scores_sha256": _sha256(candidate_scores_path),
                }
        canonical_name = str(case["canonical_name"])
        if canonical_name in seen_canonical:
            raise ValueError(f"Final selection repeats canonical sample: {canonical_name}")
        seen_canonical.add(canonical_name)
        if len(case["rollouts"]) != 4 or int(candidate_manifest["contract"]["rollouts_per_case"]) != 4:
            raise ValueError(f"Final case does not contain exactly four rollouts: {case_id}")
        if int(case["diversity"]["unique_video_sha256"]) != 4:
            raise ValueError(f"Final case contains byte-identical rollouts: {case_id}")

        display_id = f"demo_{rank:02d}"
        case_dir = output_root / "cases" / display_id
        input_destination = case_dir / "input.png"
        ground_truth_destination = case_dir / "ground_truth.mp4"
        audit_destination = case_dir / "manual_audit.jpg"
        _hardlink_or_copy(Path(case["first_frame"]), input_destination)
        _hardlink_or_copy(Path(case["ground_truth"]), ground_truth_destination)
        _hardlink_or_copy(audit_sheet, audit_destination)
        packaged_rollouts = []
        for rollout in case["rollouts"]:
            rollout_index = int(rollout["rollout_index"])
            native_source = Path(rollout["generated_path"])
            scored_source = Path(rollout["prepared_path"])
            native_destination = case_dir / f"rollout_{rollout_index:02d}_native.mp4"
            scored_destination = case_dir / f"rollout_{rollout_index:02d}_scored.mp4"
            verification_destination = (
                output_root / "verification_inputs" / f"rollout_{rollout_index:02d}" / f"{canonical_name}.mp4"
            )
            _hardlink_or_copy(native_source, native_destination)
            _hardlink_or_copy(scored_source, scored_destination)
            _hardlink_or_copy(scored_source, verification_destination)
            packaged_rollouts.append(
                {
                    "rollout_index": rollout_index,
                    "seed": int(rollout["seed"]),
                    "score": float(rollout["score"]),
                    "score_dimensions": rollout.get("score_dimensions", {}),
                    "selection_source_root": str(rollout["selection_source_root"]),
                    "selection_source_case_id": str(rollout["selection_source_case_id"]),
                    "selection_source_rollout_index": int(rollout["selection_source_rollout_index"]),
                    "native_video": _relative_url(native_destination, base=output_root),
                    "native_sha256": _sha256(native_source),
                    "scored_video": _relative_url(scored_destination, base=output_root),
                    "scored_sha256": _sha256(scored_source),
                    "verification_input": _relative_url(verification_destination, base=output_root),
                }
            )
        final_cases.append(
            {
                "rank": rank,
                "display_id": display_id,
                "source_display_id": str(item.get("display_id", case_id)),
                "source_root": str(source_root),
                "source_case_id": case_id,
                "rollout_overrides": item.get("rollout_overrides", []),
                "checkpoint": int(case["checkpoint"]),
                "sample_index": int(case["sample_index"]),
                "canonical_name": canonical_name,
                "domain": case.get("domain"),
                "task_name": case.get("task_name"),
                "video_idx": case.get("video_idx"),
                "prompt": case.get("prompt"),
                "input_image": _relative_url(input_destination, base=output_root),
                "input_sha256": _sha256(input_destination),
                "ground_truth_video": _relative_url(ground_truth_destination, base=output_root),
                "ground_truth_sha256": _sha256(ground_truth_destination),
                "audit_sheet": _relative_url(audit_destination, base=output_root),
                "manual_review": {
                    "status": "pass",
                    "reviewed_at": str(item["reviewed_at"]),
                    "note": str(item["manual_note"]),
                },
                "scores": [float(value) for value in case["scores"]],
                "score_min": float(case["score_min"]),
                "score_max": float(case["score_max"]),
                "score_mean": float(case["score_mean"]),
                "score_std": float(case["score_std"]),
                "score_range": float(case["score_range"]),
                "diversity": case["diversity"],
                "inference_load_mode": case.get("inference_load_mode"),
                "rollouts": packaged_rollouts,
            }
        )

    canonical_contract = {key: value for key, value in contracts[0].items() if key != "seed_base"}
    comparable_contracts = [
        {key: value for key, value in contract.items() if key != "seed_base"} for contract in contracts
    ]
    if any(contract != canonical_contract for contract in comparable_contracts[1:]):
        raise ValueError("Candidate roots do not share one generation contract")
    canonical_contract["seed_bases"] = sorted(seed_bases)
    score_ranges = [float(case["score_range"]) for case in final_cases]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Presentation-ready same-input four-rollout evidence for Flow-CPS RL behavior",
        "contract": canonical_contract,
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "manual_review_required": True,
        },
        "sources": list(source_records.values()),
        "aggregate_evidence": aggregate_evidence,
        "case_count": len(final_cases),
        "video_count": len(final_cases) * 4,
        "checkpoints": sorted({int(case["checkpoint"]) for case in final_cases}),
        "score_range_summary": {
            "minimum": min(score_ranges),
            "median": statistics.median(score_ranges),
            "mean": statistics.mean(score_ranges),
            "maximum": max(score_ranges),
            "count_at_least_0p5": sum(value >= 0.5 for value in score_ranges),
        },
        "cases": final_cases,
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    csv_path = output_root / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "display_id",
                "checkpoint",
                "task_name",
                "canonical_name",
                "score_0",
                "score_1",
                "score_2",
                "score_3",
                "score_range",
                "score_mean",
                "manual_review",
            ]
        )
        for case in final_cases:
            writer.writerow(
                [
                    case["rank"],
                    case["display_id"],
                    case["checkpoint"],
                    case["task_name"],
                    case["canonical_name"],
                    *case["scores"],
                    case["score_range"],
                    case["score_mean"],
                    case["manual_review"]["status"],
                ]
            )
    gallery = _write_final_gallery(output_root, final_cases, canonical_contract, aggregate_evidence)
    print(f"Packaged {len(final_cases)} cases / {len(final_cases) * 4} videos: {manifest_path}")
    print(f"Gallery: {gallery}")


def _verify(args: argparse.Namespace) -> None:
    from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256

    manifest_path = Path(args.manifest).resolve()
    manifest = _json(manifest_path)
    output_root = manifest_path.parent
    evalkit_dir = Path(args.evalkit_dir).resolve()
    actual_evalkit_sha256 = evalkit_source_sha256(evalkit_dir)
    if actual_evalkit_sha256 != args.expected_evalkit_source_sha256:
        raise ValueError(
            f"EvalKit source mismatch: expected={args.expected_evalkit_source_sha256}, actual={actual_evalkit_sha256}"
        )
    checks = []
    result_records = []
    max_score_delta = 0.0
    for rollout_index in range(4):
        result_dir = output_root / "verification_scores" / f"rollout_{rollout_index:02d}"
        matches = sorted(result_dir.glob("*_vbvr_results.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one independent result under {result_dir}, found {matches}")
        result_path = matches[0]
        result = _json(result_path)
        samples = result.get("samples")
        if not isinstance(samples, list) or len(samples) != int(manifest["case_count"]):
            raise ValueError(f"Independent result has the wrong sample count: {result_path}")
        errors = [sample for sample in samples if sample.get("error")]
        if errors:
            raise ValueError(f"Independent result contains {len(errors)} scorer errors: {result_path}")
        mapping = {_canonical_result_name(sample): sample for sample in samples}
        for case in manifest["cases"]:
            canonical_name = str(case["canonical_name"])
            sample = mapping.get(canonical_name)
            if sample is None:
                raise KeyError(f"Independent result is missing {canonical_name}")
            expected_score = float(case["rollouts"][rollout_index]["score"])
            actual_score = float(sample["score"])
            delta = abs(expected_score - actual_score)
            max_score_delta = max(max_score_delta, delta)
            if delta > args.score_tolerance:
                raise ValueError(
                    f"Independent score mismatch for rollout {rollout_index}, {canonical_name}: "
                    f"expected={expected_score}, actual={actual_score}, delta={delta}"
                )
            verification_input = output_root / case["rollouts"][rollout_index]["verification_input"]
            expected_sha256 = str(case["rollouts"][rollout_index]["scored_sha256"])
            actual_sha256 = _sha256(verification_input)
            if actual_sha256 != expected_sha256:
                raise ValueError(f"Verification input changed: {verification_input}")
            checks.append(
                {
                    "display_id": case["display_id"],
                    "canonical_name": canonical_name,
                    "rollout_index": rollout_index,
                    "expected_score": expected_score,
                    "independent_score": actual_score,
                    "absolute_delta": delta,
                    "input_sha256": actual_sha256,
                }
            )
        result_records.append(
            {
                "rollout_index": rollout_index,
                "result": str(result_path),
                "result_sha256": _sha256(result_path),
                "sample_count": len(samples),
                "error_count": 0,
            }
        )
    verification = {
        "schema_version": 1,
        "status": "pass",
        "verified_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "evalkit_dir": str(evalkit_dir),
        "evalkit_source_sha256": actual_evalkit_sha256,
        "score_tolerance": args.score_tolerance,
        "case_count": int(manifest["case_count"]),
        "score_check_count": len(checks),
        "scorer_error_count": 0,
        "max_absolute_score_delta": max_score_delta,
        "results": result_records,
        "checks": checks,
    }
    verification_path = output_root / "verification.json"
    _write_json(verification_path, verification)
    print(
        f"PASS: independently reproduced {len(checks)} scores with max delta {max_score_delta:.3g}; {verification_path}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build per-checkpoint same-input rollout job JSON files")
    build.add_argument("--selection", required=True)
    build.add_argument("--eval-json", required=True)
    build.add_argument("--formal-root", required=True)
    build.add_argument("--checkpoint-root", required=True)
    build.add_argument("--converted-root", required=True)
    build.add_argument("--converted-prefix", required=True)
    build.add_argument("--base-model", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--rollouts", type=int, default=4)
    build.add_argument("--seed-base", type=int, default=2_026_081_800)
    build.add_argument("--cps-noise-level", type=float, default=0.7)
    build.add_argument("--height", type=int, default=512)
    build.add_argument("--width", type=int, default=512)
    build.add_argument("--num-frames", type=int, default=81)
    build.add_argument("--fps", type=int, default=16)
    build.add_argument("--num-inference-steps", type=int, default=30)
    build.add_argument("--guidance-scale", type=float, default=1.0)
    build.set_defaults(func=_build)

    stage = subparsers.add_parser("stage", help="Stage prepared videos into canonical EvalKit trees")
    stage.add_argument("--manifest", required=True)
    stage.set_defaults(func=_stage)

    aggregate = subparsers.add_parser("aggregate", help="Combine rollout scores and compute visual diversity")
    aggregate.add_argument("--manifest", required=True)
    aggregate.set_defaults(func=_aggregate)

    audit = subparsers.add_parser("audit", help="Render temporal contact sheets for manual score review")
    audit.add_argument("--selection", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.set_defaults(func=_audit)

    package = subparsers.add_parser("package", help="Package a manually approved final demo gallery")
    package.add_argument("--selection", required=True)
    package.add_argument("--output-root", required=True)
    package.add_argument("--trend-root", required=True)
    package.add_argument("--expected-cases", type=int, default=20)
    package.set_defaults(func=_package)

    verify = subparsers.add_parser("verify", help="Compare an independent EvalKit pass with packaged scores")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--evalkit-dir", required=True)
    verify.add_argument("--expected-evalkit-source-sha256", required=True)
    verify.add_argument("--score-tolerance", type=float, default=1e-12)
    verify.set_defaults(func=_verify)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if hasattr(args, "rollouts") and args.rollouts < 2:
        raise ValueError("--rollouts must be at least 2")
    if hasattr(args, "seed_base") and args.seed_base < 0:
        raise ValueError("--seed-base must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
