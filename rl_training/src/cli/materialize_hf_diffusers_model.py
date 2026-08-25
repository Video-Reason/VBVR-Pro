"""Materialize and validate an immutable Hugging Face Diffusers snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from src.eval.validate_diffusers_model import validate_diffusers_model

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_METADATA_NAME = "conversion_metadata.json"


def _repo_id(value: str) -> str:
    if not _REPO_ID_RE.fullmatch(value) or ".." in value:
        raise argparse.ArgumentTypeError(f"expected a Hugging Face model repo id like owner/name, got {value!r}")
    return value


def _commit_sha(value: str) -> str:
    normalized = value.lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError(f"revision must be a full 40-hex commit SHA, got {value!r}")
    return normalized


def _sha256_digest(value: str) -> str:
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError(f"pipeline SHA-256 must be 64 lowercase hex characters, got {value!r}")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_metadata(
    *,
    repo_id: str,
    revision: str,
    pipeline_sha256: str,
    model_summary: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "huggingface_diffusers_snapshot",
        "repo_id": repo_id,
        "revision": revision,
        "source_uri": f"https://huggingface.co/{repo_id}/tree/{revision}",
        "pipeline_file": "pipeline.py",
        "pipeline_sha256": pipeline_sha256,
        "model_summary": dict(sorted(model_summary.items())),
        "materializer": "src.cli.materialize_hf_diffusers_model",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read import metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"import metadata must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_materialized_model(
    output: Path,
    *,
    repo_id: str,
    revision: str,
    pipeline_sha256: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    pipeline_path = output / "pipeline.py"
    if not pipeline_path.is_file():
        raise ValueError(f"pipeline.py is missing from the materialized snapshot: {pipeline_path}")
    actual_pipeline_sha256 = _sha256_file(pipeline_path)
    if actual_pipeline_sha256 != pipeline_sha256:
        raise ValueError(
            "reviewed pipeline SHA-256 mismatch: "
            f"expected={pipeline_sha256} actual={actual_pipeline_sha256} path={pipeline_path}"
        )
    model_summary = validate_diffusers_model(output)
    metadata = _expected_metadata(
        repo_id=repo_id,
        revision=revision,
        pipeline_sha256=pipeline_sha256,
        model_summary=model_summary,
    )
    return model_summary, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=_repo_id, required=True)
    parser.add_argument("--revision", type=_commit_sha, required=True)
    parser.add_argument("--pipeline-sha256", type=_sha256_digest, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    metadata_path = output / _METADATA_NAME

    if args.dry_run:
        print(
            f"[dry-run] repo_id={args.repo_id} revision={args.revision} "
            f"pipeline_sha256={args.pipeline_sha256} output={output}"
        )
        return 0

    if metadata_path.is_file():
        model_summary, expected_metadata = _verify_materialized_model(
            output,
            repo_id=args.repo_id,
            revision=args.revision,
            pipeline_sha256=args.pipeline_sha256,
        )
        actual_metadata = _read_json_object(metadata_path)
        if actual_metadata != expected_metadata:
            raise ValueError(
                f"existing import metadata does not match the requested immutable snapshot: {metadata_path}"
            )
        print(
            f"Reused validated Hugging Face snapshot at {output}: "
            f"components={model_summary['components']} tensors={model_summary['tensors']}"
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output,
        local_files_only=args.local_files_only,
        max_workers=args.max_workers,
    )
    if Path(snapshot_path).resolve() != output:
        raise RuntimeError(f"snapshot_download returned an unexpected path: {snapshot_path} != {output}")

    model_summary, metadata = _verify_materialized_model(
        output,
        repo_id=args.repo_id,
        revision=args.revision,
        pipeline_sha256=args.pipeline_sha256,
    )
    _write_json_atomic(metadata_path, metadata)
    print(
        f"Materialized Hugging Face snapshot at {output}: "
        f"components={model_summary['components']} tensors={model_summary['tensors']}"
    )
    print(f"Import provenance: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
