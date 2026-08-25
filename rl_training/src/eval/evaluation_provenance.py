"""Write and verify deterministic provenance manifests for evaluation stages.

The CLI records scalar settings plus fingerprints of input files and trees. It
is intentionally generic so launchers can mark a stage ``in_progress`` before
publishing outputs and ``complete`` only after their own semantic validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Provenance input file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def fingerprint_tree(path: Path) -> dict[str, object]:
    """Fingerprint a tree from relative paths, file sizes, mtimes, and links."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Provenance input tree does not exist: {resolved}")

    digest = hashlib.sha256()
    entries = 0
    total_size = 0
    for entry in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        relative = entry.relative_to(resolved).as_posix()
        if entry.is_symlink():
            payload = f"L\0{relative}\0{os.readlink(entry)}\n".encode()
        elif entry.is_file():
            stat = entry.stat()
            entries += 1
            total_size += stat.st_size
            payload = f"F\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        elif entry.is_dir():
            payload = f"D\0{relative}\n".encode()
        else:
            continue
        digest.update(payload)
    return {
        "path": str(resolved),
        "entries": entries,
        "total_size": total_size,
        "sha256": digest.hexdigest(),
    }


def fingerprint_media_tree(path: Path) -> dict[str, object]:
    """Fingerprint MP4 outputs without including non-media temporary files."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Provenance media tree does not exist: {resolved}")

    digest = hashlib.sha256()
    entries = 0
    total_size = 0
    for entry in sorted(resolved.rglob("*.mp4"), key=lambda item: item.relative_to(resolved).as_posix()):
        if not entry.is_file():
            continue
        relative = entry.relative_to(resolved).as_posix()
        stat = entry.stat()
        entries += 1
        total_size += stat.st_size
        digest.update(f"F\0{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return {
        "path": str(resolved),
        "entries": entries,
        "total_size": total_size,
        "sha256": digest.hexdigest(),
    }


def _parse_pairs(values: list[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"{option} expects KEY=VALUE, got {value!r}")
        if key in parsed:
            raise ValueError(f"Duplicate provenance key {key!r} for {option}")
        parsed[key] = item
    return parsed


def build_manifest(
    *,
    stage: str,
    values: dict[str, str],
    files: dict[str, str],
    trees: dict[str, str],
    output_files: dict[str, str] | None = None,
    output_trees: dict[str, str] | None = None,
    media_trees: dict[str, str] | None = None,
) -> dict[str, object]:
    output_files = output_files or {}
    output_trees = output_trees or {}
    media_trees = media_trees or {}
    return {
        "schema_version": _SCHEMA_VERSION,
        "stage": stage,
        "values": dict(sorted(values.items())),
        "files": {key: fingerprint_file(Path(path)) for key, path in sorted(files.items())},
        "trees": {key: fingerprint_tree(Path(path)) for key, path in sorted(trees.items())},
        "output_files": {key: fingerprint_file(Path(path)) for key, path in sorted(output_files.items())},
        "output_trees": {key: fingerprint_tree(Path(path)) for key, path in sorted(output_trees.items())},
        "media_trees": {key: fingerprint_media_tree(Path(path)) for key, path in sorted(media_trees.items())},
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_recorded_manifest(
    path: Path,
    *,
    expected_stage: str | None = None,
    require_complete: bool = False,
    sections: tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    """Recompute the artifact fingerprints stored in an existing manifest.

    Unlike :func:`manifest_matches`, this does not require callers to rebuild
    the original CLI arguments. It is intended for consumers such as sweep
    skip logic and report generation, which must not trust a provenance label
    after its recorded result or media artifacts have been replaced.
    """
    source = path.expanduser().resolve()
    if not source.is_file():
        return False, f"provenance manifest is missing: {source}"
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read provenance manifest {source}: {exc}"
    if not isinstance(manifest, dict):
        return False, f"provenance manifest is not a JSON object: {source}"
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        return False, f"unsupported provenance schema in {source}: {manifest.get('schema_version')!r}"
    if expected_stage is not None and manifest.get("stage") != expected_stage:
        return False, (
            f"provenance stage mismatch in {source}: expected={expected_stage!r}, actual={manifest.get('stage')!r}"
        )
    values = manifest.get("values")
    if not isinstance(values, dict):
        return False, f"provenance values are missing or invalid: {source}"
    if require_complete and values.get("state") != "complete":
        return False, f"provenance manifest is not complete: {source}"

    fingerprinters = {
        "files": fingerprint_file,
        "trees": fingerprint_tree,
        "output_files": fingerprint_file,
        "output_trees": fingerprint_tree,
        "media_trees": fingerprint_media_tree,
    }
    selected = sections or tuple(fingerprinters)
    unknown = [section for section in selected if section not in fingerprinters]
    if unknown:
        return False, f"unknown provenance artifact section {unknown[0]!r}"

    for section in selected:
        artifacts = manifest.get(section)
        if not isinstance(artifacts, dict):
            return False, f"provenance section {section!r} is missing or invalid: {source}"
        fingerprint = fingerprinters[section]
        for name, recorded in sorted(artifacts.items()):
            if not isinstance(recorded, dict) or not isinstance(recorded.get("path"), str):
                return False, f"invalid recorded artifact {section}.{name} in {source}"
            try:
                current = fingerprint(Path(recorded["path"]))
            except Exception as exc:
                return False, f"could not fingerprint {section}.{name} from {source}: {exc}"
            if current != recorded:
                return False, f"recorded artifact changed: {section}.{name} in {source}"
    return True, ""


def manifest_matches(path: Path, expected: dict[str, object], *, inputs_only: bool = False) -> tuple[bool, str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        return False, f"provenance manifest is missing: {source}"
    try:
        actual = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read provenance manifest {source}: {exc}"
    if not isinstance(actual, dict):
        return False, f"provenance manifest is not a JSON object: {source}"
    sections = ("schema_version", "stage", "values", "files", "trees")
    if inputs_only and all(actual.get(section) == expected.get(section) for section in sections):
        return True, ""
    if not inputs_only and actual == expected:
        return True, ""

    changed: list[str] = []
    compared_sections = sections if inputs_only else (*sections, "output_files", "output_trees", "media_trees")
    for section in compared_sections:
        if actual.get(section) != expected.get(section):
            changed.append(section)
    detail = ", ".join(changed) if changed else "schema or unknown fields"
    return False, f"provenance mismatch in {detail}: {source}"


def promote_manifest(
    path: Path,
    expected_in_progress: dict[str, object],
    complete: dict[str, object],
) -> tuple[bool, str]:
    """Promote the exact recorded input snapshot after re-verifying it."""
    matches, detail = manifest_matches(path, expected_in_progress)
    if not matches:
        return False, detail

    source = path.expanduser().resolve()
    recorded = json.loads(source.read_text(encoding="utf-8"))
    promoted = dict(recorded)
    promoted["values"] = dict(recorded["values"])
    promoted["values"]["state"] = "complete"
    for section in ("output_files", "output_trees", "media_trees"):
        promoted[section] = complete[section]
    write_manifest(source, promoted)
    return True, ""


def refresh_manifest_outputs(
    path: Path,
    expected_complete: dict[str, object],
) -> tuple[bool, str]:
    """Refresh output fingerprints while preserving exact recorded inputs.

    This is deliberately narrower than ``write``: the existing manifest must
    already be a complete record for the exact same stage, values, files, and
    trees. Callers are responsible for semantically validating outputs and
    ensuring that no writer is active before using this recovery operation.
    """
    matches, detail = manifest_matches(path, expected_complete, inputs_only=True)
    if not matches:
        return False, detail

    source = path.expanduser().resolve()
    recorded = json.loads(source.read_text(encoding="utf-8"))
    refreshed = dict(recorded)
    for section in ("output_files", "output_trees", "media_trees"):
        refreshed[section] = expected_complete[section]
    write_manifest(source, refreshed)
    return True, ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["check", "check-inputs", "write", "promote", "refresh-outputs"],
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--value", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--tree", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--output-file", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--output-tree", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--media-tree", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--quiet", action="store_true", help="Suppress check failure details")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        values = _parse_pairs(args.value, option="--value")
        files = _parse_pairs(args.file, option="--file")
        trees = _parse_pairs(args.tree, option="--tree")
        output_files = _parse_pairs(args.output_file, option="--output-file")
        output_trees = _parse_pairs(args.output_tree, option="--output-tree")
        media_trees = _parse_pairs(args.media_tree, option="--media-tree")
        manifest = build_manifest(
            stage=args.stage,
            values=values,
            files=files,
            trees=trees,
            output_files=output_files,
            output_trees=output_trees,
            media_trees=media_trees,
        )
        if args.mode == "write":
            write_manifest(args.manifest, manifest)
            print(f"Wrote {args.stage} provenance to {args.manifest}")
            return 0
        if args.mode == "promote":
            state = values.get("state", "")
            if not state.startswith("in_progress"):
                raise ValueError("promote requires an in-progress state value")
            expected_in_progress = build_manifest(stage=args.stage, values=values, files=files, trees=trees)
            complete_values = dict(values)
            complete_values["state"] = "complete"
            complete = build_manifest(
                stage=args.stage,
                values=complete_values,
                files=files,
                trees=trees,
                output_files=output_files,
                output_trees=output_trees,
                media_trees=media_trees,
            )
            matches, detail = promote_manifest(args.manifest, expected_in_progress, complete)
        elif args.mode == "refresh-outputs":
            if values.get("state") != "complete":
                raise ValueError("refresh-outputs requires state=complete")
            matches, detail = refresh_manifest_outputs(args.manifest, manifest)
        else:
            matches, detail = manifest_matches(args.manifest, manifest, inputs_only=args.mode == "check-inputs")
        if not matches and not args.quiet:
            print(f"[error] {detail}", file=sys.stderr)
        return 0 if matches else 1
    except Exception as exc:
        if not args.quiet:
            print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
