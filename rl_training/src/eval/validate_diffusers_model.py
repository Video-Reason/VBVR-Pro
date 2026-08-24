"""Validate a converted Diffusers model without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open


class ModelValidationError(ValueError):
    """Raised when a converted model is incomplete or internally inconsistent."""


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelValidationError(f"expected a JSON object in {path}")
    return value


def validate_diffusers_model(model_dir: Path) -> dict[str, int]:
    """Check component layout, shard indexes, and safetensors headers."""
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise ModelValidationError(f"model directory does not exist: {root}")

    model_index_path = root / "model_index.json"
    if not model_index_path.is_file():
        raise ModelValidationError(f"model_index.json is missing: {model_index_path}")
    model_index = _read_json_object(model_index_path)

    component_count = 0
    for name, specification in model_index.items():
        if name.startswith("_") or name in {"boundary_ratio", "expand_timesteps"}:
            continue
        if not isinstance(specification, list) or len(specification) != 2:
            continue
        if specification == [None, None]:
            continue
        component = root / name
        if not component.is_dir():
            raise ModelValidationError(f"referenced component directory is missing: {component}")
        component_count += 1

    expected_keys: dict[Path, set[str]] = {}
    index_count = 0
    for index_path in sorted(root.rglob("*.safetensors.index.json")):
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelValidationError(f"weight_map is missing or empty: {index_path}")
        index_count += 1
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise ModelValidationError(f"invalid weight_map entry in {index_path}")
            shard = (index_path.parent / shard_name).resolve()
            try:
                shard.relative_to(root)
            except ValueError as exc:
                raise ModelValidationError(f"shard escapes model directory: {shard}") from exc
            expected_keys.setdefault(shard, set()).add(tensor_name)

    safetensor_paths = sorted(path.resolve() for path in root.rglob("*.safetensors"))
    if not safetensor_paths:
        raise ModelValidationError(f"model contains no safetensors files: {root}")
    actual_shards = set(safetensor_paths)
    missing_shards = sorted(set(expected_keys) - actual_shards)
    if missing_shards:
        raise ModelValidationError(f"indexed safetensors shard is missing: {missing_shards[0]}")

    tensor_count = 0
    for shard in safetensor_paths:
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
        except Exception as exc:
            raise ModelValidationError(f"invalid safetensors file {shard}: {exc}") from exc
        if not actual_keys:
            raise ModelValidationError(f"safetensors file has no tensors: {shard}")
        indexed_keys = expected_keys.get(shard)
        if indexed_keys is not None and actual_keys != indexed_keys:
            missing = sorted(indexed_keys - actual_keys)
            extra = sorted(actual_keys - indexed_keys)
            raise ModelValidationError(
                f"safetensors index mismatch for {shard}: missing={missing[:1]} extra={extra[:1]}"
            )
        tensor_count += len(actual_keys)

    return {
        "components": component_count,
        "indexes": index_count,
        "safetensors": len(safetensor_paths),
        "tensors": tensor_count,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = validate_diffusers_model(args.model_dir)
    except ModelValidationError as exc:
        if not args.quiet:
            print(f"[error] {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            "Validated converted model: "
            f"components={summary['components']} indexes={summary['indexes']} "
            f"shards={summary['safetensors']} tensors={summary['tensors']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
