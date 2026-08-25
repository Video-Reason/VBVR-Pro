"""Symlink existing task directories into the split layout that VBVR-EvalKit expects.

Our inference writes::

    $MODEL_OUT/$SOURCE_SPLIT/{task_name}/{00000..00004}.mp4

but run_evaluation.py hard-expects::

    $MODEL_OUT/In-Domain_50/{task_name}/{idx}.mp4
    $MODEL_OUT/Out-of-Domain_50/{task_name}/{idx}.mp4

This helper reads the split classifier from an explicit external EvalKit
checkout and symlinks each task into the correct split. Idempotent: re-runs
are safe.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _module_is_from(module, directory: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(directory)
    except ValueError:
        return False
    return True


def _load_split_classifier(evalkit_dir: Path):
    directory = evalkit_dir.expanduser().resolve()
    if not (directory / "vbvr_bench").is_dir():
        raise FileNotFoundError(f"EvalKit checkout does not contain vbvr_bench: {directory}")
    root_module = sys.modules.get("vbvr_bench")
    if root_module is not None and not _module_is_from(root_module, directory):
        for name in list(sys.modules):
            if name == "vbvr_bench" or name.startswith("vbvr_bench."):
                sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    module = importlib.import_module("vbvr_bench.evaluators")
    if not _module_is_from(module, directory):
        raise ImportError(f"EvalKit split classifier resolved outside the requested checkout: {module.__file__}")
    classifier = getattr(module, "is_out_of_domain", None)
    if classifier is None:
        raise ImportError(f"EvalKit split classifier is missing from {directory / 'vbvr_bench/evaluators'}")
    return classifier


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_out", required=True, type=Path, help="Video directory root")
    ap.add_argument("--evalkit_dir", required=True, type=Path, help="External EvalKit checkout")
    ap.add_argument(
        "--source_split",
        required=True,
        help="Name of the existing subdir containing task folders (e.g. Open_60)",
    )
    args = ap.parse_args()
    try:
        is_out_of_domain = _load_split_classifier(args.evalkit_dir)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    model_out: Path = args.model_out.resolve()
    src_dir = model_out / args.source_split
    if not src_dir.is_dir():
        print(f"[error] source split dir not found: {src_dir}", file=sys.stderr)
        return 1

    in_dir = model_out / "In-Domain_50"
    out_dir = model_out / "Out-of-Domain_50"
    in_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    linked = skipped = 0
    for task_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        task = task_dir.name
        dst_parent = out_dir if is_out_of_domain(task) else in_dir
        dst = dst_parent / task

        if dst.is_symlink() or dst.exists():
            skipped += 1
            continue

        # Relative link keeps $MODEL_OUT portable.
        target = os.path.relpath(task_dir, dst_parent)
        dst.symlink_to(target)
        linked += 1

    print(
        f"{model_out.name}: linked={linked} skipped={skipped} "
        f"In-Domain_50={len(list(in_dir.iterdir()))} "
        f"Out-of-Domain_50={len(list(out_dir.iterdir()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
