r"""Materialize the official VBVR-Pro RL archives for raw I2V training.

The public ``Video-Reason/VBVR-Pro-RL`` dataset stores one compressed archive
per task under both ``VBVR-Pro-RL-Image`` and ``VBVR-Pro-RL-Video``. The video
archives already contain every field needed by ``I2VDataset`` and
``vbvr_rule``, so only that directory needs to be downloaded.

This utility safely reads the video archives without using ``tar.extract``,
restores the five training/reward-critical files into a flat VBVR-Pro tree,
and writes the standard ``dataset.json`` and ``split_manifest_rl.json``.

Example:
    .venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
        --dataset-root storage/datasets/VBVR-Pro-RL \
        --output-dir storage/datasets/VBVR-Pro-RL/materialized \
        --source-revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
        --expected-tasks 50 \
        --expected-samples 50000 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from loguru import logger

SOURCE_REPO_ID = "Video-Reason/VBVR-Pro-RL"
PINNED_SOURCE_REVISION = "ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1"
_VIDEO_ARCHIVE_DIR = "VBVR-Pro-RL-Video"
_ARCHIVE_SUFFIX = ".tar.gz"
_TASK_SPLIT = "In-Domain_50"

_FIELD_TARGETS = {
    "first_frame.png": Path("first_frame.png"),
    "metadata.json": Path("metadata.json"),
    "video/final_frame.png": Path("video/final_frame.png"),
    "video/ground_truth.mp4": Path("video/ground_truth.mp4"),
    "video/prompt.txt": Path("video/prompt.txt"),
}


@dataclass(frozen=True)
class _ArchiveResult:
    archive: Path
    task_name: str
    sample_ids: tuple[str, ...]
    written_files: int
    reused_files: int
    written_bytes: int


def _safe_component(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _task_name_from_archive(archive: Path) -> str:
    if not archive.name.endswith(_ARCHIVE_SUFFIX):
        raise ValueError(f"expected an {_ARCHIVE_SUFFIX} archive: {archive}")
    return _safe_component(archive.name.removesuffix(_ARCHIVE_SUFFIX), label="task name")


def _parse_member(member: tarfile.TarInfo, task_name: str) -> tuple[str, str] | None:
    if not member.isfile():
        return None
    member_path = PurePosixPath(member.name.removeprefix("./"))
    if member_path.is_absolute() or any(part in {"", ".", ".."} for part in member_path.parts):
        raise ValueError(f"unsafe archive member: {member.name!r}")
    if len(member_path.parts) < 4 or member_path.parts[0] != task_name:
        return None

    _safe_component(member_path.parts[1], label=f"task directory in {member.name}")
    sample_id = _safe_component(member_path.parts[2], label=f"sample ID in {member.name}")
    field = PurePosixPath(*member_path.parts[3:]).as_posix()
    if field not in _FIELD_TARGETS:
        return None
    return sample_id, field


def _streams_match(source, existing, *, chunk_size: int = 1024 * 1024) -> bool:
    while True:
        source_chunk = source.read(chunk_size)
        existing_chunk = existing.read(chunk_size)
        if source_chunk != existing_chunk:
            return False
        if not source_chunk:
            return True


def _existing_matches_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
) -> bool:
    if not target.is_file() or target.stat().st_size != member.size:
        return False
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"could not read archive member: {member.name}")
    try:
        with target.open("rb") as existing:
            return _streams_match(source, existing)
    finally:
        source.close()


def _copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
    *,
    verify_existing: bool,
) -> tuple[bool, int]:
    if (
        target.is_file()
        and target.stat().st_size == member.size
        and (not verify_existing or _existing_matches_archive_member(archive, member, target))
    ):
        return False, 0

    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"could not read archive member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.pid{os.getpid()}.thread{threading.get_ident()}.tmp")
    written = 0
    try:
        with temporary.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                written += len(chunk)
        if written != member.size:
            raise ValueError(f"short read for {member.name}: expected {member.size}, wrote {written}")
        os.replace(temporary, target)
    finally:
        source.close()
        temporary.unlink(missing_ok=True)
    return True, written


def _materialize_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    verify_existing: bool,
) -> _ArchiveResult:
    task_name = _task_name_from_archive(archive_path)
    seen: dict[str, set[str]] = {}
    written_files = 0
    reused_files = 0
    written_bytes = 0

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            parsed = _parse_member(member, task_name)
            if parsed is None:
                continue
            sample_id, field = parsed
            sample_fields = seen.setdefault(sample_id, set())
            if field in sample_fields:
                raise ValueError(f"duplicate field {field!r} for sample {sample_id} in {archive_path}")
            sample_fields.add(field)
            target = output_dir / "raw" / task_name / sample_id / _FIELD_TARGETS[field]
            written, byte_count = _copy_member(
                archive,
                member,
                target,
                verify_existing=verify_existing,
            )
            if written:
                written_files += 1
                written_bytes += byte_count
            else:
                reused_files += 1

    required = set(_FIELD_TARGETS)
    missing = {sample_id: sorted(required - fields) for sample_id, fields in seen.items() if required - fields}
    if missing:
        raise ValueError(f"{archive_path} has incomplete samples: {list(missing.items())[:5]}")
    if not seen:
        raise ValueError(f"{archive_path} contains no recognized VBVR-Pro video samples")

    return _ArchiveResult(
        archive=archive_path,
        task_name=task_name,
        sample_ids=tuple(sorted(seen)),
        written_files=written_files,
        reused_files=reused_files,
        written_bytes=written_bytes,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pid{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_training_metadata(
    output_dir: Path,
    dataset_root: Path,
    results: list[_ArchiveResult],
    *,
    source_revision: str,
) -> None:
    manifest = [
        {
            "task": result.task_name,
            "source": result.task_name,
            "split": _TASK_SPLIT,
            "rl": list(result.sample_ids),
        }
        for result in results
    ]
    manifest_path = output_dir / "split_manifest_rl.json"
    descriptor_path = output_dir / "dataset.json"
    _write_json(manifest_path, manifest)
    _write_json(
        descriptor_path,
        [
            {
                "format": "vbvr_pro",
                "split_manifest": manifest_path.name,
                "data_roots": ["raw"],
                "split": "rl",
                "allowed_task_splits": [_TASK_SPLIT],
                "check_files": False,
                "num_frames": 161,
                "height": 256,
                "width": 256,
                "fps": 16,
            }
        ],
    )
    _write_json(
        output_dir / "materialization.json",
        {
            "source_repo_id": SOURCE_REPO_ID,
            "source_revision": source_revision,
            "source_dataset_root": os.path.relpath(dataset_root, output_dir),
            "source_archive_directory": _VIDEO_ARCHIVE_DIR,
            "archives": [
                {
                    "path": os.path.relpath(result.archive, dataset_root),
                    "size": result.archive.stat().st_size,
                    "task": result.task_name,
                    "samples": len(result.sample_ids),
                }
                for result in results
            ],
            "samples": sum(len(result.sample_ids) for result in results),
            "tasks": len(results),
            "fields": sorted(_FIELD_TARGETS),
            "descriptor": descriptor_path.name,
            "split_manifest": manifest_path.name,
        },
    )


def materialize(
    dataset_root: Path,
    output_dir: Path,
    *,
    source_revision: str = PINNED_SOURCE_REVISION,
    expected_tasks: int | None = None,
    expected_samples: int | None = None,
    workers: int = 4,
    verify_existing: bool = False,
) -> list[_ArchiveResult]:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if not source_revision:
        raise ValueError("source_revision must identify the downloaded Hugging Face revision")

    archive_dir = dataset_root / _VIDEO_ARCHIVE_DIR
    archives = sorted(archive_dir.glob(f"*{_ARCHIVE_SUFFIX}"))
    if not archives:
        raise FileNotFoundError(
            f"no official video archives found beneath {archive_dir}; "
            f"download {_VIDEO_ARCHIVE_DIR} from {SOURCE_REPO_ID} first"
        )
    if expected_tasks is not None and len(archives) != expected_tasks:
        raise ValueError(f"expected {expected_tasks} task archives, found {len(archives)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Materializing {} official VBVR-Pro task archives into {} with {} workers",
        len(archives),
        output_dir,
        workers,
    )
    results: list[_ArchiveResult] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(archives))) as executor:
        futures = {
            executor.submit(
                _materialize_archive,
                archive,
                output_dir,
                verify_existing=verify_existing,
            ): archive
            for archive in archives
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                "Completed {}: {} samples, {} files written, {} reused, {:.2f} GiB written",
                result.archive.name,
                len(result.sample_ids),
                result.written_files,
                result.reused_files,
                result.written_bytes / 2**30,
            )

    results.sort(key=lambda result: result.task_name)
    sample_count = sum(len(result.sample_ids) for result in results)
    if expected_samples is not None and sample_count != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, found {sample_count}")
    _write_training_metadata(
        output_dir,
        dataset_root,
        results,
        source_revision=source_revision,
    )
    logger.info(
        "Materialization complete: {} tasks, {} samples, {} files written, {} reused, {:.2f} GiB written",
        len(results),
        sample_count,
        sum(result.written_files for result in results),
        sum(result.reused_files for result in results),
        sum(result.written_bytes for result in results) / 2**30,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        default=PINNED_SOURCE_REVISION,
        help="Hugging Face revision used for the download; recorded in materialization.json",
    )
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Byte-compare existing same-size files with their archive members before reuse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize(
        args.dataset_root,
        args.output_dir,
        source_revision=args.source_revision,
        expected_tasks=args.expected_tasks,
        expected_samples=args.expected_samples,
        workers=args.workers,
        verify_existing=args.verify_existing,
    )


if __name__ == "__main__":
    main()
