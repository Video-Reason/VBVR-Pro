#!/usr/bin/env python3
"""Extract VBVR-Pro SFT archives and build public trainer metadata.

The Hugging Face datasets are distributed as tar.gz shards. This utility
extracts them without trusting archive paths, creates DiffSynth JSONL files,
and converts the image set to the parquet schemas consumed by BAGEL and
ThinkMorph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-archives",
        type=Path,
        required=True,
        help="Directory produced by downloading VBVR-Pro-SFT-Image.",
    )
    parser.add_argument(
        "--video-archives",
        type=Path,
        required=True,
        help="Directory produced by downloading VBVR-Pro-SFT-Video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for extracted samples and generated metadata.",
    )
    parser.add_argument(
        "--parquet-shard-size",
        type=int,
        default=128,
        help="Number of samples per BAGEL/ThinkMorph parquet shard.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Replace existing generated parquet and metadata files.",
    )
    return parser.parse_args()


def validate_archive_member(member: tarfile.TarInfo, destination: Path) -> None:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe path in archive: {member.name!r}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise ValueError(f"Unsupported archive member: {member.name!r}")
    target = (destination / Path(*member_path.parts)).resolve()
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive member escapes destination: {member.name!r}") from exc


def extraction_marker(archive: Path, archive_root: Path, destination: Path) -> Path:
    relative_name = archive.relative_to(archive_root).as_posix()
    digest = hashlib.sha256(relative_name.encode("utf-8")).hexdigest()
    return destination / ".vbvr_extract_state" / f"{digest}.json"


def extract_archives(archive_root: Path, destination: Path) -> int:
    archive_root = archive_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_root}")

    archives = sorted(archive_root.rglob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz shards found under {archive_root}")

    destination.mkdir(parents=True, exist_ok=True)
    marker_dir = destination / ".vbvr_extract_state"
    marker_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for index, archive in enumerate(archives, start=1):
        marker = extraction_marker(archive, archive_root, destination)
        archive_state = {"size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == archive_state:
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        print(f"[{index}/{len(archives)}] extracting {archive.name}", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                validate_archive_member(member, destination)
            tar.extractall(destination, members=members)
        marker.write_text(json.dumps(archive_state), encoding="utf-8")
        extracted += 1
    return extracted


def frame_number(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def discover_image_samples(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for prompt_path in sorted(root.rglob("prompt.txt")):
        if prompt_path.parent.name != "image":
            continue
        sample_dir = prompt_path.parent.parent
        first_frame = sample_dir / "first_frame.png"
        frames = sorted(prompt_path.parent.glob("frame_*.png"), key=frame_number)
        if not first_frame.is_file() or not frames:
            raise FileNotFoundError(f"Incomplete image sample: {sample_dir}")
        samples.append(
            {
                "sample_dir": sample_dir,
                "first_frame": first_frame,
                "frames": frames,
                "prompt": prompt_path.read_text(encoding="utf-8").strip(),
                "metadata": read_json(sample_dir / "metadata.json"),
            }
        )
    if not samples:
        raise FileNotFoundError(f"No extracted image samples found under {root}")
    return samples


def discover_video_samples(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for prompt_path in sorted(root.rglob("prompt.txt")):
        if prompt_path.parent.name != "video":
            continue
        sample_dir = prompt_path.parent.parent
        first_frame = sample_dir / "first_frame.png"
        video = prompt_path.parent / "ground_truth.mp4"
        if not first_frame.is_file() or not video.is_file():
            raise FileNotFoundError(f"Incomplete video sample: {sample_dir}")
        samples.append(
            {
                "sample_dir": sample_dir,
                "first_frame": first_frame,
                "video": video,
                "prompt": prompt_path.read_text(encoding="utf-8").strip(),
                "metadata": read_json(sample_dir / "metadata.json"),
            }
        )
    if not samples:
        raise FileNotFoundError(f"No extracted video samples found under {root}")
    return samples


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def image_metadata_rows(samples: Iterable[dict[str, Any]], root: Path) -> Iterable[dict[str, Any]]:
    for sample in samples:
        frame_count = len(sample["frames"])
        for index, frame in enumerate(sample["frames"], start=1):
            yield {
                "image": relative_path(frame, root),
                "edit_image": relative_path(sample["first_frame"], root),
                "prompt": sample["prompt"],
                "keyframe_index": index,
                "keyframe_count": frame_count,
            }


def video_metadata_rows(samples: Iterable[dict[str, Any]], root: Path) -> Iterable[dict[str, Any]]:
    for sample in samples:
        parameters = sample["metadata"].get("parameters", {})
        frame_rate = parameters.get("video_fps", 16)
        yield {
            "video": relative_path(sample["video"], root),
            "input_image": relative_path(sample["first_frame"], root),
            "prompt": sample["prompt"],
            "frame_rate": frame_rate,
        }


def thinkmorph_outputs(sample: dict[str, Any]) -> list[str]:
    frames = sample["frames"]
    render = sample["metadata"].get("generic_declarative_render", {})
    frame_metadata = render.get("image_frames", [])
    outputs: list[str] = []
    for index in range(len(frames)):
        progress = None
        if index < len(frame_metadata) and isinstance(frame_metadata[index], dict):
            progress = frame_metadata[index].get("progress")
        progress_text = f" at progress {progress:g}" if isinstance(progress, (int, float)) else ""
        thought = (
            f"Follow the instruction and render keyframe {index + 1} of "
            f"{len(frames)}{progress_text}."
        )
        prefix = "" if index == 0 else "<image_end>"
        outputs.append(f"{prefix}<think>{thought}</think><image_start>")
    outputs.append("<image_end><answer>The requested visual sequence is complete.</answer>")
    return outputs


def prepare_parquet_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.glob("part-*.parquet"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Generated parquet already exists in {path}; pass --overwrite-metadata to replace it."
        )
    if overwrite:
        for item in existing:
            item.unlink()


def write_parquet_datasets(
    samples: list[dict[str, Any]], output_dir: Path, shard_size: int, overwrite: bool
) -> dict[str, Any]:
    if shard_size < 1:
        raise ValueError("--parquet-shard-size must be at least 1")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install it with `pip install pyarrow`.") from exc

    bagel_dir = output_dir / "bagel" / "parquet"
    thinkmorph_dir = output_dir / "thinkmorph" / "parquet"
    prepare_parquet_directory(bagel_dir, overwrite)
    prepare_parquet_directory(thinkmorph_dir, overwrite)

    bagel_schema = pa.schema(
        [
            ("image_list", pa.list_(pa.binary())),
            ("instruction_list", pa.list_(pa.list_(pa.string()))),
        ]
    )
    thinkmorph_schema = pa.schema(
        [
            ("image_list", pa.list_(pa.binary())),
            ("instruction_list", pa.list_(pa.string())),
            ("output_text_list", pa.list_(pa.string())),
        ]
    )
    bagel_info: dict[str, Any] = {}
    thinkmorph_info: dict[str, Any] = {}

    for shard_index, start in enumerate(range(0, len(samples), shard_size)):
        subset = samples[start : start + shard_size]
        bagel_rows = []
        thinkmorph_rows = []
        for sample in subset:
            images = [sample["first_frame"].read_bytes()]
            images.extend(frame.read_bytes() for frame in sample["frames"])
            bagel_rows.append(
                {
                    "image_list": images,
                    "instruction_list": [[sample["prompt"]] for _ in sample["frames"]],
                }
            )
            thinkmorph_rows.append(
                {
                    "image_list": images,
                    "instruction_list": [sample["prompt"]],
                    "output_text_list": thinkmorph_outputs(sample),
                }
            )

        filename = f"part-{shard_index:05d}.parquet"
        bagel_path = bagel_dir / filename
        thinkmorph_path = thinkmorph_dir / filename
        pq.write_table(
            pa.Table.from_pylist(bagel_rows, schema=bagel_schema),
            bagel_path,
            compression="zstd",
            row_group_size=len(bagel_rows),
        )
        pq.write_table(
            pa.Table.from_pylist(thinkmorph_rows, schema=thinkmorph_schema),
            thinkmorph_path,
            compression="zstd",
            row_group_size=len(thinkmorph_rows),
        )
        bagel_info[str(bagel_path.resolve())] = {
            "num_row_groups": pq.ParquetFile(bagel_path).num_row_groups
        }
        thinkmorph_info[str(thinkmorph_path.resolve())] = {
            "num_row_groups": pq.ParquetFile(thinkmorph_path).num_row_groups
        }
        print(f"wrote parquet shard {shard_index + 1}", flush=True)

    bagel_info_path = output_dir / "bagel" / "parquet_info.json"
    thinkmorph_info_path = output_dir / "thinkmorph" / "parquet_info.json"
    bagel_info_path.write_text(json.dumps(bagel_info, indent=2), encoding="utf-8")
    thinkmorph_info_path.write_text(json.dumps(thinkmorph_info, indent=2), encoding="utf-8")
    return {
        "bagel_parquet_dir": str(bagel_dir.resolve()),
        "bagel_parquet_info": str(bagel_info_path.resolve()),
        "thinkmorph_parquet_dir": str(thinkmorph_dir.resolve()),
        "thinkmorph_parquet_info": str(thinkmorph_info_path.resolve()),
        "parquet_shards": len(bagel_info),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    image_root = output_dir / "extracted" / "image"
    video_root = output_dir / "extracted" / "video"

    image_archives_extracted = extract_archives(args.image_archives, image_root)
    video_archives_extracted = extract_archives(args.video_archives, video_root)
    image_samples = discover_image_samples(image_root)
    video_samples = discover_video_samples(video_root)

    metadata_dir = output_dir / "metadata"
    image_metadata_path = metadata_dir / "diffsynth_image.jsonl"
    video_metadata_path = metadata_dir / "diffsynth_video.jsonl"
    image_rows = write_jsonl(image_metadata_path, image_metadata_rows(image_samples, image_root))
    video_rows = write_jsonl(video_metadata_path, video_metadata_rows(video_samples, video_root))
    parquet_summary = write_parquet_datasets(
        image_samples, output_dir, args.parquet_shard_size, args.overwrite_metadata
    )

    summary = {
        "image_archives_extracted_this_run": image_archives_extracted,
        "video_archives_extracted_this_run": video_archives_extracted,
        "image_samples": len(image_samples),
        "video_samples": len(video_samples),
        "diffsynth_image_rows": image_rows,
        "diffsynth_video_rows": video_rows,
        "image_base_path": str(image_root.resolve()),
        "video_base_path": str(video_root.resolve()),
        "diffsynth_image_metadata": str(image_metadata_path.resolve()),
        "diffsynth_video_metadata": str(video_metadata_path.resolve()),
        **parquet_summary,
    }
    summary_path = output_dir / "prepared_data.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Preparation complete. Settings: {summary_path}")


if __name__ == "__main__":
    main()
