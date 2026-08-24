"""Build a small held-out maze eval set with rendered references.

The generated JSON is consumable by ``src.cli.eval_maze`` for inference and by
``src.eval.maze_tracker_score`` for rule/tracker scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from src.precompute.maze_generator import MazeSpec, build_maze_sample
from src.precompute.maze_webdataset import _parse_difficulty_geometries, _sample_to_json_blob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate held-out maze eval set")
    p.add_argument("--output_dir", default="data/maze/eval/maze_384x384x81_perfect_v2")
    p.add_argument("--per_difficulty", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--cell_h", type=int, default=16)
    p.add_argument("--cell_w", type=int, default=16)
    p.add_argument("--cell_px", type=int, default=12)
    p.add_argument(
        "--difficulty_geometries",
        default=None,
        help=(
            "Optional comma-separated per-difficulty geometry map, e.g. "
            "'easy:8x8x24,mid:12x12x16,hard:16x16x12,xhard:16x16x12'. "
            "Entries are cell_h x cell_w x cell_px."
        ),
    )
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--difficulties", default="easy,mid,hard,xhard")
    p.add_argument("--render_mode", default="moving_ball")
    return p.parse_args()


def _build_geometry_map(args: argparse.Namespace, difficulties: list[str]) -> dict[str, tuple[int, int, int]]:
    geometry_map = _parse_difficulty_geometries(args.difficulty_geometries)
    if not geometry_map:
        return {name: (args.cell_h, args.cell_w, args.cell_px) for name in difficulties}

    missing = [name for name in difficulties if name not in geometry_map]
    if missing:
        raise ValueError(f"--difficulty_geometries is missing requested difficulties: {missing!r}")

    extra = sorted(set(geometry_map) - set(difficulties))
    if extra:
        raise ValueError(f"--difficulty_geometries contains unrequested difficulties: {extra!r}")

    image_hw: tuple[int, int] | None = None
    for name in difficulties:
        cell_h, cell_w, cell_px = geometry_map[name]
        hw = (2 * cell_h * cell_px, 2 * cell_w * cell_px)
        if image_hw is None:
            image_hw = hw
        elif hw != image_hw:
            raise ValueError(
                "All difficulty geometries must render to the same image size for one eval set; "
                f"{name} gives {hw}, expected {image_hw}"
            )

    return geometry_map


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "first_frames"
    final_frames_dir = output_dir / "final_frames"
    videos_dir = output_dir / "reference_videos"
    meta_dir = output_dir / "metadata"
    for d in (frames_dir, final_frames_dir, videos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    difficulties = [x.strip() for x in args.difficulties.split(",") if x.strip()]
    geometry_map = _build_geometry_map(args, difficulties)
    first_geometry = geometry_map[difficulties[0]]
    image_h = 2 * first_geometry[0] * first_geometry[2]
    image_w = 2 * first_geometry[1] * first_geometry[2]
    all_rows: list[dict] = []
    manifest = {
        "seed": args.seed,
        "per_difficulty": args.per_difficulty,
        "difficulties": difficulties,
        "difficulty_geometries": {
            name: {"cell_h": cell_h, "cell_w": cell_w, "cell_px": cell_px}
            for name, (cell_h, cell_w, cell_px) in geometry_map.items()
        },
        "image_h": image_h,
        "image_w": image_w,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "render_mode": args.render_mode,
        "rows": [],
    }

    for diff_idx, difficulty in enumerate(difficulties):
        diff_rows: list[dict] = []
        cell_h, cell_w, cell_px = geometry_map[difficulty]
        spec = MazeSpec(
            cell_h=cell_h,
            cell_w=cell_w,
            cell_px=cell_px,
            num_frames=args.num_frames,
            difficulty_names=(difficulty,),
            render_mode=args.render_mode,
            max_generation_attempts=1024,
        )
        for local_idx in range(args.per_difficulty):
            sample_id = f"{difficulty}_{local_idx:03d}"
            sample_seed = args.seed + diff_idx * 10_000 + local_idx
            rng = np.random.default_rng(sample_seed)
            video, sample = build_maze_sample(spec, rng, sample_seed=sample_seed)

            first_frame_path = frames_dir / f"{sample_id}.png"
            final_frame_path = final_frames_dir / f"{sample_id}.png"
            reference_video_path = videos_dir / f"{sample_id}.mp4"
            metadata_path = meta_dir / f"{sample_id}.json"

            Image.fromarray(video[0]).save(first_frame_path)
            Image.fromarray(video[-1]).save(final_frame_path)
            export_to_video([Image.fromarray(frame) for frame in video], str(reference_video_path), fps=args.fps)

            maze_blob = _sample_to_json_blob(sample, fps=args.fps)
            metadata = {
                "id": sample_id,
                "difficulty": difficulty,
                "seed": sample_seed,
                "prompt": sample.prompt,
                "first_frame": str(first_frame_path.relative_to(output_dir)),
                "final_frame": str(final_frame_path.relative_to(output_dir)),
                "reference_video": str(reference_video_path.relative_to(output_dir)),
                "metadata": str(metadata_path.relative_to(output_dir)),
                "maze": maze_blob,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            diff_rows.append(metadata)
            all_rows.append(metadata)
            manifest["rows"].append(
                {
                    "id": sample_id,
                    "difficulty": difficulty,
                    "seed": sample_seed,
                    "path_len": sample.path_len,
                    "path_ratio": sample.path_ratio,
                    "cell_h": sample.cell_h,
                    "cell_w": sample.cell_w,
                    "cell_px": sample.cell_px,
                }
            )

        diff_dir = output_dir / difficulty
        diff_dir.mkdir(parents=True, exist_ok=True)
        (diff_dir / "eval.json").write_text(json.dumps(diff_rows, indent=2) + "\n")

    (output_dir / "eval.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    dataset_info = {
        "dataset": output_dir.name,
        "format": "filesystem_images_and_videos",
        "num_samples": len(all_rows),
        "per_difficulty": args.per_difficulty,
        "difficulty_names": difficulties,
        "difficulty_geometries": manifest["difficulty_geometries"],
        "image_h": image_h,
        "image_w": image_w,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "render_mode": args.render_mode,
        "seed": args.seed,
        "files": {
            "eval_json": "eval.json",
            "manifest": "manifest.json",
            "first_frames": "first_frames/{id}.png",
            "final_frames": "final_frames/{id}.png",
            "reference_videos": "reference_videos/{id}.mp4",
            "metadata": "metadata/{id}.json",
            "per_difficulty_eval_json": "{difficulty}/eval.json",
        },
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")
    print(f"Wrote {len(all_rows)} eval samples to {output_dir}")


if __name__ == "__main__":
    main()
