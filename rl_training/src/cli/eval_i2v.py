"""Wan2.2 I2V batch evaluation — generate videos for a dataset.

Multi-GPU data-parallel: each rank loads its own pipeline and processes
a disjoint slice of the dataset.

Usage:
    # Single GPU
    .venv/bin/python -m src.cli.eval_i2v --eval_json data/eval.json --output_dir eval_out/

    # Multi-GPU
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_i2v \
        --eval_json data/eval.json --output_dir eval_out/

JSON format (compatible with training dataset):
[
    {"image": "path/to/image.jpg", "prompt": "a cat playing"},
    {"image": "img2.jpg", "prompt": "sunset over the ocean", "video": "ignored.mp4"},
    ...
]

If "image" is absent but "video" is present, the first frame of the video is used.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import decord
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

if TYPE_CHECKING:
    from diffusers import WanImageToVideoPipeline


_TEMPORARY_VIDEO_NAME_RE = re.compile(r"^\.[A-Za-z0-9_.-]+\.tmp-rank[0-9]+-pid(?P<pid>[1-9][0-9]*)\.mp4$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan2.2 I2V batch evaluation")
    parser.add_argument("--eval_json", type=str, required=True, help="JSON file with eval samples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a DCP training checkpoint directory",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load EMA shadow weights from DCP checkpoint")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt",
    )
    parser.add_argument(
        "--max_area",
        type=int,
        default=480 * 832,
        help="Max pixel area (default 480*832 for 480P)",
    )
    parser.add_argument("--height", type=int, default=None, help="Fixed output height (must be used with --width)")
    parser.add_argument("--width", type=int, default=None, help="Fixed output width (must be used with --height)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument(
        "--use_item_num_frames",
        action="store_true",
        help="Use each eval item's integer num_frames value, falling back to --num_frames when absent",
    )
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N eval samples")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate the complete output set without loading a model",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate videos even when existing outputs validate")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Hide per-sample denoising progress bars")
    parser.add_argument(
        "--parallel_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model concurrently on all ranks. Use --no-parallel_load to reduce host RAM peak.",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl"],
        help="Distributed backend for rank coordination.",
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=180,
        help="Process-group timeout in minutes. Slow serialized model loading can exceed PyTorch's default.",
    )
    return parser


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)


def _resolve_path(path: str, base_dir: Path) -> str:
    p = Path(path)
    return str(p) if p.is_absolute() else str(base_dir / p)


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _first_frame_from_video(video_path: str) -> Image.Image:
    vr = decord.VideoReader(video_path)
    frame = vr[0].numpy()
    return Image.fromarray(frame)


def _output_path(output_dir: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe output name {name!r}: expected a relative path without '..'")
    if not relative.parts or str(relative) in {"", "."}:
        raise ValueError("Output name must not be empty")
    return output_dir / relative.with_suffix(".mp4")


def _sample_name(item: dict, index: int) -> str:
    if not isinstance(item, dict):
        raise ValueError(f"Eval item {index} must be a JSON object, got {type(item).__name__}")
    for key in ("name", "id"):
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def _expected_output_paths(data: list[dict], output_dir: Path) -> tuple[Path, ...]:
    expected: dict[Path, int] = {}
    duplicates: list[str] = []
    for index, item in enumerate(data):
        relative_path = _output_path(Path(), _sample_name(item, index))
        previous = expected.get(relative_path)
        if previous is not None:
            duplicates.append(f"{relative_path} (items {previous} and {index})")
        else:
            expected[relative_path] = index
    if duplicates:
        raise ValueError(f"Duplicate eval output path(s): {', '.join(duplicates[:10])}")
    return tuple(output_dir / path for path in sorted(expected))


def _item_num_frames(item: dict, index: int, *, default: int, enabled: bool) -> int:
    value = item.get("num_frames", default) if enabled else default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Eval item {index} num_frames must be a positive integer, got {value!r}")
    return value


def _video_validation_error(path: Path, *, width: int, height: int, num_frames: int, fps: float) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return "file is missing or empty"
    try:
        reader = decord.VideoReader(str(path), num_threads=1)
        if len(reader) != num_frames:
            return f"frames={len(reader)}, expected={num_frames}"
        frame = reader[0]
        if tuple(frame.shape[:2]) != (height, width):
            return f"size={frame.shape[1]}x{frame.shape[0]}, expected={width}x{height}"
        actual_fps = float(reader.get_avg_fps())
        if not math.isfinite(actual_fps) or not math.isclose(actual_fps, fps, rel_tol=0.0, abs_tol=1e-3):
            return f"fps={actual_fps}, expected={fps}"
    except Exception as exc:
        return f"video decode failed: {exc}"
    return None


def _valid_video(path: Path, *, width: int, height: int, num_frames: int, fps: float) -> bool:
    return _video_validation_error(path, width=width, height=height, num_frames=num_frames, fps=fps) is None


def _can_resume_video(
    path: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    force: bool,
) -> bool:
    return not force and _valid_video(path, width=width, height=height, num_frames=num_frames, fps=fps)


def _temporary_video_pid(path: Path) -> int | None:
    match = _TEMPORARY_VIDEO_NAME_RE.fullmatch(path.name)
    return int(match.group("pid")) if match is not None else None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_temporary_videos(output_dir: Path) -> tuple[Path, ...]:
    if not output_dir.is_dir():
        return ()

    removed: list[Path] = []
    for path in output_dir.rglob("*.mp4"):
        if not path.is_file():
            continue
        pid = _temporary_video_pid(path)
        if pid is None or _pid_is_alive(pid):
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)
    return tuple(sorted(removed))


def _validate_output_set(
    data: list[dict],
    output_dir: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    use_item_num_frames: bool = False,
) -> int:
    _cleanup_stale_temporary_videos(output_dir)
    expected_paths = _expected_output_paths(data, output_dir)
    expected_relative = {path.relative_to(output_dir) for path in expected_paths}
    actual_relative = (
        {
            path.relative_to(output_dir)
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4" and _temporary_video_pid(path) is None
        }
        if output_dir.is_dir()
        else set()
    )

    errors: list[str] = []
    missing = sorted(expected_relative - actual_relative)
    extra = sorted(actual_relative - expected_relative)
    if missing:
        errors.append(f"missing={len(missing)}: {', '.join(str(path) for path in missing[:10])}")
    if extra:
        errors.append(f"extra={len(extra)}: {', '.join(str(path) for path in extra[:10])}")
    item_by_relative = {
        _output_path(Path(), _sample_name(item, index)): (index, item) for index, item in enumerate(data)
    }
    for relative_path in sorted(expected_relative & actual_relative):
        index, item = item_by_relative[relative_path]
        reason = _video_validation_error(
            output_dir / relative_path,
            width=width,
            height=height,
            num_frames=_item_num_frames(item, index, default=num_frames, enabled=use_item_num_frames),
            fps=fps,
        )
        if reason is not None:
            errors.append(f"invalid {relative_path}: {reason}")
    if errors:
        raise RuntimeError("Output validation failed:\n  - " + "\n  - ".join(errors))
    return len(expected_paths)


def _load_eval_data(args: argparse.Namespace) -> tuple[Path, Path, list[dict]]:
    eval_json = Path(args.eval_json)
    data = json.loads(eval_json.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {eval_json}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        data = data[: args.limit]
    return eval_json, eval_json.parent, data


def _validate_dimension_args(args: argparse.Namespace, *, require_fixed: bool) -> None:
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be specified together")
    if require_fixed and args.height is None:
        raise ValueError("--validate_only requires fixed --height and --width")
    if args.height is not None and (args.height <= 0 or args.width <= 0):
        raise ValueError("--height and --width must be positive")


def _temporary_video_path(out_path: Path, *, rank: int) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_path.stem)
    return out_path.with_name(f".{safe_stem}.tmp-rank{rank}-pid{os.getpid()}.mp4")


def _load_pipeline(args, device: torch.device, rank: int) -> WanImageToVideoPipeline:
    from diffusers import WanImageToVideoPipeline

    if rank == 0:
        print(f"Loading model from {args.model_path} ...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    pipe.to(device)
    gc.collect()

    if args.checkpoint:
        from src.trainer.checkpoint import load_dcp_into_pipeline

        if rank == 0:
            print(f"Loading DCP checkpoint from {args.checkpoint} (ema={args.use_ema}) ...", flush=True)
        load_dcp_into_pipeline(pipe, args.checkpoint, use_ema=args.use_ema)
        gc.collect()

    return pipe


def _barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _load_pipeline_rank_serialized(args, device: torch.device, rank: int, world_size: int) -> WanImageToVideoPipeline:
    if world_size == 1 or args.parallel_load:
        return _load_pipeline(args, device, rank)

    pipe = None
    if rank == 0:
        print("Loading model rank-by-rank to avoid host RAM spikes.", flush=True)
    for load_rank in range(world_size):
        if rank == load_rank:
            print(f"[rank {rank}] Loading pipeline on {device} ...", flush=True)
            pipe = _load_pipeline(args, device, rank)
            print(f"[rank {rank}] Pipeline ready on {device}", flush=True)
        _barrier(device)
    assert pipe is not None
    return pipe


def _pipeline_call_kwargs(args, generator: torch.Generator) -> dict[str, object]:
    """Return backend-specific keyword arguments for one pipeline call."""
    return {"generator": generator}


def run(args: argparse.Namespace) -> int:

    # Validation-only must remain usable without distributed, CUDA, or model setup.
    _, base_dir, data = _load_eval_data(args)
    output_dir = Path(args.output_dir)
    _validate_dimension_args(args, require_fixed=args.validate_only)
    _expected_output_paths(data, output_dir)
    _cleanup_stale_temporary_videos(output_dir)
    if args.validate_only:
        validated = _validate_output_set(
            data,
            output_dir,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            use_item_num_frames=args.use_item_num_frames,
        )
        print(f"Validated {validated} videos in {output_dir}", flush=True)
        return 0

    from diffusers.utils import export_to_video

    # ---- Distributed setup (works for both single and multi-GPU) ----
    if "RANK" in os.environ:
        dist.init_process_group(args.dist_backend, timeout=timedelta(minutes=args.dist_timeout_minutes))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Partition across ranks (round-robin for balanced load)
    my_indices = list(range(rank, len(data), world_size))
    if rank == 0:
        print(f"Eval: {len(data)} samples, {world_size} GPUs, {len(my_indices)} samples on this rank")

    # ---- Load pipeline ----
    pipe = _load_pipeline_rank_serialized(args, device, rank, world_size)

    # ---- Output dir ----
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        _barrier(device)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolution helper ----
    mod_value = pipe.vae_scale_factor_spatial * (
        pipe.transformer.config.patch_size[1]
        if pipe.transformer is not None
        else pipe.transformer_2.config.patch_size[1]
    )

    if args.height is not None and (args.height % mod_value or args.width % mod_value):
        raise ValueError(f"Fixed dimensions must be divisible by the model alignment {mod_value}")

    # ---- Generate ----
    for count, idx in enumerate(my_indices):
        item = data[idx]
        num_frames = _item_num_frames(item, idx, default=args.num_frames, enabled=args.use_item_num_frames)

        # Determine output filename
        name = _sample_name(item, idx)
        out_path = _output_path(output_dir, name)

        # Load reference image
        raw_image = item.get("image")
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        if raw_image:
            image = _load_image(_resolve_path(raw_image, base_dir))
        elif "video" in item:
            image = _first_frame_from_video(_resolve_path(item["video"], base_dir))
        else:
            print(f"[rank {rank}] Skipping index {idx}: no image or video")
            continue

        prompt = item["prompt"]

        # Compute resolution
        if args.height is not None:
            height, width = args.height, args.width
        else:
            aspect_ratio = image.height / image.width
            height = round(np.sqrt(args.max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(args.max_area / aspect_ratio)) // mod_value * mod_value
        if _can_resume_video(
            out_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=args.fps,
            force=args.force,
        ):
            print(f"[rank {rank}] Skipping {out_path} (validated)", flush=True)
            continue
        if out_path.exists():
            reason = "--force" if args.force else "invalid output"
            print(f"[rank {rank}] Replacing {out_path} ({reason})", flush=True)
        image = image.resize((width, height), Image.Resampling.LANCZOS)

        # Generate
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            **_pipeline_call_kwargs(args, generator),
        ).frames[0]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_video_path(out_path, rank=rank)
        tmp_path.unlink(missing_ok=True)
        try:
            export_to_video(frames, str(tmp_path), fps=args.fps)
            if not _valid_video(
                tmp_path,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=args.fps,
            ):
                raise RuntimeError(f"Generated video failed validation: {tmp_path}")
            tmp_path.replace(out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(
            f"[rank {rank}] [{count + 1}/{len(my_indices)}] Saved {out_path} ({num_frames} frames)",
            flush=True,
        )

    if world_size > 1:
        _barrier(device)
    validation_error: Exception | None = None
    if rank == 0 and args.height is not None:
        try:
            _validate_output_set(
                data,
                output_dir,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,
                use_item_num_frames=args.use_item_num_frames,
            )
        except Exception as exc:
            validation_error = exc
    if dist.is_initialized():
        dist.destroy_process_group()

    if validation_error is not None:
        raise validation_error

    if rank == 0:
        print(f"Done. Generated videos in {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
