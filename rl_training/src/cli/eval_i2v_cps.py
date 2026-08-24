"""Generate an I2V evaluation set with the training-time Flow-CPS sampler."""

from __future__ import annotations

import argparse
import gc
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video
from PIL import Image

from src.cli.eval_i2v import (
    _barrier,
    _can_resume_video,
    _cleanup_stale_temporary_videos,
    _expected_output_paths,
    _first_frame_from_video,
    _load_eval_data,
    _load_image,
    _output_path,
    _resolve_path,
    _sample_name,
    _temporary_video_path,
    _valid_video,
    _validate_dimension_args,
    _validate_output_set,
)
from src.inference.config import InferenceConfig
from src.inference.engine import InferenceEngine, build_model
from src.inference.inputs import PreparedInput
from src.inference.outputs import decode_batch_to_uint8


def _noise_level(value: str) -> float:
    level = float(value)
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError(f"CPS noise level must be in [0, 1], got {value}")
    return level


def _item_seed(item: dict, index: int, *, base_seed: int) -> int:
    """Resolve an optional per-item seed while preserving legacy index seeding."""
    value = item.get("seed", base_seed + index)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Eval item {index} seed must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"Eval item {index} seed must be non-negative, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2 I2V Flow-CPS batch evaluation")
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--model_path", required=True, help="Converted Diffusers model directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional DCP checkpoint loaded over model_path")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--noise_level", type=_noise_level, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--parallel_load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the model concurrently on all ranks.",
    )
    parser.add_argument("--dist_backend", default="nccl", choices=["nccl"])
    parser.add_argument("--dist_timeout_minutes", type=int, default=180)
    return parser.parse_args(argv)


def _inference_config(
    args: argparse.Namespace,
    device: torch.device,
    *,
    seed: int,
    image: str = "unused",
    prompt: str = "unused",
) -> InferenceConfig:
    return InferenceConfig(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        use_ema=args.use_ema,
        device=str(device),
        image=image,
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        mode="cps",
        num_sampling_steps=args.num_inference_steps,
        noise_scale=args.noise_level,
        cfg_scale=args.guidance_scale,
        seed=seed,
        batch_size=1,
        share_init_noise=True,
        output_dir=args.output_dir,
        fps=args.fps,
        save_steps=False,
        save_reference=False,
    )


def _load_model(args: argparse.Namespace, device: torch.device, rank: int):
    if rank == 0:
        print(f"Loading CPS model from {args.model_path} ...", flush=True)
    model = build_model(_inference_config(args, device, seed=args.seed), need_text_encoder=True)
    gc.collect()
    return model


def _load_model_rank_serialized(args: argparse.Namespace, device: torch.device, rank: int, world_size: int):
    if world_size == 1 or args.parallel_load:
        return _load_model(args, device, rank)

    model = None
    if rank == 0:
        print("Loading CPS model rank-by-rank to avoid host RAM spikes.", flush=True)
    for load_rank in range(world_size):
        if rank == load_rank:
            print(f"[rank {rank}] Loading CPS model on {device} ...", flush=True)
            model = _load_model(args, device, rank)
            print(f"[rank {rank}] CPS model ready on {device}", flush=True)
        _barrier(device)
    assert model is not None
    return model


def _prepare_input(
    model,
    image: Image.Image,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> PreparedInput:
    array = np.asarray(image, dtype=np.float32).copy()
    image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
    with torch.no_grad():
        prompt_embeds = model.encode_text([prompt], device=device)
        condition = model.prepare_condition(image_tensor, args.num_frames, args.height, args.width)
    return PreparedInput(
        condition=condition.to(device=device, dtype=torch.bfloat16),
        prompt_embeds=prompt_embeds.to(device=device, dtype=torch.bfloat16),
        source="vbvr-eval-json",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, base_dir, data = _load_eval_data(args)
    output_dir = Path(args.output_dir)
    _validate_dimension_args(args, require_fixed=True)
    _expected_output_paths(data, output_dir)
    _cleanup_stale_temporary_videos(output_dir)

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
    my_indices = list(range(rank, len(data), world_size))
    if rank == 0:
        print(
            f"CPS eval: {len(data)} samples, {world_size} GPUs, noise_level={args.noise_level}, "
            f"steps={args.num_inference_steps}, cfg={args.guidance_scale}",
            flush=True,
        )

    model = _load_model_rank_serialized(args, device, rank, world_size)
    transformer = model.transformer if model.transformer is not None else model.transformer_2
    if transformer is None:
        raise RuntimeError("CPS model has no transformer")
    alignment = model.vae_scale_factor_spatial * transformer.config.patch_size[1]
    if args.height % alignment or args.width % alignment:
        raise ValueError(f"Fixed dimensions must be divisible by model alignment {alignment}")

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        _barrier(device)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for count, index in enumerate(my_indices):
        item = data[index]
        out_path = _output_path(output_dir, _sample_name(item, index))
        if _can_resume_video(
            out_path,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            force=args.force,
        ):
            print(f"[rank {rank}] Skipping {out_path} (validated)", flush=True)
            continue

        raw_image = item.get("image")
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        if raw_image:
            image = _load_image(_resolve_path(raw_image, base_dir))
        elif item.get("video"):
            image = _first_frame_from_video(_resolve_path(item["video"], base_dir))
        else:
            raise ValueError(f"Eval item {index} has no image or video")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Eval item {index} has no prompt")
        image = image.resize((args.width, args.height), Image.Resampling.LANCZOS)

        sample_seed = _item_seed(item, index, base_seed=args.seed)
        cfg = _inference_config(args, device, seed=sample_seed)
        prepared = _prepare_input(model, image, prompt, args, device)
        result = InferenceEngine(model, cfg).sample(prepared)
        video = decode_batch_to_uint8(model, result.final_latent.to(device))[0]
        frames = [Image.fromarray(frame) for frame in video]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _temporary_video_path(out_path, rank=rank)
        tmp_path.unlink(missing_ok=True)
        try:
            export_to_video(frames, str(tmp_path), fps=args.fps)
            if not _valid_video(
                tmp_path,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,
            ):
                raise RuntimeError(f"Generated video failed validation: {tmp_path}")
            tmp_path.replace(out_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        print(
            f"[rank {rank}] [{count + 1}/{len(my_indices)}] Saved {out_path} (seed={sample_seed})",
            flush=True,
        )

        del prepared, result, video, frames
        torch.cuda.empty_cache()

    if world_size > 1:
        _barrier(device)
    validation_error: Exception | None = None
    if rank == 0:
        try:
            _validate_output_set(
                data,
                output_dir,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,
            )
        except Exception as exc:
            validation_error = exc
    if dist.is_initialized():
        dist.destroy_process_group()
    if validation_error is not None:
        raise validation_error
    if rank == 0:
        print(f"Done. Generated CPS videos in {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
