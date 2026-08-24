"""Maze evaluation with per-step video rendering.

Multi-GPU data-parallel: each rank loads its own pipeline and processes
a disjoint slice of the dataset. At each denoising step, intermediate
latents are decoded through the VAE and saved as a video.

Usage:
    # Single GPU
    .venv/bin/python -m src.cli.eval_maze \
        --eval_json /path/to/test.json --output_dir eval_out/

    # Multi-GPU (fastest)
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_maze \
        --eval_json /path/to/test.json --output_dir eval_out/

    # With checkpoint
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.eval_maze \
        --eval_json /path/to/test.json --output_dir eval_out/ \
        --checkpoint storage/checkpoints/<run>/checkpoint-1000

Output structure:
    output_dir/
      000000/
        step_00.mp4  step_01.mp4  ...  step_49.mp4
      000001/
        ...
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video
from PIL import Image

SCHEDULERS = {
    "euler": FlowMatchEulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Maze evaluation with per-step video rendering")
    parser.add_argument("--eval_json", type=str, nargs="+", required=True, help="JSON file(s) with test samples")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to DCP training checkpoint(s). When multiple are given, "
        "the base model is loaded once and each checkpoint is evaluated in turn.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load EMA shadow weights from checkpoint")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt",
    )
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=256, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_render_steps",
        type=int,
        default=None,
        help="Number of steps to render (evenly spaced). None = render all steps.",
    )
    parser.add_argument("--save_latents", action="store_true", help="Also save raw latents as .pt files")
    parser.add_argument("--copy_refs", action="store_true", help="Copy first_frame and solution image into output")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=list(SCHEDULERS.keys()),
        help="Override the default scheduler/solver (default: use model's original scheduler)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Only evaluate the first N samples from each eval json. None = use all.",
    )
    return parser.parse_args()


def _resolve_path(path: str, base_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base_dir / p


@torch.no_grad()
def decode_latents_to_frames(pipe, latents: torch.Tensor) -> list:
    """Decode a latent tensor to a list of PIL frames (same as pipeline post-processing)."""
    latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pil")
    return frames[0]  # list of PIL images for batch element 0


def _run_eval(
    pipe,
    args,
    data: list[dict],
    my_indices: list[int],
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    """Run evaluation for one checkpoint configuration (already loaded into pipe)."""

    # ---- Output dir ----
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(args._current_eval_json).parent

    # ---- Compute which steps to render ----
    total_steps = args.num_inference_steps
    if args.num_render_steps is not None and args.num_render_steps < total_steps:
        n = args.num_render_steps
        render_indices = {round(i * (total_steps - 1) / (n - 1)) for i in range(n)} if n > 1 else {total_steps - 1}
    else:
        render_indices = None

    if rank == 0 and render_indices is not None:
        print(f"Rendering {len(render_indices)} of {total_steps} steps: {sorted(render_indices)}")

    # ---- Monkey-patch scheduler to capture z0 predictions ----
    z0_predictions: dict[int, torch.Tensor] = {}
    scheduler = pipe.scheduler
    original_step = scheduler.step
    _step_counter = [0]

    def patched_step(model_output, timestep, sample, **kwargs):
        step_idx = _step_counter[0]
        if render_indices is None or step_idx in render_indices:
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)
            sigma = scheduler.sigmas[scheduler.step_index]
            z0 = sample.float() - sigma * model_output.float()
            z0_predictions[step_idx] = z0.detach().cpu()
        _step_counter[0] += 1
        return original_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = patched_step

    # ---- Generate ----
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for count, idx in enumerate(my_indices):
        item = data[idx]
        name = f"{idx:06d}"
        sample_dir = output_dir / name

        last_step_path = sample_dir / f"step_{args.num_inference_steps - 1:02d}.mp4"
        if last_step_path.exists():
            print(f"[rank {rank}] Skipping {name} (exists)")
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        first_frame_field = item.get("first_frame") or item.get("image")
        if not first_frame_field:
            print(f"[rank {rank}] Skipping {name}: no first_frame or image")
            continue
        img_path = _resolve_path(first_frame_field, base_dir)
        image = Image.open(str(img_path)).convert("RGB").resize((args.width, args.height))

        if args.copy_refs:
            shutil.copy2(str(img_path), str(sample_dir / "first_frame.png"))
            sol_field = item.get("image")
            if sol_field:
                sol_path = _resolve_path(sol_field, base_dir)
                shutil.copy2(str(sol_path), str(sample_dir / "solution.png"))

        z0_predictions.clear()
        _step_counter[0] = 0

        pipe(
            image=image,
            prompt=item["prompt"],
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type="latent",
        )

        for step_idx in sorted(z0_predictions.keys()):
            step_path = sample_dir / f"step_{step_idx:02d}.mp4"
            frames = decode_latents_to_frames(pipe, z0_predictions[step_idx])
            export_to_video(frames, str(step_path), fps=args.fps)

        if args.save_latents:
            latent_path = sample_dir / "latents.pt"
            ordered = [z0_predictions[k] for k in sorted(z0_predictions.keys())]
            torch.save(torch.stack(ordered), str(latent_path))

        print(f"[rank {rank}] [{count + 1}/{len(my_indices)}] Saved {name} ({len(z0_predictions)} step videos)")

        z0_predictions.clear()

    # Restore original scheduler step
    pipe.scheduler.step = original_step

    if rank == 0:
        print(f"Done checkpoint eval. Results in {output_dir}")


def main():
    args = parse_args()

    # ---- Distributed setup ----
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ---- Load pipeline (once) ----
    if rank == 0:
        print(f"Loading model from {args.model_path} ...")
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    if args.scheduler:
        scheduler_cls = SCHEDULERS[args.scheduler]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        if rank == 0:
            print(f"Using scheduler: {scheduler_cls.__name__}")

    pipe.to(device)

    # ---- Build checkpoint list ----
    checkpoints = args.checkpoint or [None]  # None = base model without DCP
    eval_jsons = args.eval_json
    multi_eval = len(eval_jsons) > 1

    for ckpt_idx, checkpoint in enumerate(checkpoints):
        if checkpoint is not None:
            from src.trainer.checkpoint import load_dcp_into_pipeline

            if rank == 0:
                print(
                    f"\n{'=' * 60}\n"
                    f"[{ckpt_idx + 1}/{len(checkpoints)}] Loading DCP: {checkpoint} (ema={args.use_ema})\n"
                    f"{'=' * 60}"
                )
            load_dcp_into_pipeline(pipe, checkpoint, use_ema=args.use_ema)
        elif rank == 0:
            print("Evaluating base model (no checkpoint)")

        for eval_json_path in eval_jsons:
            # ---- Load eval data ----
            eval_json = Path(eval_json_path)
            data = json.loads(eval_json.read_text())
            if args.num_samples is not None:
                data = data[: args.num_samples]
            # Temporarily set args.eval_json to current path for _run_eval's base_dir
            args._current_eval_json = eval_json_path

            my_indices = list(range(rank, len(data), world_size))
            if rank == 0:
                print(
                    f"Eval: {eval_json_path} — {len(data)} samples, {world_size} GPUs, {len(my_indices)} samples/rank"
                )
                print(
                    f"Steps: {args.num_inference_steps}, "
                    f"Resolution: {args.width}x{args.height}, Frames: {args.num_frames}"
                )

            # Derive output dir
            base_output = Path(args.output_dir)

            # Per-checkpoint subdir
            if len(checkpoints) > 1 and checkpoint is not None:
                ckpt_name = checkpoint.rstrip("/").replace("/", "_")
                if "storage/checkpoints/" in ckpt_name:
                    ckpt_name = ckpt_name.split("storage_checkpoints_", 1)[-1]
                ema_suffix = "_ema" if args.use_ema else ""
                base_output = base_output / f"{ckpt_name}{ema_suffix}"

            # Per-eval-json subdir (use parent directory name, e.g. test_data_easy)
            if multi_eval:
                eval_name = eval_json.parent.name
                output_dir = base_output / eval_name
            else:
                output_dir = base_output

            if rank == 0:
                print(f"Output: {output_dir}")

            _run_eval(pipe, args, data, my_indices, output_dir, rank, world_size, device)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nAll done. Evaluated {len(checkpoints)} checkpoint(s) x {len(eval_jsons)} eval json(s).")


if __name__ == "__main__":
    main()
