"""Wan2.2 Image-to-Video inference script using diffusers.

Usage:
    .venv/bin/python -m src.cli.infer_i2v --image path/to/image.jpg --prompt "description"
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image

SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
    "unipc": UniPCMultistepScheduler,
    "flow_match_euler": FlowMatchEulerDiscreteScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory",
    )
    parser.add_argument("--image", type=str, required=True, help="Path or URL to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        help="Negative prompt",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument(
        "--max_area",
        type=int,
        default=480 * 832,
        help="Max pixel area (default 480*832 for 480P, use 1280*720 for 720P)",
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a DCP training checkpoint directory to load (overrides transformer weights)",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load EMA shadow weights from DCP checkpoint")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=list(SCHEDULERS.keys()),
        help="Override the default scheduler/solver (default: use model's original scheduler)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print(f"Loading model from {args.model_path} ...")
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    if args.checkpoint:
        from src.trainer.checkpoint import load_dcp_into_pipeline

        print(f"Loading DCP checkpoint from {args.checkpoint} (ema={args.use_ema}) ...")
        load_dcp_into_pipeline(pipe, args.checkpoint, use_ema=args.use_ema)

    if args.scheduler:
        scheduler_cls = SCHEDULERS[args.scheduler]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        print(f"Using scheduler: {scheduler_cls.__name__}")

    pipe.to(device)

    print(f"Loading image from {args.image} ...")
    image = load_image(args.image)

    # Compute height/width that respects aspect ratio and model constraints
    max_area = args.max_area
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    print(f"Generating video: {width}x{height}, {args.num_frames} frames ...")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).frames[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(output, str(output_path), fps=args.fps)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
