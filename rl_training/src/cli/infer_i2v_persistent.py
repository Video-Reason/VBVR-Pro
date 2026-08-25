"""Persistent Wan2.2 Image-to-Video inference runner.

Loads one pipeline once, then accepts JSON jobs from stdin or a jobs file.

Examples:
    .venv/bin/python -m src.cli.infer_i2v_persistent \
        --model_path storage/models/dcp_converted/sft_maze_4_checkpoint-epoch0 \
        --max_area 147456 --num_frames 161 --num_inference_steps 50

    {
      "image": "storage/examples/input.png",
      "prompt": "A red solution line grows step-by-step through a maze.",
      "output": "out.mp4",
      "seed": 0
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, text, watermark, outer border, outside outline, "
    "perimeter path, drawing around the maze, crossing walls, disconnected path"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Wan2.2 I2V JSONL inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional DCP checkpoint to load into the pipeline after model_path",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load EMA shadow weights from a DCP checkpoint")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=list(SCHEDULERS.keys()),
        help="Override the scheduler",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="JSON, JSONL, or {jobs:[...]} file. Omit for stdin JSONL.",
    )
    parser.add_argument("--output_dir", type=str, default="storage/eval_out/i2v_persistent")
    parser.add_argument("--prompt", type=str, default=None, help="Default prompt when a job omits prompt")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=None, help="Default explicit output height")
    parser.add_argument("--width", type=int, default=None, help="Default explicit output width")
    parser.add_argument("--max_area", type=int, default=480 * 832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _read_jobs_file(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("jobs"), list):
        return parsed["jobs"]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported jobs file format: {path}")


def _stdin_jobs() -> Any:
    print(json.dumps({"status": "ready", "mode": "stdin_jsonl"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            print(json.dumps({"status": "bye"}), flush=True)
            break
        yield json.loads(line)


def _job_value(job: dict[str, Any], args: argparse.Namespace, name: str) -> Any:
    return job[name] if name in job else getattr(args, name)


def _output_path_for(job: dict[str, Any], args: argparse.Namespace, index: int, seed: int) -> Path:
    if "output" in job:
        return Path(str(job["output"]))
    image = str(job["image"])
    stem = Path(image).stem if "://" not in image else f"job_{index:06d}"
    return Path(args.output_dir) / f"{stem}_seed{seed}.mp4"


def _resize_image(
    pipe: WanImageToVideoPipeline,
    image: Any,
    job: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, int, int]:
    height = job.get("height", args.height)
    width = job.get("width", args.width)
    if height is not None and width is not None:
        height = int(height)
        width = int(width)
        return image.resize((width, height)), height, width

    max_area = int(job.get("max_area", args.max_area))
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return image.resize((width, height)), height, width


def _run_job(
    pipe: WanImageToVideoPipeline,
    job: dict[str, Any],
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    image_path = str(job["image"])
    prompt = job.get("prompt") or args.prompt
    if not prompt:
        raise ValueError("Job must include prompt, or --prompt must be set")

    seed = int(job.get("seed", args.seed))
    image = load_image(image_path)
    image, height, width = _resize_image(pipe, image, job, args)

    output_path = _output_path_for(job, args, index, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=args.device).manual_seed(seed)
    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=str(_job_value(job, args, "negative_prompt")),
        height=height,
        width=width,
        num_frames=int(_job_value(job, args, "num_frames")),
        guidance_scale=float(_job_value(job, args, "guidance_scale")),
        num_inference_steps=int(_job_value(job, args, "num_inference_steps")),
        generator=generator,
    ).frames[0]
    export_to_video(frames, str(output_path), fps=int(_job_value(job, args, "fps")))
    return {
        "output": str(output_path),
        "width": width,
        "height": height,
        "num_frames": int(_job_value(job, args, "num_frames")),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()

    print(json.dumps({"status": "loading", "model_path": args.model_path}), flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=_dtype(args.dtype))

    if args.checkpoint:
        from src.trainer.checkpoint import load_dcp_into_pipeline

        print(json.dumps({"status": "loading_checkpoint", "checkpoint": args.checkpoint}), flush=True)
        load_dcp_into_pipeline(pipe, args.checkpoint, use_ema=args.use_ema)

    if args.scheduler:
        scheduler_cls = SCHEDULERS[args.scheduler]
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        print(json.dumps({"status": "scheduler", "scheduler": args.scheduler}), flush=True)

    pipe.to(args.device)
    print(json.dumps({"status": "loaded", "device": args.device}), flush=True)

    jobs = _read_jobs_file(args.jobs) if args.jobs else _stdin_jobs()
    for index, job in enumerate(jobs):
        try:
            print(json.dumps({"status": "started", "index": index, "image": job.get("image")}), flush=True)
            result = _run_job(pipe, job, args, index)
            print(json.dumps({"status": "done", "index": index, **result}), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "error", "index": index, "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
