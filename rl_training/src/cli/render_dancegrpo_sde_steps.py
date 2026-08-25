"""Render per-step DanceGRPO SDE predicted-final videos for one rollout.

This mirrors the ODE evaluation renderer that decodes
``z0 = sample - sigma * model_output`` at each denoising step, while keeping
the actual rollout transition stochastic via the DanceGRPO RF-SDE update.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from src.cli.sample_dancegrpo_sde import (
    _decode_latents_to_uint8,
    _export_uint8_video,
    _load_checkpoint_into_model,
    _load_sample,
    _sample_summary,
    _torch_dtype,
)
from src.models.wan_i2v import WanI2VForTraining
from src.trainer.config import RLConfig, SFTConfig


def _load_render_config(path: str) -> Any:
    cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
    trainer = cfg_dict.get("trainer")
    if trainer == "dancegrpo":
        return RLConfig(**cfg_dict)
    return SFTConfig(**cfg_dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="SFT or RL config used by the checkpoint")
    p.add_argument("--checkpoint", required=True, help="DCP checkpoint directory")
    p.add_argument("--output_dir", default="storage/outputs/dancegrpo_sde_steps")
    p.add_argument("--sample_index", type=int, default=0, help="Ordinal sample index across sorted shard tar files")
    p.add_argument("--group_index", type=int, default=0, help="Rollout group index to reproduce/render")
    p.add_argument("--num_sampling_steps", type=int, default=None, help="Override cfg.grpo_num_sampling_steps")
    p.add_argument("--sde_noise_scale", type=float, default=None, help="Override cfg.grpo_sde_noise_scale")
    p.add_argument("--cfg_scale", type=float, default=None, help="Override cfg.grpo_cfg_scale")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--transformer_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--checkpoint_prefer", choices=["auto", "raw", "ema"], default="raw")
    p.add_argument("--grid_cols", type=int, default=6)
    p.add_argument("--grid_thumb_width", type=int, default=208)
    p.add_argument("--force", action="store_true", help="Overwrite existing step videos")
    return p.parse_args()


@torch.no_grad()
def _generate_dancegrpo_sde_z0_predictions(
    model: WanI2VForTraining,
    *,
    condition: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_sampling_steps: int,
    sde_noise_scale: float,
    sde_formula: str,
    cfg_scale: float,
    generator: torch.Generator,
    initial_latent: torch.Tensor | None,
) -> dict[str, Any]:
    """Run one SDE trajectory and capture the predicted x0/z0 at each step."""
    batch_size = condition.shape[0]
    device = condition.device
    if initial_latent is not None:
        latent_shape = tuple(initial_latent.shape)
    else:
        latent_shape = model.latent_shape_from_condition(condition)

    t_values = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device)
    shift = model.flow_shift
    sigmas = shift * t_values / (1.0 + (shift - 1.0) * t_values)

    # 5B TI2V: frame 0 is the frozen I2V conditioning image — never noise it.
    freeze_first_frame = model.expand_timesteps and latent_shape[2] > 1 and condition.shape[2] == latent_shape[2]
    cond_first_frame = condition[:, :, 0:1].to(torch.bfloat16) if freeze_first_frame else None

    if initial_latent is None:
        latent = torch.randn(latent_shape, device=device, dtype=torch.bfloat16, generator=generator)
    else:
        latent = initial_latent.to(device=device, dtype=torch.bfloat16).clone()
    if freeze_first_frame:
        latent[:, :, 0:1] = cond_first_frame

    z0_predictions: list[torch.Tensor] = []
    timesteps: list[float] = []
    sigma_values: list[float] = []

    for step_idx in range(num_sampling_steps):
        sigma = float(sigmas[step_idx].item())
        sigma_prev = float(sigmas[step_idx + 1].item())
        timestep_val = sigma * model.num_train_timesteps

        transformer = model._get_expert_for_timestep(timestep_val)
        model_input = model._build_model_input(latent, condition)
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.float32).expand(batch_size)
        timestep_input = model._build_timestep_input(timestep_tensor, latent, transformer)
        hidden_states, timestep_input, encoder_hidden_states = model._prepare_transformer_call(
            transformer,
            model_input,
            timestep_input,
            prompt_embeds,
        )
        model_output = transformer(
            hidden_states=hidden_states,
            timestep=timestep_input,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        if cfg_scale > 1.0:
            uncond_embeds = torch.zeros_like(encoder_hidden_states)
            uncond_output = transformer(
                hidden_states=hidden_states,
                timestep=timestep_input,
                encoder_hidden_states=uncond_embeds,
                return_dict=False,
            )[0]
            model_output = uncond_output.to(torch.float32) + cfg_scale * (
                model_output.to(torch.float32) - uncond_output.to(torch.float32)
            )

        z0 = model._predicted_clean_latent(
            latent,
            model_output,
            sigma,
            cond_first_frame=cond_first_frame,
        )
        z0_predictions.append(z0.detach().cpu())
        sigma_values.append(sigma)
        timesteps.append(timestep_val)

        noise = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=generator)
        if freeze_first_frame:
            noise[:, :, 0:1] = 0
        prev_mean, noise_scale = model._sde_transition_mean(
            sample=latent,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            sde_formula=sde_formula,
            sde_noise_scale=sde_noise_scale,
        )
        latent = (prev_mean + noise.to(torch.float32) * noise_scale).to(latent.dtype)
        if freeze_first_frame:
            latent[:, :, 0:1] = cond_first_frame

    return {
        "z0_predictions": z0_predictions,
        "final_latent": latent.detach(),
        "sigmas": [float(x) for x in sigmas.detach().cpu().tolist()],
        "step_sigmas": sigma_values,
        "timesteps": timesteps,
    }


def _save_step_contact_sheet(
    videos: list[np.ndarray],
    path: Path,
    *,
    frame_count: int = 5,
    thumb_width: int = 160,
) -> None:
    if not videos:
        return

    total_frames = videos[0].shape[0]
    frame_indices = (
        [total_frames - 1]
        if frame_count <= 1
        else np.linspace(0, total_frames - 1, frame_count).round().astype(int).tolist()
    )
    aspect = videos[0].shape[1] / videos[0].shape[2]
    thumb_height = max(1, int(round(thumb_width * aspect)))
    label_h = 18
    pad = 4
    cols = len(frame_indices)
    rows = len(videos)
    sheet_w = cols * thumb_width + (cols + 1) * pad
    sheet_h = rows * (thumb_height + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row, video in enumerate(videos):
        y = pad + row * (thumb_height + label_h + pad)
        draw.text((pad, y), f"s{row:02d}", fill=(0, 0, 0))
        for col, frame_idx in enumerate(frame_indices):
            frame = Image.fromarray(video[frame_idx]).resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
            x = pad + col * (thumb_width + pad)
            sheet.paste(frame, (x, y + label_h))
            if row == 0:
                draw.text((x + 2, y + label_h + 2), f"f{frame_idx}", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _save_step_grid_video(
    videos: list[np.ndarray],
    path: Path,
    *,
    fps: int,
    cols: int,
    thumb_width: int,
) -> None:
    if not videos:
        return

    cols = max(1, cols)
    rows = int(np.ceil(len(videos) / cols))
    frame_count = min(video.shape[0] for video in videos)
    aspect = videos[0].shape[1] / videos[0].shape[2]
    thumb_height = max(1, int(round(thumb_width * aspect)))
    label_h = 18
    pad = 4
    grid_w = cols * thumb_width + (cols + 1) * pad
    grid_h = rows * (thumb_height + label_h) + (rows + 1) * pad
    frames: list[Image.Image] = []

    for frame_idx in range(frame_count):
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        for step_idx, video in enumerate(videos):
            row = step_idx // cols
            col = step_idx % cols
            x = pad + col * (thumb_width + pad)
            y = pad + row * (thumb_height + label_h + pad)
            draw.text((x + 2, y), f"step {step_idx:02d}", fill=(0, 0, 0))
            frame = Image.fromarray(video[frame_idx]).resize(
                (thumb_width, thumb_height),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(frame, (x, y + label_h))
        frames.append(canvas)

    path.parent.mkdir(parents=True, exist_ok=True)
    from diffusers.utils import export_to_video

    export_to_video(frames, str(path), fps=fps)


def main() -> int:
    args = parse_args()
    if args.group_index < 0:
        raise ValueError(f"group_index must be non-negative, got {args.group_index}")

    cfg = _load_render_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    default_steps = int(getattr(cfg, "grpo_num_sampling_steps", 10))
    step_paths = [output_dir / f"step_{idx:02d}.mp4" for idx in range(args.num_sampling_steps or default_steps)]
    if all(path.exists() for path in step_paths) and not args.force:
        print(f"[skip] all {len(step_paths)} step videos already exist in {output_dir}", flush=True)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    num_sampling_steps = args.num_sampling_steps or default_steps
    sde_noise_scale = (
        args.sde_noise_scale if args.sde_noise_scale is not None else getattr(cfg, "grpo_sde_noise_scale", 0.0)
    )
    sde_formula = getattr(cfg, "grpo_sde_formula", "dancegrpo")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else getattr(cfg, "grpo_cfg_scale", 1.0)
    seed = args.seed if args.seed is not None else cfg.seed
    fps = args.fps if args.fps is not None else 16
    share_group_init_noise = bool(getattr(cfg, "dancegrpo_share_group_init_noise", True))

    started = time.time()
    loaded = _load_sample(str(cfg.latent_webdataset_dir), args.sample_index)
    sample = loaded.sample
    prompt_embeds = sample["prompt_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    condition = sample["condition"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    print(f"[sample] ordinal={loaded.ordinal} key={loaded.key} shard={loaded.shard}", flush=True)
    print(f"[sample] {_sample_summary(loaded.metadata)}", flush=True)

    model = WanI2VForTraining(
        cfg.model_path,
        train_experts="both",
        train_text_encoder=False,
        gradient_checkpointing=False,
        load_vae=True,
        load_text_encoder=False,
        transformer_dtype=_torch_dtype(args.transformer_dtype),
    )
    ckpt_info = _load_checkpoint_into_model(model, args.checkpoint, prefer=args.checkpoint_prefer)
    for transformer in (model.transformer, model.transformer_2):
        if transformer is not None:
            transformer.to(device)
            transformer.eval()
    if model.vae is None:
        raise RuntimeError("VAE was not loaded")
    model.vae.to(device)
    model.vae.eval()
    print(f"[checkpoint] {ckpt_info}", flush=True)

    reference_paths: list[str] = []
    if "video_latents" in sample:
        ref_path = output_dir / "reference_latent0.mp4"
        if args.force or not ref_path.exists():
            ref_latents = sample["video_latents"]
            ref_video = _decode_latents_to_uint8(model, ref_latents.unsqueeze(0).to(device))
            _export_uint8_video(ref_video, ref_path, fps)
            del ref_video
            torch.cuda.empty_cache()
        reference_paths.append(str(ref_path))

    init_generator = torch.Generator(device=device).manual_seed(seed + 17)
    shared_initial_latent = None
    if share_group_init_noise:
        latent_shape = model.latent_shape_from_condition(condition)
        shared_initial_latent = torch.randn(
            latent_shape,
            device=device,
            dtype=torch.bfloat16,
            generator=init_generator,
        )

    rollout_generator = torch.Generator(device=device).manual_seed(seed + 1009 * (args.group_index + 1))
    print(
        f"[rollout] group={args.group_index} T={num_sampling_steps} formula={sde_formula} "
        f"eta={sde_noise_scale} cfg={cfg_scale}",
        flush=True,
    )
    traj = _generate_dancegrpo_sde_z0_predictions(
        model,
        condition=condition,
        prompt_embeds=prompt_embeds,
        num_sampling_steps=num_sampling_steps,
        sde_noise_scale=sde_noise_scale,
        sde_formula=sde_formula,
        cfg_scale=cfg_scale,
        generator=rollout_generator,
        initial_latent=shared_initial_latent,
    )

    videos: list[np.ndarray] = []
    video_paths: list[str] = []
    for step_idx, z0_cpu in enumerate(traj["z0_predictions"]):
        step_path = output_dir / f"step_{step_idx:02d}.mp4"
        if args.force or not step_path.exists():
            video = _decode_latents_to_uint8(model, z0_cpu.to(device=device))
            _export_uint8_video(video, step_path, fps)
        else:
            video = None

        if video is None:
            from decord import VideoReader, cpu

            reader = VideoReader(str(step_path), ctx=cpu(0))
            video = reader.get_batch(list(range(len(reader)))).asnumpy()
        videos.append(video)
        video_paths.append(str(step_path))
        print(f"[render] step {step_idx:02d} -> {step_path}", flush=True)
        del z0_cpu
        torch.cuda.empty_cache()

    final_name = "final_ode.mp4" if abs(float(sde_noise_scale)) <= 1e-12 else "final_sde.mp4"
    final_path = output_dir / final_name
    if args.force or not final_path.exists():
        final_video = _decode_latents_to_uint8(model, traj["final_latent"].to(device=device))
        _export_uint8_video(final_video, final_path, fps)
        del final_video

    _save_step_contact_sheet(videos, output_dir / "step_contact_sheet.jpg")
    _save_step_grid_video(
        videos,
        output_dir / "steps_grid.mp4",
        fps=fps,
        cols=args.grid_cols,
        thumb_width=args.grid_thumb_width,
    )

    manifest = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_info": ckpt_info,
        "output_dir": str(output_dir),
        "sample": {
            "ordinal": loaded.ordinal,
            "key": loaded.key,
            "shard": loaded.shard,
            "summary": _sample_summary(loaded.metadata),
        },
        "sampling": {
            "group_index": args.group_index,
            "num_sampling_steps": num_sampling_steps,
            "sde_noise_scale": sde_noise_scale,
            "cfg_scale": cfg_scale,
            "share_group_init_noise": share_group_init_noise,
            "seed": seed,
            "rollout_seed": seed + 1009 * (args.group_index + 1),
            "fps": fps,
        },
        "transition_mode": "deterministic_ode" if abs(float(sde_noise_scale)) <= 1e-12 else "sde",
        "z0_definition": "predicted final latent at each SDE step: z0 = sample - sigma * model_output",
        "sde_formula": sde_formula,
        "sigmas": traj["sigmas"],
        "step_sigmas": traj["step_sigmas"],
        "timesteps": traj["timesteps"],
        "videos": video_paths,
        "references": reference_paths,
        "final_sample": str(final_path),
        "grid": str(output_dir / "steps_grid.mp4"),
        "contact_sheet": str(output_dir / "step_contact_sheet.jpg"),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "sample_metadata.json").write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(video_paths)} step videos to {output_dir}", flush=True)
    output_summary = {
        "manifest": str(output_dir / "manifest.json"),
        "grid": str(output_dir / "steps_grid.mp4"),
    }
    print(json.dumps(output_summary, indent=2), flush=True)

    del traj, videos
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
