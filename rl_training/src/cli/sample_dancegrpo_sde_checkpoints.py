"""Sample one or more latent RL examples across a directory of DCP checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.cli.sample_dancegrpo_sde import (
    _decode_latents_to_uint8,
    _export_uint8_video,
    _load_checkpoint_into_model,
    _load_config,
    _load_sample,
    _pairwise_video_metrics,
    _sample_summary,
    _save_contact_sheet,
    _torch_dtype,
)
from src.models.wan_i2v import WanI2VForTraining


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="RL config shared by the checkpoints")
    p.add_argument("--checkpoint_root", default=None)
    p.add_argument("--checkpoint", action="append", default=None, help="Specific checkpoint path; repeatable")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--sample_index", type=int, action="append", default=None)
    p.add_argument("--group_size", type=int, default=1)
    p.add_argument("--sample_batch_size", type=int, default=1)
    p.add_argument("--num_sampling_steps", type=int, default=None)
    p.add_argument("--sde_noise_scale", type=float, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--transformer_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--checkpoint_prefer", choices=["auto", "raw", "ema"], default="raw")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _discover_checkpoints(root: str | None, explicit: list[str] | None = None) -> list[Path]:
    if explicit:
        checkpoints = [Path(path) for path in explicit]
    else:
        if root is None:
            raise ValueError("Either --checkpoint_root or --checkpoint is required")
        root_path = Path(root)
        checkpoints = []
        for metadata in root_path.glob("checkpoint-*/high/.metadata"):
            checkpoints.append(metadata.parent.parent)
        for metadata in root_path.glob("checkpoint-*/.metadata"):
            checkpoints.append(metadata.parent)
    unique = {str(path): path for path in checkpoints}

    def key(path: Path) -> tuple[int, str]:
        if path.name.startswith("checkpoint-"):
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit():
                return int(suffix), str(path)
        return 10**18, str(path)

    return sorted(unique.values(), key=key)


def _safe_name(checkpoint: Path) -> str:
    return checkpoint.name


def _save_references(
    *,
    model: WanI2VForTraining,
    sample: dict[str, Any],
    out_dir: Path,
    device: torch.device,
    fps: int,
    force: bool,
) -> list[str]:
    reference_paths = []
    video_latents = sample.get("video_latents")
    if video_latents is None:
        return reference_paths
    ref_path = out_dir / "reference_latent0.mp4"
    if force or not ref_path.exists():
        ref_video = _decode_latents_to_uint8(model, video_latents.unsqueeze(0).to(device))
        _export_uint8_video(ref_video, ref_path, fps)
        del ref_video
        torch.cuda.empty_cache()
    reference_paths.append(str(ref_path))
    return reference_paths


def _run_sample(
    *,
    model: WanI2VForTraining,
    cfg,
    loaded,
    checkpoint: Path,
    ckpt_info: dict[str, Any],
    sample_index: int,
    out_dir: Path,
    device: torch.device,
    group_size: int,
    sample_batch_size: int,
    num_sampling_steps: int,
    sde_noise_scale: float,
    sde_formula: str,
    cfg_scale: float,
    seed: int,
    fps: int,
    force: bool,
) -> dict[str, Any]:
    sample = loaded.sample
    prompt_embeds = sample["prompt_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    condition = sample["condition"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    references = _save_references(model=model, sample=sample, out_dir=out_dir, device=device, fps=fps, force=force)

    init_generator = torch.Generator(device=device).manual_seed(seed + 17 + 104729 * sample_index)
    shared_initial_latent = None
    if cfg.dancegrpo_share_group_init_noise:
        latent_shape = model.latent_shape_from_condition(condition)
        shared_initial_latent = torch.randn(
            latent_shape,
            device=device,
            dtype=torch.bfloat16,
            generator=init_generator,
        )

    videos: list[np.ndarray] = []
    video_paths: list[str] = []
    started = time.time()
    for group_start in range(0, group_size, sample_batch_size):
        cur_s = min(sample_batch_size, group_size - group_start)
        out_paths = [out_dir / f"group_{g:02d}.mp4" for g in range(group_start, group_start + cur_s)]
        if all(path.exists() for path in out_paths) and not force:
            for path in out_paths:
                video_paths.append(str(path))
            continue

        cond_s = condition.repeat_interleave(cur_s, dim=0)
        pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
        initial_latent = (
            shared_initial_latent.repeat_interleave(cur_s, dim=0) if shared_initial_latent is not None else None
        )
        rollout_generator = torch.Generator(device=device).manual_seed(seed + 1009 * (sample_index + 1) + group_start)
        with torch.no_grad():
            traj = model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=num_sampling_steps,
                sde_noise_scale=sde_noise_scale,
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg_scale,
                generator=rollout_generator,
                initial_latent=initial_latent,
                sde_formula=sde_formula,
            )
            decoded = model.decode_latents(traj["latents"][-1])
            decoded_uint8 = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
            decoded_uint8 = decoded_uint8.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

        for local_idx, path in enumerate(out_paths):
            video = decoded_uint8[local_idx]
            _export_uint8_video(video, path, fps)
            videos.append(video)
            video_paths.append(str(path))
        del traj, decoded, decoded_uint8, cond_s, pe_s, initial_latent
        torch.cuda.empty_cache()

    if videos:
        _save_contact_sheet(videos, out_dir / "contact_sheet.jpg")
    metrics = _pairwise_video_metrics(videos) if len(videos) > 1 else {}
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_info": ckpt_info,
        "sample_index": sample_index,
        "sample": {
            "ordinal": loaded.ordinal,
            "key": loaded.key,
            "shard": loaded.shard,
            "summary": _sample_summary(loaded.metadata),
        },
        "sampling": {
            "group_size": group_size,
            "sample_batch_size": sample_batch_size,
            "num_sampling_steps": num_sampling_steps,
            "sde_formula": sde_formula,
            "sde_noise_scale": sde_noise_scale,
            "cfg_scale": cfg_scale,
            "share_group_init_noise": cfg.dancegrpo_share_group_init_noise,
            "seed": seed,
            "fps": fps,
        },
        "videos": video_paths,
        "references": references,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "sample_metadata.json").write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)
    sample_indices = args.sample_index or [0]
    checkpoints = _discover_checkpoints(args.checkpoint_root, args.checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {args.checkpoint_root}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_sampling_steps = args.num_sampling_steps or cfg.grpo_num_sampling_steps
    sde_noise_scale = args.sde_noise_scale if args.sde_noise_scale is not None else cfg.grpo_sde_noise_scale
    sde_formula = getattr(cfg, "grpo_sde_formula", "dancegrpo")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.grpo_cfg_scale
    seed = args.seed if args.seed is not None else cfg.seed

    loaded_samples = []
    for sample_index in sample_indices:
        loaded = _load_sample(str(cfg.latent_webdataset_dir), sample_index)
        loaded_samples.append((sample_index, loaded))
        print(f"[sample] index={sample_index} key={loaded.key} {_sample_summary(loaded.metadata)}", flush=True)

    print(f"[checkpoints] {len(checkpoints)} total: {[path.name for path in checkpoints]}", flush=True)
    print(
        f"[sampling] group_size={args.group_size} batch={args.sample_batch_size} "
        f"T={num_sampling_steps} formula={sde_formula} eta={sde_noise_scale}",
        flush=True,
    )

    model = WanI2VForTraining(
        cfg.model_path,
        train_experts="both",
        train_text_encoder=False,
        gradient_checkpointing=False,
        load_vae=True,
        load_text_encoder=False,
        transformer_dtype=_torch_dtype(args.transformer_dtype),
    )
    for transformer in (model.transformer, model.transformer_2):
        if transformer is not None:
            transformer.to(device)
            transformer.eval()
    if model.vae is None:
        raise RuntimeError("VAE was not loaded")
    model.vae.to(device).eval()

    manifests = []
    for ckpt_idx, checkpoint in enumerate(checkpoints, start=1):
        print(f"[checkpoint {ckpt_idx}/{len(checkpoints)}] loading {checkpoint}", flush=True)
        ckpt_info = _load_checkpoint_into_model(model, str(checkpoint), prefer=args.checkpoint_prefer)
        print(f"[checkpoint {ckpt_idx}/{len(checkpoints)}] {ckpt_info}", flush=True)
        for sample_index, loaded in loaded_samples:
            sample_out = output_dir / _safe_name(checkpoint) / f"sample_{sample_index:03d}"
            print(f"[run] checkpoint={checkpoint.name} sample={sample_index} -> {sample_out}", flush=True)
            manifest = _run_sample(
                model=model,
                cfg=cfg,
                loaded=loaded,
                checkpoint=checkpoint,
                ckpt_info=ckpt_info,
                sample_index=sample_index,
                out_dir=sample_out,
                device=device,
                group_size=args.group_size,
                sample_batch_size=args.sample_batch_size,
                num_sampling_steps=num_sampling_steps,
                sde_noise_scale=sde_noise_scale,
                sde_formula=sde_formula,
                cfg_scale=cfg_scale,
                seed=seed,
                fps=args.fps,
                force=args.force,
            )
            manifests.append(manifest)
        gc.collect()
        torch.cuda.empty_cache()

    root_manifest = {
        "config": args.config,
        "checkpoint_root": args.checkpoint_root,
        "output_dir": str(output_dir),
        "sample_indices": sample_indices,
        "checkpoints": [str(path) for path in checkpoints],
        "runs": manifests,
    }
    manifest_name = "manifest.json" if len(checkpoints) != 1 else f"manifest_{_safe_name(checkpoints[0])}.json"
    (output_dir / manifest_name).write_text(json.dumps(root_manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote demo outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
