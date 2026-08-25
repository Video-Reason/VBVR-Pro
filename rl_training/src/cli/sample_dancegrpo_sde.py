"""Sample a DanceGRPO SDE group from one precomputed latent example.

This is an inspection utility: it loads a 5B TI2V/DanceGRPO checkpoint,
reuses the training SDE sampler, and writes all G group rollouts for one
prompt/condition so their diversity can be compared visually and numerically.
"""

from __future__ import annotations

import argparse
import gc
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw

from src.data.vbvr_latent_dataset import _decode_sample
from src.eval.maze_tracker_score import _track_ball
from src.models.wan_i2v import WanI2VForTraining
from src.trainer.checkpoint import extract_init_weights, read_dcp_to_flat_dict, remap_for_current_model
from src.trainer.config import RLConfig


@dataclass
class LoadedSample:
    sample: dict[str, Any]
    metadata: dict[str, Any]
    key: str
    shard: str
    ordinal: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="RL config used by the checkpoint")
    p.add_argument("--checkpoint", required=True, help="DCP checkpoint directory")
    p.add_argument("--output_dir", default="storage/outputs/dancegrpo_5b_sde_group")
    p.add_argument("--sample_index", type=int, default=0, help="Ordinal sample index across sorted shard tar files")
    p.add_argument("--group_size", type=int, default=None, help="Override cfg.grpo_group_size")
    p.add_argument("--sample_batch_size", type=int, default=1, help="Rollouts per forward batch")
    p.add_argument("--num_sampling_steps", type=int, default=None, help="Override cfg.grpo_num_sampling_steps")
    p.add_argument("--sde_noise_scale", type=float, default=None, help="Override cfg.grpo_sde_noise_scale")
    p.add_argument("--cfg_scale", type=float, default=None, help="Override cfg.grpo_cfg_scale")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--transformer_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--checkpoint_prefer", choices=["auto", "raw", "ema"], default="raw")
    p.add_argument("--no_reference", action="store_true", help="Skip decoding reference latent videos")
    p.add_argument("--force", action="store_true", help="Overwrite existing rollout videos")
    return p.parse_args()


def _load_config(path: str) -> RLConfig:
    cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
    return RLConfig(**cfg_dict)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _json_metadata(raw: bytes | bytearray | None) -> dict[str, Any]:
    if raw is None:
        return {}
    data = json.loads(bytes(raw).decode("utf-8"))
    return data if isinstance(data, dict) else {}


def _load_sample(webdataset_dir: str, sample_index: int) -> LoadedSample:
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")

    shard_paths = sorted(Path(webdataset_dir).glob("shard-*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard-*.tar files found in {webdataset_dir}")

    ordinal = 0
    for shard_path in shard_paths:
        groups: dict[str, dict[str, bytes]] = {}
        key_order: list[str] = []
        with tarfile.open(shard_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                suffix = Path(member.name).suffix.lstrip(".")
                if suffix not in {"json", "safetensors"}:
                    continue
                key = Path(member.name).stem
                if key not in groups:
                    groups[key] = {}
                    key_order.append(key)
                f = tar.extractfile(member)
                if f is None:
                    continue
                groups[key][suffix] = f.read()

        for key in key_order:
            parts = groups[key]
            if "json" not in parts or "safetensors" not in parts:
                continue
            if ordinal == sample_index:
                raw_sample = {
                    "__key__": key,
                    "__url__": str(shard_path),
                    "json": parts["json"],
                    "safetensors": parts["safetensors"],
                }
                return LoadedSample(
                    sample=_decode_sample(raw_sample),
                    metadata=_json_metadata(parts["json"]),
                    key=key,
                    shard=str(shard_path),
                    ordinal=ordinal,
                )
            ordinal += 1

    raise IndexError(f"sample_index={sample_index} is out of range; found {ordinal} complete samples")


def _load_checkpoint_into_model(
    model: WanI2VForTraining,
    checkpoint: str,
    *,
    prefer: str,
) -> dict[str, Any]:
    if model.transformer is None:
        raise RuntimeError("This sampler expects the primary transformer to be loaded")

    root = Path(checkpoint)
    dcp_path = root / "high" if (root / "high" / ".metadata").exists() else root
    started = time.time()
    flat = read_dcp_to_flat_dict(dcp_path)
    weights, source_tag = extract_init_weights(flat, "transformer", prefer=prefer)
    remapped = remap_for_current_model(weights, model.transformer)
    missing, unexpected = model.transformer.load_state_dict(remapped, strict=False)
    elapsed = time.time() - started
    del flat, weights, remapped
    gc.collect()
    return {
        "dcp_path": str(dcp_path),
        "source": source_tag,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "load_seconds": elapsed,
    }


def _video_tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return video[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()


def _export_uint8_video(video: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(frame) for frame in video]
    export_to_video(frames, str(path), fps=fps)


def _decode_latents_to_uint8(model: WanI2VForTraining, latents: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        decoded = model.decode_latents(latents)
    return _video_tensor_to_uint8(decoded)


def _save_contact_sheet(videos: list[np.ndarray], path: Path, frame_count: int = 5, thumb: int = 160) -> None:
    if not videos:
        return
    total_frames = videos[0].shape[0]
    if frame_count <= 1:
        frame_indices = [total_frames - 1]
    else:
        frame_indices = np.linspace(0, total_frames - 1, frame_count).round().astype(int).tolist()

    label_h = 18
    pad = 4
    cols = len(frame_indices)
    rows = len(videos)
    sheet_w = cols * thumb + (cols + 1) * pad
    sheet_h = rows * (thumb + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for row, video in enumerate(videos):
        y = pad + row * (thumb + label_h + pad)
        draw.text((pad, y), f"g{row:02d}", fill=(0, 0, 0))
        for col, frame_idx in enumerate(frame_indices):
            frame = Image.fromarray(video[frame_idx]).resize((thumb, thumb), Image.Resampling.BILINEAR)
            x = pad + col * (thumb + pad)
            sheet.paste(frame, (x, y + label_h))
            if row == 0:
                draw.text((x + 2, y + label_h + 2), f"f{frame_idx}", fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _pairwise_video_metrics(videos: list[np.ndarray]) -> dict[str, Any]:
    if len(videos) < 2:
        return {}
    pair_means = []
    final_pair_means = []
    adjacent_means = []
    for i in range(len(videos)):
        a = videos[i].astype(np.float32)
        for j in range(i + 1, len(videos)):
            b = videos[j].astype(np.float32)
            mean_abs = float(np.abs(a - b).mean())
            pair_means.append(mean_abs)
            final_pair_means.append(float(np.abs(a[-1] - b[-1]).mean()))
            if j == i + 1:
                adjacent_means.append(mean_abs)
    stacked = np.stack(videos).astype(np.float32)
    group_mean = stacked.mean(axis=0)
    to_mean = np.abs(stacked - group_mean[None]).mean(axis=(1, 2, 3, 4))
    return {
        "pixel_absdiff_mean_all_pairs": float(np.mean(pair_means)),
        "pixel_absdiff_min_all_pairs": float(np.min(pair_means)),
        "pixel_absdiff_max_all_pairs": float(np.max(pair_means)),
        "pixel_absdiff_adjacent_mean": float(np.mean(adjacent_means)) if adjacent_means else None,
        "pixel_absdiff_final_frame_mean_all_pairs": float(np.mean(final_pair_means)),
        "pixel_absdiff_to_group_mean_per_video": [float(x) for x in to_mean],
        "pixel_std_mean": float(stacked.std(axis=0).mean()),
        "pixel_scale": "uint8 RGB, mean absolute channel difference in [0,255]",
    }


def _track_positions(video: np.ndarray, metadata: dict[str, Any]) -> np.ndarray | None:
    maze = metadata.get("maze")
    if not isinstance(maze, dict):
        return None
    render_meta = maze.get("render_metadata")
    if not isinstance(render_meta, dict):
        return None
    ball_rgb = np.asarray(render_meta.get("ball_rgb", [220, 40, 40]), dtype=np.float32)
    expected = np.asarray(maze.get("frame_positions_pix", []), dtype=np.float32)
    initial_xy = expected[0] if expected.ndim == 2 and expected.shape[0] else None
    try:
        positions, _conf, _err = _track_ball(
            video.astype(np.float32),
            ball_rgb,
            initial_xy=initial_xy,
            priors_xy=None,
            search_radius=96,
            color_slack=28.0,
        )
    except Exception:
        return None
    return positions


def _tracker_metrics(videos: list[np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    tracks = [_track_positions(video, metadata) for video in videos]
    if any(track is None for track in tracks) or len(tracks) < 2:
        return {}
    track_stack = np.stack([track for track in tracks if track is not None]).astype(np.float32)
    pair_means = []
    final_dists = []
    for i in range(track_stack.shape[0]):
        for j in range(i + 1, track_stack.shape[0]):
            d = np.linalg.norm(track_stack[i] - track_stack[j], axis=-1)
            pair_means.append(float(d.mean()))
            final_dists.append(float(d[-1]))
    return {
        "tracked_xy_pairwise_distance_px_mean": float(np.mean(pair_means)),
        "tracked_xy_pairwise_distance_px_max": float(np.max(pair_means)),
        "tracked_xy_final_distance_px_mean": float(np.mean(final_dists)),
        "tracked_xy_final_distance_px_max": float(np.max(final_dists)),
    }


def _sample_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    maze = metadata.get("maze") if isinstance(metadata, dict) else None
    generation = maze.get("generation") if isinstance(maze, dict) else None
    return {
        "prompt": metadata.get("prompt"),
        "global_index": metadata.get("global_index"),
        "split_index": metadata.get("split_index"),
        "difficulty": maze.get("difficulty") if isinstance(maze, dict) else None,
        "path_len": maze.get("path_len") if isinstance(maze, dict) else None,
        "path_ratio": maze.get("path_ratio") if isinstance(maze, dict) else None,
        "turn_count": generation.get("turn_count") if isinstance(generation, dict) else None,
    }


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = args.group_size or int(cfg.grpo_group_size or 1)
    sample_batch_size = max(1, int(args.sample_batch_size))
    num_sampling_steps = args.num_sampling_steps or cfg.grpo_num_sampling_steps
    sde_noise_scale = args.sde_noise_scale if args.sde_noise_scale is not None else cfg.grpo_sde_noise_scale
    sde_formula = getattr(cfg, "grpo_sde_formula", "dancegrpo")
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.grpo_cfg_scale
    seed = args.seed if args.seed is not None else cfg.seed
    fps = args.fps if args.fps is not None else 16

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
    model.transformer.to(device)
    model.transformer.eval()
    if model.vae is None:
        raise RuntimeError("VAE was not loaded")
    model.vae.to(device)
    model.vae.eval()
    print(f"[checkpoint] {ckpt_info}", flush=True)

    reference_paths: list[str] = []
    if not args.no_reference and "video_latents" in sample:
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
        out_paths = [output_dir / f"group_{g:02d}.mp4" for g in range(group_start, group_start + cur_s)]
        if all(path.exists() for path in out_paths) and not args.force:
            print(f"[rollout] skipping existing group {group_start}..{group_start + cur_s - 1}", flush=True)
            continue

        cond_s = condition.repeat_interleave(cur_s, dim=0)
        pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
        initial_latent = (
            shared_initial_latent.repeat_interleave(cur_s, dim=0) if shared_initial_latent is not None else None
        )
        rollout_generator = torch.Generator(device=device).manual_seed(seed + 1009 * (group_start + 1))

        print(
            f"[rollout] groups {group_start}..{group_start + cur_s - 1} "
            f"T={num_sampling_steps} formula={sde_formula} eta={sde_noise_scale}",
            flush=True,
        )
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

        for local_idx, out_path in enumerate(out_paths):
            video = decoded_uint8[local_idx]
            _export_uint8_video(video, out_path, fps)
            videos.append(video)
            video_paths.append(str(out_path))
        del traj, decoded, decoded_uint8, cond_s, pe_s, initial_latent
        torch.cuda.empty_cache()

    metrics = {}
    if videos:
        metrics.update(_pairwise_video_metrics(videos))
        metrics.update(_tracker_metrics(videos, loaded.metadata))
        _save_contact_sheet(videos, output_dir / "contact_sheet.jpg")

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
        "references": reference_paths,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "sample_metadata.json").write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(video_paths)} videos to {output_dir}", flush=True)
    print(json.dumps({"metrics": metrics, "manifest": str(output_dir / "manifest.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
