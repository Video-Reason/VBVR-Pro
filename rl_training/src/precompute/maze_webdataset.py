"""Generate a synthetic maze dataset as WebDataset tar shards of precomputed latents.

End-to-end pipeline (runnable under torchrun):

    1. Each rank owns a disjoint slice of tar shards.
    2. For every sample, the rank synthesises a maze layout + BFS path and
       renders an RGB video (CPU).
    3. The rank batch-encodes video + first-frame condition through the
       Wan2.2 VAE and the prompt through the UMT5 text encoder (GPU).
    4. Results are packed into SFT/RL ``shard-NNNNNN.tar`` files whose sample
       layout matches ``VBVRLatentDataset``:

           {key:07d}.safetensors   → prompt_embeds / latents / condition
                                     plus maze_* tensors for reward use
           {key:07d}.json          → prompt / tar / index_in_tar / seq_len
                                     plus a structured ``maze`` blob

The extra ``maze_*`` tensors and JSON fields pass through
``VBVRLatentDataset._decode_sample`` for maze-aware reward functions.

Launch (single node, 8 GPUs)::

    .venv/bin/torchrun --nproc_per_node=8 \
        -m src.precompute.maze_webdataset \
        --output_dir data/maze_synth/latents/webdataset \
        --sft_output_dir data/maze_synth/latents/webdataset/sft \
        --rl_output_dir data/maze_synth/latents/webdataset/rl \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --num_samples 100000 --samples_per_shard 1000

Set ``latent_webdataset_dir`` to either the generated SFT or RL split directory.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import random
import shutil
import tarfile
from pathlib import Path

import ftfy
import numpy as np
import regex as re
import torch
from loguru import logger
from pydantic import BaseModel, Field
from safetensors.torch import save as st_save
from tqdm import tqdm

from src.precompute.maze_generator import (
    DEFAULT_DIFFICULTIES,
    DEFAULT_PALETTES,
    MazeSample,
    MazeSpec,
    build_maze_sample,
    normalize_render_mode,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GenConfig(BaseModel):
    """Top-level config for maze dataset generation."""

    # Output
    output_dir: str
    sft_output_dir: str | None = None
    rl_output_dir: str | None = None
    preview_dir: str | None = None
    tar_tag: str = "maze_384x384x81_perfect_v2"

    # Volume
    num_samples: int = Field(ge=1)
    samples_per_shard: int = Field(default=1000, ge=1)
    shard_write_batch_size: int = Field(default=64, ge=1)
    sft_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)

    # Model
    model_path: str

    # Maze geometry
    cell_h: int = Field(default=16, ge=2)
    cell_w: int = Field(default=16, ge=2)
    cell_px: int = Field(default=12, ge=4)  # image H/W = (2 * rooms) * cell_px
    num_frames: int = Field(default=81, ge=2)
    difficulty_names: str = "easy,mid,hard,xhard"
    difficulty_geometries: str | None = None
    render_mode: str = "moving_ball"
    max_generation_attempts: int = Field(default=512, ge=1)
    max_search_steps: int = Field(default=250_000, ge=1)

    # Encoding
    vae_batch_size: int = Field(default=2, ge=1)
    text_batch_size: int = Field(default=32, ge=1)

    # Reproducibility
    seed: int = 42
    split_seed: int | None = None

    # Misc
    skip_existing: bool = False
    only_split: str | None = None
    num_preview_videos: int = Field(default=0, ge=0)
    preview_fps: int = Field(default=16, ge=1)
    dry_run: bool = False

    def difficulty_tuple(self) -> tuple[str, ...]:
        return tuple(name.strip() for name in self.difficulty_names.split(",") if name.strip())

    def maze_spec(
        self,
        *,
        difficulty_names: tuple[str, ...] | None = None,
        cell_h: int | None = None,
        cell_w: int | None = None,
        cell_px: int | None = None,
    ) -> MazeSpec:
        return MazeSpec(
            cell_h=self.cell_h if cell_h is None else cell_h,
            cell_w=self.cell_w if cell_w is None else cell_w,
            cell_px=self.cell_px if cell_px is None else cell_px,
            num_frames=self.num_frames,
            palettes=list(DEFAULT_PALETTES),
            difficulty_names=self.difficulty_tuple() if difficulty_names is None else difficulty_names,
            render_mode=normalize_render_mode(self.render_mode),
            max_generation_attempts=self.max_generation_attempts,
            max_search_steps=self.max_search_steps,
        )


# ---------------------------------------------------------------------------
# CLI parsing (auto-generated from GenConfig)
# ---------------------------------------------------------------------------


def parse_args() -> GenConfig:
    p = argparse.ArgumentParser(description="Generate maze WebDataset with precomputed latents")
    for name, field in GenConfig.model_fields.items():
        flag = f"--{name}"
        ann = field.annotation
        if ann is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif ann is int:
            p.add_argument(flag, type=int, default=None)
        elif ann is float:
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, type=str, default=None)
    args = p.parse_args()
    raw = {name: getattr(args, name) for name in GenConfig.model_fields if getattr(args, name) is not None}
    return GenConfig(**raw)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _init_distributed() -> None:
    if not _is_distributed():
        return
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


# ---------------------------------------------------------------------------
# Resolution sanity
# ---------------------------------------------------------------------------


_SPATIAL_DIVISOR = 8  # Wan2.2 VAE scale_factor_spatial
_TEMPORAL_GROUP_SIZE = 4  # Wan2.2 first-frame condition expects 4k+1 frames.


def _validate_hw(cell_h: int, cell_w: int, cell_px: int) -> tuple[int, int]:
    h = 2 * cell_h * cell_px
    w = 2 * cell_w * cell_px
    if h % _SPATIAL_DIVISOR != 0 or w % _SPATIAL_DIVISOR != 0:
        raise ValueError(
            f"Maze image resolution {h}x{w} must be divisible by {_SPATIAL_DIVISOR}. "
            f"Pick cell_px / cell counts that satisfy 2*cells*cell_px % 8 == 0."
        )
    return h, w


def _validate_num_frames(num_frames: int) -> None:
    if (num_frames - 1) % _TEMPORAL_GROUP_SIZE != 0:
        nearest = ((num_frames - 1 + _TEMPORAL_GROUP_SIZE - 1) // _TEMPORAL_GROUP_SIZE) * _TEMPORAL_GROUP_SIZE + 1
        raise ValueError(
            f"Wan2.2 first-frame conditioning requires num_frames = 4k + 1; got {num_frames}. "
            f"Use {nearest} for the next valid frame count."
        )


def _parse_difficulty_geometries(raw: str | None) -> dict[str, tuple[int, int, int]]:
    """Parse ``easy:8x8x24,mid:12x12x16`` into per-difficulty geometry."""
    if raw is None or not raw.strip():
        return {}

    out: dict[str, tuple[int, int, int]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--difficulty_geometries entries must look like name:cell_hxcell_wxcell_px; got {item!r}")
        name, spec = item.split(":", 1)
        parts = [p for p in spec.lower().replace("x", ",").split(",") if p]
        if len(parts) != 3:
            raise ValueError(
                f"--difficulty_geometries entries must provide three integers (cell_h, cell_w, cell_px); got {item!r}"
            )
        cell_h, cell_w, cell_px = (int(p) for p in parts)
        if cell_h < 2 or cell_w < 2 or cell_px < 4:
            raise ValueError(f"Invalid geometry for {name!r}: {item!r}")
        out[name.strip()] = (cell_h, cell_w, cell_px)
    return out


def _build_maze_specs(cfg: GenConfig) -> tuple[dict[str, MazeSpec], tuple[int, int], dict[str, tuple[int, int, int]]]:
    """Build either one default spec or per-difficulty specs with shared output H/W."""
    _validate_num_frames(cfg.num_frames)
    difficulty_names = cfg.difficulty_tuple()
    if not difficulty_names:
        raise ValueError("--difficulty_names must contain at least one difficulty")

    geometry_map = _parse_difficulty_geometries(cfg.difficulty_geometries)
    if not geometry_map:
        height, width = _validate_hw(cfg.cell_h, cfg.cell_w, cfg.cell_px)
        return {"__default__": cfg.maze_spec()}, (height, width), {}

    missing = [name for name in difficulty_names if name not in geometry_map]
    if missing:
        raise ValueError(f"--difficulty_geometries must include every requested difficulty; missing {missing!r}")
    extra = sorted(set(geometry_map) - set(difficulty_names))
    if extra:
        raise ValueError(f"--difficulty_geometries contains names not present in --difficulty_names: {extra!r}")

    specs: dict[str, MazeSpec] = {}
    image_hw: tuple[int, int] | None = None
    for name in difficulty_names:
        cell_h, cell_w, cell_px = geometry_map[name]
        hw = _validate_hw(cell_h, cell_w, cell_px)
        if image_hw is None:
            image_hw = hw
        elif hw != image_hw:
            raise ValueError(
                "All per-difficulty geometries must render to the same image size for batched VAE encoding; "
                f"{name} gives {hw}, expected {image_hw}"
            )
        specs[name] = cfg.maze_spec(
            difficulty_names=(name,),
            cell_h=cell_h,
            cell_w=cell_w,
            cell_px=cell_px,
        )

    assert image_hw is not None
    return specs, image_hw, geometry_map


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_vae(model_path: str, device: str) -> dict:
    import json

    from diffusers import AutoencoderKLWan

    model_dir = Path(model_path)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    vae.to(device).eval().requires_grad_(False)
    vae_cfg = vae.config
    mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)

    # 5B TI2V uses expand_timesteps first-frame conditioning (cond latents only, no
    # mask concat); A14B I2V uses mask+cond concat. Read the flag from model_index.json.
    model_index_path = model_dir / "model_index.json"
    expand_timesteps = False
    if model_index_path.exists():
        model_index = json.loads(model_index_path.read_text())
        expand_timesteps = bool(model_index.get("expand_timesteps", False))

    return {
        "vae": vae,
        "latents_mean": mean,
        "latents_std_inv": std_inv,
        "scale_spatial": vae_cfg.scale_factor_spatial,
        "scale_temporal": vae_cfg.scale_factor_temporal,
        "expand_timesteps": expand_timesteps,
    }


def _load_text_encoder(model_path: str, device: str) -> dict:
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device).eval().requires_grad_(False)
    max_length = tokenizer.model_max_length
    if max_length is None or max_length > 10_000:
        max_length = 512
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "max_length": max_length}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_video(vae_bundle: dict, video_bf16_m1p1: torch.Tensor) -> torch.Tensor:
    vae = vae_bundle["vae"]
    latents = vae.encode(video_bf16_m1p1.to(vae.dtype)).latent_dist.mode()
    mean = vae_bundle["latents_mean"].to(dtype=latents.dtype)
    std_inv = vae_bundle["latents_std_inv"].to(dtype=latents.dtype)
    return ((latents - mean) * std_inv).to(torch.bfloat16)


@torch.no_grad()
def _encode_condition(
    vae_bundle: dict,
    first_frame_bf16_m1p1: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    vae = vae_bundle["vae"]
    scale_spatial = vae_bundle["scale_spatial"]
    scale_temporal = vae_bundle["scale_temporal"]

    B = first_frame_bf16_m1p1.shape[0]

    # 5B TI2V (expand_timesteps): condition is the first-frame latent expanded across
    # the latent temporal axis, with no mask concat (z_dim channels, matches latents).
    if vae_bundle.get("expand_timesteps", False):
        cond_latents = vae.encode(first_frame_bf16_m1p1.unsqueeze(2).to(vae.dtype)).latent_dist.mode()
        mean = vae_bundle["latents_mean"].to(dtype=cond_latents.dtype)
        std_inv = vae_bundle["latents_std_inv"].to(dtype=cond_latents.dtype)
        cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)
        latent_frames = (num_frames - 1) // scale_temporal + 1
        return cond_latents.expand(B, -1, latent_frames, -1, -1).contiguous()

    cond_video = first_frame_bf16_m1p1.new_zeros((B, 3, num_frames, height, width))
    cond_video[:, :, 0] = first_frame_bf16_m1p1

    cond_latents = vae.encode(cond_video.to(vae.dtype)).latent_dist.mode()
    mean = vae_bundle["latents_mean"].to(dtype=cond_latents.dtype)
    std_inv = vae_bundle["latents_std_inv"].to(dtype=cond_latents.dtype)
    cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

    latent_h = height // scale_spatial
    latent_w = width // scale_spatial
    mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=cond_latents.device, dtype=cond_latents.dtype)
    mask[:, :, 1:] = 0
    first_frame_mask = mask[:, :, 0:1].repeat(1, 1, scale_temporal, 1, 1)
    mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
    mask = mask.view(1, -1, scale_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
    mask = mask.expand(B, -1, -1, -1, -1)
    return torch.cat([mask, cond_latents], dim=1)


@torch.no_grad()
def _encode_prompts(text_bundle: dict, prompts: list[str], device: str) -> list[torch.Tensor]:
    def _clean(t: str) -> str:
        t = ftfy.fix_text(t)
        t = html.unescape(html.unescape(t))
        t = re.sub(r"\s+", " ", t).strip()
        return t

    cleaned = [_clean(p) for p in prompts]
    tokens = text_bundle["tokenizer"](
        cleaned,
        padding="longest",
        max_length=text_bundle["max_length"],
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    embeds = text_bundle["text_encoder"](input_ids, mask).last_hidden_state.to(torch.bfloat16)
    seq_lens = mask.sum(dim=1).tolist()
    return [embeds[i, :length].contiguous() for i, length in enumerate(seq_lens)]


# ---------------------------------------------------------------------------
# Sample conversion helpers
# ---------------------------------------------------------------------------


def _video_to_tensor(video_uint8: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (3, T, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(video_uint8).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W) uint8
    return t.to(torch.float32).div_(127.5).sub_(1.0)


def _sample_to_reward_tensors(sample: MazeSample) -> dict[str, torch.Tensor]:
    """Small tensor payload for fast reward-side reads. All CPU, no bf16.

    ``maze_path`` is intentionally **excluded**: its length varies per sample
    and would break default DataLoader collation.  The BFS path is still
    available in the JSON blob if needed offline.  Fixed-shape tensors only
    in the safetensors pass-through.
    """
    render_mode_id = 1 if sample.render_mode == "growing_path_line" else 0
    return {
        "maze_grid": torch.tensor(sample.grid, dtype=torch.int8),
        "maze_frame_positions_cell": torch.tensor(sample.frame_positions_cell, dtype=torch.float32),
        "maze_frame_positions_pix": torch.tensor(sample.frame_positions_pix, dtype=torch.float32),
        "maze_start": torch.tensor(sample.start, dtype=torch.int16),
        "maze_goal": torch.tensor(sample.goal, dtype=torch.int16),
        "maze_difficulty_id": torch.tensor(sample.difficulty_id, dtype=torch.int16),
        "maze_path_len": torch.tensor(sample.path_len, dtype=torch.int16),
        "maze_manhattan_distance": torch.tensor(sample.manhattan_distance, dtype=torch.int16),
        "maze_path_ratio": torch.tensor(sample.path_ratio, dtype=torch.float32),
        "maze_wall_fraction": torch.tensor(sample.generation["wall_fraction"], dtype=torch.float32),
        "maze_branch_count": torch.tensor(sample.generation["branch_count"], dtype=torch.int16),
        "maze_branch_cells": torch.tensor(sample.generation["branch_cells"], dtype=torch.int16),
        "maze_generation_seed": torch.tensor(
            sample.generation["sample_seed"] if sample.generation.get("sample_seed") is not None else -1,
            dtype=torch.int64,
        ),
        "maze_render_mode_id": torch.tensor(render_mode_id, dtype=torch.int16),
        "maze_ball_rgb": torch.tensor(sample.palette.ball_rgb, dtype=torch.uint8),
        "maze_goal_rgb": torch.tensor(sample.palette.goal_rgb, dtype=torch.uint8),
        "maze_wall_rgb": torch.tensor(sample.palette.wall_rgb, dtype=torch.uint8),
        "maze_passage_rgb": torch.tensor(sample.palette.passage_rgb, dtype=torch.uint8),
        "maze_cell_px": torch.tensor(sample.cell_px, dtype=torch.int32),
        "maze_image_hw": torch.tensor([sample.image_h, sample.image_w], dtype=torch.int32),
    }


def _sample_to_json_blob(sample: MazeSample, *, fps: int | None = None) -> dict:
    """JSON-friendly maze record — everything human-readable for offline use."""
    return {
        "metadata_schema_version": 2,
        "reconstruction": {
            "python_function": "src.precompute.maze_generator.render_video_from_metadata",
            "note": (
                "This maze blob contains the grid, path, per-frame positions, palette, geometry, "
                "and render settings needed to reconstruct the RGB video frames."
            ),
        },
        "generation": sample.generation,
        "difficulty": sample.difficulty,
        "difficulty_id": sample.difficulty_id,
        "render_mode": sample.render_mode,
        "render_metadata": sample.render_metadata,
        "grid_h": sample.grid_h,
        "grid_w": sample.grid_w,
        "cell_h": sample.cell_h,
        "cell_w": sample.cell_w,
        "cell_px": sample.cell_px,
        "image_h": sample.image_h,
        "image_w": sample.image_w,
        "num_frames": sample.num_frames,
        "fps": fps,
        "start": list(sample.start),
        "goal": list(sample.goal),
        "path_len": sample.path_len,
        "manhattan_distance": sample.manhattan_distance,
        "path_ratio": sample.path_ratio,
        "path": [list(p) for p in sample.path],
        "frame_positions_cell": [list(p) for p in sample.frame_positions_cell],
        "frame_positions_pix": [list(p) for p in sample.frame_positions_pix],
        "grid": sample.grid,
        "palette": {
            "wall_rgb": list(sample.palette.wall_rgb),
            "passage_rgb": list(sample.palette.passage_rgb),
            "ball_rgb": list(sample.palette.ball_rgb),
            "goal_rgb": list(sample.palette.goal_rgb),
            "wall_name": sample.palette.wall_name,
            "passage_name": sample.palette.passage_name,
            "ball_name": sample.palette.ball_name,
            "goal_name": sample.palette.goal_name,
        },
    }


# ---------------------------------------------------------------------------
# Tar writing
# ---------------------------------------------------------------------------


def _tar_add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------


ShardPlanEntry = tuple[str, Path, int, list[int], int]
GeneratedSample = tuple[np.ndarray, MazeSample, int]
EncodedSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, MazeSample, int]


def _using_split_dirs(cfg: GenConfig) -> bool:
    if (cfg.sft_output_dir is None) != (cfg.rl_output_dir is None):
        raise ValueError("--sft_output_dir and --rl_output_dir must be provided together")
    return cfg.sft_output_dir is not None and cfg.rl_output_dir is not None


def _output_dirs(cfg: GenConfig) -> dict[str, Path]:
    if _using_split_dirs(cfg):
        assert cfg.sft_output_dir is not None and cfg.rl_output_dir is not None
        return {"sft": Path(cfg.sft_output_dir), "rl": Path(cfg.rl_output_dir)}
    return {"all": Path(cfg.output_dir)}


def _preview_output_dir(cfg: GenConfig) -> Path | None:
    if cfg.num_preview_videos <= 0:
        return None
    return Path(cfg.preview_dir) if cfg.preview_dir is not None else Path(cfg.output_dir) / "preview_videos"


def _shard_plan(cfg: GenConfig) -> tuple[list[ShardPlanEntry], dict[str, int]]:
    """Global plan: (split_name, output_dir, shard_id, sample_gids, split_offset)."""
    output_dirs = _output_dirs(cfg)
    split_seed = cfg.seed if cfg.split_seed is None else cfg.split_seed

    if set(output_dirs) == {"sft", "rl"}:
        gids = list(range(cfg.num_samples))
        random.Random(split_seed).shuffle(gids)
        sft_count = int(cfg.num_samples * cfg.sft_ratio)
        split_gids = {
            "sft": gids[:sft_count],
            "rl": gids[sft_count:],
        }
    else:
        split_gids = {"all": list(range(cfg.num_samples))}

    plan: list[ShardPlanEntry] = []
    split_counts: dict[str, int] = {}
    for split_name, gids in split_gids.items():
        split_counts[split_name] = len(gids)
        output_dir = output_dirs[split_name]
        for sid, start in enumerate(range(0, len(gids), cfg.samples_per_shard)):
            plan.append((split_name, output_dir, sid, gids[start : start + cfg.samples_per_shard], start))
    return plan, split_counts


def _generate_shard_samples(
    cfg: GenConfig,
    specs: dict[str, MazeSpec],
    sample_gids: list[int],
    split_name: str,
) -> list[GeneratedSample]:
    """CPU-only: synthesise ``count`` maze samples with deterministic seeds."""
    out: list[GeneratedSample] = []
    difficulty_names = cfg.difficulty_tuple()
    for gid in sample_gids:
        sample_seed = cfg.seed + gid
        rng = np.random.default_rng(sample_seed)
        if "__default__" in specs:
            spec = specs["__default__"]
        else:
            difficulty_rng = np.random.default_rng(sample_seed)
            difficulty_name = difficulty_names[int(difficulty_rng.integers(0, len(difficulty_names)))]
            spec = specs[difficulty_name]
        video, sample = build_maze_sample(spec, rng, sample_seed=sample_seed)
        _write_preview_video(cfg, gid, split_name, video, sample)
        out.append((video, sample, gid))
    return out


def _write_preview_video(
    cfg: GenConfig,
    gid: int,
    split_name: str,
    video: np.ndarray,
    sample: MazeSample,
) -> None:
    preview_dir = _preview_output_dir(cfg)
    if preview_dir is None or gid >= cfg.num_preview_videos:
        return

    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{gid:07d}_{sample.difficulty}"
    from diffusers.utils import export_to_video
    from PIL import Image

    video_path = preview_dir / f"{stem}.mp4"
    if not video_path.exists():
        export_to_video([Image.fromarray(frame) for frame in video], str(video_path), fps=cfg.preview_fps)

    meta_path = preview_dir / f"{stem}.json"
    if not meta_path.exists():
        meta = {
            "global_index": gid,
            "split": split_name,
            "prompt": sample.prompt,
            "maze": _sample_to_json_blob(sample, fps=cfg.preview_fps),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _encode_shard(
    cfg: GenConfig,
    samples: list[GeneratedSample],
    vae_bundle: dict,
    text_bundle: dict,
    device: str,
    height: int,
    width: int,
) -> list[EncodedSample]:
    """Run VAE + text encoder over a shard's samples. Returns per-sample tensors (CPU)."""
    n = len(samples)
    if n == 0:
        return []
    latents_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    condition_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]

    # Video + condition (VAE) in small batches.
    for batch_start in range(0, n, cfg.vae_batch_size):
        batch = samples[batch_start : batch_start + cfg.vae_batch_size]
        videos = torch.stack([_video_to_tensor(video) for video, _, _ in batch]).to(
            device=device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )
        first_frames = videos[:, :, 0].contiguous()
        cond = _encode_condition(vae_bundle, first_frames, cfg.num_frames, height, width)
        latents = _encode_video(vae_bundle, videos)
        for j in range(len(batch)):
            condition_all[batch_start + j] = cond[j].contiguous().cpu()
            latents_all[batch_start + j] = latents[j].contiguous().cpu()
        del videos, first_frames, cond, latents

    # Prompts (text encoder) in larger batches.
    prompt_embeds_all: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    for batch_start in range(0, n, cfg.text_batch_size):
        batch = samples[batch_start : batch_start + cfg.text_batch_size]
        prompts = [sample.prompt for _, sample, _ in batch]
        embeds = _encode_prompts(text_bundle, prompts, device)
        for j, e in enumerate(embeds):
            prompt_embeds_all[batch_start + j] = e.cpu()

    return [
        (
            latents_all[i],
            condition_all[i],
            prompt_embeds_all[i],
            samples[i][1],
            samples[i][2],
        )
        for i in range(n)
    ]


def _write_encoded_samples(
    cfg: GenConfig,
    tar: tarfile.TarFile,
    split_name: str,
    encoded: list[EncodedSample],
    split_offset: int,
    local_start: int,
) -> None:
    for local_idx, (latents, condition, prompt_embeds, sample, gid) in enumerate(encoded):
        key = f"{gid:07d}"

        st_tensors = {
            "prompt_embeds": prompt_embeds,
            "condition": condition,
            "latents": latents,
            **_sample_to_reward_tensors(sample),
        }
        st_bytes = st_save(st_tensors)

        meta = {
            "metadata_schema_version": 2,
            "prompt": sample.prompt,
            "tar": cfg.tar_tag,
            "index_in_tar": gid,
            "global_index": gid,
            "split": split_name,
            "split_index": split_offset + local_start + local_idx,
            "seq_len": int(prompt_embeds.shape[0]),
            "render_mode": sample.render_mode,
            "maze": _sample_to_json_blob(sample, fps=cfg.preview_fps),
        }
        meta_bytes = json.dumps(meta).encode()

        _tar_add_bytes(tar, f"{key}.safetensors", st_bytes)
        _tar_add_bytes(tar, f"{key}.json", meta_bytes)


def _write_shard(
    cfg: GenConfig,
    specs: dict[str, MazeSpec],
    split_name: str,
    shard_id: int,
    sample_gids: list[int],
    output_dir: Path,
    split_offset: int,
    vae_bundle: dict,
    text_bundle: dict,
    device: str,
    height: int,
    width: int,
) -> Path:
    """Generate, encode, and pack one shard into ``shard-NNNNNN.tar`` atomically."""
    final_path = output_dir / f"shard-{shard_id:06d}.tar"
    if final_path.exists():
        return final_path

    local_tmp_dir = Path(os.environ.get("WAN_TRAINER_LOCAL_TMP", "/tmp/wan_trainer_shards"))
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    local_tmp = local_tmp_dir / f"{cfg.tar_tag}_{split_name}_{shard_id:06d}.{os.getpid()}.tar"
    dest_tmp = output_dir / f"shard-{shard_id:06d}.tar.copying.{os.getpid()}"
    try:
        with tarfile.open(local_tmp, "w") as tar:
            for local_start in range(0, len(sample_gids), cfg.shard_write_batch_size):
                chunk_gids = sample_gids[local_start : local_start + cfg.shard_write_batch_size]
                samples = _generate_shard_samples(cfg, specs, chunk_gids, split_name)
                encoded = _encode_shard(cfg, samples, vae_bundle, text_bundle, device, height, width)
                _write_encoded_samples(cfg, tar, split_name, encoded, split_offset, local_start)
                del samples, encoded
        if final_path.exists():
            local_tmp.unlink(missing_ok=True)
            return final_path
        shutil.copyfile(local_tmp, dest_tmp)
        if final_path.exists():
            dest_tmp.unlink(missing_ok=True)
            return final_path
        dest_tmp.replace(final_path)
    except FileNotFoundError:
        if final_path.exists():
            return final_path
        raise
    except Exception:
        raise
    finally:
        local_tmp.unlink(missing_ok=True)
        if not final_path.exists():
            dest_tmp.unlink(missing_ok=True)
    return final_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    _init_distributed()

    rank = _get_rank()
    world_size = _get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = _output_dirs(cfg)
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = _preview_output_dir(cfg)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    specs, (height, width), geometry_map = _build_maze_specs(cfg)
    plan, split_counts = _shard_plan(cfg)
    if cfg.only_split:
        only_split = cfg.only_split.strip()
        if only_split not in output_dirs:
            raise ValueError(f"--only_split={only_split!r} is not one of {sorted(output_dirs)}")
        plan = [entry for entry in plan if entry[0] == only_split]
        output_dirs = {only_split: output_dirs[only_split]}
        split_counts = {only_split: split_counts.get(only_split, 0)}

    if rank == 0:
        logger.info(
            "Maze gen: {} samples @ {}x{} px, {} frames, render_mode={}",
            cfg.num_samples,
            height,
            width,
            cfg.num_frames,
            normalize_render_mode(cfg.render_mode),
        )
        if geometry_map:
            logger.info("Per-difficulty geometries: {}", geometry_map)
        else:
            logger.info("Geometry: grid {}x{}, cell_px {}", cfg.cell_h, cfg.cell_w, cfg.cell_px)
        logger.info("Difficulties sampled uniformly from: {}", cfg.difficulty_tuple())
        logger.info("Split counts: {}", split_counts)
        if preview_dir is not None:
            logger.info("Preview videos: first {} global samples -> {}", cfg.num_preview_videos, preview_dir)

    my_shards = plan[rank::world_size]
    if cfg.skip_existing:
        my_shards = [s for s in my_shards if not (s[1] / f"shard-{s[2]:06d}.tar").exists()]
    logger.info("[rank {}] assigned {} / {} shards", rank, len(my_shards), len(plan))

    if cfg.dry_run:
        if rank == 0:
            logger.info("dry_run=1: validated config and shard plan; not loading models")
        if _is_distributed():
            with contextlib.suppress(Exception):
                torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return

    torch.backends.cudnn.benchmark = True
    logger.info("[rank {}] loading VAE", rank)
    vae_bundle = _load_vae(cfg.model_path, device)
    logger.info("[rank {}] loading text encoder", rank)
    text_bundle = _load_text_encoder(cfg.model_path, device)

    pbar = tqdm(total=len(my_shards), desc=f"[rank {rank}]", position=local_rank, leave=True, unit="shard")
    for split_name, shard_output_dir, shard_id, sample_gids, split_offset in my_shards:
        _write_shard(
            cfg,
            specs,
            split_name,
            shard_id,
            sample_gids,
            shard_output_dir,
            split_offset,
            vae_bundle,
            text_bundle,
            device,
            height,
            width,
        )
        pbar.update(1)
    pbar.close()

    # Rank 0 writes a small dataset_info.json for downstream tooling.
    if rank == 0:
        info = {
            "tar_tag": cfg.tar_tag,
            "num_samples": cfg.num_samples,
            "samples_per_shard": cfg.samples_per_shard,
            "shard_write_batch_size": cfg.shard_write_batch_size,
            "num_shards": len(plan),
            "splits": {
                name: {
                    "output_dir": str(path),
                    "num_samples": split_counts.get(name, 0),
                    "num_shards": sum(1 for entry in plan if entry[0] == name),
                }
                for name, path in output_dirs.items()
            },
            "sft_ratio": cfg.sft_ratio if "sft" in output_dirs else None,
            "image_h": height,
            "image_w": width,
            "num_frames": cfg.num_frames,
            "render_mode": normalize_render_mode(cfg.render_mode),
            "cell_h": cfg.cell_h,
            "cell_w": cfg.cell_w,
            "cell_px": cfg.cell_px,
            "difficulty_names": list(cfg.difficulty_tuple()),
            "difficulty_geometries": {
                name: {"cell_h": values[0], "cell_w": values[1], "cell_px": values[2]}
                for name, values in geometry_map.items()
            },
            "difficulty_recipes": [d.model_dump() for d in DEFAULT_DIFFICULTIES if d.name in cfg.difficulty_tuple()],
            "seed": cfg.seed,
            "split_seed": cfg.seed if cfg.split_seed is None else cfg.split_seed,
            "preview_dir": str(preview_dir) if preview_dir is not None else None,
            "num_preview_videos": cfg.num_preview_videos,
            "model_path": cfg.model_path,
        }
        (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
        for split_name, split_dir in output_dirs.items():
            split_info = dict(info)
            split_info["active_split"] = split_name
            split_info["num_samples"] = split_counts.get(split_name, cfg.num_samples)
            split_info["num_shards"] = sum(1 for entry in plan if entry[0] == split_name)
            (split_dir / "dataset_info.json").write_text(json.dumps(split_info, indent=2))

    if _is_distributed():
        with contextlib.suppress(Exception):
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    logger.info("[rank {}] done", rank)


if __name__ == "__main__":
    main()
