"""Input adapters: turn a checkpoint-agnostic source into a :class:`PreparedInput`.

Two sources are supported, mirroring the existing sampling CLIs:

* a precomputed latent webdataset sample (``condition`` + ``prompt_embeds``,
  optionally with reference video latents), and
* a raw image + text prompt (VAE-encoded condition, text-encoded prompt).

Both produce the same :class:`PreparedInput` so the engine and renderer stay
source-agnostic.
"""

import gc
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from src.data.vbvr_latent_dataset import _decode_sample

from .config import InferenceConfig


class PreparedInput(BaseModel):
    """Everything sampling needs for one prompt/condition, plus context for output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    condition: torch.Tensor  # (1, C, T', H', W')
    prompt_embeds: torch.Tensor  # (1, 512, text_dim)
    reference_latents: list[torch.Tensor] = []  # CPU latents (C, T', H', W'), decoded for comparison
    metadata: dict[str, Any] = {}  # raw sample metadata, written to the manifest
    summary: dict[str, Any] = {}  # short human-readable digest
    source: str = ""  # e.g. "latent:<key>" or "image:<path>"


# ----------------------------------------------------------------------
# Latent webdataset source
# ----------------------------------------------------------------------
class LoadedLatentSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample: dict[str, Any]
    metadata: dict[str, Any]
    key: str
    shard: str
    ordinal: int


def _json_metadata(raw: bytes | bytearray | None) -> dict[str, Any]:
    if raw is None:
        return {}
    data = json.loads(bytes(raw).decode("utf-8"))
    return data if isinstance(data, dict) else {}


def load_latent_sample(webdataset_dir: str, sample_index: int) -> LoadedLatentSample:
    """Load the ``sample_index``-th complete sample across sorted shard tars."""
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
                return LoadedLatentSample(
                    sample=_decode_sample(raw_sample),
                    metadata=_json_metadata(parts["json"]),
                    key=key,
                    shard=str(shard_path),
                    ordinal=ordinal,
                )
            ordinal += 1

    raise IndexError(f"sample_index={sample_index} is out of range; found {ordinal} complete samples")


def summarize_latent_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Pull the human-interesting fields out of a maze/latent sample's metadata."""
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


def prepare_from_latent(cfg: InferenceConfig, device: torch.device) -> PreparedInput:
    """Build a :class:`PreparedInput` from a precomputed latent shard sample."""
    loaded = load_latent_sample(str(cfg.latent_webdataset_dir), cfg.sample_index)
    sample = loaded.sample
    prompt_embeds = sample["prompt_embeds"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    condition = sample["condition"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    reference_latents: list[torch.Tensor] = []
    video_latents = sample.get("video_latents")
    if video_latents is not None:
        reference_latents = [video_latents.detach().cpu()]

    return PreparedInput(
        condition=condition,
        prompt_embeds=prompt_embeds,
        reference_latents=reference_latents,
        metadata=loaded.metadata,
        summary={
            "ordinal": loaded.ordinal,
            "key": loaded.key,
            "shard": loaded.shard,
            **summarize_latent_metadata(loaded.metadata),
        },
        source=f"latent:{loaded.key}",
    )


# ----------------------------------------------------------------------
# Raw image + prompt source
# ----------------------------------------------------------------------
def load_image_tensor(path: str, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Load an image as a (1, 3, H, W) bf16 tensor in [-1, 1]."""
    image = Image.open(str(path)).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)


def prepare_from_image(model: Any, cfg: InferenceConfig, device: torch.device) -> PreparedInput:
    """Build a :class:`PreparedInput` from a raw image + text prompt.

    Encodes the prompt and condition once, then frees the text encoder /
    tokenizer — they are not needed again during sampling.
    """
    image = load_image_tensor(str(cfg.image), cfg.height, cfg.width, device)
    with torch.no_grad():
        prompt_embeds = model.encode_text([str(cfg.prompt)], device=device)
        condition = model.prepare_condition(image, cfg.num_frames, cfg.height, cfg.width)
    del image

    model.text_encoder = None
    model.tokenizer = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return PreparedInput(
        condition=condition.to(device=device, dtype=torch.bfloat16),
        prompt_embeds=prompt_embeds.to(device=device, dtype=torch.bfloat16),
        reference_latents=[],
        metadata={"prompt": cfg.prompt, "image": cfg.image},
        summary={
            "prompt": cfg.prompt,
            "image": cfg.image,
            "height": cfg.height,
            "width": cfg.width,
            "num_frames": cfg.num_frames,
        },
        source=f"image:{cfg.image}",
    )


def prepare_input(model: Any, cfg: InferenceConfig, device: torch.device) -> PreparedInput:
    """Dispatch to the right input adapter based on the configured source."""
    if cfg.from_image:
        return prepare_from_image(model, cfg, device)
    return prepare_from_latent(cfg, device)
