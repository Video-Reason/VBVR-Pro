"""Shared utilities for training."""

import math
import os
from unittest.mock import patch

import torch
from loguru import logger
from torch.distributed.fsdp import fully_shard

_PINNED_HUB_ATTENTION_KERNELS = {
    "_flash_3_hub": (
        "kernels-community/flash-attn3",
        "43f0bd269777115d94ff826e0d113ce9c1c9087b",
    ),
    "_flash_3_varlen_hub": (
        "kernels-community/flash-attn3",
        "43f0bd269777115d94ff826e0d113ce9c1c9087b",
    ),
}


def prefetch_diffusers_attention_backend(backend: str) -> str:
    """Download the pinned binary for a supported Hub attention backend."""
    pinned_kernel = _PINNED_HUB_ATTENTION_KERNELS.get(backend)
    if pinned_kernel is None:
        raise ValueError(f"Attention backend {backend!r} has no pinned downloadable kernel")

    repo_id, revision = pinned_kernel
    from kernels import install_kernel

    return str(install_kernel(repo_id, revision=revision, validate_dependencies=True))


def _collate_tensor_values(values: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    """Stack tensors when possible; preserve variable-shape metadata as a list."""
    first_shape = tuple(values[0].shape)
    if all(tuple(value.shape) == first_shape for value in values):
        return torch.stack(values)
    return values


def apply_liger_rms_norm(model: torch.nn.Module) -> int:
    """Replace all torch.nn.RMSNorm modules with LigerRMSNorm (fused Triton kernel)."""
    from liger_kernel.transformers import LigerRMSNorm

    count = 0
    for _parent_name, parent in list(model.named_modules()):
        for name, module in list(parent.named_children()):
            if not isinstance(module, torch.nn.RMSNorm):
                continue
            (hidden_size,) = module.normalized_shape
            replacement = LigerRMSNorm(hidden_size, eps=module.eps, elementwise_affine=module.elementwise_affine)
            if module.weight is not None:
                replacement.weight = module.weight
            setattr(parent, name, replacement)
            count += 1
    return count


def prepare_diffusers_attention_backend(backend: str | None) -> bool:
    """Validate and initialize a configured Diffusers attention backend.

    Hub-backed implementations are resolved lazily by Diffusers. Entering the
    public context once downloads/loads the selected kernel and populates the
    dispatcher registry before individual Wan attention modules store that
    backend explicitly.
    """
    if backend is None:
        return False

    from diffusers.models import attention_dispatch

    pinned_kernel = _PINNED_HUB_ATTENTION_KERNELS.get(backend)
    if pinned_kernel is None:
        with attention_dispatch.attention_backend(backend):
            pass
        return True

    # Compute nodes may have no route to the Hub. Diffusers 0.37 resolves its
    # versioned Hub backend online even when the binary is already cached, so
    # replace that exact request with kernels' official offline locked loader.
    # The repository and revision are both pinned and checked here.
    pinned_repo, pinned_revision = pinned_kernel
    from kernels import load_kernel

    def _get_pinned_kernel(repo_id, *args, **kwargs):
        if repo_id != pinned_repo:
            raise RuntimeError(f"Attention backend {backend!r} requested unexpected Hub kernel repository {repo_id!r}")
        if args:
            raise RuntimeError(f"Attention backend {backend!r} requested unsupported positional kernel arguments")

        requested_revision = kwargs.pop("revision", None)
        requested_version = kwargs.pop("version", None)
        kernel_backend = kwargs.pop("backend", None)
        if requested_revision not in (None, pinned_revision) or requested_version not in (None, 1):
            raise RuntimeError(
                f"Attention backend {backend!r} requested an unexpected kernel revision/version: "
                f"revision={requested_revision!r}, version={requested_version!r}"
            )
        if kwargs:
            raise RuntimeError(f"Attention backend {backend!r} requested unsupported kernel options: {sorted(kwargs)}")
        try:
            return load_kernel(
                repo_id,
                lockfile=None,
                revision=pinned_revision,
                backend=kernel_backend,
            )
        except FileNotFoundError as exc:
            cache_dir = os.environ.get("KERNELS_CACHE", "<Hugging Face default cache>")
            raise FileNotFoundError(
                f"Pinned attention kernel {repo_id}@{pinned_revision} is unavailable in "
                f"KERNELS_CACHE={cache_dir!r}. On a networked login node run: "
                ".venv/bin/python -m src.cli.prefetch_attention_kernel --backend _flash_3_hub"
            ) from exc

    with patch("kernels.get_kernel", _get_pinned_kernel), attention_dispatch.attention_backend(backend):
        pass
    return True


def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def collate(batch):
    """Collate function for I2V training batches."""
    collated = {}
    sample = batch[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            collated[key] = _collate_tensor_values([x[key] for x in batch])
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            # List of tensors (e.g. videos): stack each position across the batch
            collated[key] = [_collate_tensor_values([x[key][i] for x in batch]) for i in range(len(value))]
        elif isinstance(value, str | int | float | bool):
            collated[key] = [x[key] for x in batch]
    if "prompt" in sample:
        collated["prompt"] = [x["prompt"] for x in batch]
    if "index" in sample:
        collated["index"] = torch.tensor([x["index"] for x in batch], dtype=torch.long)
    return collated


def to_model_pixels(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move uint8 pixels to GPU and normalize to [-1, 1] in bf16."""
    return tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True).div(127.5).sub(1.0)


def shard_transformer(module, mesh, mp_policy):
    """Apply FSDP2 fully_shard per-block then top-level."""
    for block in module.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=mesh, mp_policy=mp_policy)


def setup_loguru(rank: int, enabled: bool | None = None) -> None:
    """Configure loguru: Rich sink on selected ranks, silence the rest."""
    from rich.console import Console
    from rich.text import Text

    logger.remove()
    if enabled is None:
        enabled = rank == 0

    if enabled:
        console = Console(stderr=True)

        _LEVEL_STYLES = {
            "DEBUG": "dim cyan",
            "INFO": "bold green",
            "SUCCESS": "bold green",
            "WARNING": "bold yellow",
            "ERROR": "bold red",
            "CRITICAL": "bold white on red",
        }

        def _rich_sink(message):
            record = message.record
            level = record["level"].name
            style = _LEVEL_STYLES.get(level, "")
            ts = record["time"].strftime("%H:%M:%S")

            line = Text()
            line.append(ts, style="dim")
            line.append(" | ", style="dim")
            if rank != 0:
                line.append(f"rank{rank} ", style="dim")
            line.append(f"{level:<8}", style=style)
            line.append(" | ", style="dim")
            line.append(str(record["message"]))
            console.print(line)

        logger.add(_rich_sink, level="INFO")
    else:
        logger.disable("src")
