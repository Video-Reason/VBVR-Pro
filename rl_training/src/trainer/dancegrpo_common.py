"""Shared data structures and partitioning helpers for DanceGRPO runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _split_group_indices(group_size: int, rollout_rank: int, rollout_world_size: int) -> list[int]:
    """Return GRPO group indices assigned to one rollout actor."""
    if rollout_world_size <= 0:
        return []
    return list(range(rollout_rank, group_size, rollout_world_size))


def _interleave_actor_ranks_by_node(actor_ranks: list[int], local_world_size: int) -> list[int]:
    """Order actors by GPU index across nodes instead of draining one node first."""
    if local_world_size <= 0:
        return list(actor_ranks)
    return sorted(actor_ranks, key=lambda rank: (rank % local_world_size, rank // local_world_size))


def _shared_prompt_assignment(
    rank: int,
    world_size: int,
    prompt_batch_size: int,
    group_size: int,
) -> tuple[int, int, int, list[int]]:
    """Assign one global prompt and a subset of its GRPO groups to a rank."""
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be > 0")
    if prompt_batch_size > world_size:
        raise ValueError(
            f"grpo_shared_prompt_batch requires batch_size <= world_size, got batch_size={prompt_batch_size} "
            f"world_size={world_size}"
        )
    if world_size % prompt_batch_size != 0:
        raise ValueError(
            "grpo_shared_prompt_batch requires world_size to be divisible by batch_size "
            f"so every prompt has the same number of ranks; got world_size={world_size}, batch_size={prompt_batch_size}"
        )
    prompt_idx = rank % prompt_batch_size
    prompt_rank = rank // prompt_batch_size
    prompt_world_size = (world_size - 1 - prompt_idx) // prompt_batch_size + 1
    if group_size % prompt_world_size != 0:
        raise ValueError(
            "grpo_shared_prompt_batch requires grpo_group_size to be divisible by ranks per prompt "
            f"so every rank has the same backward count; got group_size={group_size}, "
            f"ranks_per_prompt={prompt_world_size}"
        )
    group_indices = list(range(prompt_rank, group_size, prompt_world_size))
    if not group_indices:
        raise ValueError(
            "grpo_shared_prompt_batch requires each rank to own at least one group; "
            f"got group_size={group_size}, prompt_world_size={prompt_world_size}. "
            "Increase grpo_group_size or reduce batch_size/world_size."
        )
    return prompt_idx, prompt_rank, prompt_world_size, group_indices


def _shared_prompt_wave_ranges(prompt_batch_size: int, microbatch_size: int | None) -> list[tuple[int, int]]:
    """Split one global prompt batch into equal waves without changing its optimizer semantics."""
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size must be > 0")
    wave_size = prompt_batch_size if microbatch_size is None else int(microbatch_size)
    if wave_size <= 0:
        raise ValueError("grpo_shared_prompt_microbatch_size must be > 0")
    if wave_size > prompt_batch_size:
        raise ValueError(
            "grpo_shared_prompt_microbatch_size must be <= the runtime prompt batch, got "
            f"{wave_size} > {prompt_batch_size}"
        )
    if prompt_batch_size % wave_size != 0:
        raise ValueError(
            "runtime prompt batch must be divisible by grpo_shared_prompt_microbatch_size, got "
            f"{prompt_batch_size} % {wave_size}"
        )
    return [(start, wave_size) for start in range(0, prompt_batch_size, wave_size)]


@dataclass(slots=True)
class _SharedPromptRollout:
    """One prompt wave whose CPU rewards may still be running."""

    prompt_offset: int
    prompt_batch_size: int
    prompt_idx: int
    global_prompt_idx: int
    prompt_rank: int
    prompt_world_size: int
    group_indices: list[int]
    prompt_embeds: torch.Tensor
    condition: torch.Tensor
    local_chunks: list[tuple[dict[str, Any], list[int]]]
    pending_reward_parts: list[tuple[Any, list[int]]]
    prepare_seconds: float


@dataclass(slots=True)
class _SharedPromptStepRollout:
    """All trajectory/reward state for one future optimizer update."""

    rollout_step: int
    policy_version: int
    prompt_batch_size: int
    selected_t_idxs: list[int]
    waves: list[_SharedPromptRollout]


def _slice_meta(meta: dict[str, Any], index: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (torch.Tensor, list)):
            sliced[key] = value[index : index + 1]
        else:
            sliced[key] = value
    return sliced


def _batch_prompt_size(batch: dict[str, Any]) -> int:
    if "prompt_embeds" in batch:
        return int(batch["prompt_embeds"].shape[0])
    if "condition" in batch:
        return int(batch["condition"].shape[0])
    if "prompt" in batch:
        return len(batch["prompt"])
    if "image" in batch:
        return int(batch["image"].shape[0])
    if "videos" in batch and batch["videos"]:
        return int(batch["videos"][0].shape[0])
    raise KeyError("Could not infer prompt batch size from batch")


def _slice_prompt_batch(batch: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[index : index + 1]
        elif isinstance(value, list):
            if key == "videos" and value and all(torch.is_tensor(item) for item in value):
                sliced[key] = [item[index : index + 1] for item in value]
            elif len(value) == batch_size:
                sliced[key] = value[index : index + 1]
            else:
                sliced[key] = value
        else:
            sliced[key] = value
    return sliced


__all__ = [
    "_SharedPromptRollout",
    "_SharedPromptStepRollout",
    "_batch_prompt_size",
    "_interleave_actor_ranks_by_node",
    "_shared_prompt_assignment",
    "_shared_prompt_wave_ranges",
    "_slice_meta",
    "_slice_prompt_batch",
    "_split_group_indices",
]
