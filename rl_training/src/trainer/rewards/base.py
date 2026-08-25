"""Abstract reward interface for GRPO training.

A reward is instantiated **once** per trainer (in
``BaseGRPOTrainer._post_init``) and invoked during every GRPO rollout to
score the generated trajectory.  The signature mirrors the original
``_compute_reward_neg_loss`` so old call sites migrate 1:1.

Implementations receive a reverse reference to the trainer, giving them
access to ``trainer.model``, ``trainer.device``, ``trainer.world_size``, the
reference policy, etc., without restructuring the call graph.  Any knob a
reward needs should be declared on :class:`src.trainer.config.RLConfig` and
read via ``self.cfg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from src.trainer.base_grpo_trainer import BaseGRPOTrainer
    from src.trainer.config import RLConfig


class BaseReward:
    """Base class for all Flow-GRPO reward functions.

    Override ``__call__`` and return a 1-D float32 tensor of shape ``(B,)``
    whose entries are the per-sample reward (higher = better).

    CPU-bound rewards may additionally expose a ``submit`` method with the
    same arguments. It may return either a tensor or a handle with a
    zero-argument ``result()`` method. DanceGRPO submits all rollout chunks
    first and resolves those handles before computing group advantages, which
    lets CPU scoring overlap later GPU rollout chunks without changing reward
    order or values.

    Class attributes:
        requires_vae: set True if the reward calls ``model.decode_latents``
            (or otherwise touches the VAE).  ``BaseRLTrainer._build_model``
            consults this to force-load the VAE even when the dataset ships
            precomputed latents.
        requires_policy_forward: set True if the reward calls a trainable Wan
            transformer. Tensor-parallel trainers must then execute the reward
            on every TP rank instead of evaluating it only on TP rank 0.
    """

    requires_vae: ClassVar[bool] = False
    requires_policy_forward: ClassVar[bool] = False

    def __init__(self, trainer: BaseGRPOTrainer, cfg: RLConfig) -> None:
        self.trainer = trainer
        self.cfg = cfg

    @torch.no_grad()
    def __call__(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
