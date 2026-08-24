"""Default reward: negative flow-matching loss against the ground-truth video.

Migrated verbatim from ``BaseGRPOTrainer._compute_reward_neg_loss`` — behavior
is preserved so existing GRPO / DanceGRPO training runs remain unchanged.

For each sample the reward picks a random diffusion timestep, adds noise to
the ground-truth latent at that σ, runs one transformer forward, and returns
``-MSE(pred, target)`` where ``target = ε − x_GT``.  In expert-parallel mode
each rank contributes its expert's share; the trainer sums the two.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward


@register_reward("neg_loss")
class NegLossReward(BaseReward):
    """Reward = -flow_matching_loss of the generated latent against GT."""

    requires_policy_forward = True

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
        meta: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del meta  # neg_loss is model-internal; ignores per-sample metadata
        model = self.trainer.model
        need_dummy_forward = self.cfg.fsdp

        B = gt_video_latents.shape[0]
        device = gt_video_latents.device
        shifted_sigmas, shifted_timesteps, _bsmntw = model._get_training_buffers(device)

        if indices is None:
            if expert_filter == "high":
                indices = torch.randint(0, model.boundary_idx, (B,), device=device)
            elif expert_filter == "low":
                indices = torch.randint(model.boundary_idx, model.num_train_timesteps, (B,), device=device)
            else:
                indices = torch.randint(0, model.num_train_timesteps, (B,), device=device)

        sigmas = shifted_sigmas.index_select(0, indices).view(B, 1, 1, 1, 1)
        timesteps = shifted_timesteps.index_select(0, indices)

        noise = torch.randn_like(gt_video_latents)
        noisy = sigmas * noise + (1.0 - sigmas) * gt_video_latents
        target = noise - gt_video_latents
        model_input = model._build_model_input(noisy, condition)

        rewards = torch.zeros(B, device=device, dtype=torch.float32)

        for expert_name, selected, transformer in model._iter_transformer_selections(timesteps):
            if expert_filter is not None and expert_name != expert_filter:
                continue
            self._run_expert(
                transformer,
                selected,
                model,
                model_input,
                noisy,
                timesteps,
                prompt_embeds,
                target,
                rewards,
                need_dummy_forward=need_dummy_forward,
            )

        return rewards

    def _run_expert(
        self,
        transformer: torch.nn.Module,
        selected: torch.Tensor,
        model,
        model_input: torch.Tensor,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        target: torch.Tensor,
        rewards: torch.Tensor,
        *,
        need_dummy_forward: bool,
    ) -> None:
        """Run one expert's forward and update ``rewards`` in place.

        FSDP requires all ranks to participate in forward (all_gather).  When
        ``selected`` is empty, we run a dummy forward to stay in sync; without
        FSDP we skip entirely.
        """
        if selected.numel() > 0:
            forward_idx = selected
        elif need_dummy_forward:
            forward_idx = torch.zeros(1, device=timesteps.device, dtype=torch.long)
        else:
            return

        timestep_input = model._build_timestep_input(timesteps, latent, transformer)
        sel_input = model_input.index_select(0, forward_idx)
        sel_ts = timestep_input.index_select(0, forward_idx)
        sel_pe = prompt_embeds.index_select(0, forward_idx)
        hidden_states, sel_ts, encoder_hidden_states = model._prepare_transformer_call(
            transformer,
            sel_input,
            sel_ts,
            sel_pe,
        )
        pred = transformer(
            hidden_states=hidden_states,
            timestep=sel_ts,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        if selected.numel() > 0:
            per_sample_loss = F.mse_loss(
                pred.float(),
                target.index_select(0, selected).float(),
                reduction="none",
            )
            per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
            rewards.index_copy_(0, selected, -per_sample_loss)
