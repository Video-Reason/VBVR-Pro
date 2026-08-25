"""DanceGRPO trainer for Wan I2V.

This module keeps the algorithm-specific policy objective while execution
topologies live in focused runtime mixins.
"""

from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_grpo_trainer import BaseGRPOTrainer, _repeat_meta
from src.trainer.dancegrpo_common import (
    _batch_prompt_size,
    _interleave_actor_ranks_by_node,
    _shared_prompt_assignment,
    _shared_prompt_wave_ranges,
    _SharedPromptRollout,
    _SharedPromptStepRollout,
    _slice_meta,
    _slice_prompt_batch,
    _split_group_indices,
)
from src.trainer.dancegrpo_shared_prompt import DanceGRPOSharedPromptMixin
from src.trainer.dancegrpo_split_runtime import DanceGRPOSplitRuntimeMixin


class DanceGRPOTrainer(DanceGRPOSplitRuntimeMixin, DanceGRPOSharedPromptMixin, BaseGRPOTrainer):
    """Paper-inspired GRPO variant with shared group noise and timestep selection."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.expert_parallel:
            raise NotImplementedError(
                "DanceGRPOTrainer currently supports standard FSDP/HSDP paths only; "
                "expert_parallel remains specific to the custom GRPO trainer."
            )
        if cfg.grpo_group_size < 2:
            raise ValueError("DanceGRPO requires grpo_group_size >= 2 for group-relative advantages")
        if cfg.grpo_num_sampling_steps < 2:
            raise ValueError("DanceGRPO requires grpo_num_sampling_steps >= 2")
        cps_noise_range = cfg.grpo_cps_noise_scale_range
        if cps_noise_range is not None and cfg.grpo_sde_formula != "flowcps":
            raise ValueError("grpo_cps_noise_scale_range is only supported with grpo_sde_formula='flowcps'")
        if cps_noise_range is None and cfg.grpo_sde_noise_scale <= 0:
            raise ValueError("DanceGRPO requires grpo_sde_noise_scale > 0")
        if cfg.grpo_sde_formula == "flowcps" and cps_noise_range is None and cfg.grpo_sde_noise_scale > 1:
            raise ValueError("Flow-CPS requires grpo_sde_noise_scale <= 1")
        if self.rl_split_enabled and cfg.grpo_shared_prompt_batch:
            raise ValueError("grpo_shared_prompt_batch is for non-split all-rank GRPO; disable split RL")
        if self.expert_parallel and cfg.grpo_shared_prompt_batch:
            raise ValueError("grpo_shared_prompt_batch does not support expert_parallel")
        if self.rl_split_enabled and cfg.rl_actor_weight_sync == "lora" and cfg.lora_rank <= 0:
            raise ValueError(
                "Split DanceGRPO with rl_actor_weight_sync='lora' requires lora_rank > 0; "
                "use rl_actor_weight_sync='full' for full-finetune split RL"
            )
        if self.rl_split_enabled and cfg.rl_actor_weight_sync == "full" and not cfg.rl_async_rollout:
            raise ValueError("Split DanceGRPO full actor weight sync requires rl_async_rollout=true")
        logger.info(
            "DanceGRPO | sde_formula={} shared_group_init_noise={} timestep_selection_ratio={:.2f} "
            "train_sample_batch_size={} split_debug_logs={} cps_noise={}",
            cfg.grpo_sde_formula,
            cfg.dancegrpo_share_group_init_noise,
            cfg.dancegrpo_timestep_selection_ratio,
            cfg.grpo_train_sample_batch_size,
            cfg.rl_split_debug_logs,
            list(cps_noise_range) if cps_noise_range is not None else cfg.grpo_sde_noise_scale,
        )

    def _sample_group_initial_latents(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None:
        """Sample one x_T per prompt for reuse across the whole group."""
        if not self.cfg.dancegrpo_share_group_init_noise:
            return None
        latent_shape = self.model.latent_shape_from_condition(condition)
        return torch.randn(latent_shape, device=condition.device, dtype=torch.bfloat16, generator=generator)

    def _sample_group_cps_noise_levels(
        self,
        num_prompts: int,
        *,
        step: int,
        stream_id: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Sample one Flow-CPS coefficient per prompt/GRPO group.

        The stateless seed lets split rollout actors and shared-prompt ranks
        independently reproduce the same prompt coefficients. Callers repeat
        each value across that prompt's G rollout samples.
        """
        if self.cfg.grpo_sde_formula != "flowcps":
            return None

        noise_range = self.cfg.grpo_cps_noise_scale_range
        if noise_range is None:
            return torch.full(
                (num_prompts,),
                float(self.cfg.grpo_sde_noise_scale),
                device=device,
                dtype=torch.float32,
            )

        low, high = noise_range
        seed = (self.cfg.seed + 1_000_003 * int(step) + 104_729 * int(stream_id) + 53) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        levels = torch.rand(num_prompts, generator=generator, dtype=torch.float32)
        return (low + (high - low) * levels).to(device=device)

    def _select_training_timesteps(self, num_sampling_steps: int) -> list[int]:
        """Subsample replay timesteps following DanceGRPO's timestep selection idea.

        The selection is generated on rank 0 and broadcast to all ranks so that
        every rank replays the same timesteps.  This is required because
        ``_get_expert_for_timestep`` routes to different FSDP-wrapped modules
        depending on the timestep value — if ranks diverge, the per-parameter
        all-gather in FSDP2 will deadlock.
        """
        candidate_steps = max(1, num_sampling_steps - 1)
        keep = max(1, int(candidate_steps * self.cfg.dancegrpo_timestep_selection_ratio))
        if keep >= candidate_steps:
            return list(range(candidate_steps))
        if self.rank == 0:
            selected = torch.randperm(candidate_steps, device=self.device)[:keep].sort().values
        else:
            selected = torch.empty(keep, dtype=torch.long, device=self.device)
        dist.broadcast(selected, src=0)
        return selected.tolist()

    def _select_training_timesteps_for_step(self, num_sampling_steps: int, step: int) -> list[int]:
        """Deterministic timestep selection for split rollout/train ranks."""
        candidate_steps = max(1, num_sampling_steps - 1)
        keep = max(1, int(candidate_steps * self.cfg.dancegrpo_timestep_selection_ratio))
        if keep >= candidate_steps:
            return list(range(candidate_steps))
        generator = torch.Generator(device="cpu").manual_seed(self.cfg.seed + 1_000_003 * int(step))
        selected = torch.randperm(candidate_steps, generator=generator)[:keep].sort().values
        return selected.tolist()

    def _policy_forward(
        self,
        transformer: torch.nn.Module,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward the trainable policy with optional CFG, matching rollout."""
        B = latent.shape[0]
        device = latent.device
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.float32).expand(B)
        model_input = self.model._build_model_input(latent, condition)
        timestep_input = self.model._build_timestep_input(timestep_tensor, latent, transformer)
        hidden_states, timestep_input, encoder_hidden_states = self.model._prepare_transformer_call(
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

        if self.cfg.grpo_cfg_scale <= 1.0:
            return model_output

        uncond_output = transformer(
            hidden_states=hidden_states,
            timestep=timestep_input,
            encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
            return_dict=False,
        )[0]
        return uncond_output.to(torch.float32) + self.cfg.grpo_cfg_scale * (
            model_output.to(torch.float32) - uncond_output.to(torch.float32)
        )

    def _ref_forward_cfg(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep_val: float,
    ) -> torch.Tensor:
        """Forward the reference policy with the same CFG semantics as rollout."""
        ref_output = self._ref_forward(latent, condition, prompt_embeds, timestep_val)
        if self.cfg.grpo_cfg_scale <= 1.0:
            return ref_output

        uncond_ref_output = self._ref_forward(latent, condition, torch.zeros_like(prompt_embeds), timestep_val)
        return uncond_ref_output.to(torch.float32) + self.cfg.grpo_cfg_scale * (
            ref_output.to(torch.float32) - uncond_ref_output.to(torch.float32)
        )

    def _transition_mean(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sde_noise_scale: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        cfg = self.cfg
        return self.model._sde_transition_mean(
            sample=sample,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            sde_formula=cfg.grpo_sde_formula,
            sde_noise_scale=cfg.grpo_sde_noise_scale if sde_noise_scale is None else sde_noise_scale,
            sigma_min=cfg.grpo_sde_sigma_min,
            sigma_max=cfg.grpo_sde_sigma_max,
        )

    def _transition_log_prob(
        self,
        sample: torch.Tensor,
        mean: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        return self.model._sde_transition_log_prob(
            self.cfg.grpo_sde_formula, sample, mean, noise_scale, skip_first_frame=self.model.expand_timesteps
        )

    def _transition_kl_loss(
        self,
        mean: torch.Tensor,
        ref_mean: torch.Tensor,
        noise_scale: float,
    ) -> torch.Tensor:
        return self.model._sde_transition_kl_loss(
            self.cfg.grpo_sde_formula, mean, ref_mean, noise_scale, skip_first_frame=self.model.expand_timesteps
        )

    def _rollout_video_root(self) -> Path:
        if self.cfg.grpo_rollout_video_dir:
            return Path(self.cfg.grpo_rollout_video_dir)
        return Path(self.cfg.output_dir) / "rollout_videos"

    def _maybe_save_rollout_videos(
        self,
        *,
        final_latents: torch.Tensor,
        step_idx: int,
        prompt_idx: int,
        groups: list[int],
        saved_count: int,
    ) -> int:
        cfg = self.cfg
        if not cfg.grpo_save_rollout_videos:
            return saved_count
        if self.model.vae is None:
            raise RuntimeError("grpo_save_rollout_videos requires the VAE to be loaded")
        if step_idx % cfg.grpo_rollout_video_every_steps != 0:
            return saved_count

        remaining = max(0, cfg.grpo_rollout_video_max_per_rank - saved_count)
        if remaining <= 0:
            return saved_count
        count = min(remaining, final_latents.shape[0], len(groups))
        if count <= 0:
            return saved_count

        out_dir = self._rollout_video_root() / f"step_{step_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            decoded = self.model.decode_latents(final_latents[:count])
        decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        decoded = decoded.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

        from diffusers.utils import export_to_video
        from PIL import Image

        for local_idx in range(count):
            group_idx = int(groups[local_idx])
            path = out_dir / f"rank{self.global_rank:03d}_prompt{int(prompt_idx):03d}_group{group_idx:03d}.mp4"
            frames = [Image.fromarray(frame) for frame in decoded[local_idx]]
            export_to_video(frames, str(path), fps=cfg.grpo_rollout_video_fps)

        del decoded
        return saved_count + count

    def _grpo_step(self, batch: dict) -> dict[str, float]:
        """DanceGRPO-style GRPO step on the standard single-group path."""
        if self.cfg.grpo_delayed_replay:
            return self._grpo_step_shared_prompt_batch_delayed(batch)
        if self.cfg.grpo_shared_prompt_batch:
            return self._grpo_step_shared_prompt_batch(batch)

        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(batch)
        B = gt_video_latents.shape[0]
        prompt_cps_noise_levels = self._sample_group_cps_noise_levels(
            B,
            step=int(self.train_state.step),
            stream_id=int(self.dp_rank if self.tensor_parallel_enabled else self.global_rank),
            device=device,
        )
        selected_t_idxs = self._select_training_timesteps(T)
        if self.rank == 0:
            logger.info(
                "DanceGRPO replay timesteps: keep {}/{} steps ({})",
                len(selected_t_idxs),
                T,
                selected_t_idxs,
            )

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        all_chunk_trajs = []
        pending_reward_chunks = []
        shared_initial_latent = self._sample_group_initial_latents(condition)

        for g_start in range(0, G, S):
            cur_s = min(S, G - g_start)
            cond_s = condition.repeat_interleave(cur_s, dim=0)
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0)
            gt_s = gt_video_latents.repeat_interleave(cur_s, dim=0)
            meta_s = _repeat_meta(meta, cur_s)
            initial_latent = (
                shared_initial_latent.repeat_interleave(cur_s, dim=0) if shared_initial_latent is not None else None
            )
            chunk_cps_noise_levels = (
                prompt_cps_noise_levels.repeat_interleave(cur_s) if prompt_cps_noise_levels is not None else None
            )

            traj = self.model.sde_generate(
                condition=cond_s,
                prompt_embeds=pe_s,
                num_sampling_steps=T,
                sde_noise_scale=(
                    chunk_cps_noise_levels if chunk_cps_noise_levels is not None else cfg.grpo_sde_noise_scale
                ),
                sigma_min=cfg.grpo_sde_sigma_min,
                sigma_max=cfg.grpo_sde_sigma_max,
                cfg_scale=cfg.grpo_cfg_scale,
                initial_latent=initial_latent,
                sde_formula=cfg.grpo_sde_formula,
            )

            reward_submission = self._submit_reward(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            pending_reward_chunks.append((reward_submission, cur_s))

            del traj["noises"]
            traj["latents"] = [x.to("cpu", non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to("cpu", non_blocking=True) for x in traj["log_probs"]]
            if chunk_cps_noise_levels is not None:
                traj["cps_noise_levels"] = chunk_cps_noise_levels.to("cpu", non_blocking=True)
            all_chunk_trajs.append((traj, cur_s))

        reward_chunks = [
            self._resolve_reward(reward_submission).view(B, cur_s) for reward_submission, cur_s in pending_reward_chunks
        ]
        rewards = torch.cat(reward_chunks, dim=1)
        advantages = self._compute_advantages(rewards)
        self._offload_inference_models_for_replay()

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        num_chunks = len(all_chunk_trajs)
        metric_normalizer = max(len(selected_t_idxs) * G, 1)

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        g_offset = 0
        for chunk_idx, (traj, cur_s) in enumerate(all_chunk_trajs):
            BS = B * cur_s
            cond_s = condition.repeat_interleave(cur_s, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0).detach()
            adv_chunk = advantages[:, g_offset : g_offset + cur_s].reshape(BS)
            g_offset += cur_s

            traj["latents"] = [x.to(device, non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to(device, non_blocking=True) for x in traj["log_probs"]]
            cps_noise_levels = traj.get("cps_noise_levels")
            if cps_noise_levels is not None:
                cps_noise_levels = cps_noise_levels.to(device, non_blocking=True)

            for replay_idx, t_idx in enumerate(selected_t_idxs):
                is_last = chunk_idx == num_chunks - 1 and replay_idx == len(selected_t_idxs) - 1
                self._set_requires_gradient_sync(is_last)

                sigma = traj["sigmas"][t_idx].item()
                sigma_prev = traj["sigmas"][t_idx + 1].item()
                latent = traj["latents"][t_idx].detach()
                next_latent = traj["latents"][t_idx + 1].detach()
                old_log_prob = traj["log_probs"][t_idx].detach()
                timestep_val = traj["timesteps"][t_idx]

                transformer = self.model._get_expert_for_timestep(timestep_val)
                model_output = self._policy_forward(transformer, latent, cond_s, pe_s, timestep_val)
                prev_mean, noise_scale = self._transition_mean(
                    sample=latent,
                    model_output=model_output,
                    sigma=sigma,
                    sigma_prev=sigma_prev,
                    sde_noise_scale=cps_noise_levels,
                )
                new_log_prob = self._transition_log_prob(next_latent, prev_mean, noise_scale)

                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - cfg.grpo_clip_range, 1.0 + cfg.grpo_clip_range)
                policy_loss = torch.max(unclipped, clipped).mean()

                kl_loss = torch.tensor(0.0, device=device)
                if cfg.grpo_kl_coeff > 0:
                    with torch.no_grad():
                        ref_output = self._ref_forward_cfg(latent, cond_s, pe_s, timestep_val)
                    ref_mean, _ = self._transition_mean(
                        sample=latent,
                        model_output=ref_output,
                        sigma=sigma,
                        sigma_prev=sigma_prev,
                        sde_noise_scale=cps_noise_levels,
                    )
                    kl_loss = self._transition_kl_loss(prev_mean, ref_mean, noise_scale)

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) * (cur_s / metric_normalizer)
                loss.backward()

                total_policy_loss += policy_loss.item() * cur_s
                total_kl_loss += kl_loss.item() * cur_s

            del traj["latents"], traj["log_probs"]
            if "cps_noise_levels" in traj:
                del traj["cps_noise_levels"]

        return {
            "policy_loss": total_policy_loss / metric_normalizer,
            "kl_loss": total_kl_loss / metric_normalizer,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
        }


__all__ = [
    "DanceGRPOTrainer",
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
