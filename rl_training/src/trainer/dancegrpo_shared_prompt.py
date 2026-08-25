"""Shared-prompt and delayed-replay execution for DanceGRPO."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_grpo_trainer import _repeat_meta
from src.trainer.dancegrpo_common import (
    _batch_prompt_size,
    _shared_prompt_assignment,
    _shared_prompt_wave_ranges,
    _SharedPromptRollout,
    _SharedPromptStepRollout,
    _slice_prompt_batch,
)


class DanceGRPOSharedPromptMixin:
    """Prepare and replay shared-prompt DanceGRPO trajectories."""

    def _gather_shared_prompt_rewards(
        self,
        *,
        prompt_batch_size: int,
        prompt_idx: int,
        groups: list[int],
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        payload = {
            "prompt_idx": int(prompt_idx),
            # A TP group owns one logical rollout. Only TP rank 0 contributes
            # it to the global reward table; the partner participates in the
            # Gloo collective with an empty payload.
            "groups": [int(group) for group in groups] if not self.tensor_parallel_enabled or self.tp_rank == 0 else [],
            "rewards": rewards.detach().cpu()
            if not self.tensor_parallel_enabled or self.tp_rank == 0
            else torch.empty(0, dtype=torch.float32),
        }
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, payload, group=self._checkpoint_pg)

        full_rewards = torch.full((prompt_batch_size, self.cfg.grpo_group_size), float("nan"), dtype=torch.float32)
        active_ranks = 0
        seen: set[tuple[int, int]] = set()
        for item in gathered:
            if not isinstance(item, dict):
                continue
            item_prompt_idx = int(item["prompt_idx"])
            item_groups = [int(group) for group in item["groups"]]
            item_rewards = item["rewards"].float().view(-1)
            if item_groups:
                active_ranks += 1
            for local_idx, group_idx in enumerate(item_groups):
                key = (item_prompt_idx, group_idx)
                if key in seen:
                    raise RuntimeError(f"Duplicate shared-prompt reward for prompt={item_prompt_idx} group={group_idx}")
                full_rewards[item_prompt_idx, group_idx] = item_rewards[local_idx]
                seen.add(key)

        missing = torch.isnan(full_rewards).nonzero(as_tuple=False)
        if missing.numel() > 0:
            missing_items = [(int(row[0]), int(row[1])) for row in missing[:16]]
            raise RuntimeError(f"Missing shared-prompt rewards, first missing prompt/group entries: {missing_items}")
        return full_rewards, active_ranks

    def _grpo_step_shared_prompt_batch_legacy(self, batch: dict) -> dict[str, float]:
        """All-rank GRPO where ranks shard group samples for a fixed global prompt batch."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device
        debug_logs = self._split_debug_enabled()

        def debug_sync() -> None:
            if debug_logs and device.type == "cuda":
                torch.cuda.synchronize(device)

        step_start = time.monotonic()

        prompt_batch_size = _batch_prompt_size(batch)
        prompt_idx, prompt_rank, prompt_world_size, group_indices = _shared_prompt_assignment(
            self.dp_rank if self.tensor_parallel_enabled else self.rank,
            self.dp_size if self.tensor_parallel_enabled else self.world_size,
            prompt_batch_size,
            G,
        )
        selected_t_idxs = self._select_training_timesteps(T)
        if self.rank == 0:
            logger.info(
                "DanceGRPO shared prompt batch: prompts={} world={} ranks_per_prompt={} groups_per_rank={} "
                "sample_batch_size={} replay_t={}",
                prompt_batch_size,
                self.dp_size if self.tensor_parallel_enabled else self.world_size,
                prompt_world_size,
                len(group_indices),
                S,
                selected_t_idxs,
            )
        self._split_debug_log(
            "shared_prompt_step_start global_rank={} local_rank={} step={} prompt={}/{} prompt_rank={}/{} "
            "groups={} formula={} rollout_batch={} replay_t={}",
            self.global_rank,
            self.local_rank,
            int(self.train_state.step),
            prompt_idx,
            prompt_batch_size,
            prompt_rank,
            prompt_world_size,
            group_indices,
            cfg.grpo_sde_formula,
            S,
            selected_t_idxs,
        )

        encode_start = time.monotonic()
        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(
            _slice_prompt_batch(batch, prompt_idx, prompt_batch_size)
        )
        all_prompt_cps_noise_levels = self._sample_group_cps_noise_levels(
            prompt_batch_size,
            step=int(self.train_state.step),
            stream_id=0,
            device=device,
        )
        prompt_cps_noise_levels = (
            all_prompt_cps_noise_levels[prompt_idx : prompt_idx + 1]
            if all_prompt_cps_noise_levels is not None
            else None
        )
        debug_sync()
        self._split_debug_log(
            "shared_prompt_encode_done rank={} step={} prompt={} batch={} seconds={:.2f}",
            self.global_rank,
            int(self.train_state.step),
            prompt_idx,
            prompt_batch_size,
            time.monotonic() - encode_start,
        )

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        local_chunks = []
        pending_reward_parts = []
        saved_rollout_videos = 0
        rollout_step_idx = int(self.train_state.step) + 1
        init_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1_000_003 * self.train_state.step + 17)
        shared_initial_latent = self._sample_group_initial_latents(condition, generator=init_generator)

        for offset in range(0, len(group_indices), S):
            chunk_start = time.monotonic()
            groups = group_indices[offset : offset + S]
            cur_s = len(groups)
            self._split_debug_log(
                "shared_prompt_rollout_chunk_start rank={} step={} prompt={} groups={}",
                self.global_rank,
                int(self.train_state.step),
                prompt_idx,
                groups,
            )
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
            rollout_seed = (
                cfg.seed
                + 1_000_003 * self.train_state.step
                + 9_176 * (prompt_idx + 1)
                + 503 * (prompt_rank + 1)
                + 131 * offset
            )
            rollout_generator = torch.Generator(device=device).manual_seed(rollout_seed)
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
                generator=rollout_generator,
                initial_latent=initial_latent,
                sde_formula=cfg.grpo_sde_formula,
            )
            debug_sync()
            rollout_seconds = time.monotonic() - chunk_start
            reward_submit_start = time.monotonic()
            reward_submission = self._submit_reward(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            debug_sync()
            reward_submit_seconds = time.monotonic() - reward_submit_start
            saved_rollout_videos = self._maybe_save_rollout_videos(
                final_latents=traj["latents"][-1],
                step_idx=rollout_step_idx,
                prompt_idx=prompt_idx,
                groups=groups,
                saved_count=saved_rollout_videos,
            )
            pending_reward_parts.append((reward_submission, groups))
            del traj["noises"]
            traj["latents"] = [x.to("cpu", non_blocking=True) for x in traj["latents"]]
            traj["log_probs"] = [x.to("cpu", non_blocking=True) for x in traj["log_probs"]]
            if chunk_cps_noise_levels is not None:
                traj["cps_noise_levels"] = chunk_cps_noise_levels.to("cpu", non_blocking=True)
            local_chunks.append((traj, groups))
            debug_sync()
            self._split_debug_log(
                "shared_prompt_rollout_chunk_done rank={} step={} prompt={} groups={} rollout={:.2f}s "
                "reward_submit={:.2f}s total={:.2f}s",
                self.global_rank,
                int(self.train_state.step),
                prompt_idx,
                groups,
                rollout_seconds,
                reward_submit_seconds,
                time.monotonic() - chunk_start,
            )

        reward_drain_start = time.monotonic()
        reward_parts = []
        for reward_submission, groups in pending_reward_parts:
            reward_flat = self._resolve_reward(reward_submission)
            reward_parts.append(reward_flat.view(-1))
            self._split_debug_log(
                "shared_prompt_reward_resolved rank={} step={} prompt={} groups={} reward_mean={:.4f}",
                self.global_rank,
                int(self.train_state.step),
                prompt_idx,
                groups,
                float(reward_flat.float().mean().item()),
            )
        self._split_debug_log(
            "shared_prompt_reward_pipeline_done rank={} step={} prompt={} chunks={} drain={:.2f}s",
            self.global_rank,
            int(self.train_state.step),
            prompt_idx,
            len(pending_reward_parts),
            time.monotonic() - reward_drain_start,
        )
        local_rewards = torch.cat(reward_parts, dim=0)
        gather_start = time.monotonic()
        self._split_debug_log(
            "shared_prompt_reward_gather_start rank={} step={} prompt={} groups={}",
            self.global_rank,
            int(self.train_state.step),
            prompt_idx,
            group_indices,
        )
        rewards, active_ranks = self._gather_shared_prompt_rewards(
            prompt_batch_size=prompt_batch_size,
            prompt_idx=prompt_idx,
            groups=group_indices,
            rewards=local_rewards,
        )
        debug_sync()
        self._split_debug_log(
            "shared_prompt_reward_gather_done rank={} step={} prompt={} active_ranks={} seconds={:.2f}",
            self.global_rank,
            int(self.train_state.step),
            prompt_idx,
            active_ranks,
            time.monotonic() - gather_start,
        )
        rewards = rewards.to(device=device, non_blocking=True)
        advantages = self._compute_advantages(rewards)
        self._offload_inference_models_for_replay()

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        local_policy_sum = 0.0
        local_kl_sum = 0.0
        gradient_normalizer = max(
            len(selected_t_idxs)
            * G
            * prompt_batch_size
            / (self.dp_size if self.tensor_parallel_enabled else self.world_size),
            1.0,
        )
        num_chunks = len(local_chunks)

        for chunk_idx, (traj, groups) in enumerate(local_chunks):
            replay_chunk_start = time.monotonic()
            self._split_debug_log(
                "shared_prompt_replay_chunk_start rank={} step={} prompt={} chunk={}/{} groups={}",
                self.global_rank,
                int(self.train_state.step),
                prompt_idx,
                chunk_idx + 1,
                num_chunks,
                groups,
            )
            cur_s = len(groups)
            cond_s = condition.repeat_interleave(cur_s, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0).detach()
            adv_chunk = advantages[prompt_idx, groups].reshape(cur_s)

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

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) * (cur_s / gradient_normalizer)
                loss.backward()
                local_policy_sum += policy_loss.item() * cur_s
                local_kl_sum += kl_loss.item() * cur_s

            del traj["latents"], traj["log_probs"]
            if "cps_noise_levels" in traj:
                del traj["cps_noise_levels"]
            debug_sync()
            self._split_debug_log(
                "shared_prompt_replay_chunk_done rank={} step={} prompt={} chunk={}/{} groups={} seconds={:.2f}",
                self.global_rank,
                int(self.train_state.step),
                prompt_idx,
                chunk_idx + 1,
                num_chunks,
                groups,
                time.monotonic() - replay_chunk_start,
            )

        reduce_start = time.monotonic()
        self._split_debug_log(
            "shared_prompt_metric_reduce_start rank={} step={}",
            self.global_rank,
            int(self.train_state.step),
        )
        metric_buf = torch.tensor([local_policy_sum, local_kl_sum], device=device, dtype=torch.float32)
        dist.all_reduce(
            metric_buf,
            op=dist.ReduceOp.SUM,
            group=self._dp_pg if self.tensor_parallel_enabled else None,
        )
        debug_sync()
        self._split_debug_log(
            "shared_prompt_metric_reduce_done rank={} step={} seconds={:.2f} total_step_inner={:.2f}s",
            self.global_rank,
            int(self.train_state.step),
            time.monotonic() - reduce_start,
            time.monotonic() - step_start,
        )
        metric_normalizer = max(len(selected_t_idxs) * G * prompt_batch_size, 1)
        return {
            "policy_loss": (metric_buf[0] / metric_normalizer).item(),
            "kl_loss": (metric_buf[1] / metric_normalizer).item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "active_rollout_ranks": float(active_ranks),
        }

    def _prepare_shared_prompt_rollout_wave(
        self,
        batch: dict,
        *,
        rollout_step: int,
        prompt_offset: int,
        prompt_batch_size: int,
        total_prompt_batch_size: int,
        all_prompt_cps_noise_levels: torch.Tensor | None,
        saved_rollout_videos: int,
    ) -> tuple[_SharedPromptRollout, int]:
        """Generate one prompt wave and submit its rewards without waiting."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        device = self.device
        debug_logs = self._split_debug_enabled()

        def debug_sync() -> None:
            if debug_logs and device.type == "cuda":
                torch.cuda.synchronize(device)

        prepare_start = time.monotonic()
        prompt_idx, prompt_rank, prompt_world_size, group_indices = _shared_prompt_assignment(
            self.dp_rank if self.tensor_parallel_enabled else self.rank,
            self.dp_size if self.tensor_parallel_enabled else self.world_size,
            prompt_batch_size,
            G,
        )
        global_prompt_idx = prompt_offset + prompt_idx

        encode_start = time.monotonic()
        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(
            _slice_prompt_batch(batch, global_prompt_idx, total_prompt_batch_size)
        )
        prompt_cps_noise_levels = (
            all_prompt_cps_noise_levels[global_prompt_idx : global_prompt_idx + 1]
            if all_prompt_cps_noise_levels is not None
            else None
        )
        debug_sync()
        self._split_debug_log(
            "shared_prompt_wave_encode_done rank={} step={} wave_offset={} prompt={}/{} "
            "prompt_rank={}/{} groups={} seconds={:.2f}",
            self.global_rank,
            int(self.train_state.step),
            prompt_offset,
            global_prompt_idx,
            total_prompt_batch_size,
            prompt_rank,
            prompt_world_size,
            group_indices,
            time.monotonic() - encode_start,
        )

        local_chunks: list[tuple[dict[str, Any], list[int]]] = []
        pending_reward_parts: list[tuple[Any, list[int]]] = []
        rollout_step_idx = int(rollout_step) + 1
        init_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1_000_003 * rollout_step + 17)
        shared_initial_latent = self._sample_group_initial_latents(condition, generator=init_generator)

        for offset in range(0, len(group_indices), S):
            chunk_start = time.monotonic()
            groups = group_indices[offset : offset + S]
            cur_s = len(groups)
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
            rollout_seed = (
                cfg.seed
                + 1_000_003 * rollout_step
                + 9_176 * (global_prompt_idx + 1)
                + 503 * (prompt_rank + 1)
                + 131 * offset
            )
            rollout_generator = torch.Generator(device=device).manual_seed(rollout_seed)
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
                generator=rollout_generator,
                initial_latent=initial_latent,
                sde_formula=cfg.grpo_sde_formula,
            )
            debug_sync()
            rollout_seconds = time.monotonic() - chunk_start

            reward_submit_start = time.monotonic()
            reward_submission = self._submit_reward(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            debug_sync()
            reward_submit_seconds = time.monotonic() - reward_submit_start
            saved_rollout_videos = self._maybe_save_rollout_videos(
                final_latents=traj["latents"][-1],
                step_idx=rollout_step_idx,
                prompt_idx=global_prompt_idx,
                groups=groups,
                saved_count=saved_rollout_videos,
            )
            pending_reward_parts.append((reward_submission, groups))

            del traj["noises"]
            traj["latents"] = [tensor.to("cpu", non_blocking=True) for tensor in traj["latents"]]
            traj["log_probs"] = [tensor.to("cpu", non_blocking=True) for tensor in traj["log_probs"]]
            if chunk_cps_noise_levels is not None:
                traj["cps_noise_levels"] = chunk_cps_noise_levels.to("cpu", non_blocking=True)
            local_chunks.append((traj, groups))
            debug_sync()
            self._split_debug_log(
                "shared_prompt_wave_chunk_done rank={} step={} wave_offset={} prompt={} groups={} "
                "rollout={:.2f}s reward_submit={:.2f}s total={:.2f}s",
                self.global_rank,
                int(self.train_state.step),
                prompt_offset,
                global_prompt_idx,
                groups,
                rollout_seconds,
                reward_submit_seconds,
                time.monotonic() - chunk_start,
            )

        debug_sync()
        rollout = _SharedPromptRollout(
            prompt_offset=prompt_offset,
            prompt_batch_size=prompt_batch_size,
            prompt_idx=prompt_idx,
            global_prompt_idx=global_prompt_idx,
            prompt_rank=prompt_rank,
            prompt_world_size=prompt_world_size,
            group_indices=group_indices,
            prompt_embeds=prompt_embeds,
            condition=condition,
            local_chunks=local_chunks,
            pending_reward_parts=pending_reward_parts,
            prepare_seconds=time.monotonic() - prepare_start,
        )
        return rollout, saved_rollout_videos

    def _replay_shared_prompt_rollout_wave(
        self,
        rollout: _SharedPromptRollout,
        *,
        selected_t_idxs: list[int],
        total_prompt_batch_size: int,
        sync_on_last_backward: bool,
        clip_range: float,
    ) -> dict[str, Any]:
        """Resolve one wave and replay it while later waves keep scoring."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        device = self.device
        debug_logs = self._split_debug_enabled()

        def debug_sync() -> None:
            if debug_logs and device.type == "cuda":
                torch.cuda.synchronize(device)

        reward_drain_start = time.monotonic()
        reward_parts = []
        for reward_submission, groups in rollout.pending_reward_parts:
            reward_flat = self._resolve_reward(reward_submission)
            reward_parts.append(reward_flat.view(-1))
            self._split_debug_log(
                "shared_prompt_wave_reward_resolved rank={} step={} wave_offset={} prompt={} "
                "groups={} reward_mean={:.4f}",
                self.global_rank,
                int(self.train_state.step),
                rollout.prompt_offset,
                rollout.global_prompt_idx,
                groups,
                float(reward_flat.float().mean().item()),
            )
        local_rewards = torch.cat(reward_parts, dim=0)
        rewards, active_ranks = self._gather_shared_prompt_rewards(
            prompt_batch_size=rollout.prompt_batch_size,
            prompt_idx=rollout.prompt_idx,
            groups=rollout.group_indices,
            rewards=local_rewards,
        )
        debug_sync()
        reward_drain_seconds = time.monotonic() - reward_drain_start
        rewards = rewards.to(device=device, non_blocking=True)
        advantages = self._compute_advantages(rewards)

        local_policy_sum = 0.0
        local_kl_sum = 0.0
        local_clip_fraction_sum = 0.0
        local_ratio_sum = 0.0
        local_approx_kl_sum = 0.0
        local_ratio_abs_max = 0.0
        gradient_normalizer = max(
            len(selected_t_idxs)
            * G
            * total_prompt_batch_size
            / (self.dp_size if self.tensor_parallel_enabled else self.world_size),
            1.0,
        )
        num_chunks = len(rollout.local_chunks)
        replay_start = time.monotonic()

        for chunk_idx, (traj, groups) in enumerate(rollout.local_chunks):
            replay_chunk_start = time.monotonic()
            cur_s = len(groups)
            cond_s = rollout.condition.repeat_interleave(cur_s, dim=0).detach()
            pe_s = rollout.prompt_embeds.repeat_interleave(cur_s, dim=0).detach()
            adv_chunk = advantages[rollout.prompt_idx, groups].reshape(cur_s)

            traj["latents"] = [tensor.to(device, non_blocking=True) for tensor in traj["latents"]]
            traj["log_probs"] = [tensor.to(device, non_blocking=True) for tensor in traj["log_probs"]]
            cps_noise_levels = traj.get("cps_noise_levels")
            if cps_noise_levels is not None:
                cps_noise_levels = cps_noise_levels.to(device, non_blocking=True)

            for replay_idx, t_idx in enumerate(selected_t_idxs):
                is_last_in_wave = chunk_idx == num_chunks - 1 and replay_idx == len(selected_t_idxs) - 1
                self._set_requires_gradient_sync(sync_on_last_backward and is_last_in_wave)

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

                log_ratio = new_log_prob - old_log_prob
                ratio = torch.exp(log_ratio)
                unclipped = -adv_chunk * ratio
                clipped = -adv_chunk * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
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

                loss = (policy_loss + cfg.grpo_kl_coeff * kl_loss) * (cur_s / gradient_normalizer)
                loss.backward()
                local_policy_sum += policy_loss.item() * cur_s
                local_kl_sum += kl_loss.item() * cur_s
                local_clip_fraction_sum += (torch.abs(ratio - 1.0) > clip_range).float().mean().item() * cur_s
                local_ratio_sum += ratio.float().mean().item() * cur_s
                local_approx_kl_sum += (ratio - 1.0 - log_ratio).float().mean().item() * cur_s
                local_ratio_abs_max = max(
                    local_ratio_abs_max,
                    torch.abs(ratio - 1.0).float().max().item(),
                )

            del traj["latents"], traj["log_probs"]
            if "cps_noise_levels" in traj:
                del traj["cps_noise_levels"]
            debug_sync()
            self._split_debug_log(
                "shared_prompt_wave_replay_chunk_done rank={} step={} wave_offset={} prompt={} "
                "chunk={}/{} groups={} seconds={:.2f}",
                self.global_rank,
                int(self.train_state.step),
                rollout.prompt_offset,
                rollout.global_prompt_idx,
                chunk_idx + 1,
                num_chunks,
                groups,
                time.monotonic() - replay_chunk_start,
            )

        debug_sync()
        return {
            "local_policy_sum": local_policy_sum,
            "local_kl_sum": local_kl_sum,
            "local_clip_fraction_sum": local_clip_fraction_sum,
            "local_ratio_sum": local_ratio_sum,
            "local_approx_kl_sum": local_approx_kl_sum,
            "local_ratio_abs_max": local_ratio_abs_max,
            "rewards": rewards,
            "advantages": advantages,
            "active_ranks": active_ranks,
            "reward_drain_seconds": reward_drain_seconds,
            "replay_seconds": time.monotonic() - replay_start,
        }

    def _prepare_shared_prompt_step_rollout(
        self,
        batch: dict,
        *,
        rollout_step: int,
        policy_version: int,
        selected_t_idxs: list[int],
    ) -> _SharedPromptStepRollout:
        """Prepare every prompt wave for one future optimizer update."""
        cfg = self.cfg
        prompt_batch_size = _batch_prompt_size(batch)
        wave_ranges = _shared_prompt_wave_ranges(
            prompt_batch_size,
            cfg.grpo_shared_prompt_microbatch_size,
        )
        G = cfg.grpo_group_size
        device = self.device
        wave_size = wave_ranges[0][1]
        _prompt_idx, _prompt_rank, prompt_world_size, group_indices = _shared_prompt_assignment(
            self.dp_rank if self.tensor_parallel_enabled else self.rank,
            self.dp_size if self.tensor_parallel_enabled else self.world_size,
            wave_size,
            G,
        )
        if self.rank == 0:
            logger.info(
                "DanceGRPO shared prompt rollout: rollout_step={} policy_version={} prompts={} "
                "prompt_wave={} waves={} world={} "
                "ranks_per_prompt={} groups_per_rank={} sample_batch_size={} replay_t={}",
                rollout_step,
                policy_version,
                prompt_batch_size,
                wave_size,
                len(wave_ranges),
                self.dp_size if self.tensor_parallel_enabled else self.world_size,
                prompt_world_size,
                len(group_indices),
                cfg.grpo_sample_batch_size,
                selected_t_idxs,
            )

        all_prompt_cps_noise_levels = self._sample_group_cps_noise_levels(
            prompt_batch_size,
            step=rollout_step,
            stream_id=0,
            device=device,
        )
        for module in (self.model.transformer, self.model.transformer_2):
            if module is not None:
                module.eval()

        rollouts: list[_SharedPromptRollout] = []
        saved_rollout_videos = 0
        for prompt_offset, current_wave_size in wave_ranges:
            rollout, saved_rollout_videos = self._prepare_shared_prompt_rollout_wave(
                batch,
                rollout_step=rollout_step,
                prompt_offset=prompt_offset,
                prompt_batch_size=current_wave_size,
                total_prompt_batch_size=prompt_batch_size,
                all_prompt_cps_noise_levels=all_prompt_cps_noise_levels,
                saved_rollout_videos=saved_rollout_videos,
            )
            rollouts.append(rollout)

        return _SharedPromptStepRollout(
            rollout_step=rollout_step,
            policy_version=policy_version,
            prompt_batch_size=prompt_batch_size,
            selected_t_idxs=selected_t_idxs,
            waves=rollouts,
        )

    def _replay_shared_prompt_step_rollout(
        self,
        step_rollout: _SharedPromptStepRollout,
        *,
        clip_range: float,
    ) -> dict[str, float]:
        """Resolve and replay one prepared prompt batch."""
        cfg = self.cfg
        G = cfg.grpo_group_size
        device = self.device
        debug_logs = self._split_debug_enabled()

        def debug_sync() -> None:
            if debug_logs and device.type == "cuda":
                torch.cuda.synchronize(device)

        replay_step_start = time.monotonic()
        # Reward submission performs VAE decode synchronously, so the frozen
        # inference modules are no longer needed once all waves are prepared.
        self._offload_inference_models_for_replay()
        for module in (self.model.transformer, self.model.transformer_2):
            if module is not None:
                module.train()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        replay_results: list[dict[str, Any]] = []
        for wave_idx, rollout in enumerate(step_rollout.waves):
            replay_results.append(
                self._replay_shared_prompt_rollout_wave(
                    rollout,
                    selected_t_idxs=step_rollout.selected_t_idxs,
                    total_prompt_batch_size=step_rollout.prompt_batch_size,
                    sync_on_last_backward=wave_idx == len(step_rollout.waves) - 1,
                    clip_range=clip_range,
                )
            )

        metric_buf = torch.tensor(
            [
                sum(float(result["local_policy_sum"]) for result in replay_results),
                sum(float(result["local_kl_sum"]) for result in replay_results),
                sum(float(result["local_clip_fraction_sum"]) for result in replay_results),
                sum(float(result["local_ratio_sum"]) for result in replay_results),
                sum(float(result["local_approx_kl_sum"]) for result in replay_results),
            ],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(
            metric_buf,
            op=dist.ReduceOp.SUM,
            group=self._dp_pg if self.tensor_parallel_enabled else None,
        )
        ratio_abs_max_buf = torch.tensor(
            max(float(result["local_ratio_abs_max"]) for result in replay_results),
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(
            ratio_abs_max_buf,
            op=dist.ReduceOp.MAX,
            group=self._dp_pg if self.tensor_parallel_enabled else None,
        )
        phase_buf = torch.tensor(
            [
                sum(rollout.prepare_seconds for rollout in step_rollout.waves),
                sum(float(result["reward_drain_seconds"]) for result in replay_results),
                sum(float(result["replay_seconds"]) for result in replay_results),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(phase_buf, op=dist.ReduceOp.MAX)
        debug_sync()
        self._split_debug_log(
            "shared_prompt_waves_done rank={} step={} waves={} total={:.2f}s "
            "prepare_max={:.2f}s reward_drain_max={:.2f}s replay_max={:.2f}s",
            self.global_rank,
            int(self.train_state.step),
            len(step_rollout.waves),
            time.monotonic() - replay_step_start,
            phase_buf[0].item(),
            phase_buf[1].item(),
            phase_buf[2].item(),
        )

        rewards = torch.cat([result["rewards"] for result in replay_results], dim=0)
        advantages = torch.cat([result["advantages"] for result in replay_results], dim=0)
        metric_normalizer = max(
            len(step_rollout.selected_t_idxs) * G * step_rollout.prompt_batch_size,
            1,
        )
        return {
            "policy_loss": (metric_buf[0] / metric_normalizer).item(),
            "kl_loss": (metric_buf[1] / metric_normalizer).item(),
            "ppo_clip_fraction": (metric_buf[2] / metric_normalizer).item(),
            "ppo_ratio_mean": (metric_buf[3] / metric_normalizer).item(),
            "ppo_approx_kl": (metric_buf[4] / metric_normalizer).item(),
            "ppo_ratio_abs_max": ratio_abs_max_buf.item(),
            "ppo_clip_range": float(clip_range),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "active_rollout_ranks": float(max(int(result["active_ranks"]) for result in replay_results)),
            "shared_prompt_prepare_seconds": phase_buf[0].item(),
            "shared_prompt_reward_drain_seconds": phase_buf[1].item(),
            "shared_prompt_replay_seconds": phase_buf[2].item(),
        }

    def _grpo_step_shared_prompt_batch(self, batch: dict) -> dict[str, float]:
        """Pipeline prompt waves within one optimizer step without policy staleness."""
        cfg = self.cfg
        prompt_batch_size = _batch_prompt_size(batch)
        wave_ranges = _shared_prompt_wave_ranges(
            prompt_batch_size,
            cfg.grpo_shared_prompt_microbatch_size,
        )
        if len(wave_ranges) == 1:
            return self._grpo_step_shared_prompt_batch_legacy(batch)

        rollout_step = int(self.train_state.step)
        step_rollout = self._prepare_shared_prompt_step_rollout(
            batch,
            rollout_step=rollout_step,
            policy_version=rollout_step,
            selected_t_idxs=self._select_training_timesteps_for_step(
                cfg.grpo_num_sampling_steps,
                rollout_step,
            ),
        )
        return self._replay_shared_prompt_step_rollout(
            step_rollout,
            clip_range=float(cfg.grpo_clip_range),
        )

    def _delayed_replay_must_flush(self) -> bool:
        """Return whether the pending slot must be empty after this update."""
        cfg = self.cfg
        next_update = int(self.train_state.step) + 1
        if bool(getattr(self, "_grpo_force_delayed_replay_flush", False)):
            return True
        if cfg.max_steps is not None and next_update >= cfg.max_steps:
            return True
        return bool(cfg.save_steps > 0 and next_update % cfg.save_steps == 0)

    def _max_rank_seconds(self, seconds: float) -> float:
        buf = torch.tensor(float(seconds), device=self.device, dtype=torch.float64)
        dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        return buf.item()

    def _grpo_step_shared_prompt_batch_delayed(self, batch: dict) -> dict[str, float]:
        """Prepare the next rollout, then replay a one-slot-older trajectory."""
        cfg = self.cfg
        optimizer_step = int(self.train_state.step)
        pending: _SharedPromptStepRollout | None = getattr(self, "_delayed_shared_prompt_rollout", None)
        must_flush = self._delayed_replay_must_flush()
        delayed_clip_range = float(cfg.grpo_delayed_replay_clip_range or cfg.grpo_clip_range)
        current: _SharedPromptStepRollout | None = None
        current_prepare_seconds = 0.0

        if pending is None:
            prepare_start = time.monotonic()
            current = self._prepare_shared_prompt_step_rollout(
                batch,
                rollout_step=optimizer_step,
                policy_version=optimizer_step,
                selected_t_idxs=self._select_training_timesteps_for_step(
                    cfg.grpo_num_sampling_steps,
                    optimizer_step,
                ),
            )
            current_prepare_seconds = self._max_rank_seconds(time.monotonic() - prepare_start)
            if not must_flush:
                self._delayed_shared_prompt_rollout = current
                return {
                    "_skip_optimizer_step": True,
                    "delayed_current_prepare_seconds": current_prepare_seconds,
                }
            pending = current
            current = None
        elif not must_flush:
            prepare_start = time.monotonic()
            rollout_step = optimizer_step + 1
            current = self._prepare_shared_prompt_step_rollout(
                batch,
                rollout_step=rollout_step,
                policy_version=optimizer_step,
                selected_t_idxs=self._select_training_timesteps_for_step(
                    cfg.grpo_num_sampling_steps,
                    rollout_step,
                ),
            )
            current_prepare_seconds = self._max_rank_seconds(time.monotonic() - prepare_start)

        self._delayed_shared_prompt_rollout = current
        metrics = self._replay_shared_prompt_step_rollout(
            pending,
            clip_range=delayed_clip_range,
        )
        metrics.update(
            {
                "delayed_replay_staleness": float(max(optimizer_step - pending.policy_version, 0)),
                "delayed_current_prepare_seconds": current_prepare_seconds,
                "delayed_replay_flush": float(must_flush),
                "delayed_rollout_step": float(pending.rollout_step),
                "delayed_policy_version": float(pending.policy_version),
            }
        )
        return metrics


__all__ = ["DanceGRPOSharedPromptMixin"]
