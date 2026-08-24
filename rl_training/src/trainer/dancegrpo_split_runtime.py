"""Split train/rollout execution runtime for DanceGRPO."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from src.trainer.base_grpo_trainer import _repeat_meta
from src.trainer.dancegrpo_common import _interleave_actor_ranks_by_node, _split_group_indices
from src.trainer.utils import cosine_lr, format_eta


class DanceGRPOSplitRuntimeMixin:
    """Coordinate synchronous and asynchronous split DanceGRPO execution.

    The concrete trainer supplies the DanceGRPO rollout/replay primitives and
    :class:`BaseGRPOTrainer` supplies the shared optimizer/checkpoint runtime.
    """

    # ------------------------------------------------------------------
    # Split train/rollout execution
    # ------------------------------------------------------------------

    def _split_debug_enabled(self) -> bool:
        return bool(getattr(self.cfg, "rl_split_debug_logs", False) and self.log_enabled)

    def _split_debug_log(self, message: str, *args) -> None:
        if self._split_debug_enabled():
            logger.info(message, *args)

    @staticmethod
    def _split_policy_state_stats(policy_state: dict[str, dict[str, torch.Tensor]] | None) -> tuple[int, int, float]:
        if not policy_state:
            return 0, 0, 0.0
        tensors = 0
        total_bytes = 0
        for module_state in policy_state.values():
            tensors += len(module_state)
            for tensor in module_state.values():
                if torch.is_tensor(tensor):
                    total_bytes += int(tensor.numel()) * int(tensor.element_size())
        return len(policy_state), tensors, total_bytes / (1024.0 * 1024.0)

    @staticmethod
    def _split_payload_group_count(payload: dict[str, Any]) -> int:
        return sum(len(chunk.get("groups", [])) for chunk in payload.get("chunks") or [])

    @staticmethod
    def _split_payload_reward_mean(payload: dict[str, Any]) -> float:
        reward_sum = 0.0
        reward_count = 0
        for chunk in payload.get("chunks") or []:
            rewards = chunk.get("rewards")
            if torch.is_tensor(rewards) and rewards.numel() > 0:
                rewards_f = rewards.float()
                reward_sum += float(rewards_f.sum().item())
                reward_count += int(rewards_f.numel())
        if reward_count == 0:
            return float("nan")
        return reward_sum / reward_count

    def train(self):
        if not self.rl_split_enabled:
            return super().train()
        if self.cfg.rl_async_rollout:
            if self.is_inference_rank:
                return self._run_async_split_inference_loop()
            return self._run_async_split_train_loop()
        if self.is_inference_rank:
            return self._run_split_inference_loop()
        return self._run_split_train_loop()

    def _collect_trainable_policy_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Collect trainable policy weights on the split train root."""
        collect_start = time.monotonic()
        debug_logs = self._split_debug_enabled()
        if debug_logs:
            self._split_debug_log(
                "split_policy_collect_start role={} global_rank={} sync_mode={} fsdp={}",
                self.rl_role,
                self.global_rank,
                self.cfg.rl_actor_weight_sync,
                self.cfg.fsdp,
            )
        if self.cfg.rl_actor_weight_sync == "none":
            if debug_logs:
                self._split_debug_log(
                    "split_policy_collect_done sync_mode=none seconds={:.2f}",
                    time.monotonic() - collect_start,
                )
            return {}

        state: dict[str, dict[str, torch.Tensor]] = {}
        for module_name in ("transformer", "transformer_2"):
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            if self.cfg.fsdp:
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                    ignore_frozen_params=True,
                )
                module_state = get_model_state_dict(module, options=options)
            else:
                module_state = {
                    name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad
                }
            if self._is_train_root():
                state[module_name] = module_state
        if debug_logs:
            modules, tensors, mb = self._split_policy_state_stats(state)
            self._split_debug_log(
                "split_policy_collect_done role={} modules={} tensors={} size_mb={:.1f} seconds={:.2f}",
                self.rl_role,
                modules,
                tensors,
                mb,
                time.monotonic() - collect_start,
            )
        return state

    def _load_actor_policy_state(self, policy_state: dict[str, dict[str, torch.Tensor]]) -> None:
        if self.cfg.rl_actor_weight_sync == "none":
            return
        debug_logs = self._split_debug_enabled()
        modules = tensors = 0
        if debug_logs:
            modules, tensors, mb = self._split_policy_state_stats(policy_state)
            self._split_debug_log(
                "split_actor_policy_load_start modules={} tensors={} size_mb={:.1f}",
                modules,
                tensors,
                mb,
            )
        load_start = time.monotonic()
        for module_name, module_state in policy_state.items():
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            module.load_state_dict(module_state, strict=False)
            module.eval()
        if debug_logs:
            self._split_debug_log(
                "split_actor_policy_load_done modules={} tensors={} seconds={:.2f}",
                modules,
                tensors,
                time.monotonic() - load_start,
            )

    def _sync_split_rollout_control(
        self,
        *,
        stop: bool | None = None,
        global_step: int | None = None,
        selected_t_idxs: list[int] | None = None,
    ) -> dict[str, Any]:
        """Synchronize actor weights and per-step rollout control."""
        train_root = self.train_global_ranks[0]
        if self.is_train_rank:
            assert stop is not None and global_step is not None and selected_t_idxs is not None
            if stop:
                payload = {
                    "stop": True,
                    "global_step": int(global_step),
                    "selected_t_idxs": [],
                    "policy_state": {},
                }
            else:
                policy_state = self._collect_trainable_policy_state()
                payload = {
                    "stop": False,
                    "global_step": int(global_step),
                    "selected_t_idxs": list(selected_t_idxs),
                    "policy_state": policy_state,
                }
            obj = [payload if self._is_train_root() else None]
        else:
            obj = [None]

        dist.broadcast_object_list(obj, src=train_root)
        payload = obj[0]
        if self.is_inference_rank and not payload["stop"]:
            self._load_actor_policy_state(payload["policy_state"])
        return payload

    def _async_prefetch_steps(self) -> int:
        if self.cfg.rl_async_rollout_prefetch_steps > 0:
            return self.cfg.rl_async_rollout_prefetch_steps
        actor_jobs_per_step = max(1, math.ceil(self.cfg.grpo_group_size / self.cfg.grpo_sample_batch_size))
        return max(1, math.ceil(self.rollout_world_size / actor_jobs_per_step) + 1)

    def _actor_pair_pg(self, actor_rank: int | None = None):
        rank = self.global_rank if actor_rank is None else actor_rank
        pg = self._split_actor_pgs.get(rank)
        if pg is None:
            raise RuntimeError(f"Missing split actor process group for rank {rank}")
        return pg

    @staticmethod
    def _dtype_from_name(dtype_name: str) -> torch.dtype:
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name.removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported streamed tensor dtype: {dtype_name}")
        return dtype

    def _send_async_full_policy_state(
        self,
        actor_rank: int,
        policy_state: dict[str, dict[str, torch.Tensor]],
        pg,
    ) -> None:
        """Stream full actor weights tensor-by-tensor to avoid giant pickle payloads."""
        metadata: list[dict[str, Any]] = []
        tensors: list[torch.Tensor] = []
        for module_name, module_state in policy_state.items():
            for tensor_name, tensor in module_state.items():
                if not torch.is_tensor(tensor):
                    continue
                tensor_cpu = tensor.detach()
                if tensor_cpu.device.type != "cpu":
                    tensor_cpu = tensor_cpu.cpu()
                if not tensor_cpu.is_contiguous():
                    tensor_cpu = tensor_cpu.contiguous()
                metadata.append(
                    {
                        "module": module_name,
                        "name": tensor_name,
                        "shape": tuple(tensor_cpu.shape),
                        "dtype": str(tensor_cpu.dtype),
                    }
                )
                tensors.append(tensor_cpu)

        modules, tensor_count, mb = self._split_policy_state_stats(policy_state)
        send_start = time.monotonic()
        self._split_debug_log(
            "async_full_policy_stream_send_start actor={} modules={} tensors={} size_mb={:.1f}",
            actor_rank,
            modules,
            tensor_count,
            mb,
        )
        dist.send_object_list([metadata], dst=actor_rank, group=pg)
        for tensor in tensors:
            dist.send(tensor, dst=actor_rank, group=pg)
        self._split_debug_log(
            "async_full_policy_stream_send_done actor={} tensors={} seconds={:.2f}",
            actor_rank,
            len(tensors),
            time.monotonic() - send_start,
        )

    def _recv_async_full_policy_state(self, pg) -> dict[str, dict[str, torch.Tensor]]:
        train_root = self.train_global_ranks[0]
        recv_start = time.monotonic()
        obj = [None]
        dist.recv_object_list(obj, src=train_root, group=pg)
        metadata = obj[0] or []
        policy_state: dict[str, dict[str, torch.Tensor]] = {}
        for entry in metadata:
            tensor = torch.empty(
                tuple(entry["shape"]),
                dtype=self._dtype_from_name(str(entry["dtype"])),
                device="cpu",
            )
            dist.recv(tensor, src=train_root, group=pg)
            policy_state.setdefault(str(entry["module"]), {})[str(entry["name"])] = tensor
        modules, tensors, mb = self._split_policy_state_stats(policy_state)
        self._split_debug_log(
            "async_full_policy_stream_recv_done modules={} tensors={} size_mb={:.1f} seconds={:.2f}",
            modules,
            tensors,
            mb,
            time.monotonic() - recv_start,
        )
        return policy_state

    def _broadcast_train_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        obj = [command if self._is_train_root() else None]
        dist.broadcast_object_list(obj, group=self._train_gloo_pg, group_src=0)
        return obj[0]

    def _collect_actor_policy_state(self, version: int) -> dict[str, dict[str, torch.Tensor]]:
        self._split_debug_log("async_policy_collect_command_start version={}", version)
        command_start = time.monotonic()
        self._broadcast_train_command({"type": "collect_policy"})
        command_seconds = time.monotonic() - command_start
        collect_start = time.monotonic()
        policy_state = self._collect_trainable_policy_state()
        collect_seconds = time.monotonic() - collect_start
        if self._split_debug_enabled():
            modules, tensors, mb = self._split_policy_state_stats(policy_state)
            self._split_debug_log(
                "async_policy_collect_command_done version={} broadcast={:.2f}s collect={:.2f}s "
                "modules={} tensors={} size_mb={:.1f}",
                version,
                command_seconds,
                collect_seconds,
                modules,
                tensors,
                mb,
            )
        return policy_state

    def _apply_split_optimizer_step(self, global_step: int) -> tuple[float, float]:
        cfg = self.cfg
        self._all_reduce_gradients()
        grad_norm = self._clip_grad_norm(cfg.max_grad_norm)
        lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update()
        if self.mfu_monitor is not None:
            self.mfu_monitor.step()
        return grad_norm, lr

    def _run_async_split_inference_loop(self) -> None:
        train_root = self.train_global_ranks[0]
        pg = self._actor_pair_pg()
        self._split_debug_log(
            "async_actor_loop_start rank={} local_rank={} rollout_rank={} train_root={}",
            self.global_rank,
            self.local_rank,
            self.rollout_rank,
            train_root,
        )
        while True:
            obj = [None]
            wait_start = time.monotonic()
            self._split_debug_log(
                "async_actor_wait_job rank={} rollout_rank={}",
                self.global_rank,
                self.rollout_rank,
            )
            dist.recv_object_list(obj, src=train_root, group=pg)
            wait_seconds = time.monotonic() - wait_start
            job = obj[0]
            if job.get("stop"):
                self._split_debug_log(
                    "async_actor_stop_received rank={} wait={:.2f}s",
                    self.global_rank,
                    wait_seconds,
                )
                dist.destroy_process_group()
                return
            groups = [int(g) for g in job.get("groups", [])]
            policy_sync = bool(job.get("policy_sync", job.get("policy_state") is not None))
            policy_state = job.get("policy_state")
            if policy_sync and job.get("policy_sync_mode") == "full":
                policy_state = self._recv_async_full_policy_state(pg)
            if self._split_debug_enabled():
                _modules, tensors, mb = self._split_policy_state_stats(policy_state)
                self._split_debug_log(
                    "async_actor_recv_job step={} groups={} selected_t={} policy_sync={} "
                    "policy_tensors={} policy_mb={:.1f} wait={:.2f}s",
                    int(job["step_id"]),
                    groups,
                    len(job.get("selected_t_idxs", [])),
                    policy_sync,
                    tensors,
                    mb,
                    wait_seconds,
                )
            if policy_state is not None:
                self._load_actor_policy_state(policy_state)
            payload = self._split_actor_rollout_job(job)
            send_start = time.monotonic()
            dist.send_object_list([payload], dst=train_root, group=pg)
            if self._split_debug_enabled():
                self._split_debug_log(
                    "async_actor_send_result_done step={} groups={} chunks={} reward_mean={:.4f} send={:.2f}s",
                    int(payload["step_id"]),
                    self._split_payload_group_count(payload),
                    len(payload.get("chunks") or []),
                    self._split_payload_reward_mean(payload),
                    time.monotonic() - send_start,
                )

    def _split_actor_rollout_job(self, job: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        selected_t_idxs = list(job["selected_t_idxs"])
        global_step = int(job["global_step"])
        step_id = int(job["step_id"])
        group_indices = [int(g) for g in job["groups"]]

        payload: dict[str, Any] = {
            "rank": self.global_rank,
            "rollout_rank": self.rollout_rank,
            "step_id": step_id,
            "chunks": [],
        }
        rollout_start = time.monotonic()
        self._split_debug_log(
            "async_actor_rollout_start step={} global_step={} groups={} selected_t={} sample_batch_size={}",
            step_id,
            global_step,
            group_indices,
            len(selected_t_idxs),
            S,
        )
        if not group_indices:
            self._split_debug_log("async_actor_rollout_empty step={} seconds={:.2f}", step_id, 0.0)
            return payload

        encode_start = time.monotonic()
        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(job["batch"])
        B = gt_video_latents.shape[0]
        prompt_cps_noise_levels = self._sample_group_cps_noise_levels(
            B,
            step=global_step,
            stream_id=0,
            device=self.device,
        )
        self._split_debug_log(
            "async_actor_encode_done step={} groups={} batch={} seconds={:.2f}",
            step_id,
            group_indices,
            B,
            time.monotonic() - encode_start,
        )

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        init_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 1_000_003 * global_step + 17)
        shared_initial_latent = self._sample_group_initial_latents(condition, generator=init_generator)

        chunk_total = math.ceil(len(group_indices) / S)
        pending_reward_chunks = []
        for chunk_idx, offset in enumerate(range(0, len(group_indices), S), start=1):
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
            rollout_seed = cfg.seed + 1_000_003 * global_step + 9_176 * (min(groups) + 1) + 131 * offset
            rollout_generator = torch.Generator(device=self.device).manual_seed(rollout_seed)

            chunk_start = time.monotonic()
            self._split_debug_log(
                "async_actor_chunk_start step={} chunk={}/{} groups={} seed={}",
                step_id,
                chunk_idx,
                chunk_total,
                groups,
                rollout_seed,
            )
            generate_start = time.monotonic()
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
            generate_seconds = time.monotonic() - generate_start

            reward_submit_start = time.monotonic()
            reward_submission = self._submit_reward(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            reward_submit_seconds = time.monotonic() - reward_submit_start
            transfer_start = time.monotonic()
            chunk_payload = {
                "groups": groups,
                "latents": torch.stack([traj["latents"][idx] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
                "next_latents": torch.stack([traj["latents"][idx + 1] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
                "log_probs": torch.stack([traj["log_probs"][idx] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
            }
            if chunk_cps_noise_levels is not None:
                chunk_payload["cps_noise_levels"] = chunk_cps_noise_levels.to("cpu", non_blocking=True)
            payload["chunks"].append(chunk_payload)
            pending_reward_chunks.append((chunk_payload, reward_submission, groups, cur_s))
            if self._split_debug_enabled():
                self._split_debug_log(
                    "async_actor_chunk_queued step={} chunk={}/{} groups={} "
                    "generate={:.2f}s reward_submit={:.2f}s transfer={:.2f}s total={:.2f}s",
                    step_id,
                    chunk_idx,
                    chunk_total,
                    groups,
                    generate_seconds,
                    reward_submit_seconds,
                    time.monotonic() - transfer_start,
                    time.monotonic() - chunk_start,
                )
            del traj

        reward_drain_start = time.monotonic()
        for chunk_idx, (chunk_payload, reward_submission, groups, cur_s) in enumerate(
            pending_reward_chunks,
            start=1,
        ):
            reward_flat = self._resolve_reward(reward_submission)
            chunk_payload["rewards"] = reward_flat.view(B, cur_s).to("cpu", non_blocking=True)
            self._split_debug_log(
                "async_actor_reward_resolved step={} chunk={}/{} groups={} reward_mean={:.4f}",
                step_id,
                chunk_idx,
                chunk_total,
                groups,
                float(reward_flat.float().mean().item()),
            )
        self._split_debug_log(
            "async_actor_reward_pipeline_done step={} chunks={} drain={:.2f}s",
            step_id,
            chunk_total,
            time.monotonic() - reward_drain_start,
        )

        if self._split_debug_enabled():
            self._split_debug_log(
                "async_actor_rollout_done step={} groups={} chunks={} reward_mean={:.4f} total={:.2f}s",
                step_id,
                self._split_payload_group_count(payload),
                len(payload["chunks"]),
                self._split_payload_reward_mean(payload),
                time.monotonic() - rollout_start,
            )
        return payload

    def _split_batch_records(self, start_epoch: int, start_batch_idx: int):
        for epoch in range(start_epoch, self.cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            skip = start_batch_idx if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx < skip:
                    continue
                yield epoch, batch_idx, batch

    def _start_async_result_receiver(
        self,
        actor_rank: int,
        results: queue.Queue,
    ) -> threading.Thread:
        pg = self._actor_pair_pg(actor_rank)

        def _recv() -> None:
            obj = [None]
            dist.recv_object_list(obj, src=actor_rank, group=pg)
            results.put((actor_rank, obj[0]))

        thread = threading.Thread(target=_recv, name=f"rollout-result-{actor_rank}", daemon=False)
        thread.start()
        return thread

    def _send_async_actor_job(
        self,
        actor_rank: int,
        job: dict[str, Any],
        results: queue.Queue,
        result_threads: dict[int, threading.Thread],
    ) -> None:
        pg = self._actor_pair_pg(actor_rank)
        policy_state = job.get("policy_state")
        stream_full_policy = self.cfg.rl_actor_weight_sync == "full" and policy_state is not None
        policy_sync = bool(policy_state is not None)
        if stream_full_policy:
            job = dict(job)
            job["policy_state"] = None
            job["policy_sync"] = True
            job["policy_sync_mode"] = "full"
        else:
            job["policy_sync"] = policy_sync
            job["policy_sync_mode"] = self.cfg.rl_actor_weight_sync
        send_start = time.monotonic()
        dist.send_object_list([job], dst=actor_rank, group=pg)
        if stream_full_policy:
            self._send_async_full_policy_state(actor_rank, policy_state, pg)
        result_threads[actor_rank] = self._start_async_result_receiver(actor_rank, results)
        self._split_debug_log(
            "async_dispatch_send_done actor={} step={} groups={} policy_sync={} send={:.2f}s",
            actor_rank,
            int(job["step_id"]),
            [int(g) for g in job.get("groups", [])],
            policy_sync,
            time.monotonic() - send_start,
        )

    def _send_async_actor_stop(self, actor_rank: int) -> None:
        pg = self._actor_pair_pg(actor_rank)
        send_start = time.monotonic()
        dist.send_object_list([{"stop": True}], dst=actor_rank, group=pg)
        self._split_debug_log(
            "async_actor_stop_sent actor={} send={:.2f}s",
            actor_rank,
            time.monotonic() - send_start,
        )

    def _run_async_split_train_loop(self) -> None:
        if not self._is_train_root():
            return self._run_async_split_train_follower_loop()
        return self._run_async_split_train_root_loop()

    def _run_async_split_train_follower_loop(self) -> None:
        self._split_debug_log(
            "async_train_follower_loop_start rank={} local_rank={} train_root={}",
            self.global_rank,
            self.local_rank,
            self.train_global_ranks[0],
        )
        while True:
            command_wait_start = time.monotonic()
            command = self._broadcast_train_command(None)
            command_wait = time.monotonic() - command_wait_start
            command_type = command.get("type")
            self._split_debug_log(
                "async_train_follower_command type={} step={} wait={:.2f}s",
                command_type,
                command.get("global_step", "-"),
                command_wait,
            )
            if command_type == "stop":
                break
            if command_type == "collect_policy":
                self._collect_trainable_policy_state()
                continue
            if command_type != "train_step":
                raise RuntimeError(f"Unknown async split train command: {command_type}")

            train_start = time.monotonic()
            metrics = self._split_train_step_from_rollouts(
                batch=command["batch"],
                selected_t_idxs=list(command["selected_t_idxs"]),
                rollouts=command["rollouts"],
            )
            self._split_debug_log(
                "async_train_follower_replay_done step={} active_rollouts={} policy_loss={:.4f} "
                "kl_loss={:.4f} reward={:.4f} seconds={:.2f}",
                int(command["global_step"]),
                metrics["active_rollout_ranks"],
                metrics["policy_loss"],
                metrics["kl_loss"],
                metrics["reward_mean"],
                time.monotonic() - train_start,
            )
            del metrics
            optimizer_start = time.monotonic()
            grad_norm, _lr = self._apply_split_optimizer_step(int(command["global_step"]))
            self._split_debug_log(
                "async_train_follower_optimizer_done step={} grad_norm={:.4f} seconds={:.2f}",
                int(command["global_step"]),
                grad_norm,
                time.monotonic() - optimizer_start,
            )
            del grad_norm
            self.train_state.step = int(command["global_step"]) + 1
            self.train_state.epoch = int(command["epoch"])
            self.train_state.batch_idx = int(command["batch_idx"]) + 1

        self._split_debug_log("async_train_follower_loop_stop rank={}", self.global_rank)
        dist.destroy_process_group()

    def _run_async_split_train_root_loop(self) -> None:
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch
        start_batch_idx = self.train_state.batch_idx
        train_start_time = time.monotonic()
        train_start_step = global_step
        target_steps = self.total_steps
        prefetch_steps = self._async_prefetch_steps()
        actors = _interleave_actor_ranks_by_node(list(self.inference_global_ranks), self.local_world_size)
        actor_locations = {
            rank: {
                "node": rank // self.local_world_size,
                "local_rank": rank % self.local_world_size,
            }
            for rank in actors
        }
        if not actors:
            raise RuntimeError("Async split rollout requires at least one inference actor")

        logger.info(
            "Async split rollout enabled: actors={} G={} prefetch_steps={} max_inflight_groups={} "
            "actor_sync={} sync_interval={}",
            len(actors),
            cfg.grpo_group_size,
            prefetch_steps,
            prefetch_steps * cfg.grpo_group_size,
            cfg.rl_actor_weight_sync,
            cfg.rl_actor_weight_sync_interval,
        )
        self._split_debug_log(
            "async_train_root_start step={} target_steps={} actors={} train_ranks={} "
            "actor_locations={} sample_batch_size={} train_sample_batch_size={} output_dir={}",
            global_step,
            target_steps,
            actors,
            self.train_global_ranks,
            actor_locations,
            cfg.grpo_sample_batch_size,
            cfg.grpo_train_sample_batch_size,
            str(output_dir),
        )

        current_policy_version = global_step
        if cfg.rl_actor_weight_sync == "full":
            current_policy_state = None
            actor_versions: dict[int, int | None] = {rank: current_policy_version for rank in actors}
            logger.info(
                "Skipping initial full actor sync at step={} because actors already loaded the same "
                "init/resume weights.",
                global_step,
            )
        else:
            current_policy_state = self._collect_actor_policy_state(global_step)
            actor_versions = {rank: None for rank in actors}

        batch_iter = self._split_batch_records(start_epoch, start_batch_idx)
        next_step_to_create = global_step
        next_step_to_train = global_step
        steps: dict[int, dict[str, Any]] = {}
        free_actors: deque[int] = deque(actors)
        busy_actors: set[int] = set()
        result_threads: dict[int, threading.Thread] = {}
        results: queue.Queue = queue.Queue()
        exhausted = False

        def can_create_more_steps() -> bool:
            if exhausted:
                return False
            if target_steps is not None and next_step_to_create >= target_steps:
                return False
            return next_step_to_create < next_step_to_train + prefetch_steps

        def create_step() -> bool:
            nonlocal next_step_to_create, exhausted
            try:
                epoch, batch_idx, batch = next(batch_iter)
            except StopIteration:
                exhausted = True
                self._split_debug_log("async_step_create_exhausted next_step={}", next_step_to_create)
                return False
            selected_t_idxs = self._select_training_timesteps_for_step(
                cfg.grpo_num_sampling_steps,
                next_step_to_create,
            )
            step_id = next_step_to_create
            steps[step_id] = {
                "step_id": step_id,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch": batch,
                "selected_t_idxs": selected_t_idxs,
                "policy_version": current_policy_version,
                "policy_state": current_policy_state,
                "unassigned": deque(range(cfg.grpo_group_size)),
                "payloads": [],
                "seen": set(),
            }
            self._split_debug_log(
                "async_step_created step={} epoch={} batch_idx={} selected_t={} policy_version={} queued_steps={}",
                step_id,
                epoch,
                batch_idx,
                selected_t_idxs,
                current_policy_version,
                len(steps),
            )
            next_step_to_create += 1
            return True

        def dispatch_available() -> None:
            while can_create_more_steps():
                if not create_step():
                    break

            while free_actors:
                candidate = None
                for step_id in sorted(steps):
                    step = steps[step_id]
                    if step["unassigned"]:
                        candidate = step
                        break
                if candidate is None:
                    break

                actor_rank = free_actors.popleft()
                groups = []
                while candidate["unassigned"] and len(groups) < cfg.grpo_sample_batch_size:
                    groups.append(candidate["unassigned"].popleft())

                policy_version = int(candidate["policy_version"])
                policy_state = None
                if actor_versions[actor_rank] != policy_version:
                    policy_state = candidate["policy_state"]
                    actor_versions[actor_rank] = policy_version

                job = {
                    "stop": False,
                    "step_id": int(candidate["step_id"]),
                    "global_step": int(candidate["step_id"]),
                    "groups": groups,
                    "selected_t_idxs": list(candidate["selected_t_idxs"]),
                    "batch": candidate["batch"],
                    "policy_state": policy_state,
                }
                self._split_debug_log(
                    "async_dispatch_actor_start actor={} actor_node={} actor_local_rank={} step={} groups={} "
                    "policy_version={} "
                    "policy_sync={} free={} busy={} queued_steps={} unassigned_groups={}",
                    actor_rank,
                    actor_locations[actor_rank]["node"],
                    actor_locations[actor_rank]["local_rank"],
                    int(candidate["step_id"]),
                    groups,
                    policy_version,
                    policy_state is not None,
                    len(free_actors),
                    len(busy_actors),
                    len(steps),
                    sum(len(step["unassigned"]) for step in steps.values()),
                )
                busy_actors.add(actor_rank)
                self._send_async_actor_job(actor_rank, job, results, result_threads)

        def receive_one_result() -> None:
            wait_step = steps.get(next_step_to_train)
            wait_seen = len(wait_step["seen"]) if wait_step is not None else 0
            self._split_debug_log(
                "async_wait_result_start train_step={} seen={}/{} busy={} busy_actors={} free={} queued_steps={}",
                next_step_to_train,
                wait_seen,
                cfg.grpo_group_size,
                len(busy_actors),
                sorted(busy_actors),
                len(free_actors),
                len(steps),
            )
            wait_start = time.monotonic()
            while True:
                try:
                    actor_rank, payload = results.get(timeout=60.0)
                    break
                except queue.Empty:
                    wait_step = steps.get(next_step_to_train)
                    wait_seen = len(wait_step["seen"]) if wait_step is not None else 0
                    self._split_debug_log(
                        "async_wait_result_heartbeat train_step={} seen={}/{} busy={} busy_actors={} "
                        "free={} queued_steps={} waited={:.2f}s",
                        next_step_to_train,
                        wait_seen,
                        cfg.grpo_group_size,
                        len(busy_actors),
                        sorted(busy_actors),
                        len(free_actors),
                        len(steps),
                        time.monotonic() - wait_start,
                    )
            wait_seconds = time.monotonic() - wait_start
            thread = result_threads.pop(actor_rank, None)
            join_start = time.monotonic()
            if thread is not None:
                thread.join()
            join_seconds = time.monotonic() - join_start
            busy_actors.discard(actor_rank)
            free_actors.append(actor_rank)

            step_id = int(payload["step_id"])
            step = steps.get(step_id)
            if step is None:
                raise RuntimeError(f"Received rollout for unknown async step {step_id}")
            for chunk in payload.get("chunks") or []:
                for group_idx in chunk.get("groups", []):
                    group_idx = int(group_idx)
                    if group_idx in step["seen"]:
                        raise RuntimeError(f"Duplicate async rollout group {group_idx} for step {step_id}")
                    step["seen"].add(group_idx)
            step["payloads"].append(payload)
            self._split_debug_log(
                "async_recv_actor_result actor={} step={} groups={} chunks={} reward_mean={:.4f} "
                "seen={}/{} wait={:.2f}s join={:.2f}s busy={} free={}",
                actor_rank,
                step_id,
                self._split_payload_group_count(payload),
                len(payload.get("chunks") or []),
                self._split_payload_reward_mean(payload),
                len(step["seen"]),
                cfg.grpo_group_size,
                wait_seconds,
                join_seconds,
                len(busy_actors),
                len(free_actors),
            )

        def step_ready(step_id: int) -> bool:
            step = steps.get(step_id)
            return step is not None and len(step["seen"]) == cfg.grpo_group_size

        try:
            dispatch_available()
            while next_step_to_train < target_steps:
                while not step_ready(next_step_to_train):
                    if exhausted and next_step_to_train not in steps and not busy_actors:
                        break
                    receive_one_result()
                    dispatch_available()
                if not step_ready(next_step_to_train):
                    break

                step = steps.pop(next_step_to_train)
                B = int(step["batch"]["condition"].shape[0]) if "condition" in step["batch"] else cfg.batch_size
                pack_start = time.monotonic()
                rollouts = self._pack_split_rollouts(
                    step["payloads"],
                    B,
                    list(step["selected_t_idxs"]),
                )
                packed_groups = sum(self._split_payload_group_count(payload) for payload in step["payloads"])
                packed_reward_sum = 0.0
                packed_reward_count = 0
                for payload in step["payloads"]:
                    for chunk in payload.get("chunks") or []:
                        rewards = chunk.get("rewards")
                        if torch.is_tensor(rewards) and rewards.numel() > 0:
                            rewards_f = rewards.float()
                            packed_reward_sum += float(rewards_f.sum().item())
                            packed_reward_count += int(rewards_f.numel())
                packed_reward_mean = (
                    packed_reward_sum / packed_reward_count if packed_reward_count > 0 else float("nan")
                )
                self._split_debug_log(
                    "async_pack_rollouts_done step={} payloads={} groups={} selected_t={} "
                    "reward_mean={:.4f} seconds={:.2f}",
                    next_step_to_train,
                    len(step["payloads"]),
                    packed_groups,
                    list(step["selected_t_idxs"]),
                    packed_reward_mean,
                    time.monotonic() - pack_start,
                )

                command = {
                    "type": "train_step",
                    "global_step": int(next_step_to_train),
                    "epoch": int(step["epoch"]),
                    "batch_idx": int(step["batch_idx"]),
                    "batch": step["batch"],
                    "selected_t_idxs": list(step["selected_t_idxs"]),
                    "rollouts": rollouts,
                }
                self._split_debug_log(
                    "async_train_broadcast_start step={} payloads={} groups={}",
                    next_step_to_train,
                    len(step["payloads"]),
                    packed_groups,
                )
                broadcast_start = time.monotonic()
                self._broadcast_train_command(command)
                self._split_debug_log(
                    "async_train_broadcast_done step={} seconds={:.2f}",
                    next_step_to_train,
                    time.monotonic() - broadcast_start,
                )
                replay_start = time.monotonic()
                metrics = self._split_train_step_from_rollouts(
                    batch=step["batch"],
                    selected_t_idxs=list(step["selected_t_idxs"]),
                    rollouts=rollouts,
                )
                self._split_debug_log(
                    "async_train_replay_done step={} active_rollouts={} policy_loss={:.4f} "
                    "kl_loss={:.4f} reward={:.4f} seconds={:.2f}",
                    next_step_to_train,
                    metrics["active_rollout_ranks"],
                    metrics["policy_loss"],
                    metrics["kl_loss"],
                    metrics["reward_mean"],
                    time.monotonic() - replay_start,
                )
                optimizer_start = time.monotonic()
                grad_norm, lr = self._apply_split_optimizer_step(next_step_to_train)
                self._split_debug_log(
                    "async_optimizer_done step={} lr={:.2e} grad_norm={:.4f} seconds={:.2f}",
                    next_step_to_train,
                    lr,
                    grad_norm,
                    time.monotonic() - optimizer_start,
                )

                global_step = next_step_to_train + 1
                self.train_state.step = global_step
                self.train_state.epoch = int(step["epoch"])
                self.train_state.batch_idx = int(step["batch_idx"]) + 1

                if global_step % cfg.log_steps == 0:
                    mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                    mfu_str = f"{mfu:.1%}" if mfu is not None else "-"
                    elapsed = time.monotonic() - train_start_time
                    steps_done = global_step - train_start_step
                    secs_per_step = elapsed / max(steps_done, 1)
                    eta_str = format_eta(secs_per_step * max(target_steps - global_step, 0))
                    logger.info(
                        "step={}/{} epoch={} async_rollout={} policy_loss={:.4f} kl_loss={:.4f} "
                        "reward={:.4f}+/-{:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({:.2f} s/it)",
                        global_step,
                        target_steps,
                        step["epoch"],
                        metrics["active_rollout_ranks"],
                        metrics["policy_loss"],
                        metrics["kl_loss"],
                        metrics["reward_mean"],
                        metrics["reward_std"],
                        lr,
                        grad_norm,
                        mfu_str,
                        eta_str,
                        secs_per_step,
                    )
                    if self.use_wandb:
                        import wandb

                        log_metrics = {
                            "grpo/policy_loss": metrics["policy_loss"],
                            "grpo/kl_loss": metrics["kl_loss"],
                            "grpo/reward_mean": metrics["reward_mean"],
                            "grpo/reward_std": metrics["reward_std"],
                            "grpo/advantage_mean": metrics["advantage_mean"],
                            "grpo/active_rollout_ranks": metrics["active_rollout_ranks"],
                            "grpo/async_prefetch_steps": prefetch_steps,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                        }
                        if mfu is not None:
                            log_metrics["train/mfu"] = mfu
                        wandb.log(log_metrics, step=global_step)

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    self._split_debug_log("async_checkpoint_start step={} path={}", global_step, str(checkpoint_path))
                    checkpoint_start = time.monotonic()
                    self._save_checkpoint(checkpoint_path)
                    self._split_debug_log(
                        "async_checkpoint_done step={} seconds={:.2f} path={}",
                        global_step,
                        time.monotonic() - checkpoint_start,
                        str(checkpoint_path),
                    )

                next_step_to_train = global_step
                if global_step >= target_steps:
                    break

                steps_since_start = global_step - train_start_step
                if steps_since_start % cfg.rl_actor_weight_sync_interval == 0:
                    current_policy_state = self._collect_actor_policy_state(global_step)
                    current_policy_version = global_step
                else:
                    self._split_debug_log(
                        "async_policy_sync_skipped step={} current_policy_version={} interval={}",
                        global_step,
                        current_policy_version,
                        cfg.rl_actor_weight_sync_interval,
                    )
                dispatch_available()

            self.train_state.step = global_step
            if cfg.save_epoch_checkpoints:
                checkpoint_path = output_dir / f"checkpoint-{global_step}"
                self._split_debug_log("async_checkpoint_start step={} path={}", global_step, str(checkpoint_path))
                checkpoint_start = time.monotonic()
                self._save_checkpoint(checkpoint_path)
                self._split_debug_log(
                    "async_checkpoint_done step={} seconds={:.2f} path={}",
                    global_step,
                    time.monotonic() - checkpoint_start,
                    str(checkpoint_path),
                )
            logger.info("Async split training stopped at step={}.", global_step)
        finally:
            while busy_actors:
                receive_one_result()
            for actor_rank in actors:
                self._send_async_actor_stop(actor_rank)
            self._split_debug_log("async_train_stop_broadcast_start")
            stop_start = time.monotonic()
            self._broadcast_train_command({"type": "stop"})
            self._split_debug_log("async_train_stop_broadcast_done seconds={:.2f}", time.monotonic() - stop_start)
            if self.use_wandb:
                import wandb

                wandb.finish()
            dist.destroy_process_group()

    def _run_split_inference_loop(self) -> None:
        cfg = self.cfg
        for epoch in range(cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.dataloader:
                control = self._sync_split_rollout_control()
                if control["stop"]:
                    dist.destroy_process_group()
                    return
                self._split_actor_rollout_step(batch, control)
        dist.destroy_process_group()

    def _run_split_train_loop(self) -> None:
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = self.train_state.step
        start_epoch = self.train_state.epoch
        start_batch_idx = self.train_state.batch_idx
        train_start_time = time.monotonic()
        train_start_step = global_step
        stop_training = False

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            last_batch_idx = enum_start - 1
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
                last_batch_idx = batch_idx
                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    self._sync_split_rollout_control(
                        stop=True,
                        global_step=global_step,
                        selected_t_idxs=[],
                    )
                    stop_training = True
                    break

                selected_t_idxs = self._select_training_timesteps_for_step(
                    cfg.grpo_num_sampling_steps,
                    global_step,
                )
                self._sync_split_rollout_control(
                    stop=False,
                    global_step=global_step,
                    selected_t_idxs=selected_t_idxs,
                )
                metrics = self._split_train_step(batch, selected_t_idxs)

                self._all_reduce_gradients()
                grad_norm = self._clip_grad_norm(cfg.max_grad_norm)

                lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                for opt in self.optimizers:
                    for pg in opt.param_groups:
                        pg["lr"] = lr
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update()
                if self.mfu_monitor is not None:
                    self.mfu_monitor.step()

                global_step += 1

                if self._is_train_root() and global_step % cfg.log_steps == 0:
                    mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                    mfu_str = f"{mfu:.1%}" if mfu is not None else "-"
                    elapsed = time.monotonic() - train_start_time
                    steps_done = global_step - train_start_step
                    if steps_done > 0:
                        secs_per_step = elapsed / steps_done
                        eta_str = format_eta(secs_per_step * (self.total_steps - global_step))
                        speed_str = f"{secs_per_step:.2f}"
                    else:
                        eta_str, speed_str = "?", "?"

                    if hasattr(self.dataloader.dataset, "__len__"):
                        batches = len(self.dataloader)
                    elif cfg.dataset_size is not None:
                        batches = cfg.dataset_size // cfg.batch_size
                    else:
                        batches = None
                    fractional_epoch = epoch + (batch_idx + 1) / batches if batches else float(epoch)
                    logger.info(
                        "step={}/{} epoch={:.2f} split_rollout={} policy_loss={:.4f} kl_loss={:.4f} "
                        "reward={:.4f}+/-{:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                        global_step,
                        self.total_steps,
                        fractional_epoch,
                        metrics["active_rollout_ranks"],
                        metrics["policy_loss"],
                        metrics["kl_loss"],
                        metrics["reward_mean"],
                        metrics["reward_std"],
                        lr,
                        grad_norm,
                        mfu_str,
                        eta_str,
                        speed_str,
                    )

                    if self.use_wandb:
                        import wandb

                        log_metrics = {
                            "grpo/policy_loss": metrics["policy_loss"],
                            "grpo/kl_loss": metrics["kl_loss"],
                            "grpo/reward_mean": metrics["reward_mean"],
                            "grpo/reward_std": metrics["reward_std"],
                            "grpo/advantage_mean": metrics["advantage_mean"],
                            "grpo/active_rollout_ranks": metrics["active_rollout_ranks"],
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": fractional_epoch,
                        }
                        if mfu is not None:
                            log_metrics["train/mfu"] = mfu
                        wandb.log(log_metrics, step=global_step)

                if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                    self.train_state.step = global_step
                    self.train_state.epoch = epoch
                    self.train_state.batch_idx = batch_idx + 1
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    stop_training = True
                    self._sync_split_rollout_control(
                        stop=True,
                        global_step=global_step,
                        selected_t_idxs=[],
                    )
                    break

            if stop_training:
                self.train_state.step = global_step
                self.train_state.epoch = epoch
                self.train_state.batch_idx = max(last_batch_idx + 1, 0)
                if cfg.save_epoch_checkpoints:
                    self._save_checkpoint(output_dir / f"checkpoint-{global_step}")
                logger.info("Reached max_steps={} at step={}.", cfg.max_steps, global_step)
                break

            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            if cfg.save_epoch_checkpoints:
                self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb

            wandb.finish()
        dist.destroy_process_group()

    def _split_actor_rollout_step(self, batch: dict, control: dict[str, Any]) -> None:
        cfg = self.cfg
        G = cfg.grpo_group_size
        S = cfg.grpo_sample_batch_size
        T = cfg.grpo_num_sampling_steps
        selected_t_idxs = list(control["selected_t_idxs"])
        global_step = int(control["global_step"])
        group_indices = _split_group_indices(G, self.rollout_rank, self.rollout_world_size)

        payload: dict[str, Any] = {
            "rank": self.global_rank,
            "rollout_rank": self.rollout_rank,
            "chunks": [],
        }
        if not group_indices:
            dist.gather_object(payload, dst=self.train_global_ranks[0])
            return

        prompt_embeds, gt_video_latents, condition, meta = self._encode_batch_inputs(batch)
        B = gt_video_latents.shape[0]
        prompt_cps_noise_levels = self._sample_group_cps_noise_levels(
            B,
            step=global_step,
            stream_id=0,
            device=self.device,
        )

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.eval()

        init_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 1_000_003 * global_step + 17)
        shared_initial_latent = self._sample_group_initial_latents(condition, generator=init_generator)

        pending_reward_chunks = []
        for offset in range(0, len(group_indices), S):
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
            rollout_seed = cfg.seed + 1_000_003 * global_step + 9_176 * (self.rollout_rank + 1) + 131 * offset
            rollout_generator = torch.Generator(device=self.device).manual_seed(rollout_seed)

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

            reward_submission = self._submit_reward(
                traj["latents"][-1],
                gt_s,
                cond_s,
                pe_s,
                meta=meta_s,
            )
            chunk_payload = {
                "groups": groups,
                "latents": torch.stack([traj["latents"][idx] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
                "next_latents": torch.stack([traj["latents"][idx + 1] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
                "log_probs": torch.stack([traj["log_probs"][idx] for idx in selected_t_idxs]).to(
                    "cpu",
                    non_blocking=True,
                ),
            }
            if chunk_cps_noise_levels is not None:
                chunk_payload["cps_noise_levels"] = chunk_cps_noise_levels.to("cpu", non_blocking=True)
            payload["chunks"].append(chunk_payload)
            pending_reward_chunks.append((chunk_payload, reward_submission, cur_s))
            del traj

        for chunk_payload, reward_submission, cur_s in pending_reward_chunks:
            reward_flat = self._resolve_reward(reward_submission)
            chunk_payload["rewards"] = reward_flat.view(B, cur_s).to("cpu", non_blocking=True)

        dist.gather_object(payload, dst=self.train_global_ranks[0])

    def _receive_split_rollouts(self, B: int, selected_t_idxs: list[int]) -> dict[str, Any]:
        gathered = [None for _ in range(self.global_world_size)] if self._is_train_root() else None
        dist.gather_object(None, object_gather_list=gathered, dst=self.train_global_ranks[0])

        packed = None
        if self._is_train_root():
            packed = self._pack_split_rollouts(gathered, B, selected_t_idxs)
        obj = [packed]
        dist.broadcast_object_list(obj, group=self._train_gloo_pg, group_src=0)
        return obj[0]

    def _pack_split_rollouts(
        self,
        gathered: list[Any],
        B: int,
        selected_t_idxs: list[int],
    ) -> dict[str, Any]:
        G = self.cfg.grpo_group_size
        rewards = torch.full((B, G), float("nan"), dtype=torch.float32)
        chunks: list[dict[str, Any]] = []
        active_rollout_ranks = 0
        seen: set[int] = set()

        for item in gathered:
            if not isinstance(item, dict):
                continue
            item_chunks = item.get("chunks") or []
            if item_chunks:
                active_rollout_ranks += 1
            for chunk in item_chunks:
                groups = [int(g) for g in chunk["groups"]]
                if not groups:
                    continue
                for local_idx, group_idx in enumerate(groups):
                    if group_idx in seen:
                        raise RuntimeError(f"Duplicate rollout group index {group_idx}")
                    rewards[:, group_idx] = chunk["rewards"][:, local_idx].float()
                    seen.add(group_idx)
                chunks.append(chunk)

        missing = [idx for idx in range(G) if idx not in seen]
        if missing:
            raise RuntimeError(
                f"Missing rollout group indices {missing}; rollout_world_size={self.rollout_world_size}, group_size={G}"
            )
        chunks.sort(key=lambda chunk: min(chunk["groups"]))
        return {
            "selected_t_idxs": selected_t_idxs,
            "rewards": rewards,
            "chunks": chunks,
            "active_rollout_ranks": active_rollout_ranks,
        }

    def _split_train_step(self, batch: dict, selected_t_idxs: list[int]) -> dict[str, float]:
        prompt_embeds, _gt_video_latents, condition, _meta = self._encode_batch_inputs(batch)
        B = condition.shape[0]
        rollouts = self._receive_split_rollouts(B, selected_t_idxs)
        return self._split_train_step_from_rollouts(
            batch=batch,
            selected_t_idxs=selected_t_idxs,
            rollouts=rollouts,
            encoded=(prompt_embeds, condition),
        )

    def _coalesce_train_rollout_chunks(self, chunks: list[dict[str, Any]], B: int) -> list[dict[str, Any]]:
        max_s = int(getattr(self.cfg, "grpo_train_sample_batch_size", 1))
        if max_s <= 1 or len(chunks) <= 1:
            return chunks

        coalesced: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        pending_s = 0

        def flush_pending() -> None:
            nonlocal pending, pending_s
            if not pending:
                return
            if len(pending) == 1:
                coalesced.append(pending[0])
                pending = []
                pending_s = 0
                return

            groups: list[int] = []
            rewards = []
            tensor_parts: dict[str, list[torch.Tensor]] = {
                "latents": [],
                "next_latents": [],
                "log_probs": [],
            }
            for chunk in pending:
                chunk_groups = [int(g) for g in chunk["groups"]]
                cur_s = len(chunk_groups)
                groups.extend(chunk_groups)
                rewards.append(chunk["rewards"])
                for name in tensor_parts:
                    tensor = chunk[name]
                    tensor_parts[name].append(tensor.reshape(tensor.shape[0], B, cur_s, *tensor.shape[2:]))

            merged: dict[str, Any] = {
                "groups": groups,
                "rewards": torch.cat(rewards, dim=1),
            }
            for name, parts in tensor_parts.items():
                tensor = torch.cat(parts, dim=2)
                merged[name] = tensor.reshape(tensor.shape[0], B * len(groups), *tensor.shape[3:])
            cps_parts = [chunk.get("cps_noise_levels") for chunk in pending]
            if any(part is not None for part in cps_parts):
                if not all(part is not None for part in cps_parts):
                    raise RuntimeError("Cannot coalesce a mix of fixed/non-CPS and per-group CPS rollout chunks")
                merged["cps_noise_levels"] = torch.cat(
                    [part.reshape(B, len(chunk["groups"])) for part, chunk in zip(cps_parts, pending, strict=True)],
                    dim=1,
                ).reshape(B * len(groups))
            coalesced.append(merged)
            pending = []
            pending_s = 0

        for chunk in chunks:
            cur_s = len(chunk["groups"])
            if cur_s >= max_s:
                flush_pending()
                coalesced.append(chunk)
                continue
            if pending and pending_s + cur_s > max_s:
                flush_pending()
            pending.append(chunk)
            pending_s += cur_s
        flush_pending()
        return coalesced

    def _split_train_step_from_rollouts(
        self,
        *,
        batch: dict,
        selected_t_idxs: list[int],
        rollouts: dict[str, Any],
        encoded: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        G = cfg.grpo_group_size
        T = cfg.grpo_num_sampling_steps
        device = self.device

        if encoded is None:
            prompt_embeds, _gt_video_latents, condition, _meta = self._encode_batch_inputs(batch)
        else:
            prompt_embeds, condition = encoded
        B = condition.shape[0]
        rewards = rollouts["rewards"].to(device=device, non_blocking=True)
        advantages = self._compute_advantages(rewards)
        sigmas, timesteps, _ = self._build_sampling_schedule(T, device=device)

        for m in [self.model.transformer, self.model.transformer_2]:
            if m is not None:
                m.train()

        total_policy_loss = 0.0
        total_kl_loss = 0.0
        chunks = self._coalesce_train_rollout_chunks(rollouts["chunks"], B)
        metric_normalizer = max(len(selected_t_idxs) * G, 1)

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        for chunk_idx, chunk in enumerate(chunks):
            groups = [int(g) for g in chunk["groups"]]
            cur_s = len(groups)
            BS = B * cur_s
            cond_s = condition.repeat_interleave(cur_s, dim=0).detach()
            pe_s = prompt_embeds.repeat_interleave(cur_s, dim=0).detach()
            adv_chunk = advantages[:, groups].reshape(BS)

            latents = chunk["latents"].to(device=device, non_blocking=True)
            next_latents = chunk["next_latents"].to(device=device, non_blocking=True)
            old_log_probs = chunk["log_probs"].to(device=device, non_blocking=True)
            cps_noise_levels = chunk.get("cps_noise_levels")
            if cps_noise_levels is not None:
                cps_noise_levels = cps_noise_levels.to(device=device, non_blocking=True)

            for replay_idx, t_idx in enumerate(selected_t_idxs):
                is_last = chunk_idx == len(chunks) - 1 and replay_idx == len(selected_t_idxs) - 1
                self._set_requires_gradient_sync(is_last)

                sigma = sigmas[t_idx].item()
                sigma_prev = sigmas[t_idx + 1].item()
                latent = latents[replay_idx].detach()
                next_latent = next_latents[replay_idx].detach()
                old_log_prob = old_log_probs[replay_idx].detach()
                timestep_val = timesteps[t_idx]

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

            del latents, next_latents, old_log_probs, cps_noise_levels

        return {
            "policy_loss": total_policy_loss / metric_normalizer,
            "kl_loss": total_kl_loss / metric_normalizer,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "active_rollout_ranks": float(rollouts["active_rollout_ranks"]),
        }


__all__ = ["DanceGRPOSplitRuntimeMixin"]
