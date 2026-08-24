"""Wan2.2 I2V Trainer with FSDP2 + Distributed Checkpoint (DCP)."""

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from src.trainer.base_trainer import BaseTrainer
from src.trainer.config import TrainConfig
from src.trainer.flops import MFUMonitor, compute_wan_seq_len, estimate_wan_forward_flops, get_gpu_peak_flops_bf16
from src.trainer.utils import cosine_lr, format_eta, to_model_pixels


class I2VTrainer(BaseTrainer):
    def _post_init(self, cfg: TrainConfig) -> None:
        self.mfu_monitor = self._setup_mfu()

    def _setup_mfu(self) -> MFUMonitor | None:
        """Pre-compute FLOPs and create MFU monitor. Returns None if GPU is unrecognized."""
        gpu_peak = get_gpu_peak_flops_bf16()
        if gpu_peak is None:
            return None

        bi = self.model.boundary_idx
        N = self.model.num_train_timesteps
        experts = []
        # A single-transformer model (e.g. 5B TI2V) routes every sample through the
        # one transformer; only a true dual-expert model (A14B) splits samples between
        # experts by timestep, so apply the boundary-fraction weighting only then.
        both = self._effective_train_experts == "both"
        has_high = self.model.transformer is not None
        has_low = self.model.transformer_2 is not None
        if has_high:
            prob = (bi / N) if (both and has_low) else 1.0
            experts.append((prob, self.model.transformer))
        if has_low:
            prob = ((N - bi) / N) if (both and has_high) else 1.0
            experts.append((prob, self.model.transformer_2))

        # Determine seq_len: from dataset config or latent shape
        latent_seq_len: int | None = None
        if hasattr(self.dataset, "_configs"):
            est_cfg = self.dataset._configs[0]
            if est_cfg.fixed_height is not None and est_cfg.fixed_width is not None:
                est_h, est_w = est_cfg.fixed_height, est_cfg.fixed_width
            else:
                est_h = est_w = int(est_cfg.max_area**0.5)
            ref_t = experts[0][1] if experts else None
            if ref_t is not None:
                t_cfg = ref_t.config
                latent_seq_len = compute_wan_seq_len(
                    est_cfg.num_frames,
                    est_h,
                    est_w,
                    patch_size=tuple(t_cfg.patch_size),
                    vae_temporal_factor=self.model.vae_scale_factor_temporal,
                    vae_spatial_factor=self.model.vae_scale_factor_spatial,
                )
        else:
            # Latent dataset: peek at one sample to get (C, T', H', W')
            try:
                sample = next(iter(self.dataset))
                latent = sample["video_latents"]  # (C, T', H', W')
                _, t_lat, h_lat, w_lat = latent.shape
                ref_t = experts[0][1] if experts else None
                if ref_t is not None:
                    p_t, p_h, p_w = ref_t.config.patch_size
                    latent_seq_len = (t_lat // p_t) * (h_lat // p_h) * (w_lat // p_w)
            except Exception as e:
                logger.info("MFU monitor: skipped (cannot peek latent sample: {})", e)

        if latent_seq_len is None:
            logger.info("MFU monitor: skipped (no resolution info available)")
            return None

        weighted_fwd_flops = 0.0
        for prob, t in experts:
            t_cfg = t.config
            fwd = estimate_wan_forward_flops(
                num_layers=t_cfg.num_layers,
                num_heads=t_cfg.num_attention_heads,
                head_dim=t_cfg.attention_head_dim,
                ffn_dim=t_cfg.ffn_dim,
                seq_len=latent_seq_len,
            )
            weighted_fwd_flops += prob * fwd

        flops_per_step = 3 * weighted_fwd_flops * self.cfg.batch_size * self.cfg.gradient_accumulation_steps

        logger.info(
            "MFU monitor: seq_len={}, forward={:.2e} FLOPs/sample, step={:.2e} FLOPs, GPU={} ({:.0f} TFLOPS bf16)",
            latent_seq_len,
            weighted_fwd_flops,
            flops_per_step,
            torch.cuda.get_device_name(0),
            gpu_peak / 1e12,
        )
        return MFUMonitor(flops_per_step, gpu_peak)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
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
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                stop_training = True
                break
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)

            enum_start = start_batch_idx if epoch == start_epoch else 0
            for batch_idx, batch in enumerate(self.dataloader, start=enum_start):
                profile_step = self._profile_raw_step_enabled()
                profile_t0 = None
                if profile_step:
                    torch.cuda.synchronize(self.device)
                    profile_t0 = time.perf_counter()
                is_last_micro_step = (batch_idx + 1) % cfg.gradient_accumulation_steps == 0
                self._set_requires_gradient_sync(is_last_micro_step)
                loss = self._train_step(batch)
                if not torch.isfinite(loss).all().item():
                    raise FloatingPointError(
                        f"Non-finite loss on rank={self.rank} expert={self._effective_train_experts} "
                        f"at step={global_step} epoch={epoch} batch_idx={batch_idx}: {loss.item()}"
                    )
                scaled_loss = loss / cfg.gradient_accumulation_steps
                if cfg.detect_anomaly:
                    with torch.autograd.detect_anomaly(check_nan=True):
                        scaled_loss.backward()
                else:
                    backward_t0 = None
                    if profile_step:
                        torch.cuda.synchronize(self.device)
                        backward_t0 = time.perf_counter()
                    scaled_loss.backward()
                    if profile_step:
                        torch.cuda.synchronize(self.device)
                        self._last_step_profile["backward"] = time.perf_counter() - backward_t0

                if is_last_micro_step:
                    optimizer_t0 = None
                    if profile_step:
                        torch.cuda.synchronize(self.device)
                        optimizer_t0 = time.perf_counter()
                    self._all_reduce_gradients()
                    context = f"at step={global_step} epoch={epoch} batch_idx={batch_idx}"
                    if cfg.skip_nonfinite_gradients and not self._gradients_are_finite_across_ranks(context=context):
                        for opt in self.optimizers:
                            opt.zero_grad(set_to_none=True)
                        global_step += 1
                        if self.mfu_monitor is not None:
                            self.mfu_monitor.step()
                        if cfg.max_steps is not None and global_step >= cfg.max_steps:
                            stop_training = True
                            break
                        continue
                    self._last_grad_norm = self._clip_grad_norm_or_raise(cfg.max_grad_norm, context=context)

                    lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, cfg.learning_rate)
                    for opt in self.optimizers:
                        base = getattr(opt, "_base_lr", cfg.learning_rate)
                        opt_lr = cosine_lr(global_step, cfg.warmup_steps, self.total_steps, base)
                        for pg in opt.param_groups:
                            pg["lr"] = opt_lr
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update()
                    if profile_step:
                        torch.cuda.synchronize(self.device)
                        self._last_step_profile["optimizer_and_clip"] = time.perf_counter() - optimizer_t0
                        self._last_step_profile["compute_step_total"] = time.perf_counter() - profile_t0
                        self._log_step_profile(global_step, epoch, batch_idx)
                    global_step += 1
                    if self.mfu_monitor is not None:
                        self.mfu_monitor.step()

                    if self.log_enabled and global_step % cfg.log_steps == 0:
                        mfu = self.mfu_monitor.flush() if self.mfu_monitor is not None else None
                        mfu_str = f"{mfu:.1%}" if mfu is not None else "-"

                        # ETA
                        elapsed = time.monotonic() - train_start_time
                        steps_done = global_step - train_start_step
                        if steps_done > 0:
                            secs_per_step = elapsed / steps_done
                            eta_secs = secs_per_step * (self.total_steps - global_step)
                            eta_str = format_eta(eta_secs)
                            s_it_str = f"{secs_per_step:.1f}"
                        else:
                            eta_str = "?"
                            s_it_str = "?"

                        if hasattr(self.dataloader.dataset, "__len__"):
                            batches = len(self.dataloader)
                        elif cfg.dataset_size is not None:
                            dp = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
                            batches = cfg.dataset_size // (dp * cfg.batch_size)
                        else:
                            batches = None
                        fractional_epoch = epoch + (batch_idx + 1) / batches if batches else float(epoch)
                        logger.info(
                            "step={}/{} epoch={:.2f} loss={:.4f} lr={:.2e} grad_norm={:.4f} mfu={} eta={} ({} s/it)",
                            global_step,
                            self.total_steps,
                            fractional_epoch,
                            loss.item(),
                            lr,
                            self._last_grad_norm,
                            mfu_str,
                            eta_str,
                            s_it_str,
                        )

                        if self.use_wandb:
                            import wandb

                            log_metrics = {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": fractional_epoch,
                                "train/grad_norm": self._last_grad_norm,
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
                        break

            if stop_training:
                self.train_state.step = global_step
                self.train_state.epoch = epoch
                self.train_state.batch_idx = batch_idx + 1 if "batch_idx" in locals() else 0
                ckpt_path = output_dir / f"checkpoint-{global_step}"
                if cfg.save_final_checkpoint and global_step > 0 and not ckpt_path.exists():
                    self._save_checkpoint(ckpt_path)
                logger.info("Reached max_steps={} at step={}.", cfg.max_steps, global_step)
                break

            # End-of-epoch save
            self.train_state.step = global_step
            self.train_state.epoch = epoch + 1
            self.train_state.batch_idx = 0
            self._save_checkpoint(output_dir / f"checkpoint-epoch{epoch}")
            logger.info("Epoch {} done.", epoch)

        if self.use_wandb:
            import wandb

            wandb.finish()
        import torch.distributed as dist

        dist.destroy_process_group()

    @staticmethod
    def _profile_raw_step_enabled() -> bool:
        return os.environ.get("WAN_TRAINER_PROFILE_RAW_STEP", "").lower() in {"1", "true", "yes", "on"}

    def _time_profile_phase(self, name: str, fn):
        if not self._profile_raw_step_enabled():
            return fn()
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(self.device)
        self._last_step_profile[name] = time.perf_counter() - t0
        return out

    def _log_step_profile(self, global_step: int, epoch: int, batch_idx: int) -> None:
        if not self._profile_raw_step_enabled() or not self._last_step_profile:
            return
        keys = [
            "t5_encode",
            "video_h2d",
            "image_h2d",
            "vae_video",
            "vae_condition",
            "transformer_loss",
            "backward",
            "optimizer_and_clip",
            "compute_step_total",
        ]
        vals = torch.tensor(
            [float(self._last_step_profile.get(k, 0.0)) for k in keys],
            device=self.device,
            dtype=torch.float64,
        )
        group = self._dp_pg if self.expert_parallel and self._dp_pg is not None else None
        vals_sum = vals.clone()
        vals_min = vals.clone()
        vals_max = vals.clone()
        dist.all_reduce(vals_sum, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(vals_min, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(vals_max, op=dist.ReduceOp.MAX, group=group)
        denom = self.dp_size if self.expert_parallel else self.world_size
        means = vals_sum / max(denom, 1)
        total = max(float(vals_max[keys.index("compute_step_total")].item()), 1e-9)
        if self.log_enabled:
            parts = []
            for idx, key in enumerate(keys):
                mx = float(vals_max[idx].item())
                mn = float(vals_min[idx].item())
                avg = float(means[idx].item())
                pct = 100.0 * mx / total
                parts.append(f"{key}=max:{mx:.2f}s mean:{avg:.2f}s min:{mn:.2f}s ({pct:.1f}%)")
            logger.info(
                "raw_step_profile step={} epoch={} batch_idx={} group={} | {}",
                global_step,
                epoch,
                batch_idx,
                self.expert_group,
                " | ".join(parts),
            )

    def _train_step(self, batch: dict) -> torch.Tensor:
        """Single forward pass: encode frozen inputs, compute loss.

        Supports two modes:
        - Raw data: batch has "videos", "image", "prompt" — encode on-the-fly.
        - Precomputed: batch has "video_latents", "condition", "prompt_embeds" tensors.
        """
        self._last_step_profile = {}
        self._last_batch_debug = {
            key: batch[key]
            for key in (
                "sample_key",
                "sample_url",
                "sample_tar",
                "sample_index_in_tar",
                "sample_seq_len",
                "sample_prompt",
                "prompt",
                "index",
            )
            if key in batch
        }
        if "prompt_embeds" in batch:
            # Precomputed latents path
            prompt_embeds = batch["prompt_embeds"].to(self.device)
            video_latents = batch["video_latents"].to(self.device)
            condition = batch["condition"].to(self.device)
        else:
            # Raw data path — encode on-the-fly
            prompt_embeds = self._time_profile_phase(
                "t5_encode",
                lambda: self.model.encode_text(batch["prompt"], self.device),
            )
            video = self._time_profile_phase("video_h2d", lambda: to_model_pixels(batch["videos"][-1], self.device))
            image = self._time_profile_phase("image_h2d", lambda: to_model_pixels(batch["image"], self.device))
            video_latents = self._time_profile_phase("vae_video", lambda: self.model.encode_video(video))
            condition = self._time_profile_phase(
                "vae_condition",
                lambda: self.model.prepare_condition(image, video.shape[2], video.shape[-2], video.shape[-1]),
            )
        return self._time_profile_phase(
            "transformer_loss",
            lambda: self.model.compute_loss(
                video_latents,
                condition,
                prompt_embeds,
                prompt_dropout=self.cfg.prompt_dropout,
            ),
        )
