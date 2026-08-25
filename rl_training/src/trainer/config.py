"""Training configuration."""

import math
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class TrainConfig(BaseModel):
    """Base training configuration shared by all trainers."""

    # Model
    model_path: str = "storage/models/Wan2.2-I2V-A14B-Diffusers"

    # Data
    dataset_json: str = "data/train.json"
    latent_webdataset_dir: str | None = None  # dir of precomputed latent tar shards; skips VAE/T5 loading
    dataset_size: int | None = None  # total samples in dataset (required for webdataset / IterableDataset)
    num_frames: int | None = None  # override dataset JSON config; default 81
    max_area: int | None = None  # override dataset JSON config; default 480*832
    height: int | None = None  # override dataset JSON config; fixed height
    width: int | None = None  # override dataset JSON config; fixed width
    fps: int | None = None  # override dataset JSON config; default 16
    shuffle_raw_indices: bool = False  # one-time full index shuffle for JSON/parquet raw datasets
    shuffle_raw_indices_seed: int | None = None  # None = use seed
    raw_remote_prefetch_lookahead: int = 0  # bounded S3 lookahead per DataLoader worker; 0 disables
    raw_remote_prefetch_workers: int = 1  # background downloader threads per DataLoader worker
    num_workers: int = 16
    persistent_workers: bool = True
    prefetch_factor: int = 2
    dataloader_timeout_seconds: int = 0  # 0 keeps PyTorch's default no-timeout behavior
    dataloader_in_order: bool = True  # False lets workers return whichever batch is ready first
    dataloader_item_trace_seconds: int = 0  # raw I2V worker stack dump threshold; 0 disables

    # Training
    output_dir: str = "storage/checkpoints"
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_steps: int | None = None  # optional hard cap on optimizer steps
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    skip_nonfinite_gradients: bool = True
    detect_anomaly: bool = False  # debug only; very slow
    warmup_steps: int = 100
    save_steps: int = 500
    save_final_checkpoint: bool = True
    log_steps: int = 10
    seed: int = 42
    ema_decay: float = 0.0  # 0 = disabled; typical value: 0.9999
    distributed_timeout_minutes: int = 240

    # Optimizer
    optimizer: Literal["adamw", "muon"] = "adamw"
    # AdamW
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_fused: bool = True
    # Muon
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = None
    muon_fallback_lr: float | None = None  # AdamW lr for non-2D params; None = use learning_rate

    # Which components to train
    train_experts: Literal["both", "high", "low"] = "both"
    train_text_encoder: bool = False
    gradient_checkpointing: bool = True

    # FSDP2
    fsdp: bool = True  # False = no sharding, manual gradient all-reduce (faster for LoRA / small models)
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"
    transformer_load_dtype: Literal["auto", "bfloat16", "float32"] = "auto"
    # auto: full fine-tuning loads trainable transformers in fp32 for fp32 Adam states;
    # LoRA keeps the frozen base in bf16 unless explicitly overridden.
    prompt_dropout: float = 0.0  # whole-prompt dropout for CFG/unconditional branch training

    # Liger Kernel (fused Triton kernels)
    use_liger_kernel: bool = False

    # torch.compile
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str | None = None  # e.g. "reduce-overhead", "max-autotune"
    # cuDNN SDPA backward can return NaN for Wan low-noise training on H100/PyTorch 2.11.
    disable_cudnn_sdp: bool = True
    # Explicit Diffusers attention backend, e.g. "_native_flash". None uses Diffusers/PyTorch dispatch.
    attention_backend: str | None = None

    # LoRA (set lora_rank > 0 to enable)
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Checkpoint
    resume_from: str | None = None
    auto_resume: bool = True  # auto-detect latest checkpoint in output_dir
    # None = auto (reset when resume_from is set, keep when auto-resuming from output_dir)
    reset_dataloader: bool | None = None
    # Local rank within the DCP process group. None keeps PyTorch's default
    # coordinator rank 0; negative values count from the end.
    checkpoint_dcp_coordinator_rank: int | None = None
    checkpoint_dcp_thread_count: int = 4
    checkpoint_dcp_sync_files: bool = True

    # Logging
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # HSDP: shard within node, replicate across nodes.
    # Keeps NCCL IB traffic to intra-node NVLink; cross-node uses only gradient
    # all-reduce (much less locked-memory / IB registration pressure).
    hsdp: bool = False  # Hybrid Sharded Data Parallel; shard within node, replicate across nodes
    hsdp_replicate_backend: Literal["nccl", "gloo"] | None = None  # None = auto (prefer nccl)

    # Expert parallel: split MoE experts across GPU sub-groups
    expert_parallel: bool = False  # each expert gets world_size/2 GPUs with independent FSDP
    expert_parallel_data_mode: Literal["duplicate", "split"] = "duplicate"
    # duplicate: each expert group iterates a full copy of the dataset (old SFT behavior)
    # split: shard data across all ranks so expert-parallel uses full global throughput

    @field_validator("prompt_dropout")
    @classmethod
    def _validate_prompt_dropout(cls, v: float):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"prompt_dropout must be in [0, 1], got {v}")
        return v


class SFTConfig(TrainConfig):
    """SFT training configuration."""

    trainer: Literal["i2v"] = "i2v"


class RLConfig(TrainConfig):
    """RL training configuration. Defaults to HSDP (falls back to plain FSDP on single node)."""

    hsdp: bool = True

    # Intra-model tensor parallelism for Wan attention/FFN projections. A
    # single-node TP2 run over 8 GPUs composes this with four-way FSDP data
    # parallelism (TP2 x FSDP4).
    tensor_parallel_size: int = 1

    # Smoke / bounded RL runs. Defaults preserve the normal epoch-based loop.
    max_steps: int | None = None
    save_epoch_checkpoints: bool = True

    # Trainer selection
    trainer: Literal["dancegrpo"] = "dancegrpo"

    # Split RL execution: first N nodes train with FSDP, remaining nodes run
    # rollout/reward actors. 0 keeps the legacy all-ranks-train path.
    rl_train_node_count: int = 0
    # Optional rank-level override for single-node smoke runs or non-node-aligned
    # layouts. 0 means derive train ranks from rl_train_node_count.
    rl_train_rank_count: int = 0
    # lora = sync only LoRA adapter tensors; full = sync all trainable policy
    # tensors for full fine-tuning; none = actor weights stay at init/resume state.
    rl_actor_weight_sync: Literal["lora", "full", "none"] = "lora"
    # Sync rollout actor weights every N optimizer steps in async split mode.
    # Full-model sync is expensive, so large full-FT jobs should use N > 1.
    rl_actor_weight_sync_interval: int = 1
    # Async split rollout keeps each optimizer step's prompt batch unchanged,
    # but lets rollout actors pre-generate future steps through a bounded queue.
    rl_async_rollout: bool = False
    # 0 = auto: enough future steps to give every rollout actor at least one
    # task when rollout_world_size > grpo_group_size.
    rl_async_rollout_prefetch_steps: int = 0
    # Emit per-node split-RL progress logs from local_rank=0 on every node.
    rl_split_debug_logs: bool = False

    # GRPO (set grpo_group_size > 0 to enable Flow-GRPO training)
    grpo_group_size: int | None = None  # G: number of samples per prompt. None = SFT mode
    # Non-split multi-node GRPO mode: every rank reads the same prompt batch,
    # then ranks shard each prompt's group samples. In this mode `batch_size`
    # is the global prompt batch size, not per-rank prompt batch size.
    grpo_shared_prompt_batch: bool = False
    # Optional prompt-wave size inside one shared-prompt optimizer step. A
    # value smaller than batch_size prepares all waves before replaying them,
    # so later reward work overlaps earlier replay without using stale-policy
    # trajectories. None keeps the original single-wave execution.
    grpo_shared_prompt_microbatch_size: int | None = None
    # Experimental one-update-stale pipeline. The first batch pre-fills a
    # trajectory slot; steady-state iterations prepare the next rollout and
    # then replay the previous slot while its successor's CPU reward runs.
    # Checkpoint/final boundaries flush the slot before saving.
    grpo_delayed_replay: bool = False
    # Optional wider PPO clip used only by grpo_delayed_replay. None reuses
    # grpo_clip_range, allowing scheduling and clip-width ablations separately.
    grpo_delayed_replay_clip_range: float | None = None
    grpo_sample_batch_size: int = 1  # how many G samples to batch together during rollout
    # How many already-generated G samples to replay together during the train
    # update. This does not increase prompt batch size or rollout work per step.
    grpo_train_sample_batch_size: int = 1
    # Move the frozen text encoder to CPU after raw prompt encoding and move
    # the VAE to CPU after rollout rewards, freeing replay/optimizer headroom
    # for full A14B fine-tuning. They are restored before the next raw batch.
    grpo_offload_inference_models: bool = False
    # Full fine-tuning can retain unsharded gradients across every replay
    # backward when FSDP synchronization is delayed until the final replay.
    # Synchronizing each backward bounds that peak at the cost of additional
    # reduce-scatter collectives.
    grpo_fsdp_sync_each_backward: bool = False
    grpo_num_sampling_steps: int = 10  # T: denoising steps during SDE sampling
    grpo_clip_range: float = 1e-3  # PPO clipping epsilon
    grpo_kl_coeff: float = 0.004  # beta: KL penalty coefficient against reference policy
    grpo_sde_formula: Literal["flowgrpo", "dancegrpo", "flowcps"] = "dancegrpo"
    grpo_sde_noise_scale: float = 0.3  # DanceGRPO eta / exploration noise level
    # Flow-CPS only: when set, sample one coefficient uniformly from this range
    # per prompt/GRPO group. All G rollouts for that prompt share the sampled
    # value. None preserves the fixed grpo_sde_noise_scale behavior.
    grpo_cps_noise_scale_range: tuple[float, float] | None = None
    grpo_sde_sigma_min: float = 0.0  # noise floor for SDE std
    grpo_sde_sigma_max: float = 1.0  # noise ceiling for SDE std
    grpo_adv_clip_max: float = 5.0  # clamp advantages to [-max, max]
    grpo_reward_fn: str = "neg_loss"  # reward function name
    grpo_cfg_scale: float = 1.0  # classifier-free guidance scale during sampling
    grpo_save_rollout_videos: bool = False  # debug: decode and save generated rollout videos
    grpo_rollout_video_dir: str | None = None  # default: <output_dir>/rollout_videos
    grpo_rollout_video_every_steps: int = 1
    grpo_rollout_video_max_per_rank: int = 1
    grpo_rollout_video_fps: int = 16

    # DanceGRPO (paper-inspired variant of GRPO)
    dancegrpo_share_group_init_noise: bool = True
    dancegrpo_timestep_selection_ratio: float = 0.6

    @field_validator("dancegrpo_timestep_selection_ratio")
    @classmethod
    def _validate_dancegrpo_timestep_selection_ratio(cls, v: float):
        if not (0.0 < v <= 1.0):
            raise ValueError(f"dancegrpo_timestep_selection_ratio must be in (0, 1], got {v}")
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def _validate_tensor_parallel_size(cls, v: int):
        if v <= 0:
            raise ValueError(f"tensor_parallel_size must be > 0, got {v}")
        return v

    @field_validator("grpo_cps_noise_scale_range")
    @classmethod
    def _validate_grpo_cps_noise_scale_range(cls, v: tuple[float, float] | None):
        if v is None:
            return v
        low, high = v
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"grpo_cps_noise_scale_range must satisfy 0 <= min < max <= 1, got ({low}, {high})")
        return v

    @model_validator(mode="after")
    def _validate_flowcps_noise_configuration(self):
        if self.grpo_cps_noise_scale_range is not None and self.grpo_sde_formula != "flowcps":
            raise ValueError("grpo_cps_noise_scale_range requires grpo_sde_formula='flowcps'")
        if (
            self.grpo_sde_formula == "flowcps"
            and self.grpo_cps_noise_scale_range is None
            and not (0.0 < self.grpo_sde_noise_scale <= 1.0)
        ):
            raise ValueError(f"fixed Flow-CPS grpo_sde_noise_scale must be in (0, 1], got {self.grpo_sde_noise_scale}")
        return self

    @field_validator("rl_train_node_count")
    @classmethod
    def _validate_rl_train_node_count(cls, v: int):
        if v < 0:
            raise ValueError(f"rl_train_node_count must be >= 0, got {v}")
        return v

    @field_validator("rl_train_rank_count")
    @classmethod
    def _validate_rl_train_rank_count(cls, v: int):
        if v < 0:
            raise ValueError(f"rl_train_rank_count must be >= 0, got {v}")
        return v

    @field_validator("rl_async_rollout_prefetch_steps")
    @classmethod
    def _validate_rl_async_rollout_prefetch_steps(cls, v: int):
        if v < 0:
            raise ValueError(f"rl_async_rollout_prefetch_steps must be >= 0, got {v}")
        return v

    @field_validator("grpo_train_sample_batch_size")
    @classmethod
    def _validate_grpo_train_sample_batch_size(cls, v: int):
        if v <= 0:
            raise ValueError(f"grpo_train_sample_batch_size must be > 0, got {v}")
        return v

    @field_validator("grpo_shared_prompt_microbatch_size")
    @classmethod
    def _validate_shared_prompt_microbatch_size(cls, v: int | None):
        if v is not None and v <= 0:
            raise ValueError(f"grpo_shared_prompt_microbatch_size must be > 0, got {v}")
        return v

    @field_validator("grpo_clip_range", "grpo_delayed_replay_clip_range")
    @classmethod
    def _validate_grpo_clip_ranges(cls, v: float | None):
        if v is not None and v <= 0:
            raise ValueError(f"PPO clip ranges must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_shared_prompt_microbatch_configuration(self):
        if self.grpo_shared_prompt_microbatch_size is not None and not self.grpo_shared_prompt_batch:
            raise ValueError("grpo_shared_prompt_microbatch_size requires grpo_shared_prompt_batch=true")
        if (
            self.grpo_shared_prompt_microbatch_size is not None
            and self.grpo_shared_prompt_microbatch_size > self.batch_size
        ):
            raise ValueError(
                "grpo_shared_prompt_microbatch_size must be <= batch_size, got "
                f"{self.grpo_shared_prompt_microbatch_size} > {self.batch_size}"
            )
        if (
            self.grpo_shared_prompt_microbatch_size is not None
            and self.batch_size % self.grpo_shared_prompt_microbatch_size != 0
        ):
            raise ValueError(
                "batch_size must be divisible by grpo_shared_prompt_microbatch_size, got "
                f"{self.batch_size} % {self.grpo_shared_prompt_microbatch_size}"
            )
        if self.grpo_delayed_replay and not self.grpo_shared_prompt_batch:
            raise ValueError("grpo_delayed_replay requires grpo_shared_prompt_batch=true")
        return self

    @field_validator("grpo_rollout_video_every_steps", "grpo_rollout_video_max_per_rank", "grpo_rollout_video_fps")
    @classmethod
    def _validate_positive_rollout_video_fields(cls, v: int):
        if v <= 0:
            raise ValueError(f"rollout video fields must be > 0, got {v}")
        return v

    # ------------------------------------------------------------------
    # Maze reward (grpo_reward_fn: "maze")
    # ------------------------------------------------------------------
    # Number of frames to VAE-decode per rollout (cost is B·G·K decodes).
    maze_reward_num_frames: int = 6
    # Component weights — raw reward = w_traj·r_traj + w_onpath·r_onpath + w_goal·r_goal
    # (GRPO z-scores within group, so absolute scale doesn't matter, only ratios).
    maze_reward_w_traj: float = 1.0
    maze_reward_w_onpath: float = 0.5
    maze_reward_w_goal: float = 1.0
    # r_goal: 1 if end-frame ball is within this many cells of goal, else 0.
    maze_reward_goal_cells: float = 0.75

    # ------------------------------------------------------------------
    # Maze growing-line reward (grpo_reward_fn: "maze_line")
    # ------------------------------------------------------------------
    # Scores the generated red/yellow/blue/pink path-line mask against the GT
    # decoded line mask, plus a simple "line reaches goal" term.
    maze_line_reward_num_frames: int = 6
    maze_line_reward_color_threshold: float = 90.0
    maze_line_reward_color_temperature: float = 12.0
    maze_line_reward_w_mask: float = 1.0
    maze_line_reward_w_goal: float = 0.5
    maze_line_reward_goal_cells: float = 1.0

    # ------------------------------------------------------------------
    # Maze tracker/eval reward (grpo_reward_fn: "maze_tracker")
    # ------------------------------------------------------------------
    # Mirrors src.eval.maze_tracker_score overall:
    # 0.35*traj + 0.25*on_path + 0.25*goal + 0.15*progress.
    maze_tracker_reward_num_frames: int = 21
    maze_tracker_reward_search_radius: int = 96
    maze_tracker_reward_color_slack: float = 28.0
    maze_tracker_reward_goal_tolerance_cells: float = 1.0
    maze_tracker_reward_max_mean_error_cells: float = 4.0
    maze_tracker_reward_w_traj: float = 0.35
    maze_tracker_reward_w_onpath: float = 0.25
    maze_tracker_reward_w_goal: float = 0.25
    maze_tracker_reward_w_progress: float = 0.15

    # ------------------------------------------------------------------
    # Standalone VLM judge reward (grpo_reward_fn: "vbvr_vlm")
    # ------------------------------------------------------------------
    # The launcher may override endpoint/model/key through WAN_TRAINER_VLM_*.
    vlm_reward_base_url: str = "http://127.0.0.1:18080/v1"
    vlm_reward_model: str = "qwen3.6-27b"
    vlm_reward_api_key: str = "EMPTY"
    # task_specific selects the pinned 100-task EvalKit-derived prompt map.
    # custom preserves the generic start/final/generated-video contract below.
    vlm_reward_prompt_mode: Literal["task_specific", "custom"] = "task_specific"
    # Custom-mode prompt overrides. A text file takes precedence over the
    # inline value; when both are absent, the generic in-repo prompt is used.
    vlm_reward_system_prompt: str | None = None
    vlm_reward_system_prompt_path: str | None = None
    # The decoded rollout is encoded completely at this FPS before vLLM's
    # single uniform sampling pass.
    vlm_reward_video_fps: int = 16
    # vLLM uniformly selects this many frames from the complete MP4. The HF
    # processor's own sampling is disabled so the video is sampled only once.
    vlm_reward_video_num_frames: int = 32
    vlm_reward_include_gt_first_frame: bool = True
    vlm_reward_decode_batch_size: int = 1
    vlm_reward_concurrency: int = 2
    # Maximum decoded samples waiting for service responses per scoring rank.
    # Zero selects max(decode_batch_size, 2*concurrency).
    vlm_reward_max_pending_jobs: int = 0
    vlm_reward_request_timeout_seconds: float = 180.0
    # Retry transient HTTP failures and, independently, task-output semantic
    # validation failures such as rubric weights that do not sum to 100.
    vlm_reward_max_retries: int = 2
    vlm_reward_retry_backoff_seconds: float = 1.0
    vlm_reward_max_new_tokens: int = 1024
    # This is a downscale-only safety bound, not a target resize. First-frame
    # JPEGs and generated-video frames at or below it retain native resolution.
    vlm_reward_image_max_edge: int = 512
    vlm_reward_jpeg_quality: int = 85
    vlm_reward_score_max: float = 100.0
    # Fixed JSON-schema output is available only in custom mode; task-specific
    # prompts have distinct exact line-oriented schemas.
    vlm_reward_use_structured_output: bool = False
    vlm_reward_validate_service: bool = True
    # Production training is fail-open: exhausted request/schema failures use
    # error_score so one judge failure cannot tear down the distributed job.
    # Strict smoke/preflight configs may override this to true.
    vlm_reward_fail_on_error: bool = False
    vlm_reward_error_score: float = 0.0
    vlm_reward_log_first_n: int = 2

    @field_validator(
        "vlm_reward_video_fps",
        "vlm_reward_video_num_frames",
        "vlm_reward_decode_batch_size",
        "vlm_reward_concurrency",
        "vlm_reward_max_new_tokens",
        "vlm_reward_image_max_edge",
    )
    @classmethod
    def _validate_positive_vlm_reward_integer_fields(cls, v: int):
        if v <= 0:
            raise ValueError(f"VLM reward positive integer fields must be > 0, got {v}")
        return v

    @field_validator("vlm_reward_max_pending_jobs", "vlm_reward_max_retries", "vlm_reward_log_first_n")
    @classmethod
    def _validate_nonnegative_vlm_reward_integer_fields(cls, v: int):
        if v < 0:
            raise ValueError(f"VLM reward nonnegative integer fields must be >= 0, got {v}")
        return v

    @field_validator("vlm_reward_request_timeout_seconds", "vlm_reward_score_max")
    @classmethod
    def _validate_positive_vlm_reward_float_fields(cls, v: float):
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"VLM reward positive float fields must be > 0, got {v}")
        return v

    @field_validator("vlm_reward_retry_backoff_seconds")
    @classmethod
    def _validate_nonnegative_vlm_reward_backoff(cls, v: float):
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"vlm_reward_retry_backoff_seconds must be >= 0, got {v}")
        return v

    @field_validator("vlm_reward_jpeg_quality")
    @classmethod
    def _validate_vlm_reward_jpeg_quality(cls, v: int):
        if not 1 <= v <= 100:
            raise ValueError(f"vlm_reward_jpeg_quality must be in [1, 100], got {v}")
        return v

    @field_validator("vlm_reward_error_score")
    @classmethod
    def _validate_vlm_reward_error_score(cls, v: float):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"vlm_reward_error_score must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_vlm_reward_service_configuration(self):
        if self.grpo_reward_fn != "vbvr_vlm":
            return self
        if not self.vlm_reward_base_url.strip():
            raise ValueError("grpo_reward_fn='vbvr_vlm' requires vlm_reward_base_url")
        if not self.vlm_reward_model.strip():
            raise ValueError("grpo_reward_fn='vbvr_vlm' requires vlm_reward_model")
        if self.vlm_reward_prompt_mode == "task_specific":
            if not self.vlm_reward_include_gt_first_frame:
                raise ValueError("task-specific VLM prompts require vlm_reward_include_gt_first_frame=true")
            if not math.isclose(self.vlm_reward_score_max, 100.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("task-specific VLM prompts require vlm_reward_score_max=100")
            if self.vlm_reward_use_structured_output:
                raise ValueError(
                    "task-specific VLM prompts use per-task line schemas and require "
                    "vlm_reward_use_structured_output=false"
                )
        return self

    # ------------------------------------------------------------------
    # VBVR rule reward (grpo_reward_fn: "vbvr_rule")
    # ------------------------------------------------------------------
    # Decode generated latents, apply the final-eval video preparation contract,
    # and score through the pinned main_v2 EvalKit entrypoint.
    vbvr_reward_evalkit_dir: str | None = None
    vbvr_reward_evalkit_source_sha256: str | None = None
    vbvr_reward_easyocr_module_path: str | None = None
    vbvr_reward_device: str = "cpu"
    vbvr_reward_fps: int = 16
    vbvr_reward_prepared_width: int = 1024
    vbvr_reward_prepared_height: int = 1024
    vbvr_reward_max_duration_seconds: float = 5.0
    vbvr_reward_prepare_crf: int = 12
    vbvr_reward_decode_batch_size: int = 1
    # Maximum decoded samples waiting for CPU video preparation/scoring on
    # each reward-producing rank. Zero selects max(decode_batch_size, 2*workers).
    vbvr_reward_max_pending_jobs: int = 0
    # Spawned scorer processes per reward-producing rank. Keep this low: on
    # eight-GPU nodes, one worker per rank already gives eight CPU scorers.
    vbvr_reward_cpu_workers: int = 1
    vbvr_reward_cpu_threads_per_worker: int = 1
    vbvr_reward_use_process_pool: bool = True
    vbvr_reward_task_specific_only: bool = True
    vbvr_reward_fail_on_error: bool = True
    vbvr_reward_tmp_dir: str = "storage/tmp/vbvr_rule_reward"
    vbvr_reward_keep_tmp: bool = False
    vbvr_reward_unsupported_score: float = 0.0

    @field_validator("vbvr_reward_evalkit_source_sha256")
    @classmethod
    def _validate_vbvr_reward_evalkit_source_sha256(cls, v: str | None):
        if v is None:
            return v
        normalized = v.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("vbvr_reward_evalkit_source_sha256 must be a 64-character hexadecimal digest")
        return normalized

    @model_validator(mode="after")
    def _validate_vbvr_reward_evalkit_pin(self):
        if self.grpo_reward_fn != "vbvr_rule":
            return self
        if not self.vbvr_reward_evalkit_dir:
            raise ValueError("grpo_reward_fn='vbvr_rule' requires vbvr_reward_evalkit_dir")
        if not self.vbvr_reward_evalkit_source_sha256:
            raise ValueError("grpo_reward_fn='vbvr_rule' requires vbvr_reward_evalkit_source_sha256")
        return self

    @field_validator(
        "vbvr_reward_fps",
        "vbvr_reward_prepared_width",
        "vbvr_reward_prepared_height",
        "vbvr_reward_decode_batch_size",
        "vbvr_reward_cpu_workers",
        "vbvr_reward_cpu_threads_per_worker",
    )
    @classmethod
    def _validate_positive_vbvr_reward_fields(cls, v: int):
        if v <= 0:
            raise ValueError(f"VBVR reward positive integer fields must be > 0, got {v}")
        return v

    @field_validator("vbvr_reward_max_pending_jobs")
    @classmethod
    def _validate_nonnegative_vbvr_reward_queue_size(cls, v: int):
        if v < 0:
            raise ValueError(f"vbvr_reward_max_pending_jobs must be >= 0, got {v}")
        return v

    @field_validator("vbvr_reward_max_duration_seconds")
    @classmethod
    def _validate_positive_vbvr_reward_duration(cls, v: float):
        if v <= 0:
            raise ValueError(f"vbvr_reward_max_duration_seconds must be > 0, got {v}")
        return v

    @field_validator("vbvr_reward_prepare_crf")
    @classmethod
    def _validate_vbvr_reward_crf(cls, v: int):
        if not 0 <= v <= 51:
            raise ValueError(f"vbvr_reward_prepare_crf must be in [0, 51], got {v}")
        return v
