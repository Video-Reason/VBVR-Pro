"""Base RL trainer with independent infrastructure for FSDP2 + DCP training.

This is the RL-side counterpart of BaseTrainer (which serves SFT).
The two hierarchies are intentionally kept separate so the RL side can
evolve independently — e.g. toward an inference/training GPU split
architecture — without affecting SFT trainers.
"""

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.i2v_dataset import I2VDataset
from src.data.vbvr_latent_dataset import VBVRLatentDataset
from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.checkpoint_runtime import CheckpointRuntimeMixin
from src.trainer.config import RLConfig
from src.trainer.ema import EMA
from src.trainer.optimizer import build_optimizer
from src.trainer.tensor_parallel import parallelize_wan_transformer
from src.trainer.utils import (
    apply_liger_rms_norm,
    collate,
    prepare_diffusers_attention_backend,
    setup_loguru,
    shard_transformer,
)


def _delayed_replay_optimizer_steps(
    *,
    batches_per_epoch: int,
    num_epochs: int,
    save_steps: int,
    max_steps: int | None,
) -> int:
    """Count updates after delayed-replay prefill and drain bubbles.

    The pending trajectory slot is deliberately empty at every checkpoint and
    epoch boundary. The next non-boundary batch therefore pre-fills the slot
    without updating the optimizer. Simulating that two-state machine keeps the
    LR schedule and ETA aligned with the number of updates the dataloader can
    actually produce.
    """
    optimizer_step = 0
    pending = False
    for _epoch in range(num_epochs):
        pending = False
        for batch_idx in range(batches_per_epoch):
            if max_steps is not None and optimizer_step >= max_steps:
                return optimizer_step

            next_update = optimizer_step + 1
            must_flush = (
                batch_idx == batches_per_epoch - 1
                or (max_steps is not None and next_update >= max_steps)
                or (save_steps > 0 and next_update % save_steps == 0)
            )
            if not pending and not must_flush:
                pending = True
                continue

            optimizer_step += 1
            pending = not must_flush

        if pending:
            raise AssertionError("delayed replay step accounting left a pending epoch slot")
    return optimizer_step


class BaseRLTrainer(CheckpointRuntimeMixin):
    """Independent RL training infrastructure: distributed init, model, FSDP2,
    EMA, compile, dataset, optimizer, DCP checkpointing, wandb, and resume.

    Separated from BaseTrainer so the RL training pipeline can evolve
    independently (e.g. inference/training GPU split).

    Subclasses implement ``train()`` and any algorithm-specific setup.
    Override hooks:
        ``_pre_fsdp_setup``  — called after model build, before FSDP sharding.
        ``_setup_fsdp``      — override to shard additional modules (call super).
        ``_post_init``       — called after all base init, before resume.
        ``_compute_total_steps`` — controls total optimizer steps.
    """

    def __init__(self, cfg: RLConfig):
        self.cfg = cfg

        # ---- Distributed ----
        self._dist_timeout = timedelta(minutes=cfg.distributed_timeout_minutes)
        dist_backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(dist_backend, timeout=self._dist_timeout)
        self.global_rank = dist.get_rank()
        self.global_world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        self._init_rl_split_groups(cfg, dist_backend)

        # ---- Expert parallel (must run before model build) ----
        self._init_expert_parallel(cfg)
        self._init_tensor_parallel(cfg)
        seed_rank = self.dp_rank if self.tensor_parallel_enabled else self.global_rank
        torch.manual_seed(cfg.seed + seed_rank)

        split_debug_logs = bool(getattr(cfg, "rl_split_debug_logs", False))
        self.log_enabled = (
            self._is_train_root()
            or (self.rl_split_enabled and split_debug_logs and self.local_rank == 0)
            or (self.expert_parallel and getattr(self, "dp_rank", -1) == 0)
        )
        setup_loguru(self.global_rank, enabled=self.log_enabled)
        if self.rl_split_enabled:
            hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"
            logger.info(
                "RL split: train_ranks={} inference_ranks={} role={} local_train_world_size={} "
                "global_rank={} local_rank={} host={} split_debug_logs={}",
                self.train_global_ranks,
                self.inference_global_ranks,
                self.rl_role,
                self.world_size,
                self.global_rank,
                self.local_rank,
                hostname,
                split_debug_logs,
            )
        else:
            logger.info("World size: {}", self.world_size)
        if self.tensor_parallel_enabled:
            logger.info(
                "Tensor parallel topology: DP={} x TP={} (dp_rank={}, tp_rank={})",
                self.dp_size,
                self.tensor_parallel_size,
                self.dp_rank,
                self.tp_rank,
            )
        self._configure_attention_backend(cfg)

        # ---- Model ----
        self.model = self._build_model(cfg)
        self._configure_model_attention_backend(cfg)

        # ---- Subclass hook (e.g. create reference policy copies) ----
        self._pre_fsdp_setup(cfg)

        # ---- FSDP2 ----
        if cfg.fsdp and self.is_train_rank:
            self.mesh, self.mp_policy = self._create_device_mesh(cfg)
            self.sync_modules = self._setup_fsdp(cfg)
        else:
            if cfg.expert_parallel and self.is_train_rank:
                raise ValueError("expert_parallel requires fsdp=True")
            # Move transformers to GPU (FSDP normally handles this via fully_shard).
            # Rollout-only ranks intentionally keep a full, unsharded policy copy.
            for m in [self.model.transformer, self.model.transformer_2]:
                if m is not None:
                    m.to(self.device)
            self.mesh, self.mp_policy = None, None
            self.sync_modules = []
            self._dp_pg = None
            if self.is_train_rank:
                self._dcp_pg = None
                logger.info("FSDP disabled — using manual gradient all-reduce")
            else:
                for m in [self.model.transformer, self.model.transformer_2]:
                    if m is not None:
                        m.requires_grad_(False)
                        m.eval()
                logger.info("Rollout actor rank initialized with an unsharded policy copy")

        # ---- EMA ----
        self.ema = self._setup_ema(cfg) if self.is_train_rank else None

        # ---- torch.compile ----
        if cfg.torch_compile:
            self._compile_modules(cfg)

        # ---- Dataset / DataLoader ----
        self._effective_dataset_size = None
        self.dataset, self.sampler = self._build_dataset(cfg)
        self.dataloader = self._build_dataloader(self.dataset, cfg)

        # ---- Optimizer ----
        if self.is_train_rank:
            (
                self.params,
                self.optimizers,
                self.optimizer_te,
                self.optimizer_1,
                self.optimizer_2,
                self.fallback_te,
                self.fallback_1,
                self.fallback_2,
            ) = self._build_optimizers(cfg)
        else:
            self.params = []
            self.optimizers = []
            self.optimizer_te = self.optimizer_1 = self.optimizer_2 = None
            self.fallback_te = self.fallback_1 = self.fallback_2 = None

        # ---- Total steps ----
        self.total_steps = self._compute_total_steps()
        if hasattr(self.dataset, "__len__"):
            logger.info(
                "Dataset: {} samples, {} batches/epoch, {} total optimizer steps",
                len(self.dataset),
                len(self.dataloader),
                self.total_steps,
            )
        else:
            logger.info(
                "Dataset: streaming (IterableDataset), {} total optimizer steps",
                self.total_steps,
            )

        # ---- DCP state ----
        if self.is_train_rank:
            self.train_state = TrainState(
                text_encoder=self.model.text_encoder
                if (cfg.train_text_encoder and self.model.text_encoder is not None)
                else None,
                transformer=self.model.transformer,
                transformer_2=self.model.transformer_2,
            )
        else:
            self.train_state = TrainState(None, None, None)

        # ---- Subclass hook (e.g. MFU monitor) ----
        self._post_init(cfg)

        # ---- Wandb ----
        self.use_wandb = cfg.wandb_project is not None and self._is_train_root()
        if self.use_wandb:
            import wandb

            wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.model_dump())

        # ---- Resume ----
        auto_resume_path = self._find_latest_checkpoint() if cfg.auto_resume else None
        resume_path = auto_resume_path or cfg.resume_from
        if resume_path:
            is_auto_resume = auto_resume_path is not None
            if cfg.reset_dataloader is None:
                self._reset_on_load = not is_auto_resume
            else:
                self._reset_on_load = cfg.reset_dataloader
            if self.is_train_rank:
                self._load_checkpoint(resume_path)
            elif self.rl_split_enabled:
                prefer = "auto" if self._reset_on_load else "raw"
                self._load_actor_checkpoint(resume_path, prefer=prefer)
                if not self._reset_on_load:
                    self._load_actor_dataloader_state(resume_path)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _init_rl_split_groups(self, cfg: RLConfig, dist_backend: str) -> None:
        """Partition global ranks into train and rollout roles for split RL."""
        self.rl_split_enabled = cfg.rl_train_node_count > 0 or cfg.rl_train_rank_count > 0
        self.rl_role = "train"
        self.is_train_rank = True
        self.is_inference_rank = False
        self.train_global_ranks = list(range(self.global_world_size))
        self.inference_global_ranks: list[int] = []
        self.rollout_rank = -1
        self.rollout_world_size = 0
        self._train_pg = None
        self._train_gloo_pg = None
        self._dcp_pg = None
        self._split_actor_pgs: dict[int, dist.ProcessGroup] = {}

        if not self.rl_split_enabled:
            self.rank = self.global_rank
            self.world_size = self.global_world_size
            self._checkpoint_pg = dist.new_group(
                ranks=list(range(self.world_size)),
                backend="gloo",
                timeout=self._dist_timeout,
            )
            return

        if cfg.expert_parallel:
            raise ValueError("rl_train_node_count split mode does not support expert_parallel")
        if cfg.rl_actor_weight_sync == "lora" and cfg.lora_rank <= 0:
            raise ValueError(
                "rl_actor_weight_sync='lora' requires lora_rank > 0; "
                "use rl_actor_weight_sync='full' for full-finetune split RL "
                "or 'none' for stale-actor plumbing tests"
            )
        if cfg.rl_actor_weight_sync == "full" and not cfg.rl_async_rollout:
            raise ValueError("rl_actor_weight_sync='full' currently requires rl_async_rollout=true")
        if cfg.rl_actor_weight_sync_interval <= 0:
            raise ValueError("rl_actor_weight_sync_interval must be >= 1")
        if self.local_world_size <= 0:
            raise ValueError("Could not determine LOCAL_WORLD_SIZE for split RL")
        if self.global_world_size % self.local_world_size != 0:
            raise ValueError(
                f"global_world_size={self.global_world_size} must be divisible by "
                f"local_world_size={self.local_world_size}"
            )

        if cfg.rl_train_rank_count > 0:
            train_world_size = cfg.rl_train_rank_count
            if train_world_size >= self.global_world_size:
                raise ValueError(
                    "rl_train_rank_count must leave at least one rollout rank; "
                    f"got train_ranks={train_world_size}, total_ranks={self.global_world_size}"
                )
        else:
            node_count = self.global_world_size // self.local_world_size
            if cfg.rl_train_node_count >= node_count:
                raise ValueError(
                    "rl_train_node_count must leave at least one rollout node; "
                    f"got train_nodes={cfg.rl_train_node_count}, total_nodes={node_count}"
                )
            train_world_size = cfg.rl_train_node_count * self.local_world_size
        self.train_global_ranks = list(range(train_world_size))
        self.inference_global_ranks = list(range(train_world_size, self.global_world_size))
        self.rollout_world_size = len(self.inference_global_ranks)
        self._train_pg = dist.new_group(
            ranks=self.train_global_ranks,
            backend=dist_backend,
            timeout=self._dist_timeout,
        )
        self._train_gloo_pg = dist.new_group(
            ranks=self.train_global_ranks,
            backend="gloo",
            timeout=self._dist_timeout,
        )
        train_root = self.train_global_ranks[0]
        for actor_rank in self.inference_global_ranks:
            pg = dist.new_group(
                ranks=[train_root, actor_rank],
                backend="gloo",
                timeout=self._dist_timeout,
            )
            if self.global_rank in (train_root, actor_rank):
                self._split_actor_pgs[actor_rank] = pg

        if self.global_rank in self.train_global_ranks:
            self.rank = dist.get_rank(self._train_pg)
            self.world_size = train_world_size
            self._checkpoint_pg = self._train_gloo_pg
        else:
            self.rl_role = "inference"
            self.is_train_rank = False
            self.is_inference_rank = True
            self.rank = self.global_rank
            # In split mode ``world_size`` means the train mesh size. Rollout
            # fan-out is tracked separately by ``rollout_world_size``.
            self.world_size = train_world_size
            self.rollout_rank = self.global_rank - train_world_size
            self._checkpoint_pg = None

    def _is_train_root(self) -> bool:
        return self.is_train_rank and self.global_rank == self.train_global_ranks[0]

    def _pre_fsdp_setup(self, cfg: RLConfig) -> None:
        """Called after model build, before FSDP. Override to create ref models etc."""

    def _init_expert_parallel(self, cfg: RLConfig) -> None:
        """Set up expert-parallel state: split GPUs into two groups, one per MoE expert."""
        self.expert_parallel = cfg.expert_parallel
        self.peer_rank = -1
        if not self.expert_parallel:
            self._effective_train_experts = cfg.train_experts
            self.expert_group = -1
            self.dp_rank = self.rank
            self.dp_size = self.world_size
            return

        assert cfg.train_experts == "both", f"expert_parallel requires train_experts='both', got '{cfg.train_experts}'"
        assert self.world_size >= 2 and self.world_size % 2 == 0, (
            f"expert_parallel requires even world_size >= 2, got {self.world_size}"
        )

        half = self.world_size // 2
        self.expert_group = 0 if self.rank < half else 1
        self.dp_rank = self.rank % half
        self.dp_size = half
        self.peer_rank = self.rank + half if self.expert_group == 0 else self.rank - half
        self._effective_train_experts = "high" if self.expert_group == 0 else "low"
        self._expert_log_peer = half if self.expert_group == 0 else 0

    def _init_tensor_parallel(self, cfg: RLConfig) -> None:
        """Validate and record logical DP/TP ranks before model construction."""
        self.tensor_parallel_size = int(cfg.tensor_parallel_size)
        self.tensor_parallel_enabled = self.tensor_parallel_size > 1
        self.tp_rank = 0
        self.tp_mesh = None
        self.parallel_mesh = None
        self._tp_pg = None
        if not self.tensor_parallel_enabled:
            return

        incompatible: list[str] = []
        if self.rl_split_enabled:
            incompatible.append("split RL")
        if self.expert_parallel:
            incompatible.append("expert_parallel")
        if not cfg.fsdp:
            incompatible.append("fsdp=False")
        if cfg.hsdp:
            incompatible.append("hsdp=True")
        if cfg.lora_rank > 0:
            incompatible.append("LoRA")
        if cfg.train_text_encoder:
            incompatible.append("train_text_encoder")
        if incompatible:
            raise ValueError("tensor_parallel_size > 1 currently does not support: " + ", ".join(incompatible))
        if self.world_size % self.tensor_parallel_size != 0:
            raise ValueError(
                f"train world_size={self.world_size} must be divisible by "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )

        self.dp_size = self.world_size // self.tensor_parallel_size
        self.dp_rank = self.rank // self.tensor_parallel_size
        self.tp_rank = self.rank % self.tensor_parallel_size
        shared_prompt_wave_size = cfg.grpo_shared_prompt_microbatch_size or cfg.batch_size
        if cfg.grpo_shared_prompt_batch and shared_prompt_wave_size > self.dp_size:
            raise ValueError(
                "grpo_shared_prompt_batch assigns each prompt wave over data-parallel replicas, so the "
                "effective prompt-wave size must be <= DP size; got "
                f"wave_size={shared_prompt_wave_size}, batch_size={cfg.batch_size}, "
                f"DP={self.dp_size}, TP={self.tensor_parallel_size}"
            )

    def _get_expert_parallel_sampler_seed(self, cfg: RLConfig) -> int:
        """Sampler seed for expert-parallel mode."""
        return cfg.seed

    def _expert_parallel_duplicates_data(self, cfg: RLConfig) -> bool:
        return self.expert_parallel and cfg.expert_parallel_data_mode == "duplicate"

    def _post_init(self, cfg: RLConfig) -> None:
        """Called after base init, before wandb/resume. Override for MFU etc."""

    def _compute_total_steps(self) -> int:
        """Total optimizer steps. Override for different accumulation strategies."""
        dataset_size = self._effective_dataset_size
        if dataset_size is None:
            dataset_size = self.cfg.dataset_size
        if dataset_size is not None:
            if self.rl_split_enabled or self.cfg.grpo_shared_prompt_batch:
                dp = 1
            else:
                dp = (
                    self.dp_size
                    if self.tensor_parallel_enabled or self._expert_parallel_duplicates_data(self.cfg)
                    else self.world_size
                )
            batches_per_epoch = dataset_size // (dp * self.cfg.batch_size)
        else:
            batches_per_epoch = len(self.dataloader)
        if self.cfg.grpo_delayed_replay:
            return _delayed_replay_optimizer_steps(
                batches_per_epoch=batches_per_epoch,
                num_epochs=self.cfg.num_epochs,
                save_steps=self.cfg.save_steps,
                max_steps=self.cfg.max_steps,
            )
        total = self.cfg.num_epochs * batches_per_epoch
        if self.cfg.max_steps is not None:
            total = self.cfg.max_steps if total <= 0 else min(total, self.cfg.max_steps)
        return total

    def _configure_attention_backend(self, cfg: RLConfig) -> None:
        if not cfg.disable_cudnn_sdp:
            return
        if not torch.cuda.is_available() or not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            return
        torch.backends.cuda.enable_cudnn_sdp(False)
        logger.info(
            "Disabled cuDNN SDPA backend (flash={}, mem_efficient={}, math={})",
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )

    def _configure_model_attention_backend(self, cfg: RLConfig) -> None:
        if cfg.attention_backend is None:
            return
        prepare_diffusers_attention_backend(cfg.attention_backend)
        count = 0
        for module in (self.model.transformer, self.model.transformer_2):
            if module is None:
                continue
            for child in module.modules():
                if hasattr(child, "set_attention_backend") and getattr(child, "processor", None) is not None:
                    child.set_attention_backend(cfg.attention_backend)
                    count += 1
        logger.info("Set Diffusers attention backend to {} on {} modules", cfg.attention_backend, count)

    def train(self):
        """Main training loop. Must be implemented by subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @staticmethod
    def _reward_needs_vae(cfg: RLConfig) -> bool:
        """Return True if the configured reward declares ``requires_vae = True``.

        Looked up by name in the rewards registry so model-building can force
        the VAE to load before the reward object is constructed (which happens
        later in ``_post_init``).
        """
        reward_name = getattr(cfg, "grpo_reward_fn", None)
        if not reward_name:
            return False
        from src.trainer.rewards.registry import _REGISTRY

        reward_cls = _REGISTRY.get(reward_name)
        return reward_cls is not None and getattr(reward_cls, "requires_vae", False)

    def _build_model(self, cfg: RLConfig) -> WanI2VForTraining:
        train_experts = self._effective_train_experts
        lora_cfg = (
            LoRATrainConfig(rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
            if cfg.lora_rank > 0
            else None
        )
        transformer_dtype = self._resolve_transformer_dtype(cfg)
        gradient_checkpointing_autocast_dtype = self._resolve_gradient_checkpointing_autocast_dtype(
            cfg,
            transformer_dtype,
        )
        use_precomputed = cfg.latent_webdataset_dir is not None
        load_vae = not use_precomputed
        load_text_encoder = not use_precomputed or cfg.train_text_encoder
        # Rewards that need pixel-space evaluation (e.g. maze) force VAE load
        # even when the dataset ships precomputed latents.
        if self._reward_needs_vae(cfg):
            if not load_vae:
                logger.info("Reward '{}' requires VAE — loading it despite precomputed latents", cfg.grpo_reward_fn)
            load_vae = True
        if cfg.grpo_save_rollout_videos:
            if not load_vae:
                logger.info("Rollout video saving enabled — loading VAE despite precomputed latents")
            load_vae = True
        logger.info(
            "Loading model from {} (lora_rank={}, experts={}{}, load_vae={}, load_text_encoder={}, "
            "transformer_dtype={}, checkpoint_autocast={}) ...",
            cfg.model_path,
            cfg.lora_rank,
            train_experts,
            f", expert_group={self.expert_group}" if self.expert_parallel else "",
            load_vae,
            load_text_encoder,
            transformer_dtype,
            gradient_checkpointing_autocast_dtype,
        )
        model = WanI2VForTraining(
            cfg.model_path,
            lora_config=lora_cfg,
            train_experts=train_experts,
            train_text_encoder=cfg.train_text_encoder,
            gradient_checkpointing=cfg.gradient_checkpointing,
            load_vae=load_vae,
            load_text_encoder=load_text_encoder,
            transformer_dtype=transformer_dtype,
            gradient_checkpointing_autocast_dtype=gradient_checkpointing_autocast_dtype,
        )
        if cfg.use_liger_kernel:
            count = 0
            for m in [model.transformer, model.transformer_2]:
                if m is not None:
                    count += apply_liger_rms_norm(m)
            logger.info("Liger Kernel: replaced {} RMSNorm modules", count)
        if model.text_encoder is not None:
            model.text_encoder.to(self.device)
        if model.vae is not None:
            model.vae.to(self.device)
        return model

    def _resolve_transformer_dtype(self, cfg: RLConfig) -> torch.dtype:
        if cfg.transformer_load_dtype == "float32":
            return torch.float32
        if cfg.transformer_load_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32 if cfg.lora_rank == 0 else torch.bfloat16

    def _resolve_gradient_checkpointing_autocast_dtype(
        self,
        cfg: RLConfig,
        transformer_dtype: torch.dtype,
    ) -> torch.dtype | None:
        if not cfg.gradient_checkpointing:
            return None
        if cfg.param_dtype == "bfloat16" and transformer_dtype == torch.float32:
            return torch.bfloat16
        return None

    # ------------------------------------------------------------------
    # FSDP2
    # ------------------------------------------------------------------

    def _create_device_mesh(self, cfg: RLConfig):
        if self.tensor_parallel_enabled:
            self.parallel_mesh = init_device_mesh(
                "cuda",
                (self.dp_size, self.tensor_parallel_size),
                mesh_dim_names=("dp", "tp"),
            )
            self.tp_mesh = self.parallel_mesh["tp"]
            mesh = self.parallel_mesh["dp"]
            self._tp_pg = self.tp_mesh.get_group()
            self._dp_pg = mesh.get_group()
            self._dcp_pg = None
            logger.info(
                "Initialized 2D training mesh: {} DP replicas x {} TP ranks",
                self.dp_size,
                self.tensor_parallel_size,
            )
        elif self.rl_split_enabled and not self.expert_parallel:
            if self._train_pg is None:
                raise RuntimeError("split RL train rank is missing its train process group")
            if cfg.hsdp and self.world_size > self.local_world_size:
                logger.warning("split RL currently uses a 1D train FSDP mesh; ignoring hsdp=True")
            mesh = DeviceMesh.from_group(
                self._train_pg,
                "cuda",
                mesh=torch.arange(self.world_size, device="cpu"),
                mesh_dim_names=("dp",),
            )
            self._dp_pg = self._train_pg
            self._dcp_pg = self._train_gloo_pg
        elif self.expert_parallel and cfg.hsdp:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
            # On a single node, both expert groups share the node, so each
            # expert's shard group only owns half of LOCAL_WORLD_SIZE.
            local_size = min(local_world_size, self.dp_size)
            assert self.dp_size % local_size == 0, (
                f"dp_size ({self.dp_size}) must be divisible by local GPUs ({local_size})"
            )
            num_replicas = self.dp_size // local_size
            if num_replicas <= 1:
                logger.warning("expert_parallel+hsdp but only 1 node per expert group, HSDP disabled")
                mesh_2d = init_device_mesh(
                    "cuda",
                    (2, self.dp_size),
                    mesh_dim_names=("expert", "dp"),
                )
                mesh = mesh_2d["dp"]
            else:
                rep_backend = cfg.hsdp_replicate_backend or "nccl"
                mesh_3d = init_device_mesh(
                    "cuda",
                    (2, num_replicas, local_size),
                    mesh_dim_names=("expert", "replicate", "shard"),
                    backend_override={"replicate": rep_backend},
                )
                mesh = mesh_3d["replicate", "shard"]
                logger.info(
                    "EP+HSDP 3D mesh: 2 experts × {} replicas × {} shards (replicate={})",
                    num_replicas,
                    local_size,
                    rep_backend,
                )
            half = self.world_size // 2
            dp_ranks = list(range(half)) if self.expert_group == 0 else list(range(half, self.world_size))
            self._dp_pg = dist.new_group(ranks=dp_ranks, timeout=self._dist_timeout)
            self._dcp_pg = self._create_expert_dcp_pg()
            logger.info(
                "Expert parallel: group={} dp_rank={}/{}",
                self.expert_group,
                self.dp_rank,
                self.dp_size,
            )
        elif self.expert_parallel:
            mesh_2d = init_device_mesh(
                "cuda",
                (2, self.dp_size),
                mesh_dim_names=("expert", "dp"),
            )
            mesh = mesh_2d["dp"]
            self._dp_pg = mesh.get_group()
            self._dcp_pg = self._create_expert_dcp_pg()
            logger.info(
                "Expert parallel mesh: group={} dp_rank={}/{}",
                self.expert_group,
                self.dp_rank,
                self.dp_size,
            )
        elif cfg.hsdp:
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
            assert self.world_size % local_size == 0, (
                f"world_size ({self.world_size}) must be divisible by local GPUs ({local_size})"
            )
            num_replicas = self.world_size // local_size
            if num_replicas <= 1:
                logger.warning("hsdp=True but only 1 node detected, falling back to plain FSDP")
                mesh = init_device_mesh("cuda", (self.world_size,))
            else:
                rep_backend = cfg.hsdp_replicate_backend or "nccl"
                mesh = init_device_mesh(
                    "cuda",
                    (num_replicas, local_size),
                    mesh_dim_names=("replicate", "shard"),
                    backend_override={"replicate": rep_backend},
                )
                logger.info(
                    "HSDP mesh: {} replicas × {} shards (shard=nccl, replicate={})",
                    num_replicas,
                    local_size,
                    rep_backend,
                )
            self._dp_pg = None
            self._dcp_pg = None
        else:
            mesh = init_device_mesh("cuda", (self.world_size,))
            self._dp_pg = None
            self._dcp_pg = None

        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype_map[cfg.param_dtype],
            reduce_dtype=dtype_map[cfg.reduce_dtype],
        )
        return mesh, mp_policy

    def _setup_fsdp(self, cfg: RLConfig) -> list[torch.nn.Module]:
        """Shard trainable modules with FSDP2. Override to shard additional modules."""
        if cfg.train_text_encoder and self.model.text_encoder is not None:
            fully_shard(self.model.text_encoder, mesh=self.mesh, mp_policy=self.mp_policy)
        if self.model.transformer is not None:
            self._parallelize_transformer_for_tp(self.model.transformer, "transformer")
            shard_transformer(self.model.transformer, self.mesh, self.mp_policy)
        if self.model.transformer_2 is not None:
            self._parallelize_transformer_for_tp(self.model.transformer_2, "transformer_2")
            shard_transformer(self.model.transformer_2, self.mesh, self.mp_policy)
        return [
            m
            for m in [
                self.model.text_encoder if (cfg.train_text_encoder and self.model.text_encoder is not None) else None,
                self.model.transformer,
                self.model.transformer_2,
            ]
            if m is not None
        ]

    def _parallelize_transformer_for_tp(self, transformer: torch.nn.Module, name: str) -> None:
        if not self.tensor_parallel_enabled:
            return
        if self.tp_mesh is None:
            raise RuntimeError("TP mesh must be initialized before parallelizing the Wan transformer")
        stats = parallelize_wan_transformer(transformer, self.tp_mesh)
        logger.info(
            "Tensor-parallelized {}: blocks={} attentions={} linears={} global_rms_norms={} "
            "converted_liger_rms_norms={}",
            name,
            stats.blocks,
            stats.attentions,
            stats.linears,
            stats.rms_norms,
            stats.liger_rms_norms,
        )
        if stats.liger_rms_norms:
            logger.warning(
                "TP converted {} Liger Q/K RMSNorm modules in {} to the collective-aware implementation; "
                "local Liger normalization would change Wan semantics",
                stats.liger_rms_norms,
                name,
            )

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _setup_ema(self, cfg: RLConfig) -> EMA | None:
        if cfg.ema_decay <= 0:
            return None
        ema_models: dict[str, torch.nn.Module] = {}
        if cfg.train_text_encoder and self.model.text_encoder is not None:
            ema_models["text_encoder"] = self.model.text_encoder
        if self.model.transformer is not None:
            ema_models["transformer"] = self.model.transformer
        if self.model.transformer_2 is not None:
            ema_models["transformer_2"] = self.model.transformer_2
        ema = EMA(ema_models, decay=cfg.ema_decay)
        logger.info("EMA enabled (decay={}, {} shadow params)", cfg.ema_decay, len(ema.shadow))
        return ema

    # ------------------------------------------------------------------
    # torch.compile
    # ------------------------------------------------------------------

    def _compile_modules(self, cfg: RLConfig) -> None:
        compile_kwargs = {"backend": cfg.torch_compile_backend}
        if cfg.torch_compile_mode is not None:
            compile_kwargs["mode"] = cfg.torch_compile_mode
        if self.model.vae is not None:
            self.model.vae.compile(**compile_kwargs)
            logger.info("Compiled vae")
        if not cfg.train_text_encoder and self.model.text_encoder is not None:
            self.model.text_encoder.compile(**compile_kwargs)
            logger.info("Compiled text_encoder")
        if self.model.transformer is not None:
            self.model.transformer.compile(**compile_kwargs)
            logger.info("Compiled transformer")
        if self.model.transformer_2 is not None:
            self.model.transformer_2.compile(**compile_kwargs)
            logger.info("Compiled transformer_2")
        logger.info("torch.compile enabled (backend={}, mode={})", cfg.torch_compile_backend, cfg.torch_compile_mode)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg: RLConfig) -> tuple:
        if cfg.latent_webdataset_dir is not None:
            if self.rl_split_enabled or cfg.grpo_shared_prompt_batch:
                # Split/shared-prompt GRPO intentionally feeds the same prompt
                # batch to every participating rank. Ranks partition G across
                # cards, so batch_size is the effective global prompt batch.
                data_rank = 0
                data_world_size = 1
                data_seed = cfg.seed
            else:
                if self.tensor_parallel_enabled or self._expert_parallel_duplicates_data(cfg):
                    data_rank = self.dp_rank
                    data_world_size = self.dp_size
                else:
                    data_rank = self.rank
                    data_world_size = self.world_size
                data_seed = (
                    self._get_expert_parallel_sampler_seed(cfg)
                    if self._expert_parallel_duplicates_data(cfg)
                    else cfg.seed
                )
            epoch_length = None
            if cfg.dataset_size is not None:
                epoch_length = cfg.dataset_size // data_world_size
            allowed_task_names = None
            if cfg.grpo_reward_fn == "vbvr_rule":
                from src.trainer.rewards.vbvr_rule import evalkit_supported_task_names

                allowed_task_names = evalkit_supported_task_names(cfg.vbvr_reward_evalkit_dir)
                if self.rank == 0:
                    logger.info(
                        "Filtering VBVR RL dataset to {} EvalKit-supported tasks",
                        len(allowed_task_names),
                    )
                if epoch_length is not None:
                    source_task_dir = Path("data/vbvr/VBVR-Dataset/tars")
                    source_task_names = {path.stem for path in source_task_dir.glob("*.tar")}
                    dataset_appears_prefiltered = "supported" in str(Path(cfg.latent_webdataset_dir)).lower()
                    if dataset_appears_prefiltered:
                        if self.rank == 0:
                            logger.info(
                                "VBVR RL dataset path appears prefiltered; keeping configured epoch_length={}",
                                epoch_length,
                            )
                    elif source_task_names:
                        supported_source_tasks = source_task_names & allowed_task_names
                        task_ratio = len(supported_source_tasks) / len(source_task_names)
                        filtered_epoch_length = max(1, int(epoch_length * task_ratio))
                        if self.rank == 0:
                            logger.info(
                                "VBVR RL dataset filter keeps {}/{} source tasks; epoch_length per data rank: {} -> {}",
                                len(supported_source_tasks),
                                len(source_task_names),
                                epoch_length,
                                filtered_epoch_length,
                            )
                        epoch_length = filtered_epoch_length
                        self._effective_dataset_size = epoch_length * data_world_size
                    elif self.rank == 0:
                        logger.warning(
                            "Could not find {}; VBVR task filtering is enabled but dataset_size is unchanged",
                            source_task_dir,
                        )
            dataset = VBVRLatentDataset(
                cfg.latent_webdataset_dir,
                epoch_length=epoch_length,
                seed=data_seed,
                allowed_task_names=allowed_task_names,
                node_rank=data_rank,
                node_world_size=data_world_size,
            )
            return dataset, None  # IterableDataset; shard splitting handled by wds
        else:
            raw_prefetch_stride = 1
            if cfg.raw_remote_prefetch_lookahead > 0:
                if self.rl_split_enabled or cfg.grpo_shared_prompt_batch:
                    raw_prefetch_stride = 1
                elif self.expert_parallel:
                    if self._expert_parallel_duplicates_data(cfg):
                        raw_prefetch_stride = self.dp_size
                    else:
                        raw_prefetch_stride = self.world_size
                else:
                    raw_prefetch_stride = self.dp_size if self.tensor_parallel_enabled else self.world_size
            dataset = I2VDataset(
                json_path=cfg.dataset_json,
                num_frames=cfg.num_frames,
                max_area=cfg.max_area,
                height=cfg.height,
                width=cfg.width,
                fps=cfg.fps,
                shuffle_indices=cfg.shuffle_raw_indices,
                shuffle_seed=cfg.shuffle_raw_indices_seed if cfg.shuffle_raw_indices_seed is not None else cfg.seed,
                remote_prefetch_lookahead=cfg.raw_remote_prefetch_lookahead,
                remote_prefetch_workers=cfg.raw_remote_prefetch_workers,
                remote_prefetch_stride=raw_prefetch_stride,
                item_trace_seconds=cfg.dataloader_item_trace_seconds,
            )
        sampler_shuffle = not cfg.shuffle_raw_indices
        if self.rl_split_enabled or cfg.grpo_shared_prompt_batch:
            sampler = DistributedSampler(
                dataset,
                num_replicas=1,
                rank=0,
                shuffle=sampler_shuffle,
                seed=cfg.seed,
            )
        elif self.expert_parallel:
            seed = self._get_expert_parallel_sampler_seed(cfg)
            replicas = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
            rank = self.dp_rank if self._expert_parallel_duplicates_data(cfg) else self.rank
            sampler = DistributedSampler(
                dataset,
                num_replicas=replicas,
                rank=rank,
                shuffle=sampler_shuffle,
                seed=seed,
            )
        elif self.tensor_parallel_enabled:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dp_size,
                rank=self.dp_rank,
                shuffle=sampler_shuffle,
                seed=cfg.seed,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=sampler_shuffle,
                seed=cfg.seed,
            )
        return dataset, sampler

    def _build_dataloader(self, dataset, cfg: RLConfig) -> StatefulDataLoader:
        kwargs = dict(
            dataset=dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=True,
        )
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler
        if cfg.num_workers > 0:
            kwargs["persistent_workers"] = cfg.persistent_workers
            kwargs["prefetch_factor"] = cfg.prefetch_factor
            kwargs["in_order"] = cfg.dataloader_in_order
            if cfg.dataloader_timeout_seconds > 0:
                kwargs["timeout"] = cfg.dataloader_timeout_seconds
        return StatefulDataLoader(**kwargs)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizers(self, cfg: RLConfig):
        optimizer_te = None
        optimizer_1 = None
        optimizer_2 = None
        fallback_te = None
        fallback_1 = None
        fallback_2 = None
        params = []
        total_params = 0

        if cfg.train_text_encoder and self.model.text_encoder is not None:
            params_te = [p for p in self.model.text_encoder.parameters() if p.requires_grad]
            params.extend(params_te)
            total_params += sum(p.numel() for p in self.model.text_encoder.parameters())
            optimizer_te, extras = build_optimizer(params_te, cfg)
            if extras:
                fallback_te = extras[0]

        if self.model.transformer is not None:
            params_1 = [p for p in self.model.transformer.parameters() if p.requires_grad]
            params.extend(params_1)
            total_params += sum(p.numel() for p in self.model.transformer.parameters())
            optimizer_1, extras = build_optimizer(params_1, cfg)
            if extras:
                fallback_1 = extras[0]

        if self.model.transformer_2 is not None:
            params_2 = [p for p in self.model.transformer_2.parameters() if p.requires_grad]
            params.extend(params_2)
            total_params += sum(p.numel() for p in self.model.transformer_2.parameters())
            optimizer_2, extras = build_optimizer(params_2, cfg)
            if extras:
                fallback_2 = extras[0]

        trainable_count = sum(p.numel() for p in params)
        logger.info(
            "Trainable: {:.1f}M / {:.1f}M ({:.2f}%)",
            trainable_count / 1e6,
            total_params / 1e6,
            100 * trainable_count / total_params,
        )
        optimizers = [
            opt
            for opt in [optimizer_te, optimizer_1, optimizer_2, fallback_te, fallback_1, fallback_2]
            if opt is not None
        ]
        return params, optimizers, optimizer_te, optimizer_1, optimizer_2, fallback_te, fallback_1, fallback_2

    # ------------------------------------------------------------------
    # Gradient sync
    # ------------------------------------------------------------------

    def _set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        if self.cfg.grpo_fsdp_sync_each_backward and self.cfg.fsdp:
            requires_gradient_sync = True
        for module in self.sync_modules:
            if hasattr(module, "set_requires_gradient_sync"):
                module.set_requires_gradient_sync(requires_gradient_sync, recurse=True)

    def _all_reduce_gradients(self) -> None:
        """Manual gradient all-reduce for non-FSDP mode. No-op when FSDP is active."""
        if self.cfg.fsdp or self.world_size <= 1:
            return
        for p in self.params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self._train_pg)

    def _clip_grad_norm(self, max_norm: float) -> float:
        """Clip RL gradients, including mixed 1D-FSDP and 2D-DP/TP DTensors.

        ``torch.nn.utils.clip_grad_norm_`` cannot combine scalar norms from
        DTensors backed by different meshes. Wan TP leaves top-level
        parameters on the 1D DP mesh while projection parameters live on the
        composed 2D DP/TP mesh, so we aggregate local squared norms manually.
        TP-replicated parameters contribute from TP rank 0 only.
        """
        if not self.tensor_parallel_enabled:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm).item()

        local_square_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        gradients: list[torch.Tensor] = []
        for parameter in self.params:
            grad = parameter.grad
            if grad is None:
                continue
            gradients.append(grad)

            include_local_shard = True
            local_grad = grad
            if isinstance(grad, DTensor):
                local_grad = grad.to_local()
                mesh_names = grad.device_mesh.mesh_dim_names
                if mesh_names is not None and "tp" in mesh_names:
                    tp_dim = mesh_names.index("tp")
                    tp_placement = grad.placements[tp_dim]
                    if tp_placement.is_partial():
                        raise RuntimeError(
                            "Cannot clip a TP gradient with a Partial placement; enable FSDP gradient sync "
                            "before optimizer clipping"
                        )
                    if tp_placement.is_replicate() and self.tp_rank != 0:
                        include_local_shard = False
                elif self.tp_rank != 0:
                    # A DP-only DTensor is duplicated in every TP column.
                    include_local_shard = False
            elif self.global_rank != 0:
                # A plain gradient is replicated over the whole training mesh.
                include_local_shard = False

            if include_local_shard:
                local_square_sum += local_grad.detach().double().square().sum()

        dist.all_reduce(local_square_sum, op=dist.ReduceOp.SUM)
        total_norm = local_square_sum.sqrt()
        clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for grad in gradients:
            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            local_grad.mul_(clip_coefficient.to(dtype=local_grad.dtype))
        return total_norm.item()

    def _barrier(self) -> None:
        if self._checkpoint_pg is None:
            return
        dist.barrier(group=self._checkpoint_pg)

    def _checkpoint_rank(self) -> int:
        if self.rl_split_enabled:
            return self.rank
        if self.expert_parallel:
            return self.dp_rank
        if self.mesh is not None and self.mesh.ndim == 2:
            return self.mesh.get_local_rank("shard")
        return self.rank

    def _create_expert_dcp_pg(self):
        """Create per-expert Gloo groups for DCP metadata collectives."""

        half = self.world_size // 2
        high_pg = dist.new_group(ranks=list(range(half)), backend="gloo", timeout=self._dist_timeout)
        low_pg = dist.new_group(
            ranks=list(range(half, self.world_size)),
            backend="gloo",
            timeout=self._dist_timeout,
        )
        return high_pg if self.expert_group == 0 else low_pg

    # ------------------------------------------------------------------
    # Split actor checkpoint init
    # ------------------------------------------------------------------

    def _load_actor_checkpoint(self, path: str, *, prefer: str = "auto") -> None:
        """Weight-only checkpoint load for rollout actors.

        Actors are not part of the train FSDP/DCP process group, so they read
        the checkpoint as a single-process CPU state dict and load it into the
        local unsharded policy copy. Per-step actor sync then keeps actors on
        the latest train policy.
        """
        from src.trainer.checkpoint import extract_init_weights, read_dcp_to_flat_dict, remap_for_current_model

        ckpt = Path(path)
        layout = []
        if (ckpt / "high" / ".metadata").exists():
            layout.append(("transformer", ckpt / "high"))
        if (ckpt / "low" / ".metadata").exists():
            layout.append(("transformer_2", ckpt / "low"))
        if not layout and (ckpt / ".metadata").exists():
            layout = [("transformer", ckpt), ("transformer_2", ckpt)]
        if not layout:
            raise ValueError(f"Checkpoint at {path} has no DCP metadata for actor initialization")

        logger.info("Actor initializing weights from {} (prefer={})", path, prefer)
        for model_key, dcp_path in layout:
            model = getattr(self.model, model_key, None)
            if model is None:
                continue
            flat = read_dcp_to_flat_dict(dcp_path)
            try:
                weights, source_tag = extract_init_weights(flat, model_key, prefer=prefer)
            except RuntimeError as e:
                logger.debug("No {} data in {}: {}", model_key, dcp_path, e)
                continue
            remapped = remap_for_current_model(weights, model)
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            logger.info(
                "Actor loaded {} {} weights from {} (missing={}, unexpected={})",
                model_key,
                source_tag,
                dcp_path,
                len(missing),
                len(unexpected),
            )

    def _load_actor_dataloader_state(self, path: str) -> None:
        ckpt = Path(path)
        candidates = [
            ckpt / "high" / "dataloader_rank0.pt",
            ckpt / "low" / "dataloader_rank0.pt",
            ckpt / "dataloader_rank0.pt",
        ]
        dl_state_path = next((p for p in candidates if p.exists()), None)
        if dl_state_path is None:
            logger.warning("Actor found no dataloader_rank0.pt under {}; starting dataloader fresh.", ckpt)
            return
        self.dataloader.load_state_dict(torch.load(dl_state_path, weights_only=False))
        logger.info("Actor restored dataloader state from {}", dl_state_path)

    # ------------------------------------------------------------------
    # DCP checkpointing — provided by CheckpointRuntimeMixin
    # ------------------------------------------------------------------
