"""Base trainer with shared infrastructure for FSDP2 + DCP training."""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from src.data.i2v_dataset import I2VDataset
from src.data.vbvr_latent_dataset import VBVRLatentDataset
from src.models.wan_i2v import LoRATrainConfig, WanI2VForTraining
from src.trainer.checkpoint import TrainState
from src.trainer.checkpoint_runtime import CheckpointRuntimeMixin
from src.trainer.config import TrainConfig
from src.trainer.ema import EMA
from src.trainer.optimizer import build_optimizer
from src.trainer.utils import (
    apply_liger_rms_norm,
    collate,
    prepare_diffusers_attention_backend,
    setup_loguru,
    shard_transformer,
)


class BaseTrainer(CheckpointRuntimeMixin):
    """Shared infrastructure: distributed init, model, FSDP2, EMA, compile,
    dataset, optimizer, DCP checkpointing, wandb, and resume logic.

    Subclasses implement ``train()`` and any training-mode-specific setup.
    Override hooks:
        ``_pre_fsdp_setup``  — called after model build, before FSDP sharding.
        ``_setup_fsdp``      — override to shard additional modules (call super).
        ``_post_init``       — called after all base init, before resume.
        ``_compute_total_steps`` — controls total optimizer steps.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # ---- Distributed ----
        self._dist_timeout = timedelta(minutes=cfg.distributed_timeout_minutes)
        dist.init_process_group("nccl", timeout=self._dist_timeout)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(cfg.seed + self.rank)
        self._dcp_pg = None
        self._checkpoint_pg = dist.new_group(
            ranks=list(range(self.world_size)),
            backend="gloo",
            timeout=self._dist_timeout,
        )

        # ---- Expert parallel (must run before model build) ----
        self._init_expert_parallel(cfg)

        log_enabled = self.rank == 0 or (self.expert_parallel and self.dp_rank == 0)
        self.log_enabled = log_enabled
        setup_loguru(self.rank, enabled=log_enabled)
        logger.info("World size: {}", self.world_size)
        self._configure_attention_backend(cfg)
        if self.expert_parallel:
            effective_data_replicas = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
            logger.info(
                "Expert parallel: data_mode={} effective_data_replicas={}",
                cfg.expert_parallel_data_mode,
                effective_data_replicas,
            )

        # ---- Model ----
        self.model = self._build_model(cfg)
        self._configure_model_attention_backend(cfg)
        if hasattr(self.model, "set_sync_seed"):
            self.model.set_sync_seed(cfg.seed, self.device)
        logger.info("Model loaded")

        # ---- Subclass hook (e.g. create reference policy copies) ----
        self._pre_fsdp_setup(cfg)

        # ---- FSDP2 ----
        if cfg.fsdp:
            logger.info("Creating device mesh")
            self.mesh, self.mp_policy = self._create_device_mesh(cfg)
            logger.info("Device mesh ready")
            logger.info("Applying FSDP sharding")
            self.sync_modules = self._setup_fsdp(cfg)
            logger.info("FSDP sharding ready")
        else:
            if cfg.expert_parallel:
                raise ValueError("expert_parallel requires fsdp=True")
            # Move transformers to GPU (FSDP normally handles this via fully_shard)
            for m in [self.model.transformer, self.model.transformer_2]:
                if m is not None:
                    m.to(self.device)
            self.mesh, self.mp_policy = None, None
            self.sync_modules = []
            self._dp_pg = None
            self._dcp_pg = None
            logger.info("FSDP disabled — using manual gradient all-reduce")

        # ---- EMA ----
        self.ema = self._setup_ema(cfg)

        # ---- torch.compile ----
        if cfg.torch_compile:
            self._compile_modules(cfg)

        # ---- Dataset / DataLoader ----
        self.dataset, self.sampler = self._build_dataset(cfg)
        self.dataloader = self._build_dataloader(self.dataset, cfg)

        # ---- Optimizer ----
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
        self.train_state = TrainState(
            text_encoder=self.model.text_encoder
            if (cfg.train_text_encoder and self.model.text_encoder is not None)
            else None,
            transformer=self.model.transformer,
            transformer_2=self.model.transformer_2,
        )

        # ---- Subclass hook (e.g. MFU monitor) ----
        self._post_init(cfg)

        # ---- Wandb ----
        self.use_wandb = cfg.wandb_project is not None and self.rank == 0
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
            self._load_checkpoint(resume_path)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _pre_fsdp_setup(self, cfg: TrainConfig) -> None:
        """Called after model build, before FSDP. Override to create ref models etc."""

    def _init_expert_parallel(self, cfg: TrainConfig) -> None:
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
        # Peer rank for expert log exchange (rank 0 of each expert group)
        self._expert_log_peer = half if self.expert_group == 0 else 0

    def _get_expert_parallel_sampler_seed(self, cfg: TrainConfig) -> int:
        """Sampler seed for expert-parallel mode.

        Both expert groups use the same seed when duplicate-data mode is active.
        """
        return cfg.seed

    def _expert_parallel_duplicates_data(self, cfg: TrainConfig) -> bool:
        return self.expert_parallel and cfg.expert_parallel_data_mode == "duplicate"

    def _post_init(self, cfg: TrainConfig) -> None:
        """Called after base init, before wandb/resume. Override for MFU etc."""

    def _compute_total_steps(self) -> int:
        """Total optimizer steps. Override for different accumulation strategies."""
        if self.cfg.dataset_size is not None:
            dp = self.dp_size if self._expert_parallel_duplicates_data(self.cfg) else self.world_size
            batches_per_epoch = self.cfg.dataset_size // (dp * self.cfg.batch_size)
            total = self.cfg.num_epochs * batches_per_epoch // self.cfg.gradient_accumulation_steps
        else:
            total = self.cfg.num_epochs * len(self.dataloader) // self.cfg.gradient_accumulation_steps
        if self.cfg.max_steps is not None:
            total = self.cfg.max_steps if total <= 0 else min(total, self.cfg.max_steps)
        return total

    def _configure_attention_backend(self, cfg: TrainConfig) -> None:
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

    def _configure_model_attention_backend(self, cfg: TrainConfig) -> None:
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

    def _build_model(self, cfg: TrainConfig) -> WanI2VForTraining:
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

    def _resolve_transformer_dtype(self, cfg: TrainConfig) -> torch.dtype:
        if cfg.transformer_load_dtype == "float32":
            return torch.float32
        if cfg.transformer_load_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32 if cfg.lora_rank == 0 else torch.bfloat16

    def _resolve_gradient_checkpointing_autocast_dtype(
        self,
        cfg: TrainConfig,
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

    def _create_device_mesh(self, cfg: TrainConfig):
        if self.expert_parallel and cfg.hsdp:
            # EP + HSDP: per-expert-group 2D mesh (replicate, shard).
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
            num_replicas = (
                self.dp_size // local_size if self.dp_size >= local_size and self.dp_size % local_size == 0 else 0
            )
            if num_replicas <= 1:
                # Single node per expert group — fall back to EP without HSDP
                logger.warning("expert_parallel+hsdp but only 1 node per expert group, HSDP disabled")
                mesh_2d = init_device_mesh(
                    "cuda",
                    (2, self.dp_size),
                    mesh_dim_names=("expert", "dp"),
                )
                mesh = mesh_2d["dp"]
            else:
                rep_backend = cfg.hsdp_replicate_backend or "nccl"
                if rep_backend == "nccl":
                    mesh_3d = init_device_mesh(
                        "cuda",
                        (2, num_replicas, local_size),
                        mesh_dim_names=("expert", "replicate", "shard"),
                    )
                    mesh = mesh_3d["replicate", "shard"]
                else:
                    # Build per-expert-group 2D meshes directly instead of slicing
                    # a 3D mixed-backend mesh. PyTorch has had issues with mixed-
                    # backend submesh extraction in this path.
                    half = self.world_size // 2
                    g0 = torch.arange(0, half).reshape(num_replicas, local_size)
                    g1 = torch.arange(half, self.world_size).reshape(num_replicas, local_size)
                    bo = (
                        (rep_backend, None),  # replicate dim
                        (None, None),  # shard dim (default nccl)
                    )
                    mesh_g0 = DeviceMesh(
                        "cuda",
                        g0,
                        mesh_dim_names=("replicate", "shard"),
                        backend_override=bo,
                    )
                    mesh_g1 = DeviceMesh(
                        "cuda",
                        g1,
                        mesh_dim_names=("replicate", "shard"),
                        backend_override=bo,
                    )
                    mesh = mesh_g0 if self.expert_group == 0 else mesh_g1
                logger.info(
                    "EP+HSDP: 2 experts × {} replicas × {} shards (replicate={})",
                    num_replicas,
                    local_size,
                    rep_backend,
                )
            # Flat process group for DCP checkpointing (covers all ranks in this expert group)
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
            # HSDP: shard within node (NVLink), replicate across nodes (all-reduce).
            local_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
            assert self.world_size % local_size == 0, (
                f"world_size ({self.world_size}) must be divisible by local GPUs ({local_size})"
            )
            num_replicas = self.world_size // local_size
            if num_replicas <= 1:
                logger.warning("hsdp=True but only 1 node detected, falling back to plain FSDP")
                mesh = init_device_mesh("cuda", (self.world_size,))
            else:
                # Prefer NCCL for cross-node replicate traffic; low-memlock setups
                # should switch NCCL transport away from IB rather than falling back
                # to gloo for the hot path.
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

    def _setup_fsdp(self, cfg: TrainConfig) -> list[torch.nn.Module]:
        """Shard trainable modules with FSDP2. Override to shard additional modules."""
        if cfg.train_text_encoder and self.model.text_encoder is not None:
            fully_shard(self.model.text_encoder, mesh=self.mesh, mp_policy=self.mp_policy)
        if self.model.transformer is not None:
            shard_transformer(self.model.transformer, self.mesh, self.mp_policy)
        if self.model.transformer_2 is not None:
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

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _setup_ema(self, cfg: TrainConfig) -> EMA | None:
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

    def _compile_modules(self, cfg: TrainConfig) -> None:
        compile_kwargs = {"backend": cfg.torch_compile_backend}
        if cfg.torch_compile_mode is not None:
            compile_kwargs["mode"] = cfg.torch_compile_mode
        if self.model.vae is not None:
            self.model.vae = torch.compile(self.model.vae, **compile_kwargs)
            logger.info("Compiled vae")
        if not cfg.train_text_encoder and self.model.text_encoder is not None:
            self.model.text_encoder = torch.compile(self.model.text_encoder, **compile_kwargs)
            logger.info("Compiled text_encoder")
        if self.model.transformer is not None:
            self.model.transformer = torch.compile(self.model.transformer, **compile_kwargs)
            logger.info("Compiled transformer")
        if self.model.transformer_2 is not None:
            self.model.transformer_2 = torch.compile(self.model.transformer_2, **compile_kwargs)
            logger.info("Compiled transformer_2")
        logger.info("torch.compile enabled (backend={}, mode={})", cfg.torch_compile_backend, cfg.torch_compile_mode)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg: TrainConfig) -> tuple:
        if cfg.latent_webdataset_dir is not None:
            # Compute per-rank epoch length so every rank produces the same
            # number of batches — prevents FSDP/NCCL deadlocks at epoch end.
            if self._expert_parallel_duplicates_data(cfg):
                data_rank = self.dp_rank
                data_world_size = self.dp_size
            else:
                data_rank = self.rank
                data_world_size = self.world_size
            data_seed = (
                self._get_expert_parallel_sampler_seed(cfg) if self._expert_parallel_duplicates_data(cfg) else cfg.seed
            )
            epoch_length = None
            if cfg.dataset_size is not None:
                epoch_length = cfg.dataset_size // data_world_size
            dataset = VBVRLatentDataset(
                cfg.latent_webdataset_dir,
                epoch_length=epoch_length,
                seed=data_seed,
                node_rank=data_rank,
                node_world_size=data_world_size,
            )
            return dataset, None  # IterableDataset; shard splitting handled by wds
        else:
            raw_prefetch_stride = 1
            if cfg.raw_remote_prefetch_lookahead > 0:
                raw_prefetch_stride = self.dp_size if self._expert_parallel_duplicates_data(cfg) else self.world_size
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
        if self.expert_parallel:
            if self._expert_parallel_duplicates_data(cfg):
                seed = self._get_expert_parallel_sampler_seed(cfg)
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.dp_size,
                    rank=self.dp_rank,
                    shuffle=sampler_shuffle,
                    seed=seed,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
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

    def _build_dataloader(self, dataset, cfg: TrainConfig) -> StatefulDataLoader:
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

    def _build_optimizers(self, cfg: TrainConfig):
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
        for module in self.sync_modules:
            if hasattr(module, "set_requires_gradient_sync"):
                module.set_requires_gradient_sync(requires_gradient_sync, recurse=True)

    def _all_reduce_gradients(self) -> None:
        """Manual gradient all-reduce for non-FSDP mode. No-op when FSDP is active."""
        if self.cfg.fsdp or self.world_size <= 1:
            return
        for p in self.params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _iter_trainable_named_parameters(self):
        """Yield trainable parameters with stable-ish names for diagnostics."""
        seen: set[int] = set()
        modules = (
            ("text_encoder", self.model.text_encoder if self.cfg.train_text_encoder else None),
            ("transformer", self.model.transformer),
            ("transformer_2", self.model.transformer_2),
        )
        for prefix, module in modules:
            if module is None:
                continue
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                yield f"{prefix}.{name}", param

        # Fallback in case a trainable parameter is owned outside the modules
        # above.  This keeps the diagnostic complete without changing the
        # optimizer parameter list.
        for idx, param in enumerate(getattr(self, "params", [])):
            param_id = id(param)
            if param_id in seen or not param.requires_grad:
                continue
            seen.add(param_id)
            yield f"params[{idx}]", param

    @staticmethod
    def _local_tensor_for_diagnostics(tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(tensor, "to_local"):
            return tensor.to_local()
        return tensor

    def _format_nonfinite_gradient_report(self, max_items: int = 8) -> str:
        """Return a compact report of local non-finite gradients."""
        bad: list[str] = []
        checked = 0
        with torch.no_grad():
            for name, param in self._iter_trainable_named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                checked += 1
                local_grad = self._local_tensor_for_diagnostics(grad.detach())
                finite = torch.isfinite(local_grad)
                if bool(finite.all().item()):
                    continue

                nan_count = int(torch.isnan(local_grad).sum().item())
                inf_count = int(torch.isinf(local_grad).sum().item())
                finite_count = int(finite.sum().item())
                finite_absmax = float(local_grad[finite].abs().max().item()) if finite_count > 0 else float("nan")
                bad.append(
                    f"{name}: shape={tuple(local_grad.shape)} dtype={local_grad.dtype} "
                    f"nan={nan_count} inf={inf_count} finite_absmax={finite_absmax:.6g}"
                )
                if len(bad) >= max_items:
                    break

        if not bad:
            return f"no local non-finite gradients found among {checked} gradients"
        suffix = "" if len(bad) < max_items else f"\n... truncated to first {max_items} tensors"
        report = "\n".join(bad) + suffix
        batch_report = self._format_last_batch_debug()
        loss_report = self._format_last_loss_debug()
        extras = "\n".join(x for x in (batch_report, loss_report) if x)
        return f"{extras}\n{report}" if extras else report

    def _format_last_batch_debug(self, max_items: int = 8) -> str:
        batch = getattr(self, "_last_batch_debug", None)
        if not batch:
            return ""
        parts = []
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if isinstance(value, list) and len(value) > max_items:
                value = value[:max_items] + [f"... +{len(value) - max_items} more"]
            parts.append(f"{key}={value!r}")
        return "batch: " + ", ".join(parts)

    def _format_last_loss_debug(self) -> str:
        debug = getattr(self.model, "_last_loss_debug", None)
        if not debug:
            return ""
        return "loss_debug: " + repr(debug)

    def _local_gradients_are_finite(self) -> bool:
        with torch.no_grad():
            for _name, param in self._iter_trainable_named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                local_grad = self._local_tensor_for_diagnostics(grad.detach())
                if not bool(torch.isfinite(local_grad).all().item()):
                    return False
        return True

    def _emit_nonfinite_gradient_warning(self, context: str) -> None:
        message = (
            f"Skipping optimizer step for non-finite gradients on rank={self.rank} "
            f"expert={self._effective_train_experts} {context}\n{self._format_nonfinite_gradient_report()}"
        )
        if self.log_enabled:
            logger.warning(message)
        else:
            print(f"[rank{self.rank}] {message}", file=sys.stderr, flush=True)

    def _gradients_are_finite_across_ranks(self, *, context: str) -> bool:
        local_finite = self._local_gradients_are_finite()
        if dist.is_available() and dist.is_initialized():
            finite_flag = torch.tensor(int(local_finite), device=self.device, dtype=torch.int32)
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            global_finite = bool(finite_flag.item())
        else:
            global_finite = local_finite

        if global_finite:
            return True
        if local_finite:
            if self.log_enabled:
                logger.warning("Skipping optimizer step because another rank has non-finite gradients {}", context)
        else:
            self._emit_nonfinite_gradient_warning(context)
        return False

    def _clip_grad_norm_or_raise(self, max_norm: float, *, context: str) -> float:
        """Clip gradients, raising with parameter-level diagnostics on NaN/Inf."""
        try:
            return torch.nn.utils.clip_grad_norm_(
                self.params,
                max_norm,
                error_if_nonfinite=True,
            ).item()
        except RuntimeError as exc:
            report = self._format_nonfinite_gradient_report()
            raise FloatingPointError(
                f"Non-finite gradients before clipping on rank={self.rank} "
                f"expert={self._effective_train_experts} {context}\n{report}"
            ) from exc

    def _barrier(self) -> None:
        dist.barrier(group=self._checkpoint_pg)

    def _checkpoint_rank(self) -> int:
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
    # DCP checkpointing — provided by CheckpointRuntimeMixin
    # ------------------------------------------------------------------
