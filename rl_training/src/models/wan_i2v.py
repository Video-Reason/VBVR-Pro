"""Wan2.2 I2V/TI2V model wrapper for training.

Wraps frozen (text_encoder, vae) and trainable transformer components from the
Wan pipelines. Wan2.2 A14B uses high/low expert transformers; Wan2.2 5B uses a
single transformer with expanded per-token timesteps.
"""

import html
import json
import math
from pathlib import Path

import ftfy
import regex as re
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from diffusers.models import WanTransformer3DModel
from loguru import logger
from peft import LoraConfig
from pydantic import BaseModel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, UMT5EncoderModel


class LoRATrainConfig(BaseModel):
    """LoRA configuration for Wan I2V transformers."""

    rank: int = 16
    lora_alpha: int = 16
    target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_dropout: float = 0.0


def _clean_prompt(text: str) -> str:
    """Replicate the pipeline's prompt_clean: ftfy + html unescape + whitespace."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_flow_shift(model_dir: Path) -> float:
    """Read the scheduler flow shift used by inference."""
    scheduler_config_path = model_dir / "scheduler" / "scheduler_config.json"
    if not scheduler_config_path.exists():
        logger.warning("Scheduler config not found at {}; falling back to flow_shift=5.0", scheduler_config_path)
        return 5.0
    with open(scheduler_config_path) as f:
        scheduler_config = json.load(f)
    return float(scheduler_config.get("flow_shift", scheduler_config.get("shift", 5.0)))


def _make_autocast_checkpoint_func(dtype: torch.dtype):
    """Checkpoint blocks under the same autocast context for forward and recompute."""

    def _gradient_checkpointing_func(module, *args):
        device_type = "cuda"
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_type = arg.device.type
                break

        def run_checkpoint():
            return checkpoint(
                module.__call__,
                *args,
                use_reentrant=False,
                determinism_check="none",
            )

        if device_type != "cuda":
            return run_checkpoint()

        # Non-reentrant checkpointing records the ambient autocast state and
        # restores it for recomputation. Keeping autocast around checkpoint()
        # therefore gives forward/recompute the same bf16 behavior without a
        # context_fn. PyTorch 2.11 Dynamo only accepts TorchDispatchMode-based
        # checkpoint context functions, and rejects our former nested autocast
        # context_fn before the first compiled Wan block can run.
        with torch.autocast(device_type=device_type, dtype=dtype):
            return run_checkpoint()

    return _gradient_checkpointing_func


class WanI2VForTraining:
    """Wan2.2 I2V/TI2V model for flow-matching training.

    Frozen: text_encoder, vae.
    Trainable: transformer and, for A14B, optional transformer_2 low-noise expert.
    """

    def __init__(
        self,
        model_path: str,
        lora_config: LoRATrainConfig | None = None,
        train_experts: str = "both",
        train_text_encoder: bool = False,
        gradient_checkpointing: bool = True,
        load_vae: bool = True,
        load_text_encoder: bool = True,
        transformer_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing_autocast_dtype: torch.dtype | None = None,
    ):
        assert train_experts in ("both", "high", "low"), (
            f"train_experts must be 'both', 'high', or 'low', got '{train_experts}'"
        )
        self.train_experts = train_experts
        self.train_text_encoder = train_text_encoder

        model_dir = Path(model_path)

        # ---- Read pipeline config for model topology ----
        with open(model_dir / "model_index.json") as f:
            pipe_config = json.load(f)
        self.num_train_timesteps = 1000
        self.expand_timesteps = bool(pipe_config.get("expand_timesteps", False))
        transformer_2_entry = pipe_config.get("transformer_2")
        self.has_low_noise_expert = (model_dir / "transformer_2").exists() and transformer_2_entry not in (
            None,
            [None, None],
        )
        boundary_ratio = pipe_config.get("boundary_ratio")
        if boundary_ratio is None:
            boundary_ratio = 0.9
        self.boundary_timestep = int(boundary_ratio * self.num_train_timesteps)  # 900
        # boundary_idx is computed below after shifted_timesteps is built.
        self.flow_shift = _read_flow_shift(model_dir)

        # ---- Load components sequentially ----
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None

        if load_text_encoder:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
            logger.info("Loaded tokenizer")

            # ---- Text encoder ----
            self.text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
            logger.info("Loaded text_encoder")
            if train_text_encoder:
                self.text_encoder.train()
                if gradient_checkpointing:
                    self.text_encoder.gradient_checkpointing_enable()
            else:
                self.text_encoder.requires_grad_(False)
                self.text_encoder.eval()
        else:
            logger.info("Skipped text_encoder (using precomputed prompt embeddings)")

        # ---- VAE (always frozen) ----
        if load_vae:
            self.vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.bfloat16)
            logger.info("Loaded vae")
            self.vae.requires_grad_(False)
            self.vae.eval()
        else:
            logger.info("Skipped vae (using precomputed latents)")

        # ---- Transformers (only load what we need) ----
        self.transformer: WanTransformer3DModel | None = None
        self.transformer_2: WanTransformer3DModel | None = None
        load_primary_transformer = train_experts in ("both", "high") or not self.has_low_noise_expert
        if load_primary_transformer:
            self.transformer = WanTransformer3DModel.from_pretrained(
                model_dir / "transformer", torch_dtype=transformer_dtype
            )
            logger.info("Loaded transformer")
        if train_experts in ("both", "low") and self.has_low_noise_expert:
            self.transformer_2 = WanTransformer3DModel.from_pretrained(
                model_dir / "transformer_2", torch_dtype=transformer_dtype
            )
            logger.info("Loaded transformer_2")
        elif train_experts in ("both", "low") and not self.has_low_noise_expert:
            logger.info(
                "Model has no transformer_2; routing requested '{}' training through transformer",
                train_experts,
            )

        # ---- LoRA or full fine-tuning ----
        self.lora_config = lora_config
        if lora_config is not None:
            peft_config = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
            )
            for m in [self.transformer, self.transformer_2]:
                if m is None:
                    continue
                m.add_adapter(peft_config)
                m.requires_grad_(False)
                for name, param in m.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

        for m in [self.transformer, self.transformer_2]:
            if m is None:
                continue
            # Ensure uniform dtype for FSDP2 (some params load as float32)
            m.to(transformer_dtype)
            m.train()
            if gradient_checkpointing:
                if gradient_checkpointing_autocast_dtype is None:
                    m.enable_gradient_checkpointing()
                else:
                    m.enable_gradient_checkpointing(
                        gradient_checkpointing_func=_make_autocast_checkpoint_func(
                            gradient_checkpointing_autocast_dtype
                        )
                    )
                    logger.info(
                        "Enabled gradient checkpointing with {} autocast recompute",
                        gradient_checkpointing_autocast_dtype,
                    )

        # ---- VAE normalization constants ----
        if load_vae:
            vae_cfg = self.vae.config
        else:
            # Load VAE config from disk without loading weights
            vae_config_path = model_dir / "vae" / "config.json"
            with open(vae_config_path) as f:
                vae_cfg_dict = json.load(f)

            class _VaeCfg:
                pass

            vae_cfg = _VaeCfg()
            vae_cfg.latents_mean = vae_cfg_dict["latents_mean"]
            vae_cfg.latents_std = vae_cfg_dict["latents_std"]
            vae_cfg.z_dim = vae_cfg_dict["z_dim"]
            vae_cfg.scale_factor_spatial = vae_cfg_dict.get("scale_factor_spatial", 8)
            vae_cfg.scale_factor_temporal = vae_cfg_dict.get("scale_factor_temporal", 4)

        self.latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1)
        self.latents_std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1)

        # ---- Scale factors (from VAE config, not hardcoded) ----
        self.vae_scale_factor_spatial: int = vae_cfg.scale_factor_spatial
        self.vae_scale_factor_temporal: int = vae_cfg.scale_factor_temporal

        # ---- Shifted sigma schedule (match the inference scheduler) ----
        shift = self.flow_shift
        linear_sigmas = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1)[:-1]
        self.shifted_sigmas = shift * linear_sigmas / (1 + (shift - 1) * linear_sigmas)
        # Derive timesteps from shifted sigmas (for passing to transformer)
        self.shifted_timesteps = (self.shifted_sigmas * self.num_train_timesteps).float()

        # Compute boundary_idx from shifted schedule (accounts for nonlinear shift)
        self.boundary_idx = int((self.shifted_timesteps >= self.boundary_timestep).sum().item())
        logger.info(
            "Flow schedule: flow_shift={} boundary_timestep={} boundary_idx={}/{} expand_timesteps={} low_expert={}",
            self.flow_shift,
            self.boundary_timestep,
            self.boundary_idx,
            self.num_train_timesteps,
            self.expand_timesteps,
            self.has_low_noise_expert,
        )

        # ---- BSMNTW loss weighting (Gaussian centered at t=500) ----
        bsmntw = torch.exp(-2.0 * ((self.shifted_timesteps - 500.0) / 1000.0) ** 2)
        bsmntw = bsmntw - bsmntw.min()
        self.bsmntw = bsmntw * (self.num_train_timesteps / bsmntw.sum())
        self._latent_stat_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._training_buffer_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._condition_mask_cache: dict[tuple[str, int, int, int, torch.dtype], torch.Tensor] = {}
        self._first_frame_mask_cache: dict[tuple[str, int, int, int, torch.dtype], torch.Tensor] = {}
        # Rank-synchronized generator for timestep sampling. Under non-EP FSDP,
        # both experts live on every rank; per-rank timestep sampling can route
        # all samples to one expert and skip the other, desyncing FSDP
        # collectives and hanging NCCL. Trainer seeds this with a rank-invariant
        # seed via set_sync_seed after device placement.
        self._sync_generator: torch.Generator | None = None

    def set_sync_seed(self, seed: int, device: torch.device) -> None:
        self._sync_generator = torch.Generator(device=device).manual_seed(int(seed))

    def _sync_randint(self, low: int, high: int, size: tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self._sync_generator is None:
            return torch.randint(low, high, size, device=device)
        return torch.randint(low, high, size, generator=self._sync_generator, device=device)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return a list (not generator) of all trainable parameters."""
        params = []
        if self.train_text_encoder:
            params.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
        for m in [self.transformer, self.transformer_2]:
            if m is not None:
                params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def _get_latent_stats(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype))
        cached = self._latent_stat_cache.get(key)
        if cached is None:
            cached = (
                self.latents_mean.to(device=device, dtype=dtype),
                self.latents_std_inv.to(device=device, dtype=dtype),
            )
            self._latent_stat_cache[key] = cached
        return cached

    def _get_training_buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._training_buffer_cache.get(key)
        if cached is None:
            cached = (
                self.shifted_sigmas.to(device=device),
                self.shifted_timesteps.to(device=device),
                self.bsmntw.to(device=device),
            )
            self._training_buffer_cache[key] = cached
        return cached

    def _get_condition_mask_template(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        key = (str(device), num_frames, height, width, dtype)
        cached = self._condition_mask_cache.get(key)
        if cached is not None:
            return cached

        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        mask[:, :, 1:] = 0
        first_frame_mask = mask[:, :, 0:1].repeat(1, 1, self.vae_scale_factor_temporal, 1, 1)
        mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
        cached = mask.view(1, -1, self.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
        self._condition_mask_cache[key] = cached
        return cached

    def _get_first_frame_latent_mask_template(
        self,
        device: torch.device,
        dtype: torch.dtype,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        key = (str(device), latent_frames, latent_height, latent_width, dtype)
        cached = self._first_frame_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.ones(1, 1, latent_frames, latent_height, latent_width, device=device, dtype=dtype)
        mask[:, :, 0] = 0
        self._first_frame_mask_cache[key] = mask
        return mask

    def _reference_transformer(self) -> WanTransformer3DModel:
        transformer = self.transformer if self.transformer is not None else self.transformer_2
        if transformer is None:
            raise RuntimeError("No transformer is loaded")
        return transformer

    @staticmethod
    def _transformer_input_dtype(transformer: WanTransformer3DModel) -> torch.dtype:
        for param in transformer.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.bfloat16

    def _prepare_transformer_call(
        self,
        transformer: WanTransformer3DModel,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self._transformer_input_dtype(transformer)
        return (
            hidden_states.to(dtype=dtype),
            timestep,
            encoder_hidden_states.to(device=hidden_states.device, dtype=dtype),
        )

    def _build_model_input(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Build transformer hidden_states for A14B concat conditioning or 5B TI2V masking."""
        if not self.expand_timesteps:
            return torch.cat([latent, condition.to(device=latent.device, dtype=latent.dtype)], dim=1)

        if condition.shape[1] != latent.shape[1]:
            raise ValueError(
                "expand_timesteps models expect condition channels to match latent channels "
                f"({latent.shape[1]}), got condition shape {tuple(condition.shape)}. "
                "Recompute latents/condition with the 5B model."
            )
        mask = self._get_first_frame_latent_mask_template(
            latent.device,
            latent.dtype,
            latent.shape[2],
            latent.shape[3],
            latent.shape[4],
        )
        return (1.0 - mask) * condition.to(device=latent.device, dtype=latent.dtype) + mask * latent

    def _build_timestep_input(
        self,
        timesteps: torch.Tensor,
        latent: torch.Tensor,
        transformer: WanTransformer3DModel | None = None,
    ) -> torch.Tensor:
        if not self.expand_timesteps:
            return timesteps.to(torch.bfloat16)

        transformer = transformer or self._reference_transformer()
        p_t, p_h, p_w = transformer.config.patch_size
        mask = self._get_first_frame_latent_mask_template(
            latent.device,
            timesteps.dtype,
            latent.shape[2],
            latent.shape[3],
            latent.shape[4],
        )
        token_mask = mask[0, 0, ::p_t, ::p_h, ::p_w].flatten()
        return (token_mask.unsqueeze(0) * timesteps.view(-1, 1)).to(torch.bfloat16)

    def _iter_transformer_selections(
        self,
        timesteps: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor, WanTransformer3DModel]]:
        device = timesteps.device
        all_selected = torch.arange(timesteps.shape[0], device=device)
        if self.transformer_2 is None:
            return [("single", all_selected, self._reference_transformer())]

        experts: list[tuple[str, torch.Tensor, WanTransformer3DModel]] = []
        if self.transformer is not None:
            experts.append(
                ("high", (timesteps >= self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer)
            )
        if self.transformer_2 is not None:
            experts.append(
                ("low", (timesteps < self.boundary_timestep).nonzero(as_tuple=False).flatten(), self.transformer_2)
            )
        return experts

    def latent_shape_from_condition(self, condition: torch.Tensor) -> tuple[int, int, int, int, int]:
        """Infer rollout latent shape from a condition tensor."""
        if self.expand_timesteps:
            if condition.shape[2] <= 1:
                raise ValueError(
                    "Cannot infer full latent sequence length from a single-frame 5B condition. "
                    "Pass initial_latent or precompute condition expanded to the target latent length."
                )
            return (condition.shape[0], condition.shape[1], *condition.shape[2:])
        return (condition.shape[0], condition.shape[1] - 4, *condition.shape[2:])

    # ------------------------------------------------------------------
    # Encoding helpers (all run under torch.no_grad)
    # ------------------------------------------------------------------

    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """Encode prompts to text embeddings. Returns (B, 512, text_dim)."""
        max_length = 512
        prompts = [_clean_prompt(p) for p in prompts]

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)

        with torch.set_grad_enabled(self.train_text_encoder):
            embeds = self.text_encoder(input_ids, mask).last_hidden_state
        embeds = embeds.masked_fill(~mask.bool().unsqueeze(-1), 0)
        return embeds.to(torch.bfloat16)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode pixel video to normalized latents.

        Args:
            video: (B, C, T, H, W) in [-1, 1].

        Returns:
            (B, z_dim, T', H', W') normalized latents.
        """
        latents = self.vae.encode(video.to(self.vae.dtype)).latent_dist.mode()
        mean, std_inv = self._get_latent_stats(latents.device, latents.dtype)
        return ((latents - mean) * std_inv).to(torch.bfloat16)

    def prepare_condition(
        self,
        image: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build the model condition tensor.

        A14B I2V returns the channel-concatenated condition tensor [mask, cond_latents].
        5B TI2V (``expand_timesteps``) returns first-frame latents expanded to
        the target latent sequence length; masking happens in ``_build_model_input``.

        Replicates the WanImageToVideoPipeline.prepare_latents condition path:
        - Encodes [first_frame, zeros...] with VAE (mode, not sample)
        - Normalizes with latents_mean / latents_std
        - Constructs first-frame mask with temporal expansion

        Args:
            image: (B, C, H, W) first frame in [-1, 1].
            num_frames: Number of pixel-space frames (e.g. 81).
            height: Pixel height.
            width: Pixel width.

        Returns:
            A14B: (B, 4 + z_dim, T', H', W') condition tensor.
            5B:  (B, z_dim, T', H', W') first-frame latent condition.
        """
        B = image.shape[0]

        if self.expand_timesteps:
            with torch.no_grad():
                cond_latents = self.vae.encode(image.unsqueeze(2).to(self.vae.dtype)).latent_dist.mode()
            mean, std_inv = self._get_latent_stats(cond_latents.device, cond_latents.dtype)
            cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)
            latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            return cond_latents.expand(B, -1, latent_frames, -1, -1).contiguous()

        # ---- Encode condition video: [first_frame, zeros...] ----
        cond_video = image.new_zeros((B, 3, num_frames, height, width))
        cond_video[:, :, 0] = image
        # Use mode() (argmax) like the pipeline does for condition
        with torch.no_grad():
            cond_latents = self.vae.encode(cond_video.to(self.vae.dtype)).latent_dist.mode()
        mean, std_inv = self._get_latent_stats(cond_latents.device, cond_latents.dtype)
        cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

        # ---- Construct mask (replicate pipeline_wan_i2v.py L468-481) ----
        mask = self._get_condition_mask_template(image.device, cond_latents.dtype, num_frames, height, width).expand(
            B, -1, -1, -1, -1
        )

        return torch.cat([mask, cond_latents], dim=1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_dropout: float = 0.0,
    ) -> torch.Tensor:
        """Compute flow-matching loss for one training step.

        Flow matching formulation (shifted sigma schedule):
            sigma = shift*s / (1 + (shift - 1)*s) where s = linear_sigma
            noisy = sigma * noise + (1 - sigma) * x0
            target = noise - x0  (velocity)
            loss = BSMNTW_weight * MSE(model_pred, target)

        MoE routing:
            timestep >= boundary (900) -> transformer  (high-noise expert)
            timestep <  boundary (900) -> transformer_2 (low-noise expert)

        Args:
            video_latents: (B, z_dim, T', H', W') normalized latents.
            condition: condition tensor from prepare_condition / latent precompute.
            prompt_embeds: (B, 512, text_dim).

        Returns:
            Scalar loss.
        """
        B = video_latents.shape[0]
        device = video_latents.device
        shifted_sigmas, shifted_timesteps, bsmntw = self._get_training_buffers(device)
        prompt_embeds = self._apply_prompt_dropout(prompt_embeds, prompt_dropout)

        # Sample random timestep indices, then look up shifted sigma / timestep
        if self.train_experts == "high":
            indices = self._sync_randint(0, self.boundary_idx, (B,), device=device)
        elif self.train_experts == "low":
            indices = self._sync_randint(self.boundary_idx, self.num_train_timesteps, (B,), device=device)
        else:
            indices = self._sync_randint(0, self.num_train_timesteps, (B,), device=device)

        sigmas = shifted_sigmas.index_select(0, indices).view(B, 1, 1, 1, 1)
        timesteps = shifted_timesteps.index_select(0, indices)
        weights = bsmntw.index_select(0, indices)
        with torch.no_grad():
            self._last_loss_debug = {
                "train_experts": self.train_experts,
                "indices": indices.detach().cpu().tolist(),
                "timesteps": [round(float(x), 4) for x in timesteps.detach().cpu()],
                "sigmas": [round(float(x), 6) for x in sigmas.flatten().detach().cpu()],
            }

        # Flow matching: noisy = sigma * noise + (1 - sigma) * x0
        noise = torch.randn_like(video_latents)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * video_latents

        # Target velocity: v = noise - x0
        target = noise - video_latents

        model_input = self._build_model_input(noisy_latents, condition)

        # Route to the correct MoE expert(s)
        experts = self._iter_transformer_selections(timesteps)
        self._last_loss_debug["selected_counts"] = {
            "high": int((timesteps >= self.boundary_timestep).sum().item()),
            "low": int((timesteps < self.boundary_timestep).sum().item()),
            "single": B if self.transformer_2 is None else 0,
        }

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_weight = torch.tensor(0.0, device=device, dtype=torch.float32)

        for _name, selected, transformer in experts:
            if selected.numel() == 0:
                continue
            timestep_input = self._build_timestep_input(timesteps, noisy_latents, transformer)
            hidden_states, timestep_input, encoder_hidden_states = self._prepare_transformer_call(
                transformer,
                model_input.index_select(0, selected),
                timestep_input.index_select(0, selected),
                prompt_embeds.index_select(0, selected),
            )
            pred = transformer(
                hidden_states=hidden_states,
                timestep=timestep_input,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            # Per-sample MSE weighted by BSMNTW
            per_sample_loss = F.mse_loss(pred.float(), target.index_select(0, selected).float(), reduction="none")
            per_sample_loss = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))
            selected_weights = weights.index_select(0, selected)
            loss = loss + (per_sample_loss * selected_weights).sum()
            total_weight = total_weight + selected_weights.sum()

        return loss / total_weight if total_weight > 0 else loss

    def _apply_prompt_dropout(self, prompt_embeds: torch.Tensor, dropout: float) -> torch.Tensor:
        """Drop whole prompt embeddings to train the unconditional CFG branch."""
        if dropout <= 0.0:
            return prompt_embeds
        if dropout >= 1.0:
            return torch.zeros_like(prompt_embeds)
        keep = torch.rand((prompt_embeds.shape[0],), device=prompt_embeds.device) >= dropout
        if keep.all():
            return prompt_embeds
        dropped = prompt_embeds.clone()
        dropped[~keep] = 0
        return dropped

    # ------------------------------------------------------------------
    # Flow-GRPO: SDE sampling & log-probability computation
    # ------------------------------------------------------------------

    def _get_expert_for_timestep(self, timestep: float) -> "WanTransformer3DModel":
        """Route a single scalar timestep to the correct MoE expert."""
        if self.transformer is not None and self.transformer_2 is not None:
            return self.transformer if timestep >= self.boundary_timestep else self.transformer_2
        return self.transformer if self.transformer is not None else self.transformer_2

    @staticmethod
    def _flowgrpo_transition_mean(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        """Flow-GRPO transition mean and effective Gaussian noise std."""
        dt = sigma_prev - sigma  # negative (denoising direction)

        # SDE noise std: interpolate between sigma_min and sigma_max based on sigma
        std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma

        # Transition mean (SDE drift = ODE drift + score correction)
        # prev_mean = x + [v + std²/(2σ) * (x + (1-σ)*v)] * dt
        if sigma > 1e-8:
            score_coeff = std_dev_t**2 / (2.0 * sigma)
            prev_sample_mean = (
                sample * (1.0 + score_coeff * dt)
                + model_output * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt
            )
        else:
            # At sigma ≈ 0, skip score correction to avoid division by zero
            prev_sample_mean = sample + model_output * dt

        noise_scale = std_dev_t * math.sqrt(max(-dt, 0.0))
        return prev_sample_mean, noise_scale

    def _sde_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sde_noise_scale: float = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single SDE denoising step with log-probability computation.

        Converts the deterministic ODE step into a stochastic SDE step following
        Flow-GRPO (arXiv:2505.05470). The transition becomes Gaussian, enabling
        tractable log-probability and importance ratio computation.

        Args:
            sample: Current noisy latent x_t. Shape (B, C, T, H, W).
            model_output: Velocity prediction v_θ(x_t, t). Same shape.
            sigma: Current noise level (going from 1→0 during denoising).
            sigma_prev: Next noise level (closer to 0).
            sde_noise_scale: Controls exploration. 'a' in σ_sde = a*√(t/(1-t)).
            sigma_min: Floor for SDE noise std.
            sigma_max: Ceiling for SDE noise std.
            noise: Pre-sampled noise (for reproducibility). If None, sampled here.

        Returns:
            (prev_sample, prev_sample_mean, log_prob):
            - prev_sample: x_{t-1} after stochastic step
            - prev_sample_mean: deterministic mean (for KL computation)
            - log_prob: per-sample log probability, shape (B,)
        """
        prev_sample_mean, noise_scale = self._flowgrpo_transition_mean(
            sample=sample,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        if noise is None:
            noise = torch.randn_like(sample)
        prev_sample = prev_sample_mean + noise_scale * noise

        # Log probability under the Gaussian transition
        log_prob = self._sde_transition_log_prob(
            "flowgrpo", prev_sample, prev_sample_mean, noise_scale, skip_first_frame=self.expand_timesteps
        )

        return prev_sample, prev_sample_mean, log_prob

    @staticmethod
    def _reduce_per_sample(per_elem: torch.Tensor, skip_first_frame: bool = False) -> torch.Tensor:
        """Mean a per-element tensor over all non-batch dims, returning shape (B,).

        When ``skip_first_frame`` is set the conditioning first latent frame
        (``[:, :, 0]``) is dropped before reducing.  For ``expand_timesteps``
        (5B TI2V) models that frame is the frozen I2V condition image, not a
        generated action, so it must not contribute to per-step log-probs / KL.
        """
        if skip_first_frame and per_elem.ndim >= 3 and per_elem.shape[2] > 1:
            per_elem = per_elem[:, :, 1:]
        return per_elem.mean(dim=tuple(range(1, per_elem.ndim)))

    @staticmethod
    def _gaussian_transition_log_prob(
        sample: torch.Tensor,
        mean: torch.Tensor,
        std_dev_t: float,
        skip_first_frame: bool = False,
    ) -> torch.Tensor:
        """Per-sample Gaussian transition log-probability in fp32."""
        if std_dev_t <= 1e-8:
            return torch.zeros(sample.shape[0], device=sample.device)
        log_prob = (
            -((sample.detach().to(torch.float32) - mean.to(torch.float32)) ** 2) / (2.0 * std_dev_t**2)
            - math.log(std_dev_t)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return WanI2VForTraining._reduce_per_sample(log_prob, skip_first_frame)

    @staticmethod
    def _dancegrpo_transition_mean(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        eta: float,
    ) -> tuple[torch.Tensor, float]:
        """DanceGRPO rectified-flow SDE transition mean and noise std.

        This matches the official implementation's ``flux_step``: start from
        the rectified-flow ODE update, then add the SDE score correction using a
        constant noise level ``eta``.
        """
        sample_fp32 = sample.to(torch.float32)
        model_output_fp32 = model_output.to(torch.float32)
        dsigma = sigma_prev - sigma
        delta_t = sigma - sigma_prev
        std_dev_t = eta * math.sqrt(max(delta_t, 0.0))

        prev_sample_mean = sample_fp32 + dsigma * model_output_fp32
        if sigma > 1e-8:
            pred_original_sample = sample_fp32 - sigma * model_output_fp32
            score_estimate = -(sample_fp32 - pred_original_sample * (1.0 - sigma)) / (sigma**2)
            prev_sample_mean = prev_sample_mean + (-0.5 * eta**2 * score_estimate) * dsigma

        return prev_sample_mean, std_dev_t

    @staticmethod
    def _flowcps_transition_mean(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        noise_level: float | torch.Tensor,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        """Coefficients-Preserving Sampling transition mean and noise std."""
        sample_fp32 = sample.to(torch.float32)
        model_output_fp32 = model_output.to(torch.float32)

        if torch.is_tensor(noise_level):
            levels = noise_level.to(device=sample.device, dtype=torch.float32)
            if levels.ndim == 0:
                levels = levels.expand(sample.shape[0])
            if levels.ndim != 1 or levels.shape[0] != sample.shape[0]:
                raise ValueError(
                    "Batched Flow-CPS noise_level must have shape (B,), "
                    f"got {tuple(levels.shape)} for batch size {sample.shape[0]}"
                )
            if not bool(torch.isfinite(levels).all()) or bool(((levels < 0.0) | (levels > 1.0)).any()):
                raise ValueError("Flow-CPS noise_level tensor values must be finite and in [0, 1]")
            levels = levels.reshape(sample.shape[0], *([1] * (sample.ndim - 1)))
            std_dev_t = sigma_prev * torch.sin(levels * (math.pi / 2.0))
            coeff = torch.sqrt(torch.clamp(sigma_prev**2 - std_dev_t**2, min=0.0))
        else:
            if not (0.0 <= noise_level <= 1.0):
                raise ValueError(f"Flow-CPS noise_level must be in [0, 1], got {noise_level}")
            std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2.0)
            coeff = math.sqrt(max(sigma_prev**2 - std_dev_t**2, 0.0))

        pred_original_sample = sample_fp32 - sigma * model_output_fp32
        noise_estimate = sample_fp32 + model_output_fp32 * (1.0 - sigma)
        prev_sample_mean = pred_original_sample * (1.0 - sigma_prev) + noise_estimate * coeff
        return prev_sample_mean, std_dev_t

    @staticmethod
    def _predicted_clean_latent(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        *,
        cond_first_frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the flow-matching clean endpoint prediction for display.

        ``expand_timesteps`` TI2V rollouts keep latent frame zero fixed to the
        encoded input image.  The transformer's frame-zero velocity is not used
        by the actual transition, so it must not leak into a rendered preview.
        """
        pred_x0 = sample.to(torch.float32) - sigma * model_output.to(torch.float32)
        if cond_first_frame is not None:
            pred_x0[:, :, 0:1] = cond_first_frame.to(device=pred_x0.device, dtype=pred_x0.dtype)
        return pred_x0

    @staticmethod
    def _flowcps_transition_log_prob(
        sample: torch.Tensor, mean: torch.Tensor, skip_first_frame: bool = False
    ) -> torch.Tensor:
        """Flow-CPS log-probability surrogate used by the official implementation."""
        log_prob = -((sample.detach().to(torch.float32) - mean.to(torch.float32)) ** 2)
        return WanI2VForTraining._reduce_per_sample(log_prob, skip_first_frame)

    @staticmethod
    def _sde_transition_log_prob(
        sde_formula: str,
        sample: torch.Tensor,
        mean: torch.Tensor,
        noise_scale: float,
        skip_first_frame: bool = False,
    ) -> torch.Tensor:
        if sde_formula == "flowcps":
            return WanI2VForTraining._flowcps_transition_log_prob(sample, mean, skip_first_frame)
        return WanI2VForTraining._gaussian_transition_log_prob(sample, mean, noise_scale, skip_first_frame)

    @staticmethod
    def _sde_transition_kl_loss(
        sde_formula: str,
        mean: torch.Tensor,
        ref_mean: torch.Tensor,
        noise_scale: float,
        skip_first_frame: bool = False,
    ) -> torch.Tensor:
        mse = WanI2VForTraining._reduce_per_sample((mean - ref_mean) ** 2, skip_first_frame).mean()
        if sde_formula == "flowcps":
            return mse
        if noise_scale <= 1e-8:
            return torch.zeros((), device=mean.device, dtype=mean.dtype)
        return mse / (2.0 * noise_scale**2)

    @staticmethod
    def _sde_transition_mean(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        *,
        sde_formula: str,
        sde_noise_scale: float | torch.Tensor = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        if sde_formula == "flowgrpo":
            return WanI2VForTraining._flowgrpo_transition_mean(
                sample=sample,
                model_output=model_output,
                sigma=sigma,
                sigma_prev=sigma_prev,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        if sde_formula == "dancegrpo":
            return WanI2VForTraining._dancegrpo_transition_mean(
                sample=sample,
                model_output=model_output,
                sigma=sigma,
                sigma_prev=sigma_prev,
                eta=sde_noise_scale,
            )
        if sde_formula == "flowcps":
            return WanI2VForTraining._flowcps_transition_mean(
                sample=sample,
                model_output=model_output,
                sigma=sigma,
                sigma_prev=sigma_prev,
                noise_level=sde_noise_scale,
            )
        raise ValueError(f"Unknown SDE formula: {sde_formula}")

    def _dancegrpo_sde_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        eta: float,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single official-style DanceGRPO RF-SDE step with log-probability."""
        prev_sample_mean, std_dev_t = self._dancegrpo_transition_mean(
            sample=sample,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            eta=eta,
        )
        if noise is None:
            noise = torch.randn_like(sample)
        prev_sample = (prev_sample_mean + noise.to(torch.float32) * std_dev_t).to(sample.dtype)
        log_prob = self._sde_transition_log_prob(
            "dancegrpo", prev_sample, prev_sample_mean, std_dev_t, skip_first_frame=self.expand_timesteps
        )
        return prev_sample, prev_sample_mean, log_prob

    def _flowcps_sde_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        noise_level: float | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single Flow-CPS step with the official log-probability surrogate."""
        prev_sample_mean, std_dev_t = self._flowcps_transition_mean(
            sample=sample,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            noise_level=noise_level,
        )
        if noise is None:
            noise = torch.randn_like(sample)
        prev_sample = (prev_sample_mean + noise.to(torch.float32) * std_dev_t).to(sample.dtype)
        log_prob = self._sde_transition_log_prob(
            "flowcps", prev_sample, prev_sample_mean, std_dev_t, skip_first_frame=self.expand_timesteps
        )
        return prev_sample, prev_sample_mean, log_prob

    @torch.no_grad()
    def sde_generate(
        self,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        num_sampling_steps: int = 10,
        sde_noise_scale: float | torch.Tensor = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
        initial_latent: torch.Tensor | None = None,
        sde_formula: str = "flowgrpo",
        return_pred_x0: bool = False,
    ) -> dict:
        """Generate video latents via SDE sampling, storing per-step data for GRPO.

        Runs the full denoising loop using the SDE formulation, collecting
        intermediate latents and log-probabilities needed for policy gradient.

        Args:
            condition: condition tensor from prepare_condition / latent precompute.
            prompt_embeds: (B, 512, text_dim) text embeddings.
            num_sampling_steps: Number of denoising steps T.
            sde_noise_scale: SDE noise parameter. Flow-CPS additionally accepts
                a tensor of shape ``(B,)`` for per-sample coefficients.
            sigma_min: Noise floor.
            sigma_max: Noise ceiling.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            generator: Optional RNG for reproducibility.
            initial_latent: Optional pre-sampled x_T. When provided, all group
                members can share the same initial noise as in DanceGRPO.
            sde_formula: ``"flowgrpo"`` keeps the existing trainer behavior;
                ``"dancegrpo"`` uses the official DanceGRPO RF-SDE update;
                ``"flowcps"`` uses Coefficients-Preserving Sampling.
            return_pred_x0: When True, also capture the predicted clean latent
                ``z0 = x_t - sigma * v`` at each step (offloaded to CPU). Used by
                the inference renderer to preview the denoising trajectory.

        Returns:
            dict with keys:
                - latents: list of T+1 latents [x_T, x_{T-1}, ..., x_0]
                - log_probs: list of T per-step log probabilities
                - timesteps: list of T timestep values
                - sigmas: list of T+1 sigma values
                - noises: list of T noise vectors (for recomputation)
                - pred_x0: list of T predicted-clean latents (only when
                  ``return_pred_x0``; empty otherwise)
        """
        if sde_formula not in {"flowgrpo", "dancegrpo", "flowcps"}:
            raise ValueError(f"Unknown SDE formula: {sde_formula}")

        B = condition.shape[0]
        device = condition.device
        latent_shape = (
            tuple(initial_latent.shape) if initial_latent is not None else self.latent_shape_from_condition(condition)
        )

        # Build sigma schedule for sampling: T+1 values from 1→0
        # Use linspace in [0, 1] then apply the shifted schedule
        t_values = torch.linspace(1.0, 0.0, num_sampling_steps + 1, device=device)
        shift = self.flow_shift
        sigmas = shift * t_values / (1.0 + (shift - 1.0) * t_values)

        # 5B TI2V (expand_timesteps): latent frame 0 is the frozen I2V conditioning
        # image. It must never be noised — pin it to the clean condition latent and
        # exclude it from per-step log-prob / KL (the step reductions skip frame 0).
        freeze_first_frame = self.expand_timesteps and latent_shape[2] > 1 and condition.shape[2] == latent_shape[2]
        cond_first_frame = condition[:, :, 0:1].to(torch.bfloat16) if freeze_first_frame else None

        # Start from pure noise unless a caller provides a shared x_T.
        if initial_latent is None:
            latent = torch.randn(latent_shape, device=device, dtype=torch.bfloat16, generator=generator)
        else:
            latent = initial_latent.to(device=device, dtype=torch.bfloat16).clone()
        if freeze_first_frame:
            latent[:, :, 0:1] = cond_first_frame

        all_latents = [latent]
        all_log_probs = []
        all_timesteps = []
        all_noises = []
        all_pred_x0: list[torch.Tensor] = []

        for i in range(num_sampling_steps):
            sigma = sigmas[i].item()
            sigma_prev = sigmas[i + 1].item()
            timestep_val = sigma * self.num_train_timesteps

            # Select expert based on timestep
            transformer = self._get_expert_for_timestep(timestep_val)

            model_input = self._build_model_input(latent, condition)

            # Forward pass through transformer
            timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.float32).expand(B)
            timestep_input = self._build_timestep_input(timestep_tensor, latent, transformer)
            hidden_states, timestep_input, encoder_hidden_states = self._prepare_transformer_call(
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

            # CFG (if scale > 1)
            if cfg_scale > 1.0:
                # Unconditional forward with zero prompt
                uncond_embeds = torch.zeros_like(encoder_hidden_states)
                uncond_output = transformer(
                    hidden_states=hidden_states,
                    timestep=timestep_input,
                    encoder_hidden_states=uncond_embeds,
                    return_dict=False,
                )[0]
                if sde_formula in {"dancegrpo", "flowcps"}:
                    model_output = uncond_output.to(torch.float32) + cfg_scale * (
                        model_output.to(torch.float32) - uncond_output.to(torch.float32)
                    )
                else:
                    model_output = uncond_output + cfg_scale * (model_output - uncond_output)

            # Predicted clean latent z0 = x_t - sigma * v (post-CFG, before stepping).
            # Same quantity the step renderer decodes for per-step previews.
            if return_pred_x0:
                pred_x0 = self._predicted_clean_latent(
                    latent,
                    model_output,
                    sigma,
                    cond_first_frame=cond_first_frame,
                )
                all_pred_x0.append(pred_x0.detach().cpu())

            # SDE step
            if sde_formula == "dancegrpo":
                noise = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=generator)
                latent, prev_mean, log_prob = self._dancegrpo_sde_step(
                    sample=latent,
                    model_output=model_output,
                    sigma=sigma,
                    sigma_prev=sigma_prev,
                    eta=sde_noise_scale,
                    noise=noise,
                )
            elif sde_formula == "flowcps":
                noise = torch.randn(latent.shape, device=device, dtype=torch.float32, generator=generator)
                latent, prev_mean, log_prob = self._flowcps_sde_step(
                    sample=latent,
                    model_output=model_output,
                    sigma=sigma,
                    sigma_prev=sigma_prev,
                    noise_level=sde_noise_scale,
                    noise=noise,
                )
            else:
                noise = torch.randn_like(latent, generator=generator)
                latent, prev_mean, log_prob = self._sde_step(
                    sample=latent,
                    model_output=model_output,
                    sigma=sigma,
                    sigma_prev=sigma_prev,
                    sde_noise_scale=sde_noise_scale,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    noise=noise,
                )

            if freeze_first_frame:
                # Re-pin frame 0 to the clean conditioning latent and zero its stored
                # noise so the trajectory never carries first-frame noise downstream.
                latent[:, :, 0:1] = cond_first_frame
                noise[:, :, 0:1] = 0

            all_latents.append(latent)
            all_log_probs.append(log_prob)
            all_timesteps.append(timestep_val)
            all_noises.append(noise)

        return {
            "latents": all_latents,
            "log_probs": all_log_probs,
            "timesteps": all_timesteps,
            "sigmas": sigmas,
            "noises": all_noises,
            "pred_x0": all_pred_x0,
        }

    def compute_log_prob_and_kl(
        self,
        latent: torch.Tensor,
        next_latent: torch.Tensor,
        noise: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        sigma: float,
        sigma_prev: float,
        sde_noise_scale: float | torch.Tensor = 0.7,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        sde_formula: str = "flowgrpo",
        use_ref: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-probability and KL for a stored (x_t, x_{t-1}) pair.

        Used in the GRPO training phase to compute importance ratios and KL
        divergence against the reference policy.

        Args:
            latent: x_t, the noisy latent at step t. Shape (B, C, T, H, W).
            next_latent: x_{t-1}, the denoised latent at step t-1.
            noise: The noise vector used in the original SDE step.
            condition: (B, 4+z_dim, T', H', W').
            prompt_embeds: (B, 512, text_dim).
            sigma: Noise level at step t.
            sigma_prev: Noise level at step t-1.
            sde_noise_scale: SDE noise parameter.
            sigma_min: Noise floor.
            sigma_max: Noise ceiling.
            sde_formula: Transition formula: flowgrpo, dancegrpo, or flowcps.
            use_ref: If True, disable LoRA adapters to use reference policy.

        Returns:
            (log_prob, kl_div, prev_sample_mean):
            - log_prob: Per-sample log probability under current/ref policy, shape (B,).
            - kl_div: Per-sample KL divergence between current and ref means, shape (B,).
            - prev_sample_mean: The predicted mean (for KL computation).
        """
        timestep_val = sigma * self.num_train_timesteps
        B = latent.shape[0]
        device = latent.device

        # Select expert
        transformer = self._get_expert_for_timestep(timestep_val)

        # Optionally switch to reference policy
        if use_ref:
            transformer.disable_adapters()

        # Forward pass
        model_input = self._build_model_input(latent, condition)
        timestep_tensor = torch.tensor([timestep_val], device=device, dtype=torch.float32).expand(B)
        timestep_input = self._build_timestep_input(timestep_tensor, latent, transformer)
        hidden_states, timestep_input, encoder_hidden_states = self._prepare_transformer_call(
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

        # Re-enable adapters
        if use_ref:
            transformer.enable_adapters()

        prev_sample_mean, noise_scale = self._sde_transition_mean(
            sample=latent,
            model_output=model_output,
            sigma=sigma,
            sigma_prev=sigma_prev,
            sde_formula=sde_formula,
            sde_noise_scale=sde_noise_scale,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        log_prob = self._sde_transition_log_prob(
            sde_formula, next_latent, prev_sample_mean, noise_scale, skip_first_frame=self.expand_timesteps
        )

        # KL divergence placeholder (computed externally between current and ref means)
        kl_div = torch.zeros(B, device=device)

        return log_prob, kl_div, prev_sample_mean

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents back to pixel space via VAE.

        Args:
            latents: (B, z_dim, T', H', W') normalized latents.

        Returns:
            (B, C, T, H, W) pixel-space video in [-1, 1].
        """
        mean, std_inv = self._get_latent_stats(latents.device, latents.dtype)
        # Undo normalization: latents = (raw - mean) * std_inv → raw = latents / std_inv + mean
        raw_latents = latents / std_inv + mean
        return self.vae.decode(raw_latents.to(self.vae.dtype)).sample
