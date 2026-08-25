"""Model lifecycle + the DRY sampling core.

``build_model`` loads any DCP checkpoint into a :class:`WanI2VForTraining`
(reusing the training-time loader, which handles both experts, EMA/raw and
LoRA). :class:`InferenceEngine` runs exactly one ``sde_generate`` call — the
same loop the trainer uses — and captures the per-step predicted-clean preview,
so ODE / SDE / CPS all share one code path.
"""

import torch
from pydantic import BaseModel, ConfigDict

from src.models.wan_i2v import WanI2VForTraining
from src.trainer.checkpoint import load_dcp_into_pipeline

from .config import InferenceConfig
from .inputs import PreparedInput


def _torch_dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.bfloat16


def build_model(cfg: InferenceConfig, *, need_text_encoder: bool) -> WanI2VForTraining:
    """Construct the model, load the checkpoint, and move everything to the device.

    ``load_dcp_into_pipeline`` is duck-typed on ``.transformer`` / ``.transformer_2``
    and works directly on :class:`WanI2VForTraining`. For single-expert (5B)
    models ``transformer_2 is None`` and the low-expert load is skipped.
    """
    model = WanI2VForTraining(
        cfg.model_path,
        train_experts="both",
        train_text_encoder=False,
        gradient_checkpointing=False,
        load_vae=True,
        load_text_encoder=need_text_encoder,
        transformer_dtype=_torch_dtype(cfg.transformer_dtype),
    )
    if cfg.checkpoint:
        load_dcp_into_pipeline(model, cfg.checkpoint, use_ema=cfg.use_ema)

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    for transformer in (model.transformer, model.transformer_2):
        if transformer is not None:
            transformer.to(device)
            transformer.eval()
    if model.vae is not None:
        model.vae.to(device)
        model.vae.eval()
    if model.text_encoder is not None:
        model.text_encoder.to(device)
        model.text_encoder.eval()
    return model


class StepwiseResult(BaseModel):
    """One rollout's output: final latents + the per-step z0 trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_latent: torch.Tensor  # (B, C, T', H', W') on device
    pred_x0: list[torch.Tensor] = []  # T predicted-clean latents (B, C, T', H', W') on CPU
    sigmas: list[float] = []  # T + 1 schedule values
    timesteps: list[float] = []  # T timestep values


class InferenceEngine:
    """Runs ODE / SDE / CPS sampling for a prepared input."""

    def __init__(self, model: WanI2VForTraining, cfg: InferenceConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    @torch.no_grad()
    def sample(self, prepared: PreparedInput) -> StepwiseResult:
        cfg = self.cfg
        device = self.device
        bsz = cfg.batch_size

        condition = prepared.condition.repeat_interleave(bsz, dim=0)
        prompt_embeds = prepared.prompt_embeds.repeat_interleave(bsz, dim=0)

        # Optional shared x_T (DanceGRPO-style): all batch members start from the
        # same noise. Seed offsets mirror the existing samplers so --mode sde
        # reproduces sample_dancegrpo_sde.py group 0.
        initial_latent = None
        if cfg.share_init_noise:
            latent_shape = self.model.latent_shape_from_condition(prepared.condition)
            init_generator = torch.Generator(device=device).manual_seed(cfg.seed + 17)
            shared = torch.randn(latent_shape, device=device, dtype=torch.bfloat16, generator=init_generator)
            initial_latent = shared.repeat_interleave(bsz, dim=0)

        rollout_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1009)
        traj = self.model.sde_generate(
            condition=condition,
            prompt_embeds=prompt_embeds,
            num_sampling_steps=cfg.num_sampling_steps,
            sde_noise_scale=cfg.effective_noise_scale,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            cfg_scale=cfg.cfg_scale,
            generator=rollout_generator,
            initial_latent=initial_latent,
            sde_formula=cfg.sde_formula,
            return_pred_x0=cfg.save_steps,
        )

        sigmas = traj["sigmas"]
        return StepwiseResult(
            final_latent=traj["latents"][-1].detach(),
            pred_x0=list(traj.get("pred_x0", [])),
            sigmas=[float(s) for s in sigmas.detach().cpu().tolist()],
            timesteps=[float(t) for t in traj["timesteps"]],
        )
