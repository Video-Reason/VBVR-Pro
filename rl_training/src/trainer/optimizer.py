"""Optimizer factory.

When using Muon, parameters are split by dimensionality:
  - exactly 2D parameters → Muon (Newton-Schulz orthogonalization)
  - non-2D parameters (1D biases/norms, 3D+ tensors) → AdamW fallback

The first optimizer returned is the "primary" used for DCP checkpointing.
Any extra optimizers (the AdamW fallback for 1D params) are stepped during
training but not checkpointed — their state is cheap to recompute on resume.
"""

from collections.abc import Iterable

import torch
from loguru import logger
from torch.nn import Parameter

from src.trainer.config import TrainConfig


def _log_trainable_param_dtypes(params: list[Parameter], cfg: TrainConfig) -> None:
    dtype_counts: dict[torch.dtype, int] = {}
    for p in params:
        dtype_counts[p.dtype] = dtype_counts.get(p.dtype, 0) + p.numel()
    summary = ", ".join(f"{dtype}: {count / 1e6:.1f}M" for dtype, count in sorted(dtype_counts.items(), key=str))
    logger.info("Optimizer trainable param dtypes: {}", summary or "none")
    if cfg.optimizer == "adamw" and cfg.lora_rank == 0 and any(dtype != torch.float32 for dtype in dtype_counts):
        logger.warning(
            "Full fine-tune AdamW has non-fp32 trainable params; Adam moments will follow param dtype. "
            "Set transformer_load_dtype: float32 to keep optimizer state in fp32."
        )


def build_optimizer(
    params: Iterable[Parameter], cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, list[torch.optim.Optimizer]]:
    """Create optimizer(s) for the given parameters.

    Returns:
        (primary_optimizer, extra_optimizers) — primary is used for DCP
        checkpointing; extras are stepped but not checkpointed.
    """
    params = list(params)
    _log_trainable_param_dtypes(params, cfg)

    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.adamw_betas,
            fused=cfg.adamw_fused,
        )
        opt._base_lr = cfg.learning_rate  # type: ignore[attr-defined]
        return opt, []

    elif cfg.optimizer == "muon":
        params_2d = [p for p in params if p.ndim == 2]
        params_other = [p for p in params if p.ndim != 2]

        if not params_2d:
            raise ValueError("Muon requires at least some 2D parameters, but none were found")

        muon = torch.optim.Muon(
            params_2d,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps,
            adjust_lr_fn=cfg.muon_adjust_lr_fn,
        )
        muon._base_lr = cfg.learning_rate  # type: ignore[attr-defined]

        extras: list[torch.optim.Optimizer] = []
        if params_other:
            fallback_lr = cfg.muon_fallback_lr if cfg.muon_fallback_lr is not None else cfg.learning_rate
            adamw_fallback = torch.optim.AdamW(
                params_other,
                lr=fallback_lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.adamw_betas,
                fused=cfg.adamw_fused,
            )
            adamw_fallback._base_lr = fallback_lr  # type: ignore[attr-defined]
            extras.append(adamw_fallback)
            logger.info(
                "Muon: {} params (2D, lr={}) + AdamW fallback: {} params (non-2D, lr={})",
                len(params_2d),
                cfg.learning_rate,
                len(params_other),
                fallback_lr,
            )
        else:
            logger.info("Muon: {} params (all 2D, no fallback needed)", len(params_2d))

        return muon, extras

    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
