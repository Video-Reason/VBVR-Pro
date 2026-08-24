"""Wan2.2 I2V evaluation with deterministic rectified-flow Euler steps.

This keeps the batching, distributed execution, media validation, and output
contract from :mod:`src.cli.eval_i2v`, but replaces the model's configured
UniPC scheduler with Diffusers' ``FlowMatchEulerDiscreteScheduler``.
"""

from __future__ import annotations

from typing import Any

import torch

from src.cli import eval_i2v as _base

_BASE_LOAD_PIPELINE = _base._load_pipeline


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def install_flowmatch_euler_scheduler(pipe: Any) -> float:
    """Replace ``pipe.scheduler`` and return the preserved flow shift."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    config = pipe.scheduler.config
    flow_shift = float(_config_value(config, "flow_shift", _config_value(config, "shift", 1.0)))
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(_config_value(config, "num_train_timesteps", 1000)),
        shift=flow_shift,
        use_dynamic_shifting=bool(_config_value(config, "use_dynamic_shifting", False)),
        base_shift=_config_value(config, "base_shift", 0.5),
        max_shift=_config_value(config, "max_shift", 1.15),
        base_image_seq_len=int(_config_value(config, "base_image_seq_len", 256)),
        max_image_seq_len=int(_config_value(config, "max_image_seq_len", 4096)),
        invert_sigmas=bool(_config_value(config, "invert_sigmas", False)),
        shift_terminal=_config_value(config, "shift_terminal"),
        use_karras_sigmas=bool(_config_value(config, "use_karras_sigmas", False)),
        use_exponential_sigmas=bool(_config_value(config, "use_exponential_sigmas", False)),
        use_beta_sigmas=bool(_config_value(config, "use_beta_sigmas", False)),
        time_shift_type=str(_config_value(config, "time_shift_type", "exponential")),
        stochastic_sampling=False,
    )
    return flow_shift


def _load_pipeline(args: Any, device: torch.device, rank: int):
    pipe = _BASE_LOAD_PIPELINE(args, device, rank)
    flow_shift = install_flowmatch_euler_scheduler(pipe)
    if rank == 0:
        print(
            f"Using FlowMatchEulerDiscreteScheduler (deterministic first-order Euler, flow shift={flow_shift:g}).",
            flush=True,
        )
    return pipe


def main(argv: list[str] | None = None) -> int:
    original_loader = _base._load_pipeline
    _base._load_pipeline = _load_pipeline
    try:
        return _base.main(argv)
    finally:
        _base._load_pipeline = original_loader


if __name__ == "__main__":
    raise SystemExit(main())
