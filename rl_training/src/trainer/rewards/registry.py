"""Reward-function registry.

Modules that define a reward subclass should decorate it with
``@register_reward("name")`` and be imported from
``src.trainer.rewards.__init__`` so the decorator fires at package load.
The trainer then builds the selected reward by name via ``build_reward``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trainer.base_grpo_trainer import BaseGRPOTrainer
    from src.trainer.config import RLConfig
    from src.trainer.rewards.base import BaseReward


_REGISTRY: dict[str, type[BaseReward]] = {}


def register_reward(name: str):
    """Class decorator: register ``cls`` under ``name``."""

    def _deco(cls: type[BaseReward]) -> type[BaseReward]:
        if name in _REGISTRY:
            raise ValueError(f"Reward '{name}' is already registered (by {_REGISTRY[name].__name__})")
        _REGISTRY[name] = cls
        return cls

    return _deco


def build_reward(name: str, trainer: BaseGRPOTrainer, cfg: RLConfig) -> BaseReward:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown reward '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](trainer, cfg)


def list_rewards() -> list[str]:
    return sorted(_REGISTRY)
