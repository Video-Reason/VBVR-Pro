"""Reward-function package for Flow-GRPO training.

To add a new reward:
    1. Write a ``BaseReward`` subclass in a new module under this package and
       decorate it with ``@register_reward("your_name")``.
    2. Import the module below so the decorator fires at package-import time.
    3. Set ``grpo_reward_fn: your_name`` in the training YAML.
"""

from src.trainer.rewards import maze as _maze  # noqa: F401  — register side-effect
from src.trainer.rewards import maze_line as _maze_line  # noqa: F401  — register side-effect
from src.trainer.rewards import maze_tracker as _maze_tracker  # noqa: F401  — register side-effect
from src.trainer.rewards import neg_loss as _neg_loss  # noqa: F401  — register side-effect
from src.trainer.rewards import vbvr_rule as _vbvr_rule  # noqa: F401  — register side-effect
from src.trainer.rewards import vbvr_vlm as _vbvr_vlm  # noqa: F401  — register side-effect
from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import build_reward, list_rewards, register_reward

__all__ = ["BaseReward", "build_reward", "list_rewards", "register_reward"]
