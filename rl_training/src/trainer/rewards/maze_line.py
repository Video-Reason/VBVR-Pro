"""Maze growing-line reward for synthetic maze RL.

This reward matches the ``growing_path_line`` render mode: the solution is a
colored line that progressively traces the path to the goal.  It intentionally
uses a simple, robust signal:

    reward = w_mask * F1(generated_line_mask, gt_line_mask) + w_goal * goal_score

The line masks are soft color-threshold masks against the per-sample path
color stored as ``maze_ball_rgb``.  ``goal_score`` is the strongest generated
line-mask response inside a small window around the goal cell in the final
scored frame.

Expert-parallel GRPO sums low/high reward branches; this model-free reward
emits its full signal only on the low branch and zeros on the high branch.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.maze import _as_batched_tensor
from src.trainer.rewards.registry import register_reward

_REQUIRED_META_KEYS = (
    "maze_ball_rgb",
    "maze_goal",
    "maze_cell_px",
)


def _soft_color_mask(
    pixel_video: torch.Tensor,
    color_rgb: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    """Return a soft mask for pixels close to ``color_rgb``.

    Args:
        pixel_video: ``(B, 3, T, H, W)`` RGB pixels in ``[0, 255]``.
        color_rgb: ``(B, 3)`` target RGB values in ``[0, 255]``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    B = pixel_video.shape[0]
    dist = (pixel_video.float() - color_rgb.view(B, 3, 1, 1, 1).float()).pow(2).sum(dim=1).sqrt()
    return torch.sigmoid((float(threshold) - dist) / float(temperature))


def _soft_f1(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft F1 over spatial dimensions for masks shaped ``(B, K, H, W)``."""
    intersection = (pred_mask * target_mask).sum(dim=(2, 3))
    pred_mass = pred_mask.sum(dim=(2, 3))
    target_mass = target_mask.sum(dim=(2, 3))
    precision = intersection / (pred_mass + eps)
    recall = intersection / (target_mass + eps)
    return (2.0 * precision * recall / (precision + recall + eps)).mean(dim=1)


def _goal_region_score(
    final_mask: torch.Tensor,
    goal_ij: torch.Tensor,
    cell_px: torch.Tensor,
    *,
    goal_cells: float,
) -> torch.Tensor:
    """Max final-frame line-mask response inside a goal-centered window."""
    B, H, W = final_mask.shape
    device = final_mask.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    goal_x = (goal_ij[:, 1].float() + 0.5) * cell_px.float()
    goal_y = (goal_ij[:, 0].float() + 0.5) * cell_px.float()
    radius = float(goal_cells) * cell_px.float().clamp(min=1.0)
    dist2 = (xx.view(1, H, W) - goal_x.view(B, 1, 1)).pow(2) + (yy.view(1, H, W) - goal_y.view(B, 1, 1)).pow(2)
    in_goal = dist2 <= radius.view(B, 1, 1).pow(2)
    return final_mask.masked_fill(~in_goal, 0.0).amax(dim=(1, 2))


@register_reward("maze_line")
class MazeLineReward(BaseReward):
    """Decoded pixel-space reward for growing path-line maze videos."""

    requires_vae: ClassVar[bool] = True

    @torch.no_grad()
    def __call__(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
        meta: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del condition, prompt_embeds, indices

        B = generated_latents.shape[0]
        device = generated_latents.device
        if expert_filter == "high":
            return torch.zeros(B, device=device, dtype=torch.float32)

        self._check_meta(meta)
        assert meta is not None

        line_rgb = _as_batched_tensor(meta["maze_ball_rgb"], device=device).float()
        goal_ij = _as_batched_tensor(meta["maze_goal"], device=device).long()
        cell_px = _as_batched_tensor(meta["maze_cell_px"], device=device).float()

        gen_video = self.trainer.model.decode_latents(generated_latents)
        gt_video = self.trainer.model.decode_latents(gt_video_latents)

        _, _, T, _, _ = gen_video.shape
        K = max(1, min(int(self.cfg.maze_line_reward_num_frames), T))
        frame_idxs = torch.linspace(0, T - 1, K, device=device).round().long()
        gen_video = ((gen_video[:, :, frame_idxs] + 1.0) * 127.5).clamp(0.0, 255.0).float()
        gt_video = ((gt_video[:, :, frame_idxs] + 1.0) * 127.5).clamp(0.0, 255.0).float()

        gen_mask = _soft_color_mask(
            gen_video,
            line_rgb,
            threshold=self.cfg.maze_line_reward_color_threshold,
            temperature=self.cfg.maze_line_reward_color_temperature,
        )
        gt_mask = _soft_color_mask(
            gt_video,
            line_rgb,
            threshold=self.cfg.maze_line_reward_color_threshold,
            temperature=self.cfg.maze_line_reward_color_temperature,
        )

        r_mask = _soft_f1(gen_mask, gt_mask)
        r_goal = _goal_region_score(
            gen_mask[:, -1],
            goal_ij,
            cell_px,
            goal_cells=self.cfg.maze_line_reward_goal_cells,
        )
        reward = self.cfg.maze_line_reward_w_mask * r_mask + self.cfg.maze_line_reward_w_goal * r_goal
        return reward.to(torch.float32)

    @staticmethod
    def _check_meta(meta: dict[str, torch.Tensor] | None) -> None:
        if meta is None:
            raise RuntimeError("MazeLineReward requires per-sample maze metadata tensors in the batch.")
        missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
        if missing:
            raise RuntimeError(f"MazeLineReward missing maze metadata tensor(s): {missing}")
