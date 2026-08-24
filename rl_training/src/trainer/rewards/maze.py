"""Maze reward V1 — pixel-space ball detection against a known trajectory.

Signal (per sample, higher = better)::

    r = w_traj · r_traj  +  w_onpath · r_onpath  +  w_goal · r_goal

where, after VAE-decoding the generated latents and selecting ``K`` evenly
spaced frames,

    r_traj   : -mean_k ‖detected_xy[k] - expected_xy[k]‖ / image_diag
               (negative normalised L2 — trajectory match)
    r_onpath : fraction of the K frames where the detected ball sits on a
               passage cell (i.e. not a wall) in the true maze grid
    r_goal   : 1 if the detected end-frame ball lies within
               ``maze_reward_goal_cells`` of the goal centre, else 0

Ball detection: argmin over pixels of squared distance to
``maze_ball_rgb`` in RGB space.  Works reliably on our synthetic mazes
because the palette is fixed per-sample and the background is flat.

GRPO z-scores rewards within each group, so only the *ordering* of rewards
matters — raw scale is not load-bearing.

Legacy expert-parallel reward callers summed the rewards returned on the
``low`` and ``high`` branches, so model-free rewards keep emitting zeros when
``expert_filter == 'high'`` to avoid doubling if that path is revived.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward

_REQUIRED_META_KEYS = (
    "maze_frame_positions_pix",
    "maze_grid",
    "maze_goal",
    "maze_ball_rgb",
    "maze_cell_px",
    "maze_image_hw",
)


def _as_batched_tensor(
    value: torch.Tensor | list[torch.Tensor],
    *,
    device: torch.device,
    pad_value: int | float = 0,
) -> torch.Tensor:
    """Move a collated tensor/list to device, padding variable-shape metadata."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if not value:
        raise RuntimeError("MazeReward received empty metadata list")
    if not all(isinstance(item, torch.Tensor) for item in value):
        raise TypeError(f"MazeReward metadata must be tensors, got {type(value[0]).__name__}")

    first_shape = tuple(value[0].shape)
    if all(tuple(item.shape) == first_shape for item in value):
        return torch.stack(value).to(device)

    ndim = value[0].ndim
    if any(item.ndim != ndim for item in value):
        shapes = [tuple(item.shape) for item in value]
        raise RuntimeError(f"MazeReward cannot pad metadata tensors with different ranks: {shapes}")

    max_shape = tuple(max(item.shape[dim] for item in value) for dim in range(ndim))
    out = value[0].new_full((len(value), *max_shape), pad_value)
    for idx, item in enumerate(value):
        slices = (idx, *[slice(0, size) for size in item.shape])
        out[slices] = item
    return out.to(device)


@register_reward("maze")
class MazeReward(BaseReward):
    """Pixel-space maze reward (V1): trajectory + on-path + goal components."""

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
        del gt_video_latents, condition, prompt_embeds, indices

        B = generated_latents.shape[0]
        device = generated_latents.device

        # EP double-count guard — trainer sums low+high in expert-parallel
        # mode, so a model-free reward must stay silent on one branch.
        if expert_filter == "high":
            return torch.zeros(B, device=device, dtype=torch.float32)

        self._check_meta(meta)
        assert meta is not None  # narrow for type checker

        fp_pix = _as_batched_tensor(meta["maze_frame_positions_pix"], device=device).float()  # (B, T, 2) — (x, y)
        grid = _as_batched_tensor(meta["maze_grid"], device=device, pad_value=1).long()  # (B, Hg, Wg)
        goal_ij = _as_batched_tensor(meta["maze_goal"], device=device).long()  # (B, 2) — (i, j)
        ball_rgb = _as_batched_tensor(meta["maze_ball_rgb"], device=device).float()  # (B, 3) in [0, 255]
        cell_px = _as_batched_tensor(meta["maze_cell_px"], device=device).long()  # (B,)
        image_hw = _as_batched_tensor(meta["maze_image_hw"], device=device).float()  # (B, 2) — (H, W)

        # ---- 1. Decode latents to pixel space ----
        pixel_video = self.trainer.model.decode_latents(generated_latents)
        pixel_video = ((pixel_video + 1.0) * 127.5).clamp(0.0, 255.0).float()
        # pixel_video shape: (B, 3, T, H, W)
        _, _, T, H, W = pixel_video.shape

        # ---- 2. Pick K frame indices evenly spaced over [0, T-1] ----
        K = min(int(self.cfg.maze_reward_num_frames), T)
        if K < 2:
            K = max(1, T)
        frame_idxs = torch.linspace(0, T - 1, K, device=device).round().long()

        # ---- 3. Locate ball in each selected frame (argmin of RGB L2) ----
        selected = pixel_video[:, :, frame_idxs]  # (B, 3, K, H, W)
        diff = (selected - ball_rgb.view(B, 3, 1, 1, 1)).pow(2).sum(dim=1)  # (B, K, H, W)
        flat = diff.view(B, K, H * W)
        best = flat.argmin(dim=-1)  # (B, K)
        det_y = (best // W).float()  # (B, K) pixel row
        det_x = (best % W).float()  # (B, K) pixel col

        # ---- 4. r_traj: normalised pixel distance to expected position ----
        expected_xy = fp_pix.index_select(1, frame_idxs)  # (B, K, 2) — (x, y)
        dx = det_x - expected_xy[..., 0]
        dy = det_y - expected_xy[..., 1]
        traj_dist = (dx * dx + dy * dy).sqrt()  # (B, K) pixels
        diag = (image_hw.pow(2).sum(dim=1) + 1e-8).sqrt()  # (B,)
        r_traj = -traj_dist.mean(dim=1) / diag  # (B,)

        # ---- 5. r_onpath: fraction of K frames landing on a passage cell ----
        cell_px_f = cell_px.float().view(B, 1).clamp(min=1.0)
        cell_i = (det_y / cell_px_f).long().clamp(0, grid.shape[1] - 1)  # (B, K) row cell
        cell_j = (det_x / cell_px_f).long().clamp(0, grid.shape[2] - 1)  # (B, K) col cell
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)
        cell_val = grid[batch_idx, cell_i, cell_j]  # (B, K)
        r_onpath = (cell_val == 0).float().mean(dim=1)  # (B,)

        # ---- 6. r_goal: end-frame ball within goal-tolerance of goal centre ----
        goal_x = (goal_ij[:, 1].float() + 0.5) * cell_px.float()  # (B,)
        goal_y = (goal_ij[:, 0].float() + 0.5) * cell_px.float()  # (B,)
        end_dx = det_x[:, -1] - goal_x
        end_dy = det_y[:, -1] - goal_y
        end_dist = (end_dx * end_dx + end_dy * end_dy).sqrt()  # (B,) pixels
        r_goal = (end_dist < self.cfg.maze_reward_goal_cells * cell_px.float()).float()

        reward = (
            self.cfg.maze_reward_w_traj * r_traj
            + self.cfg.maze_reward_w_onpath * r_onpath
            + self.cfg.maze_reward_w_goal * r_goal
        )
        return reward.to(torch.float32)

    @staticmethod
    def _check_meta(meta: dict[str, torch.Tensor] | None) -> None:
        if meta is None:
            raise RuntimeError(
                "MazeReward requires per-sample 'maze_*' tensors in the batch. "
                "Generate data with src/precompute/maze_webdataset.py so that "
                "VBVRLatentDataset can surface them as batch metadata."
            )
        missing = [k for k in _REQUIRED_META_KEYS if k not in meta]
        if missing:
            raise RuntimeError(
                f"MazeReward missing maze metadata tensor(s): {missing}. "
                "Re-run scripts/precompute/maze_webdataset.fish with the latest "
                "src/precompute/maze_webdataset.py (it now ships maze_cell_px "
                "and maze_image_hw alongside the other reward tensors)."
            )
