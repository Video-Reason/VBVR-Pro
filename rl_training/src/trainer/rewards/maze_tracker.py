"""Maze tracker reward matching src.eval.maze_tracker_score overall."""

from __future__ import annotations

from typing import ClassVar

import torch

from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.maze import _as_batched_tensor
from src.trainer.rewards.registry import register_reward

_REQUIRED_META_KEYS = (
    "maze_frame_positions_pix",
    "maze_grid",
    "maze_goal",
    "maze_ball_rgb",
    "maze_cell_px",
    "maze_path",
    "maze_path_len",
)


def _weighted_best_xy(dist: torch.Tensor, *, color_slack: float, x_offset: int = 0, y_offset: int = 0):
    min_val = dist.min()
    mask = dist <= min_val + float(color_slack) * float(color_slack)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    weights = 1.0 / (dist[ys, xs] + 1.0)
    weight_sum = weights.sum().clamp(min=1e-8)
    x = ((xs.float() + float(x_offset)) * weights).sum() / weight_sum
    y = ((ys.float() + float(y_offset)) * weights).sum() / weight_sum
    return torch.stack((x, y)), min_val


def _track_color_object(
    pixel_video: torch.Tensor,
    ball_rgb: torch.Tensor,
    initial_xy: torch.Tensor,
    *,
    search_radius: int,
    color_slack: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch port of src.eval.maze_tracker_score._track_ball."""
    B, _, T, H, W = pixel_video.shape
    positions = pixel_video.new_empty((B, T, 2))
    confidences = pixel_video.new_empty((B, T))
    prev_xy = initial_xy.float()
    slack = float(color_slack)

    for t in range(T):
        frame = pixel_video[:, :, t]
        dist = (frame - ball_rgb.view(B, 3, 1, 1).float()).pow(2).sum(dim=1)
        next_prev = []
        for b in range(B):
            global_xy, global_min = _weighted_best_xy(dist[b], color_slack=slack)
            px, py = prev_xy[b]
            x0 = max(0, int(round(float(px.item()))) - int(search_radius))
            x1 = min(W, int(round(float(px.item()))) + int(search_radius) + 1)
            y0 = max(0, int(round(float(py.item()))) - int(search_radius))
            y1 = min(H, int(round(float(py.item()))) + int(search_radius) + 1)

            xy = global_xy
            min_val = global_min
            if x1 > x0 and y1 > y0:
                local_xy, local_min = _weighted_best_xy(
                    dist[b, y0:y1, x0:x1],
                    color_slack=slack,
                    x_offset=x0,
                    y_offset=y0,
                )
                local_limit = torch.maximum(global_min.sqrt() + slack, global_min.new_tensor(slack * 3.0))
                if local_min.sqrt() <= local_limit:
                    xy = local_xy
                    min_val = local_min

            positions[b, t] = xy
            color_error = min_val.sqrt()
            confidences[b, t] = torch.exp(-color_error / 80.0)
            next_prev.append(xy)
        prev_xy = torch.stack(next_prev, dim=0)

    return positions, confidences


def _path_progress_score(
    det_xy: torch.Tensor,
    path_ij: torch.Tensor,
    path_len: torch.Tensor,
    cell_px: torch.Tensor,
) -> torch.Tensor:
    B, K, _ = det_xy.shape
    _, P, _ = path_ij.shape
    path_len = path_len.long().clamp(min=1, max=P)
    cell_px_f = cell_px.float().view(B, 1)
    path_x = (path_ij[..., 1].float() + 0.5) * cell_px_f
    path_y = (path_ij[..., 0].float() + 0.5) * cell_px_f
    path_xy = torch.stack((path_x, path_y), dim=-1)

    dists = torch.cdist(det_xy.float(), path_xy.float())
    valid = torch.arange(P, device=det_xy.device).view(1, 1, P) < path_len.view(B, 1, 1)
    dists = dists.masked_fill(~valid, float("inf"))
    nearest = dists.argmin(dim=-1).float()
    if K > 1:
        monotonic_fraction = (nearest[:, 1:] - nearest[:, :-1] >= -2).float().mean(dim=1)
    else:
        monotonic_fraction = torch.ones(B, device=det_xy.device, dtype=det_xy.dtype)
    progress = ((nearest[:, -1] - nearest[:, 0]) / (path_len.float() - 1.0).clamp(min=1.0)).clamp(0.0, 1.0)
    return progress * monotonic_fraction


@register_reward("maze_tracker")
class MazeTrackerReward(BaseReward):
    """Pixel-space maze reward using the same overall formula as maze_tracker_score.py."""

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
        if expert_filter == "high":
            return torch.zeros(B, device=device, dtype=torch.float32)

        self._check_meta(meta)
        assert meta is not None

        fp_pix = _as_batched_tensor(meta["maze_frame_positions_pix"], device=device).float()
        grid = _as_batched_tensor(meta["maze_grid"], device=device, pad_value=1).long()
        goal_ij = _as_batched_tensor(meta["maze_goal"], device=device).long()
        ball_rgb = _as_batched_tensor(meta["maze_ball_rgb"], device=device).float()
        cell_px = _as_batched_tensor(meta["maze_cell_px"], device=device).float()
        path_ij = _as_batched_tensor(meta["maze_path"], device=device, pad_value=0).long()
        path_len = _as_batched_tensor(meta["maze_path_len"], device=device).long()

        pixel_video = self.trainer.model.decode_latents(generated_latents)
        pixel_video = ((pixel_video + 1.0) * 127.5).clamp(0.0, 255.0).float()
        _, _, T, _, _ = pixel_video.shape
        K = min(max(1, int(self.cfg.maze_tracker_reward_num_frames)), T)
        frame_idxs = torch.linspace(0, T - 1, K, device=device).round().long()

        expected_source_idxs = torch.linspace(0, fp_pix.shape[1] - 1, T, device=device).round().long()
        expected_video_xy = fp_pix.index_select(1, expected_source_idxs)
        expected_xy = expected_video_xy.index_select(1, frame_idxs)

        det_all, confidence_all = _track_color_object(
            pixel_video,
            ball_rgb,
            expected_video_xy[:, 0],
            search_radius=int(self.cfg.maze_tracker_reward_search_radius),
            color_slack=float(self.cfg.maze_tracker_reward_color_slack),
        )
        det_xy = det_all.index_select(1, frame_idxs)
        confidence = confidence_all.index_select(1, frame_idxs)
        del confidence

        dists = (det_xy - expected_xy).pow(2).sum(dim=-1).sqrt()
        mean_error_px = dists.mean(dim=1)
        max_mean_error_px = float(self.cfg.maze_tracker_reward_max_mean_error_cells) * cell_px.clamp(min=1.0)
        traj_score = (1.0 - mean_error_px / max_mean_error_px).clamp(min=0.0)

        cell_px_b = cell_px.view(B, 1).clamp(min=1.0)
        cell_i = (det_xy[..., 1] / cell_px_b).long().clamp(0, grid.shape[1] - 1)
        cell_j = (det_xy[..., 0] / cell_px_b).long().clamp(0, grid.shape[2] - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)
        on_path_fraction = (grid[batch_idx, cell_i, cell_j] == 0).float().mean(dim=1)

        goal_x = (goal_ij[:, 1].float() + 0.5) * cell_px
        goal_y = (goal_ij[:, 0].float() + 0.5) * cell_px
        end_error_px = ((det_xy[:, -1, 0] - goal_x).pow(2) + (det_xy[:, -1, 1] - goal_y).pow(2)).sqrt()
        goal_score = (end_error_px <= float(self.cfg.maze_tracker_reward_goal_tolerance_cells) * cell_px).float()

        progress_score = _path_progress_score(det_xy, path_ij, path_len, cell_px)
        overall = (
            self.cfg.maze_tracker_reward_w_traj * traj_score
            + self.cfg.maze_tracker_reward_w_onpath * on_path_fraction
            + self.cfg.maze_tracker_reward_w_goal * goal_score
            + self.cfg.maze_tracker_reward_w_progress * progress_score
        )
        return overall.to(torch.float32)

    @staticmethod
    def _check_meta(meta: dict[str, torch.Tensor] | None) -> None:
        if meta is None:
            raise RuntimeError("MazeTrackerReward requires per-sample maze metadata tensors in the batch.")
        missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
        if missing:
            raise RuntimeError(f"MazeTrackerReward missing maze metadata tensor(s): {missing}")
