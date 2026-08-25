"""Tracker-style rule scoring for generated maze videos.

Scores whether the ball follows the known maze path, stays on passages, and
reaches the goal.  The eval JSON should be produced by
``src.precompute.maze_evalset`` and contain the ``maze`` metadata blob.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score maze videos with a color/continuity tracker")
    p.add_argument("--eval_json", required=True, nargs="+", help="Eval JSON file(s)")
    p.add_argument(
        "--predictions_dir",
        default=None,
        help="Generated-video root. Defaults to scoring each row's reference_video.",
    )
    p.add_argument(
        "--prediction_pattern",
        default=None,
        help="Optional format string relative to predictions_dir, e.g. '{index:06d}/step_49.mp4'.",
    )
    p.add_argument("--video_field", default="reference_video", help="JSON field used when predictions_dir is absent")
    p.add_argument("--output_json", default="maze_tracker_scores.json")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--num_frames", type=int, default=21, help="Frames to score uniformly over the video")
    p.add_argument("--search_radius", type=int, default=96, help="Tracker local-search radius in pixels")
    p.add_argument("--color_slack", type=float, default=28.0, help="RGB L2 slack around the best color match")
    p.add_argument("--goal_tolerance_cells", type=float, default=1.0)
    p.add_argument("--max_mean_error_cells", type=float, default=4.0)
    p.add_argument("--min_mean_tracker_confidence", type=float, default=0.30)
    p.add_argument(
        "--min_ball_like_fraction",
        type=float,
        default=0.80,
        help="Minimum fraction of scored frames whose tracked foreground is compact enough to be a ball.",
    )
    p.add_argument(
        "--min_ball_component_area_multiplier",
        type=float,
        default=0.10,
        help="Minimum tracked color-component area as a multiple of the expected rendered ball area.",
    )
    p.add_argument(
        "--max_ball_component_area_multiplier",
        type=float,
        default=6.0,
        help="Maximum tracked color-component area as a multiple of the expected rendered ball area.",
    )
    p.add_argument(
        "--max_ball_component_axis_cells",
        type=float,
        default=2.0,
        help="Maximum tracked color-component width/height in maze-cell units.",
    )
    p.add_argument(
        "--changed_pixel_threshold",
        type=float,
        default=40.0,
        help="RGB L2 threshold used to measure how much each frame differs from the first generated frame.",
    )
    p.add_argument(
        "--max_changed_area_multiplier",
        type=float,
        default=12.0,
        help="Maximum changed-pixel area as a multiple of the expected rendered ball area.",
    )
    p.add_argument(
        "--expected_guided_tracking",
        action="store_true",
        help="Debug mode: search near the expected trajectory instead of only using temporal continuity.",
    )
    return p.parse_args()


def _read_rows(paths: list[str]) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    bases: list[Path] = []
    for p in paths:
        path = Path(p)
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        rows.extend(data)
        bases.extend([path.parent] * len(data))
    return rows, bases


def _resolve_video_path(args: argparse.Namespace, row: dict, base_dir: Path, index: int) -> Path:
    if args.predictions_dir is None:
        field = row.get(args.video_field)
        if not field:
            raise FileNotFoundError(f"row {index} has no video field '{args.video_field}'")
        p = Path(field)
        return p if p.is_absolute() else base_dir / p

    root = Path(args.predictions_dir)
    if args.prediction_pattern:
        rel = args.prediction_pattern.format(
            index=index,
            id=row.get("id", f"{index:06d}"),
            difficulty=row.get("difficulty", ""),
        )
        return root / rel

    sample_dir = root / f"{index:06d}"
    if sample_dir.is_dir():
        step_videos = sorted(
            sample_dir.glob("step_*.mp4"),
            key=lambda p: int(re.search(r"step_(\d+)", p.stem).group(1)) if re.search(r"step_(\d+)", p.stem) else -1,
        )
        if step_videos:
            return step_videos[-1]

    row_id = row.get("id", f"{index:06d}")
    candidates = [
        root / f"{row_id}.mp4",
        root / f"{index:06d}.mp4",
        sample_dir / "output.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_video(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    video = iio.imread(path)
    if video.ndim != 4:
        raise ValueError(f"Expected video array (T,H,W,C), got {video.shape} from {path}")
    if video.shape[-1] > 3:
        video = video[..., :3]
    return video.astype(np.float32)


def _cell_center_xy(cell_ij: tuple[float, float], cell_px: float) -> tuple[float, float]:
    return ((cell_ij[1] + 0.5) * cell_px, (cell_ij[0] + 0.5) * cell_px)


def _track_ball(
    frames: np.ndarray,
    ball_rgb: np.ndarray,
    *,
    initial_xy: np.ndarray | None,
    priors_xy: np.ndarray | None = None,
    search_radius: int,
    color_slack: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions: list[tuple[float, float]] = []
    confidences: list[float] = []
    color_errors: list[float] = []
    prev_xy: tuple[float, float] | None = None
    slack_sq = color_slack * color_slack

    def best_xy(search: np.ndarray, x_off: int, y_off: int) -> tuple[float, float, float]:
        min_val = float(search.min())
        mask = search <= min_val + slack_sq
        if mask.sum() > 0:
            ys, xs = np.nonzero(mask)
            weights = 1.0 / (search[ys, xs] + 1.0)
            x = float(np.average(xs + x_off, weights=weights))
            y = float(np.average(ys + y_off, weights=weights))
            return x, y, min_val

        flat_idx = int(search.argmin())
        y_local, x_local = divmod(flat_idx, search.shape[1])
        return float(x_local + x_off), float(y_local + y_off), min_val

    for frame_idx, frame in enumerate(frames):
        dist = np.sum((frame - ball_rgb.reshape(1, 1, 3)) ** 2, axis=-1)
        if priors_xy is not None:
            px, py = priors_xy[frame_idx]
        elif prev_xy is not None:
            px, py = prev_xy
        elif initial_xy is not None:
            px, py = initial_xy
        else:
            px = py = None

        x, y, min_val = best_xy(dist, 0, 0)
        if px is not None and py is not None:
            x0 = max(0, int(round(px)) - search_radius)
            x1 = min(dist.shape[1], int(round(px)) + search_radius + 1)
            y0 = max(0, int(round(py)) - search_radius)
            y1 = min(dist.shape[0], int(round(py)) + search_radius + 1)
            local = dist[y0:y1, x0:x1]
            if local.size:
                lx, ly, local_min = best_xy(local, x0, y0)
                # Prefer temporal continuity when the local color match is
                # plausible. Fall back to the global best if the tracked object
                # appears to have left the search radius.
                if math.sqrt(local_min) <= max(math.sqrt(min_val) + color_slack, color_slack * 3.0):
                    x, y, min_val = lx, ly, local_min

        prev_xy = (x, y)
        positions.append((x, y))
        color_error = float(math.sqrt(min_val))
        color_errors.append(color_error)
        confidences.append(float(math.exp(-color_error / 80.0)))

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(confidences, dtype=np.float32),
        np.asarray(color_errors, dtype=np.float32),
    )


def _nearest_mask_pixel(mask: np.ndarray, xy: np.ndarray) -> tuple[int, int] | None:
    h, w = mask.shape
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if 0 <= x < w and 0 <= y < h and bool(mask[y, x]):
        return y, x

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    d2 = (xs.astype(np.float32) - float(xy[0])) ** 2 + (ys.astype(np.float32) - float(xy[1])) ** 2
    idx = int(d2.argmin())
    return int(ys[idx]), int(xs[idx])


def _component_stats_at_xy(mask: np.ndarray, xy: np.ndarray) -> tuple[int, int, int] | None:
    """Return area, width, height for the mask component nearest ``xy``."""
    seed = _nearest_mask_pixel(mask, xy)
    if seed is None:
        return None

    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    stack = [seed]
    visited[seed] = True
    area = 0
    min_y = max_y = seed[0]
    min_x = max_x = seed[1]

    while stack:
        y, x = stack.pop()
        area += 1
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        for yy in range(max(0, y - 1), min(h, y + 2)):
            for xx in range(max(0, x - 1), min(w, x + 2)):
                if yy == y and xx == x:
                    continue
                if not visited[yy, xx] and mask[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))

    return area, max_x - min_x + 1, max_y - min_y + 1


def _ball_component_metrics(
    args: argparse.Namespace,
    frames: np.ndarray,
    det_xy: np.ndarray,
    ball_rgb: np.ndarray,
    color_error: np.ndarray,
    cell_px: float,
) -> dict:
    expected_area = math.pi * (cell_px / 3.0) ** 2
    area_ratios: list[float] = []
    axis_cells: list[float] = []
    ball_like: list[bool] = []

    for frame, xy, err in zip(frames, det_xy, color_error, strict=True):
        dist = np.sum((frame - ball_rgb.reshape(1, 1, 3)) ** 2, axis=-1)
        mask = dist <= float(err * err + args.color_slack * args.color_slack)
        stats = _component_stats_at_xy(mask, xy)
        if stats is None:
            area_ratio = 0.0
            max_axis_cells = float("inf")
        else:
            area, width, height = stats
            area_ratio = float(area / max(expected_area, 1e-6))
            max_axis_cells = float(max(width, height) / max(cell_px, 1e-6))

        area_ratios.append(area_ratio)
        axis_cells.append(max_axis_cells)
        ball_like.append(
            args.min_ball_component_area_multiplier <= area_ratio <= args.max_ball_component_area_multiplier
            and max_axis_cells <= args.max_ball_component_axis_cells
        )

    finite_axes = [x for x in axis_cells if math.isfinite(x)]
    return {
        "ball_like_fraction": float(np.mean(ball_like)) if ball_like else 0.0,
        "mean_ball_component_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "max_ball_component_area_ratio": float(np.max(area_ratios)) if area_ratios else 0.0,
        "max_ball_component_axis_cells": float(np.max(finite_axes)) if finite_axes else float("inf"),
    }


def _changed_area_metrics(args: argparse.Namespace, frames: np.ndarray, cell_px: float) -> dict:
    expected_area = math.pi * (cell_px / 3.0) ** 2
    diff = np.linalg.norm(frames - frames[0:1], axis=-1)
    changed_ratios = (diff > args.changed_pixel_threshold).sum(axis=(1, 2)) / max(expected_area, 1e-6)
    max_changed = float(np.max(changed_ratios)) if len(changed_ratios) else 0.0
    compact_motion_score = min(1.0, args.max_changed_area_multiplier / max(max_changed, 1e-6))
    return {
        "compact_motion_score": compact_motion_score,
        "mean_changed_area_ratio": float(np.mean(changed_ratios)) if len(changed_ratios) else 0.0,
        "max_changed_area_ratio": max_changed,
    }


def _score_one(args: argparse.Namespace, row: dict, video_path: Path) -> dict:
    maze = row["maze"]
    video = _load_video(video_path)
    frame_idxs = np.linspace(0, video.shape[0] - 1, min(args.num_frames, video.shape[0])).round().astype(np.int64)

    expected_all = np.asarray(maze["frame_positions_pix"], dtype=np.float32)
    video_expected_idxs = np.linspace(0, expected_all.shape[0] - 1, video.shape[0]).round().astype(np.int64)
    expected_video_xy = expected_all[video_expected_idxs]
    expected_xy = expected_video_xy[frame_idxs]

    palette = maze["palette"]
    ball_rgb = np.asarray(palette["ball_rgb"], dtype=np.float32)
    priors_xy = expected_video_xy if args.expected_guided_tracking else None
    det_all, confidence_all, color_error_all = _track_ball(
        video,
        ball_rgb,
        initial_xy=expected_video_xy[0],
        priors_xy=priors_xy,
        search_radius=args.search_radius,
        color_slack=args.color_slack,
    )
    det_xy = det_all[frame_idxs]
    confidence = confidence_all[frame_idxs]
    color_error = color_error_all[frame_idxs]

    grid = np.asarray(maze["grid"], dtype=np.int64)
    cell_px = float(maze["cell_px"])
    dists = np.linalg.norm(det_xy - expected_xy, axis=1)
    mean_error_px = float(dists.mean())
    max_error_px = float(dists.max())

    cell_i = np.clip((det_xy[:, 1] / cell_px).astype(np.int64), 0, grid.shape[0] - 1)
    cell_j = np.clip((det_xy[:, 0] / cell_px).astype(np.int64), 0, grid.shape[1] - 1)
    on_path_fraction = float((grid[cell_i, cell_j] == 0).mean())

    goal_ij = tuple(maze["goal"])
    goal_xy = np.asarray(_cell_center_xy(goal_ij, cell_px), dtype=np.float32)
    end_error_px = float(np.linalg.norm(det_xy[-1] - goal_xy))
    goal_success = bool(end_error_px <= args.goal_tolerance_cells * cell_px)

    path = np.asarray([_cell_center_xy(tuple(p), cell_px) for p in maze["path"]], dtype=np.float32)
    nearest_path_idx = np.linalg.norm(det_xy[:, None, :] - path[None, :, :], axis=-1).argmin(axis=1)
    if len(nearest_path_idx) > 1:
        diffs = np.diff(nearest_path_idx)
        monotonic_fraction = float((diffs >= -2).mean())
    else:
        monotonic_fraction = 1.0
    progress = float((nearest_path_idx[-1] - nearest_path_idx[0]) / max(1, len(path) - 1))
    progress = max(0.0, min(1.0, progress))

    traj_score = max(0.0, 1.0 - mean_error_px / (args.max_mean_error_cells * cell_px))
    goal_score = 1.0 if goal_success else 0.0
    progress_score = progress * monotonic_fraction
    overall_raw = 0.35 * traj_score + 0.25 * on_path_fraction + 0.25 * goal_score + 0.15 * progress_score
    ball_metrics = _ball_component_metrics(args, video[frame_idxs], det_xy, ball_rgb, color_error, cell_px)
    changed_metrics = _changed_area_metrics(args, video[frame_idxs], cell_px)
    ball_like_fraction = ball_metrics["ball_like_fraction"]
    compact_motion_score = changed_metrics["compact_motion_score"]
    overall = overall_raw * ball_like_fraction * compact_motion_score
    mean_tracker_confidence = float(confidence.mean())
    passed = bool(
        goal_success
        and on_path_fraction >= 0.85
        and progress >= 0.85
        and mean_error_px <= args.max_mean_error_cells * cell_px
        and mean_tracker_confidence >= args.min_mean_tracker_confidence
        and ball_like_fraction >= args.min_ball_like_fraction
        and changed_metrics["max_changed_area_ratio"] <= args.max_changed_area_multiplier
    )

    return {
        "id": row.get("id"),
        "difficulty": row.get("difficulty", maze.get("difficulty")),
        "video": str(video_path),
        "passed": passed,
        "overall": overall,
        "overall_raw": overall_raw,
        "traj_score": traj_score,
        "on_path_fraction": on_path_fraction,
        "goal_success": goal_success,
        "progress": progress,
        "monotonic_fraction": monotonic_fraction,
        **ball_metrics,
        **changed_metrics,
        "mean_error_px": mean_error_px,
        "max_error_px": max_error_px,
        "end_error_px": end_error_px,
        "mean_tracker_confidence": mean_tracker_confidence,
        "min_tracker_confidence": float(confidence.min()),
        "mean_color_error": float(color_error.mean()),
        "max_color_error": float(color_error.max()),
        "num_scored_frames": int(len(frame_idxs)),
    }


def main() -> None:
    args = parse_args()
    rows, bases = _read_rows(args.eval_json)
    results: list[dict] = []
    failures = 0

    for idx, (row, base_dir) in enumerate(tqdm(list(zip(rows, bases, strict=True)), desc="Scoring maze videos")):
        try:
            video_path = _resolve_video_path(args, row, base_dir, idx)
            results.append(_score_one(args, row, video_path))
        except Exception as exc:
            failures += 1
            results.append(
                {
                    "id": row.get("id"),
                    "difficulty": row.get("difficulty"),
                    "video": None,
                    "passed": False,
                    "error": repr(exc),
                }
            )

    by_diff: dict[str, list[dict]] = {}
    for r in results:
        by_diff.setdefault(str(r.get("difficulty")), []).append(r)

    summary_by_diff = {}
    for diff, group in by_diff.items():
        valid = [g for g in group if "overall" in g]
        summary_by_diff[diff] = {
            "count": len(group),
            "valid": len(valid),
            "pass_rate": sum(bool(g.get("passed")) for g in group) / max(1, len(group)),
            "overall_mean": float(np.mean([g["overall"] for g in valid])) if valid else 0.0,
            "goal_success_rate": float(np.mean([g["goal_success"] for g in valid])) if valid else 0.0,
            "on_path_mean": float(np.mean([g["on_path_fraction"] for g in valid])) if valid else 0.0,
        }

    valid_all = [r for r in results if "overall" in r]
    summary = {
        "count": len(results),
        "valid": len(valid_all),
        "failures": failures,
        "pass_rate": sum(bool(r.get("passed")) for r in results) / max(1, len(results)),
        "overall_mean": float(np.mean([r["overall"] for r in valid_all])) if valid_all else 0.0,
        "by_difficulty": summary_by_diff,
    }
    payload = {"summary": summary, "results": results}

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    with open(output_csv, "w", newline="") as f:
        fieldnames = sorted({k for r in results for k in r})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
