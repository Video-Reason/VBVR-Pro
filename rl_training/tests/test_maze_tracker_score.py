import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

from src.eval.maze_tracker_score import _score_one
from src.precompute.maze_generator import (
    MazeSpec,
    build_line_waypoint_from_sample,
    build_maze_sample,
)
from src.precompute.maze_webdataset import _sample_to_json_blob


def _score_args() -> argparse.Namespace:
    return argparse.Namespace(
        num_frames=21,
        search_radius=96,
        color_slack=28.0,
        goal_tolerance_cells=1.0,
        max_mean_error_cells=4.0,
        min_mean_tracker_confidence=0.30,
        min_ball_like_fraction=0.80,
        min_ball_component_area_multiplier=0.10,
        max_ball_component_area_multiplier=6.0,
        max_ball_component_axis_cells=2.0,
        changed_pixel_threshold=40.0,
        max_changed_area_multiplier=12.0,
        expected_guided_tracking=False,
    )


def _write_video(path: Path, video: np.ndarray) -> None:
    export_to_video([Image.fromarray(frame) for frame in video], str(path), fps=16)


class TestMazeTrackerScore(unittest.TestCase):
    def test_path_line_is_not_scored_as_moving_ball(self):
        rng = np.random.default_rng(321)
        spec = MazeSpec(
            cell_h=6,
            cell_w=6,
            cell_px=12,
            num_frames=41,
            difficulty_names=("easy",),
            render_mode="moving_ball",
            max_generation_attempts=512,
        )
        ball_video, ball_sample = build_maze_sample(spec, rng, sample_seed=321)
        line_video, _ = build_line_waypoint_from_sample(ball_sample, completion_fraction=1.0)
        row = {
            "id": "sample",
            "difficulty": "easy",
            "maze": _sample_to_json_blob(ball_sample, fps=16),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ball_path = tmp / "ball.mp4"
            line_path = tmp / "line.mp4"
            _write_video(ball_path, ball_video)
            _write_video(line_path, line_video)

            ball_score = _score_one(_score_args(), row, ball_path)
            line_score = _score_one(_score_args(), row, line_path)

        self.assertTrue(ball_score["passed"])
        self.assertGreater(ball_score["overall"], 0.95)
        self.assertGreaterEqual(ball_score["ball_like_fraction"], 0.95)

        self.assertFalse(line_score["passed"])
        self.assertLess(line_score["ball_like_fraction"], 0.80)
        self.assertGreater(line_score["max_changed_area_ratio"], 12.0)
        self.assertLess(line_score["overall"], line_score["overall_raw"])


if __name__ == "__main__":
    unittest.main()
