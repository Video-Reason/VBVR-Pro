import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as st_load

from src.precompute.maze_generator import MazeSpec, build_maze_sample
from src.precompute.maze_webdataset import GenConfig, _write_encoded_samples


class TestMazeWebDataset(unittest.TestCase):
    def test_encoded_sample_uses_single_latent_schema(self):
        rng = np.random.default_rng(123)
        spec = MazeSpec(
            cell_h=6,
            cell_w=6,
            cell_px=8,
            num_frames=9,
            difficulty_names=("easy",),
            render_mode="moving_ball",
            max_generation_attempts=512,
        )
        _video, sample = build_maze_sample(spec, rng, sample_seed=123)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GenConfig(output_dir=tmpdir, model_path="unused", num_samples=1)
            tar_path = Path(tmpdir) / "sample.tar"
            encoded = [
                (
                    torch.ones(2, 1, 1, 1),
                    torch.zeros(1, 1, 1, 1),
                    torch.zeros(3, 4),
                    sample,
                    0,
                )
            ]
            with tarfile.open(tar_path, "w") as tar:
                _write_encoded_samples(cfg, tar, "sft", encoded, split_offset=0, local_start=0)

            with tarfile.open(tar_path, "r") as tar:
                st_member = tar.extractfile("0000000.safetensors")
                json_member = tar.extractfile("0000000.json")
                assert st_member is not None
                assert json_member is not None
                tensors = st_load(st_member.read())
                metadata = json.loads(json_member.read().decode("utf-8"))

        self.assertIn("latents", tensors)
        self.assertFalse(any(key.startswith("latents_") for key in tensors))
        self.assertEqual(metadata["prompt"], sample.prompt)


if __name__ == "__main__":
    unittest.main()
