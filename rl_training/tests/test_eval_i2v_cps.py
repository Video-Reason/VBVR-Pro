from pathlib import Path

import pytest
import torch
from PIL import Image

from src.cli.eval_i2v_cps import _inference_config, _item_seed, _prepare_input, parse_args


def _args(tmp_path: Path, noise_level: str = "0.7"):
    return parse_args(
        [
            "--eval_json",
            str(tmp_path / "eval.json"),
            "--model_path",
            str(tmp_path / "model"),
            "--output_dir",
            str(tmp_path / "output"),
            "--noise_level",
            noise_level,
            "--height",
            "256",
            "--width",
            "256",
        ]
    )


@pytest.mark.parametrize("level", ["0", "0.1", "0.3", "0.7", "0.9", "1"])
def test_parse_args_accepts_cps_noise_range(tmp_path: Path, level: str) -> None:
    assert _args(tmp_path, level).noise_level == float(level)


@pytest.mark.parametrize("level", ["-0.1", "1.1"])
def test_parse_args_rejects_cps_noise_outside_range(tmp_path: Path, level: str) -> None:
    with pytest.raises(SystemExit):
        _args(tmp_path, level)


def test_inference_config_uses_flowcps_and_training_sampling_defaults(tmp_path: Path) -> None:
    cfg = _inference_config(_args(tmp_path), torch.device("cuda:3"), seed=17)

    assert cfg.mode == "cps"
    assert cfg.sde_formula == "flowcps"
    assert cfg.effective_noise_scale == 0.7
    assert cfg.num_sampling_steps == 30
    assert cfg.cfg_scale == 1.0
    assert cfg.seed == 17
    assert cfg.share_init_noise
    assert not cfg.save_steps


def test_item_seed_preserves_legacy_index_seeding_and_accepts_explicit_seed() -> None:
    assert _item_seed({}, 7, base_seed=100) == 107
    assert _item_seed({"seed": 9001}, 7, base_seed=100) == 9001


@pytest.mark.parametrize("value", [True, 1.5, "17", -1])
def test_item_seed_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="seed"):
        _item_seed({"seed": value}, 0, base_seed=0)


def test_prepare_input_encodes_prompt_and_first_frame(tmp_path: Path) -> None:
    class FakeModel:
        def encode_text(self, prompts, device):
            assert prompts == ["test prompt"]
            assert device == torch.device("cpu")
            return torch.ones(1, 4, 8, dtype=torch.bfloat16)

        def prepare_condition(self, image, num_frames, height, width):
            assert image.shape == (1, 3, 256, 256)
            assert image.dtype == torch.bfloat16
            assert (num_frames, height, width) == (161, 256, 256)
            return torch.ones(1, 48, 41, 32, 32, dtype=torch.bfloat16)

    prepared = _prepare_input(
        FakeModel(),
        Image.new("RGB", (256, 256), "white"),
        "test prompt",
        _args(tmp_path),
        torch.device("cpu"),
    )

    assert prepared.condition.shape == (1, 48, 41, 32, 32)
    assert prepared.prompt_embeds.shape == (1, 4, 8)
