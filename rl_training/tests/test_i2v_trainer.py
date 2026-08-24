from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from src.models.wan_i2v import _make_autocast_checkpoint_func
from src.trainer.config import SFTConfig
from src.trainer.i2v_trainer import I2VTrainer


def test_sft_config_rejects_removed_trainer_modes():
    with pytest.raises(ValidationError):
        SFTConfig(trainer="cos")


def test_i2v_train_step_uses_single_target_latent():
    class _Model:
        def compute_loss(self, video_latents, condition, prompt_embeds, prompt_dropout=0.0):
            self.video_latents = video_latents
            self.condition = condition
            self.prompt_embeds = prompt_embeds
            self.prompt_dropout = prompt_dropout
            return video_latents.float().mean()

    trainer = I2VTrainer.__new__(I2VTrainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = SimpleNamespace(prompt_dropout=0.25)
    trainer.model = _Model()

    target = torch.ones(2, 3, 4, 5, 6)
    batch = {
        "prompt_embeds": torch.zeros(2, 8, 16),
        "video_latents": target,
        "condition": torch.zeros(2, 3, 4, 5, 6),
    }

    loss = trainer._train_step(batch)

    assert loss.item() == 1.0
    assert trainer.model.video_latents is target
    assert trainer.model.prompt_dropout == 0.25


def test_autocast_checkpoint_function_is_torch_compile_compatible():
    class CheckpointedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.checkpoint = _make_autocast_checkpoint_func(torch.bfloat16)

        def forward(self, inputs):
            return self.checkpoint(self.linear, inputs)

    model = CheckpointedLinear()
    model.compile(backend="eager")
    inputs = torch.randn(2, 4, requires_grad=True)

    model(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert model.linear.weight.grad is not None
