from types import SimpleNamespace

import pytest
import torch

from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.base_rl_trainer import BaseRLTrainer
from src.trainer.config import RLConfig


def _tp_cfg(**overrides) -> RLConfig:
    values = {
        "tensor_parallel_size": 2,
        "fsdp": True,
        "hsdp": False,
        "lora_rank": 0,
        "use_liger_kernel": False,
        "torch_compile": False,
        "train_text_encoder": False,
        "expert_parallel": False,
        "batch_size": 4,
        "grpo_shared_prompt_batch": False,
    }
    values.update(overrides)
    return RLConfig(**values)


def _trainer_state(rank: int = 0, world_size: int = 8):
    return SimpleNamespace(
        rl_split_enabled=False,
        expert_parallel=False,
        world_size=world_size,
        rank=rank,
        dp_rank=rank,
        dp_size=world_size,
    )


@pytest.mark.parametrize(
    ("rank", "expected_dp_rank", "expected_tp_rank"),
    [(0, 0, 0), (1, 0, 1), (2, 1, 0), (5, 2, 1), (7, 3, 1)],
)
def test_tp2_rank_mapping(rank: int, expected_dp_rank: int, expected_tp_rank: int):
    trainer = _trainer_state(rank=rank)
    BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg())

    assert trainer.tensor_parallel_enabled
    assert trainer.dp_size == 4
    assert trainer.dp_rank == expected_dp_rank
    assert trainer.tp_rank == expected_tp_rank


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"fsdp": False}, "fsdp=False"),
        ({"hsdp": True}, "hsdp=True"),
        ({"lora_rank": 8}, "LoRA"),
        ({"train_text_encoder": True}, "train_text_encoder"),
    ],
)
def test_tp_rejects_unsupported_combinations(override: dict, message: str):
    trainer = _trainer_state()
    with pytest.raises(ValueError, match=message):
        BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg(**override))


def test_tp_requires_world_size_divisibility():
    trainer = _trainer_state(world_size=7)
    with pytest.raises(ValueError, match="must be divisible"):
        BaseRLTrainer._init_tensor_parallel(trainer, _tp_cfg())


def test_tp2_four_node_topology_preserves_bs16_with_local_batch_one():
    cfg = _tp_cfg(batch_size=1)
    trainer = _trainer_state(rank=31, world_size=32)

    BaseRLTrainer._init_tensor_parallel(trainer, cfg)

    assert trainer.dp_size == 16
    assert trainer.dp_rank == 15
    assert trainer.tp_rank == 1
    assert cfg.batch_size * trainer.dp_size == 16


def test_tp_accepts_liger_and_torch_compile():
    trainer = _trainer_state()
    BaseRLTrainer._init_tensor_parallel(
        trainer,
        _tp_cfg(use_liger_kernel=True, torch_compile=True),
    )

    assert trainer.tensor_parallel_enabled
    assert trainer.dp_size == 4


def test_compile_modules_preserves_module_identity():
    class RecordingModule:
        def __init__(self):
            self.compile_kwargs = None

        def compile(self, **kwargs):
            self.compile_kwargs = kwargs

    vae = RecordingModule()
    text_encoder = RecordingModule()
    transformer = RecordingModule()
    transformer_2 = RecordingModule()
    model = SimpleNamespace(
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
    )
    trainer = SimpleNamespace(model=model)
    cfg = _tp_cfg(torch_compile=True, torch_compile_backend="eager", torch_compile_mode=None)

    BaseRLTrainer._compile_modules(trainer, cfg)

    assert trainer.model.vae is vae
    assert trainer.model.text_encoder is text_encoder
    assert trainer.model.transformer is transformer
    assert trainer.model.transformer_2 is transformer_2
    assert vae.compile_kwargs == {"backend": "eager"}
    assert text_encoder.compile_kwargs == {"backend": "eager"}
    assert transformer.compile_kwargs == {"backend": "eager"}
    assert transformer_2.compile_kwargs == {"backend": "eager"}


def test_shared_prompt_batch_is_bounded_by_dp_size():
    trainer = _trainer_state()
    with pytest.raises(ValueError, match="prompt-wave size must be <= DP size"):
        BaseRLTrainer._init_tensor_parallel(
            trainer,
            _tp_cfg(grpo_shared_prompt_batch=True, batch_size=8),
        )


def test_shared_prompt_microbatch_is_bounded_by_dp_size():
    trainer = _trainer_state()

    BaseRLTrainer._init_tensor_parallel(
        trainer,
        _tp_cfg(
            grpo_shared_prompt_batch=True,
            batch_size=8,
            grpo_shared_prompt_microbatch_size=4,
        ),
    )

    assert trainer.dp_size == 4


def test_tensor_parallel_size_must_be_positive():
    with pytest.raises(ValueError, match="tensor_parallel_size must be > 0"):
        RLConfig(tensor_parallel_size=0)


def test_replay_offload_moves_frozen_inference_models_to_cpu():
    class RecordingModule:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))
            return self

    text_encoder = RecordingModule()
    vae = RecordingModule()
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(grpo_offload_inference_models=True),
        model=SimpleNamespace(text_encoder=text_encoder, vae=vae),
        device=torch.device("cpu"),
    )

    BaseGRPOTrainer._offload_inference_models_for_replay(trainer)

    assert text_encoder.devices == ["cpu"]
    assert vae.devices == ["cpu"]


def test_reward_restores_offloaded_vae_for_latent_batches():
    class RecordingModule:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(str(device))
            return self

    class VAEReward:
        requires_vae = True
        requires_policy_forward = False

        def __call__(self, generated, _gt, _condition, _prompt, *, meta):
            assert meta == {}
            return torch.zeros(generated.shape[0])

    vae = RecordingModule()
    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    trainer.cfg = SimpleNamespace(grpo_offload_inference_models=True)
    trainer.reward_fn = VAEReward()
    trainer.tensor_parallel_enabled = False
    trainer.tp_rank = 0
    trainer.model = SimpleNamespace(vae=vae)
    trainer.device = torch.device("cpu")
    generated = torch.zeros(2, 1)

    rewards = trainer._compute_reward(
        generated,
        generated,
        generated,
        generated,
        meta={},
    )

    assert vae.devices == ["cpu"]
    assert rewards.shape == (2,)


def test_reward_submit_defers_async_result_until_resolve():
    events = []

    class PendingReward:
        def result(self):
            events.append("resolve")
            return torch.tensor([0.25, 0.75])

    class AsyncReward:
        requires_vae = False
        requires_policy_forward = False

        def submit(self, generated, _gt, _condition, _prompt, *, meta):
            assert meta == {"sample": "metadata"}
            events.append(("submit", generated.shape[0]))
            return PendingReward()

    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    trainer.cfg = SimpleNamespace(grpo_offload_inference_models=False)
    trainer.reward_fn = AsyncReward()
    trainer.tensor_parallel_enabled = False
    trainer.tp_rank = 0
    trainer.model = SimpleNamespace(vae=None)
    trainer.device = torch.device("cpu")
    generated = torch.zeros(2, 1)

    submission = trainer._submit_reward(
        generated,
        generated,
        generated,
        generated,
        meta={"sample": "metadata"},
    )

    assert events == [("submit", 2)]
    rewards = trainer._resolve_reward(submission)
    assert events == [("submit", 2), "resolve"]
    assert torch.equal(rewards, torch.tensor([0.25, 0.75]))
