from types import SimpleNamespace

import pytest
import torch

from src.models.wan_i2v import WanI2VForTraining
from src.trainer.base_grpo_trainer import BaseGRPOTrainer
from src.trainer.base_rl_trainer import _delayed_replay_optimizer_steps
from src.trainer.config import RLConfig
from src.trainer.dancegrpo_trainer import (
    DanceGRPOTrainer,
    _interleave_actor_ranks_by_node,
    _shared_prompt_assignment,
    _shared_prompt_wave_ranges,
    _split_group_indices,
)
from src.trainer.rewards.neg_loss import NegLossReward


def test_split_group_indices_exact_fanout():
    assigned = [_split_group_indices(24, rank, 24) for rank in range(24)]

    assert all(len(groups) == 1 for groups in assigned)
    assert sorted(group for groups in assigned for group in groups) == list(range(24))


def test_split_group_indices_more_actors_than_group():
    assigned = [_split_group_indices(24, rank, 56) for rank in range(56)]

    assert sorted(group for groups in assigned for group in groups) == list(range(24))
    assert sum(1 for groups in assigned if groups) == 24


def test_split_group_indices_fewer_actors_than_group():
    assigned = [_split_group_indices(24, rank, 7) for rank in range(7)]

    assert sorted(group for groups in assigned for group in groups) == list(range(24))
    assert [len(groups) for groups in assigned] == [4, 4, 4, 3, 3, 3, 3]


def test_interleave_actor_ranks_by_node_spreads_dispatch_across_nodes():
    actors = list(range(32, 64))

    ordered = _interleave_actor_ranks_by_node(actors, local_world_size=8)

    assert ordered[:8] == [32, 40, 48, 56, 33, 41, 49, 57]
    assert [rank // 8 for rank in ordered[:4]] == [4, 5, 6, 7]
    assert sorted(ordered) == actors


def test_shared_prompt_assignment_shards_prompts_and_groups_across_world():
    assignments = [_shared_prompt_assignment(rank, 64, 8, 24) for rank in range(64)]

    assert [item[0] for item in assignments[:8]] == list(range(8))
    assert assignments[0] == (0, 0, 8, [0, 8, 16])
    assert assignments[8] == (0, 1, 8, [1, 9, 17])
    for prompt_idx in range(8):
        groups = sorted(group for item in assignments if item[0] == prompt_idx for group in item[3])
        assert groups == list(range(24))


def test_shared_prompt_wave_ranges_preserve_global_prompt_batch():
    assert _shared_prompt_wave_ranges(32, None) == [(0, 32)]
    assert _shared_prompt_wave_ranges(32, 16) == [(0, 16), (16, 16)]


@pytest.mark.parametrize("prompt_batch_size,microbatch_size", [(32, 0), (32, 33), (32, 12)])
def test_shared_prompt_wave_ranges_reject_invalid_sizes(prompt_batch_size, microbatch_size):
    with pytest.raises(ValueError, match="grpo_shared_prompt_microbatch_size"):
        _shared_prompt_wave_ranges(prompt_batch_size, microbatch_size)


@pytest.mark.parametrize(
    ("world_size", "expected_ranks_per_prompt", "expected_groups_per_rank"),
    [(32, 2, 16), (64, 4, 8)],
)
def test_shared_prompt_waves_cover_every_prompt_group(
    world_size,
    expected_ranks_per_prompt,
    expected_groups_per_rank,
):
    seen: dict[int, list[int]] = {prompt_idx: [] for prompt_idx in range(32)}
    for prompt_offset, wave_size in _shared_prompt_wave_ranges(32, 16):
        assignments = [
            _shared_prompt_assignment(rank, world_size, wave_size, group_size=32) for rank in range(world_size)
        ]
        assert all(item[2] == expected_ranks_per_prompt for item in assignments)
        assert all(len(item[3]) == expected_groups_per_rank for item in assignments)
        for prompt_idx, _prompt_rank, _prompt_world_size, groups in assignments:
            seen[prompt_offset + prompt_idx].extend(groups)

    assert all(sorted(groups) == list(range(32)) for groups in seen.values())


def test_rl_config_validates_shared_prompt_microbatch_size():
    cfg = RLConfig(
        grpo_shared_prompt_batch=True,
        batch_size=32,
        grpo_shared_prompt_microbatch_size=16,
    )

    assert cfg.grpo_shared_prompt_microbatch_size == 16
    with pytest.raises(ValueError, match="requires grpo_shared_prompt_batch=true"):
        RLConfig(batch_size=32, grpo_shared_prompt_microbatch_size=16)
    with pytest.raises(ValueError, match="must be <= batch_size"):
        RLConfig(grpo_shared_prompt_batch=True, batch_size=32, grpo_shared_prompt_microbatch_size=64)
    with pytest.raises(ValueError, match="batch_size must be divisible"):
        RLConfig(grpo_shared_prompt_batch=True, batch_size=32, grpo_shared_prompt_microbatch_size=12)


def test_rl_config_validates_delayed_replay_switch_and_clip():
    cfg = RLConfig(
        grpo_shared_prompt_batch=True,
        grpo_delayed_replay=True,
        grpo_delayed_replay_clip_range=1.0e-2,
    )

    assert cfg.grpo_delayed_replay is True
    assert cfg.grpo_delayed_replay_clip_range == 1.0e-2
    with pytest.raises(ValueError, match="requires grpo_shared_prompt_batch=true"):
        RLConfig(grpo_delayed_replay=True)
    with pytest.raises(ValueError, match="PPO clip ranges must be > 0"):
        RLConfig(grpo_clip_range=0)
    with pytest.raises(ValueError, match="PPO clip ranges must be > 0"):
        RLConfig(grpo_delayed_replay_clip_range=-0.1)


@pytest.mark.parametrize(
    ("batches_per_epoch", "num_epochs", "save_steps", "max_steps", "expected"),
    [
        (3, 1, 0, None, 2),
        (10, 2, 3, None, 14),
        (1562, 5, 100, None, 7728),
        (1562, 5, 100, 3, 3),
    ],
)
def test_delayed_replay_optimizer_step_accounting(
    batches_per_epoch,
    num_epochs,
    save_steps,
    max_steps,
    expected,
):
    assert (
        _delayed_replay_optimizer_steps(
            batches_per_epoch=batches_per_epoch,
            num_epochs=num_epochs,
            save_steps=save_steps,
            max_steps=max_steps,
        )
        == expected
    )


def test_shared_prompt_waves_prepare_before_replay_and_sync_only_final_wave(monkeypatch):
    events = []

    class RecordingOptimizer:
        def zero_grad(self, *, set_to_none):
            events.append(("zero_grad", set_to_none))

    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        grpo_shared_prompt_microbatch_size=16,
        grpo_group_size=32,
        grpo_num_sampling_steps=30,
        grpo_sample_batch_size=8,
        grpo_clip_range=1.0e-4,
    )
    trainer.device = torch.device("cpu")
    trainer.tensor_parallel_enabled = False
    trainer.rank = 0
    trainer.global_rank = 0
    trainer.local_rank = 0
    trainer.world_size = 32
    trainer.dp_rank = 0
    trainer.dp_size = 32
    trainer.train_state = SimpleNamespace(step=0)
    trainer.model = SimpleNamespace(transformer=torch.nn.Identity(), transformer_2=None)
    trainer.optimizers = [RecordingOptimizer()]
    trainer._dp_pg = None
    trainer._split_debug_enabled = lambda: False
    trainer._split_debug_log = lambda *args, **kwargs: None
    trainer._select_training_timesteps_for_step = lambda _steps, _step: [0, 1]
    trainer._sample_group_cps_noise_levels = lambda *args, **kwargs: None

    def prepare_wave(
        _batch,
        *,
        rollout_step,
        prompt_offset,
        prompt_batch_size,
        total_prompt_batch_size,
        all_prompt_cps_noise_levels,
        saved_rollout_videos,
    ):
        assert total_prompt_batch_size == 32
        assert all_prompt_cps_noise_levels is None
        assert rollout_step == 0
        events.append(("prepare", prompt_offset))
        return (
            SimpleNamespace(
                prompt_offset=prompt_offset,
                prompt_batch_size=prompt_batch_size,
                prepare_seconds=1.0,
            ),
            saved_rollout_videos,
        )

    def replay_wave(
        rollout,
        *,
        selected_t_idxs,
        total_prompt_batch_size,
        sync_on_last_backward,
        clip_range,
    ):
        assert selected_t_idxs == [0, 1]
        assert total_prompt_batch_size == 32
        assert clip_range == 1.0e-4
        events.append(("replay", rollout.prompt_offset, sync_on_last_backward))
        shape = (rollout.prompt_batch_size, 32)
        return {
            "local_policy_sum": 1.0,
            "local_kl_sum": 0.0,
            "local_clip_fraction_sum": 0.0,
            "local_ratio_sum": 1024.0,
            "local_approx_kl_sum": 0.0,
            "local_ratio_abs_max": 0.0,
            "rewards": torch.zeros(shape),
            "advantages": torch.zeros(shape),
            "active_ranks": 32,
            "reward_drain_seconds": 2.0,
            "replay_seconds": 3.0,
        }

    trainer._prepare_shared_prompt_rollout_wave = prepare_wave
    trainer._replay_shared_prompt_rollout_wave = replay_wave
    trainer._offload_inference_models_for_replay = lambda: events.append(("offload",))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    metrics = trainer._grpo_step_shared_prompt_batch({"prompt": [f"prompt-{idx}" for idx in range(32)]})

    assert events == [
        ("prepare", 0),
        ("prepare", 16),
        ("offload",),
        ("zero_grad", True),
        ("replay", 0, False),
        ("replay", 16, True),
    ]
    assert metrics["shared_prompt_prepare_seconds"] == 2.0
    assert metrics["shared_prompt_reward_drain_seconds"] == 4.0
    assert metrics["shared_prompt_replay_seconds"] == 6.0
    assert metrics["ppo_clip_range"] == 1.0e-4
    assert metrics["ppo_clip_fraction"] == 0.0
    assert metrics["ppo_ratio_mean"] == 1.0


def test_delayed_replay_prefills_runs_one_update_stale_and_flushes_at_max_steps():
    events = []
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        grpo_num_sampling_steps=30,
        grpo_clip_range=1.0e-4,
        grpo_delayed_replay_clip_range=1.0e-2,
        max_steps=3,
        save_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.train_state = SimpleNamespace(step=0)
    trainer._grpo_force_delayed_replay_flush = False
    trainer._select_training_timesteps_for_step = lambda _steps, step: [step]
    trainer._max_rank_seconds = lambda seconds: seconds

    def prepare_step(_batch, *, rollout_step, policy_version, selected_t_idxs):
        assert selected_t_idxs == [rollout_step]
        events.append(("prepare", rollout_step, policy_version))
        return SimpleNamespace(
            rollout_step=rollout_step,
            policy_version=policy_version,
        )

    def replay_step(step_rollout, *, clip_range):
        events.append(("replay", step_rollout.rollout_step, clip_range))
        return {
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "advantage_mean": 0.0,
            "active_rollout_ranks": 1.0,
            "ppo_clip_range": clip_range,
        }

    trainer._prepare_shared_prompt_step_rollout = prepare_step
    trainer._replay_shared_prompt_step_rollout = replay_step

    prefill = trainer._grpo_step_shared_prompt_batch_delayed({"prompt": ["a"]})
    assert prefill["_skip_optimizer_step"] is True

    first = trainer._grpo_step_shared_prompt_batch_delayed({"prompt": ["b"]})
    assert first["delayed_replay_staleness"] == 0.0
    assert first["delayed_replay_flush"] == 0.0

    trainer.train_state.step = 1
    second = trainer._grpo_step_shared_prompt_batch_delayed({"prompt": ["c"]})
    assert second["delayed_replay_staleness"] == 1.0
    assert second["delayed_replay_flush"] == 0.0

    trainer.train_state.step = 2
    final = trainer._grpo_step_shared_prompt_batch_delayed({"prompt": ["unused"]})
    assert final["delayed_replay_staleness"] == 1.0
    assert final["delayed_replay_flush"] == 1.0
    assert trainer._delayed_shared_prompt_rollout is None
    assert events == [
        ("prepare", 0, 0),
        ("prepare", 1, 0),
        ("replay", 0, 1.0e-2),
        ("prepare", 2, 1),
        ("replay", 1, 1.0e-2),
        ("replay", 2, 1.0e-2),
    ]


def test_delayed_replay_flushes_before_checkpoint_and_epoch_boundary():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(max_steps=100, save_steps=10)
    trainer.train_state = SimpleNamespace(step=9)
    trainer._grpo_force_delayed_replay_flush = False

    assert trainer._delayed_replay_must_flush() is True

    trainer.train_state.step = 8
    assert trainer._delayed_replay_must_flush() is False

    trainer._grpo_force_delayed_replay_flush = True
    assert trainer._delayed_replay_must_flush() is True


def test_rl_config_allows_full_actor_sync_without_lora():
    cfg = RLConfig(rl_actor_weight_sync="full", lora_rank=0)

    assert cfg.rl_actor_weight_sync == "full"
    assert cfg.lora_rank == 0


def test_rl_config_allows_flowcps_sde_formula():
    cfg = RLConfig(grpo_sde_formula="flowcps", grpo_cps_noise_scale_range=[0.0, 1.0])

    assert cfg.grpo_sde_formula == "flowcps"
    assert cfg.grpo_cps_noise_scale_range == (0.0, 1.0)


@pytest.mark.parametrize(
    "noise_range",
    [(-0.1, 0.5), (0.2, 0.2), (0.8, 0.2), (0.0, 1.1)],
)
def test_rl_config_rejects_invalid_flowcps_noise_range(noise_range):
    with pytest.raises(ValueError, match="grpo_cps_noise_scale_range"):
        RLConfig(grpo_sde_formula="flowcps", grpo_cps_noise_scale_range=noise_range)


def test_rl_config_rejects_cps_noise_range_for_other_sde_formulas():
    with pytest.raises(ValueError, match="requires grpo_sde_formula='flowcps'"):
        RLConfig(grpo_sde_formula="dancegrpo", grpo_cps_noise_scale_range=(0.0, 1.0))


def test_rl_config_requires_explicit_vbvr_evalkit_pin():
    with pytest.raises(ValueError, match="requires vbvr_reward_evalkit_dir"):
        RLConfig(grpo_reward_fn="vbvr_rule")
    with pytest.raises(ValueError, match="requires vbvr_reward_evalkit_source_sha256"):
        RLConfig(grpo_reward_fn="vbvr_rule", vbvr_reward_evalkit_dir="evalkit")


def test_rl_config_validates_vbvr_evalkit_digest():
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        RLConfig(
            grpo_reward_fn="vbvr_rule",
            vbvr_reward_evalkit_dir="evalkit",
            vbvr_reward_evalkit_source_sha256="not-a-digest",
        )
    cfg = RLConfig(
        grpo_reward_fn="vbvr_rule",
        vbvr_reward_evalkit_dir="evalkit",
        vbvr_reward_evalkit_source_sha256="A" * 64,
    )
    assert cfg.vbvr_reward_evalkit_source_sha256 == "a" * 64


def test_rl_config_validates_vbvr_reward_queue_bound():
    assert RLConfig().vbvr_reward_max_pending_jobs == 0
    with pytest.raises(ValueError, match="vbvr_reward_max_pending_jobs must be >= 0"):
        RLConfig(vbvr_reward_max_pending_jobs=-1)


def test_dancegrpo_samples_deterministic_cps_noise_once_per_prompt_group():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        grpo_sde_formula="flowcps",
        grpo_sde_noise_scale=0.7,
        grpo_cps_noise_scale_range=(0.0, 1.0),
        seed=42,
    )

    levels = trainer._sample_group_cps_noise_levels(3, step=11, stream_id=0, device=torch.device("cpu"))
    repeated = levels.repeat_interleave(4).view(3, 4)
    same_seed = trainer._sample_group_cps_noise_levels(3, step=11, stream_id=0, device=torch.device("cpu"))
    other_stream = trainer._sample_group_cps_noise_levels(3, step=11, stream_id=1, device=torch.device("cpu"))

    assert torch.equal(levels, same_seed)
    assert not torch.equal(levels, other_stream)
    assert bool(((levels >= 0.0) & (levels < 1.0)).all())
    assert torch.equal(repeated, levels[:, None].expand(-1, 4))
    assert torch.unique(levels).numel() == 3


def test_dancegrpo_fixed_cps_noise_remains_supported():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        grpo_sde_formula="flowcps",
        grpo_sde_noise_scale=0.7,
        grpo_cps_noise_scale_range=None,
        seed=42,
    )

    levels = trainer._sample_group_cps_noise_levels(3, step=11, stream_id=0, device=torch.device("cpu"))

    assert torch.equal(levels, torch.full((3,), 0.7))


def test_split_rollout_coalescing_preserves_per_prompt_cps_noise_levels():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    trainer.cfg = SimpleNamespace(grpo_train_sample_batch_size=2)
    chunks = []
    for group in [0, 1]:
        chunks.append(
            {
                "groups": [group],
                "rewards": torch.zeros(2, 1),
                "latents": torch.zeros(1, 2, 1),
                "next_latents": torch.zeros(1, 2, 1),
                "log_probs": torch.zeros(1, 2),
                "cps_noise_levels": torch.tensor([0.1, 0.7]),
            }
        )

    merged = trainer._coalesce_train_rollout_chunks(chunks, B=2)

    assert len(merged) == 1
    assert merged[0]["groups"] == [0, 1]
    assert torch.allclose(merged[0]["cps_noise_levels"], torch.tensor([0.1, 0.1, 0.7, 0.7]))


def test_flowcps_transition_matches_coefficients_preserving_formula():
    sample = torch.tensor([[[[[0.4, -0.2]]]]], dtype=torch.float32)
    model_output = torch.tensor([[[[[0.5, -0.3]]]]], dtype=torch.float32)
    sigma = 0.6
    sigma_prev = 0.25
    noise_level = 0.3

    mean, std = WanI2VForTraining._flowcps_transition_mean(
        sample=sample,
        model_output=model_output,
        sigma=sigma,
        sigma_prev=sigma_prev,
        noise_level=noise_level,
    )

    expected_std = sigma_prev * torch.sin(torch.tensor(noise_level * torch.pi / 2.0)).item()
    expected_coeff = (sigma_prev**2 - expected_std**2) ** 0.5
    expected_x0 = sample - sigma * model_output
    expected_x1 = sample + model_output * (1.0 - sigma)
    expected_mean = expected_x0 * (1.0 - sigma_prev) + expected_x1 * expected_coeff

    assert abs(std - expected_std) < 1e-6
    assert torch.allclose(mean, expected_mean)


def test_predicted_clean_latent_pins_expand_timestep_first_frame():
    sample = torch.tensor(
        [[[[[0.4, -0.2]], [[0.1, 0.3]]], [[[0.8, -0.6]], [[-0.4, 0.2]]]]],
        dtype=torch.bfloat16,
    )
    model_output = torch.full_like(sample, 0.5)
    cond_first_frame = torch.tensor([[[[[1.25, -1.5]]], [[[0.75, -0.25]]]]], dtype=torch.bfloat16)

    pred_x0 = WanI2VForTraining._predicted_clean_latent(
        sample,
        model_output,
        sigma=0.6,
        cond_first_frame=cond_first_frame,
    )
    unpinned = sample.float() - 0.6 * model_output.float()

    assert pred_x0.dtype is torch.float32
    assert torch.equal(pred_x0[:, :, 0:1], cond_first_frame.float())
    assert torch.allclose(pred_x0[:, :, 1:], unpinned[:, :, 1:])
    assert not torch.equal(pred_x0[:, :, 0:1], unpinned[:, :, 0:1])


def test_flowcps_transition_supports_per_sample_noise_levels():
    sample = torch.tensor([0.4, 0.4], dtype=torch.float32).reshape(2, 1, 1, 1, 1)
    model_output = torch.tensor([0.5, 0.5], dtype=torch.float32).reshape(2, 1, 1, 1, 1)
    noise_levels = torch.tensor([0.0, 1.0])
    sigma = 0.6
    sigma_prev = 0.25

    mean, std = WanI2VForTraining._flowcps_transition_mean(
        sample=sample,
        model_output=model_output,
        sigma=sigma,
        sigma_prev=sigma_prev,
        noise_level=noise_levels,
    )

    expected_std = torch.tensor([0.0, sigma_prev]).reshape(2, 1, 1, 1, 1)
    x0 = sample - sigma * model_output
    x1 = sample + model_output * (1.0 - sigma)
    expected_coeff = torch.tensor([sigma_prev, 0.0]).reshape(2, 1, 1, 1, 1)
    expected_mean = x0 * (1.0 - sigma_prev) + x1 * expected_coeff

    assert torch.allclose(std, expected_std, atol=1e-6)
    assert torch.allclose(mean, expected_mean, atol=1e-6)


def test_flowcps_step_applies_each_samples_noise_level():
    model = WanI2VForTraining.__new__(WanI2VForTraining)
    model.expand_timesteps = False
    sample = torch.zeros(2, 1, 1, 1, 1)
    model_output = torch.zeros_like(sample)
    noise = torch.ones_like(sample)

    prev_sample, mean, log_prob = model._flowcps_sde_step(
        sample=sample,
        model_output=model_output,
        sigma=0.6,
        sigma_prev=0.25,
        noise_level=torch.tensor([0.0, 1.0]),
        noise=noise,
    )

    assert torch.allclose(mean, torch.zeros_like(mean))
    assert torch.allclose(prev_sample.flatten(), torch.tensor([0.0, 0.25]), atol=1e-6)
    assert torch.allclose(log_prob, torch.tensor([0.0, -(0.25**2)]), atol=1e-6)


def test_flowcps_log_prob_and_kl_use_official_unscaled_surrogates():
    mean = torch.zeros(2, 1, 1, 1, 2)
    sample = torch.ones_like(mean) * 0.25
    ref_mean = torch.ones_like(mean) * 0.5

    log_prob = WanI2VForTraining._sde_transition_log_prob("flowcps", sample, mean, noise_scale=0.1)
    kl_loss = WanI2VForTraining._sde_transition_kl_loss("flowcps", mean, ref_mean, noise_scale=0.1)

    assert torch.allclose(log_prob, torch.full((2,), -(0.25**2)))
    assert torch.allclose(kl_loss, torch.tensor(0.25))


def test_split_rollout_actor_skips_full_ft_reference_copies():
    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    trainer.rl_split_enabled = True
    trainer.is_inference_rank = True

    trainer._pre_fsdp_setup(SimpleNamespace(lora_rank=0, grpo_kl_coeff=0.004))

    assert trainer.is_lora is False
    assert trainer.ref_transformers == {}


def test_streamed_full_policy_dtype_parser_accepts_torch_names():
    assert DanceGRPOTrainer._dtype_from_name("torch.float32") is torch.float32
    assert DanceGRPOTrainer._dtype_from_name("bfloat16") is torch.bfloat16


class _DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_weight = torch.nn.Parameter(torch.empty((), dtype=torch.float32), requires_grad=False)
        self.seen_hidden_shapes = []
        self.seen_timestep_shapes = []
        self.seen_hidden_dtypes = []

    def forward(self, *, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        self.seen_timestep_shapes.append(tuple(timestep.shape))
        self.seen_hidden_dtypes.append(hidden_states.dtype)
        return (torch.zeros_like(hidden_states),)


class _DummyExpandTimestepModel:
    boundary_timestep = 900

    def __init__(self, transformer):
        self.transformer = transformer
        self.transformer_2 = None
        self.latent_shape_called = False
        self.build_model_input_called = False
        self.build_timestep_input_called = False
        self.prepare_call_called = False

    def latent_shape_from_condition(self, condition):
        self.latent_shape_called = True
        return tuple(condition.shape)

    def _build_model_input(self, latent, condition):
        self.build_model_input_called = True
        if condition.shape != latent.shape:
            raise ValueError("expand-timestep condition and latent shapes must match")
        return latent + condition

    def _build_timestep_input(self, timesteps, latent, transformer):
        self.build_timestep_input_called = True
        return torch.ones(latent.shape[0], 3, device=latent.device, dtype=torch.bfloat16) * timesteps[:, None]

    def _prepare_transformer_call(self, transformer, hidden_states, timestep, encoder_hidden_states):
        self.prepare_call_called = True
        return hidden_states, timestep, encoder_hidden_states

    def _get_expert_for_timestep(self, timestep_val):
        return self.transformer

    def _iter_transformer_selections(self, timesteps):
        selected = torch.arange(timesteps.shape[0], device=timesteps.device)
        return [("single", selected, self.transformer)]


class _DummyNegLossModel(_DummyExpandTimestepModel):
    boundary_timestep = 900
    num_train_timesteps = 2

    def _get_training_buffers(self, device):
        return (
            torch.tensor([0.2, 0.8], device=device),
            torch.tensor([950.0, 950.0], device=device),
            None,
        )

    def _prepare_transformer_call(self, transformer, hidden_states, timestep, encoder_hidden_states):
        self.prepare_call_called = True
        dtype = next(transformer.parameters()).dtype
        return hidden_states.to(dtype=dtype), timestep, encoder_hidden_states.to(dtype=dtype)


def test_dancegrpo_shared_initial_latents_use_model_latent_shape_for_5b():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(dancegrpo_share_group_init_noise=True)

    condition = torch.zeros(2, 48, 41, 24, 24)
    latents = trainer._sample_group_initial_latents(condition)

    assert model.latent_shape_called
    assert tuple(latents.shape) == tuple(condition.shape)


def test_dancegrpo_policy_forward_uses_model_input_helpers_for_5b():
    trainer = DanceGRPOTrainer.__new__(DanceGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(grpo_cfg_scale=1.0)

    latent = torch.zeros(2, 48, 3, 4, 4)
    condition = torch.ones_like(latent)
    prompt_embeds = torch.zeros(2, 8, 16)

    out = trainer._policy_forward(transformer, latent, condition, prompt_embeds, timestep_val=500.0)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == tuple(latent.shape)
    assert transformer.seen_hidden_shapes == [tuple(latent.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]


def test_neg_loss_reward_uses_model_helpers_and_transformer_dtype_for_5b():
    transformer = _DummyTransformer().to(dtype=torch.bfloat16)
    model = _DummyNegLossModel(transformer)
    trainer = SimpleNamespace(model=model)
    reward = NegLossReward(trainer, SimpleNamespace(fsdp=False))

    latents = torch.zeros(2, 48, 3, 4, 4, dtype=torch.float32)
    condition = torch.ones_like(latents)
    prompt_embeds = torch.zeros(2, 8, 16, dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.long)

    out = reward(latents, latents, condition, prompt_embeds, indices=indices)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == (2,)
    assert transformer.seen_hidden_shapes == [tuple(latents.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]
    assert transformer.seen_hidden_dtypes == [torch.bfloat16]


def test_grpo_reference_forward_uses_model_input_helpers_for_5b_fallback_ref():
    trainer = BaseGRPOTrainer.__new__(BaseGRPOTrainer)
    transformer = _DummyTransformer()
    model = _DummyExpandTimestepModel(transformer)
    trainer.model = model
    trainer.is_lora = False
    trainer.expert_parallel = False
    trainer.ref_transformers = {"transformer": transformer}

    latent = torch.zeros(2, 48, 3, 4, 4)
    condition = torch.ones_like(latent)
    prompt_embeds = torch.zeros(2, 8, 16)

    out = trainer._ref_forward(latent, condition, prompt_embeds, timestep_val=100.0)

    assert model.build_model_input_called
    assert model.build_timestep_input_called
    assert model.prepare_call_called
    assert tuple(out.shape) == tuple(latent.shape)
    assert transformer.seen_hidden_shapes == [tuple(latent.shape)]
    assert transformer.seen_timestep_shapes == [(2, 3)]
