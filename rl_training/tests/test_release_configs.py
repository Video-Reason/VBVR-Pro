from __future__ import annotations

from pathlib import Path

import yaml

from scripts.dev.validate_grpo_parameter_update import _load_config
from src.trainer import RLConfig, SFTConfig
from src.trainer.dancegrpo_trainer import _shared_prompt_assignment, _shared_prompt_wave_ranges

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_RL_CONFIGS = {
    "train_rl_5b_cps.yaml": "vbvr_rule",
    "train_rl_5b_sde.yaml": "vbvr_rule",
    "train_rl_5b_vlm.yaml": "vbvr_vlm",
    "train_rl_a14b_rule.yaml": "vbvr_rule",
}


def test_release_ships_only_the_selected_sft_config():
    paths = sorted((_REPO_ROOT / "configs").glob("train_sft_*.yaml"))

    assert [path.name for path in paths] == ["train_sft_vbvr_5e-6.yaml"]
    values = yaml.safe_load(paths[0].read_text(encoding="utf-8"))
    config = SFTConfig(**values)
    assert config.learning_rate == 5.0e-6
    assert config.latent_webdataset_dir == "data/vbvr/latents/sft"
    assert config.dataset_size == 800_000
    assert config.fsdp
    assert config.expert_parallel
    assert config.train_experts == "both"


def test_release_ships_only_the_selected_rl_configs():
    found: dict[str, str] = {}
    for path in sorted((_REPO_ROOT / "configs").glob("*.yaml")):
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if values.get("trainer") != "dancegrpo":
            continue
        config = RLConfig(**values)
        found[path.name] = config.grpo_reward_fn

    assert found == _EXPECTED_RL_CONFIGS


def test_one_gpu_smoke_is_derived_from_the_release_config():
    config = _load_config(
        _REPO_ROOT / "configs/train_rl_5b_cps.yaml",
        one_gpu_smoke=True,
        model_path=Path("storage/models/Wan2.2-TI2V-5B-Diffusers"),
        dataset_json=Path("storage/smoke/i2v_512x512x81/dataset.json"),
        output_dir=Path("storage/smoke/checkpoints/rl_5b_update"),
    )

    assert config.max_steps == 1
    assert config.grpo_reward_fn == "neg_loss"
    assert config.grpo_group_size == 2
    assert config.lora_rank == 16
    assert not config.fsdp
    assert not config.hsdp
    assert not config.torch_compile


def test_5b_rule_references_are_a_paired_cps_sde_variant():
    values_by_name = {
        name: yaml.safe_load((_REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
        for name in ("train_rl_5b_cps.yaml", "train_rl_5b_sde.yaml")
    }
    cps_values = values_by_name["train_rl_5b_cps.yaml"]
    sde_values = values_by_name["train_rl_5b_sde.yaml"]

    assert cps_values["grpo_sde_formula"] == "flowcps"
    assert sde_values["grpo_sde_formula"] == "dancegrpo"
    assert cps_values["grpo_sde_noise_scale"] == 0.7
    assert sde_values["grpo_sde_noise_scale"] == 0.3

    variant_fields = {
        "grpo_sde_formula",
        "grpo_sde_noise_scale",
        "output_dir",
        "vbvr_reward_tmp_dir",
        "wandb_run_name",
    }
    assert {k: v for k, v in cps_values.items() if k not in variant_fields} == {
        k: v for k, v in sde_values.items() if k not in variant_fields
    }


def test_5b_reference_topologies_assign_every_rollout_once():
    reference_world_sizes = {
        "train_rl_5b_cps.yaml": (128,),
        "train_rl_5b_sde.yaml": (128,),
        "train_rl_5b_vlm.yaml": (32, 64, 128),
    }
    for name, world_sizes in reference_world_sizes.items():
        values = yaml.safe_load((_REPO_ROOT / "configs" / name).read_text(encoding="utf-8"))
        config = RLConfig(**values)
        wave_size = _shared_prompt_wave_ranges(
            config.batch_size,
            config.grpo_shared_prompt_microbatch_size,
        )[0][1]
        for world_size in world_sizes:
            assignments = [
                _shared_prompt_assignment(rank, world_size, wave_size, config.grpo_group_size)
                for rank in range(world_size)
            ]
            for prompt_index in range(wave_size):
                owned_groups = sorted(
                    group_index
                    for assigned_prompt, _prompt_rank, _prompt_world, groups in assignments
                    if assigned_prompt == prompt_index
                    for group_index in groups
                )
                assert owned_groups == list(range(config.grpo_group_size))


def test_a14b_reference_is_tp2_over_four_data_replicas():
    values = yaml.safe_load((_REPO_ROOT / "configs/train_rl_a14b_rule.yaml").read_text(encoding="utf-8"))
    config = RLConfig(**values)

    assert config.tensor_parallel_size == 2
    assert 8 // config.tensor_parallel_size == 4
    assert config.batch_size * 4 == 16
    assert config.fsdp
    assert not config.hsdp
