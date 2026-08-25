"""Run a bounded single-rank GRPO config and prove trainable weights changed.

This is stricter than checking for a successful process exit: it snapshots all
trainable tensors before ``trainer.train()`` and fails if the optimizer step
leaves them byte-identical.

Example:
    .venv/bin/torchrun --standalone --nproc_per_node=1 \
        -m scripts.dev.validate_grpo_parameter_update \
        --config configs/train_rl_5b_cps.yaml \
        --one-gpu-smoke \
        --model-path storage/models/Wan2.2-TI2V-5B-Diffusers \
        --dataset-json storage/smoke/i2v_512x512x81/dataset.json \
        --output-dir storage/smoke/checkpoints/rl_5b_update
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from src.trainer import DanceGRPOTrainer, RLConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--one-gpu-smoke",
        action="store_true",
        help="derive a bounded LoRA/neg_loss smoke from the selected release config",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load_config(
    config_path: Path,
    *,
    one_gpu_smoke: bool = False,
    model_path: Path | None = None,
    dataset_json: Path | None = None,
    output_dir: Path | None = None,
) -> RLConfig:
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError(f"GRPO config must contain a mapping: {config_path}")

    supplied_paths = (model_path, dataset_json, output_dir)
    if not one_gpu_smoke and any(path is not None for path in supplied_paths):
        raise ValueError("--model-path, --dataset-json, and --output-dir require --one-gpu-smoke")
    if not one_gpu_smoke:
        return RLConfig(**values)
    if any(path is None for path in supplied_paths):
        raise ValueError("--one-gpu-smoke requires --model-path, --dataset-json, and --output-dir")

    # Keep this profile in code so the release has one source config per real
    # training mode. These values are the smallest settings that still cover
    # raw encoding, grouped Flow-CPS rollout, replay, backward, and an update.
    values.update(
        {
            "model_path": str(model_path),
            "dataset_json": str(dataset_json),
            "dataset_size": 4,
            "shuffle_raw_indices": False,
            "output_dir": str(output_dir),
            "resume_from": None,
            "reset_dataloader": True,
            "auto_resume": False,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "max_steps": 1,
            "learning_rate": 1.0e-6,
            "warmup_steps": 0,
            "save_steps": 0,
            "save_final_checkpoint": False,
            "save_epoch_checkpoints": False,
            "log_steps": 1,
            "transformer_load_dtype": "bfloat16",
            "num_workers": 0,
            "persistent_workers": False,
            "fsdp": False,
            "hsdp": False,
            "tensor_parallel_size": 1,
            "expert_parallel": False,
            "lora_rank": 16,
            "lora_alpha": 16,
            "use_liger_kernel": False,
            "attention_backend": None,
            "torch_compile": False,
            "rl_train_node_count": 0,
            "rl_train_rank_count": 0,
            "rl_actor_weight_sync": "none",
            "rl_async_rollout": False,
            "grpo_shared_prompt_batch": True,
            "grpo_shared_prompt_microbatch_size": 1,
            "grpo_delayed_replay": False,
            "grpo_group_size": 2,
            "grpo_sample_batch_size": 2,
            "grpo_train_sample_batch_size": 1,
            "grpo_fsdp_sync_each_backward": False,
            "grpo_offload_inference_models": True,
            "grpo_num_sampling_steps": 2,
            "grpo_kl_coeff": 0.0,
            "grpo_reward_fn": "neg_loss",
            "grpo_sde_formula": "flowcps",
            "grpo_sde_noise_scale": 0.7,
            "grpo_cps_noise_scale_range": None,
            "dancegrpo_timestep_selection_ratio": 1.0,
            "wandb_project": None,
            "wandb_run_name": None,
        }
    )
    return RLConfig(**values)


def _named_trainable_parameters(trainer: DanceGRPOTrainer):
    modules = (
        ("text_encoder", trainer.model.text_encoder),
        ("transformer", trainer.model.transformer),
        ("transformer_2", trainer.model.transformer_2),
    )
    for module_name, module in modules:
        if module is None:
            continue
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                yield f"{module_name}.{parameter_name}", parameter


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("validate_grpo_parameter_update.py is intentionally single-rank")

    cfg = _load_config(
        args.config,
        one_gpu_smoke=args.one_gpu_smoke,
        model_path=args.model_path,
        dataset_json=args.dataset_json,
        output_dir=args.output_dir,
    )
    if cfg.max_steps != 1:
        raise ValueError(f"parameter-update validation requires max_steps: 1, got {cfg.max_steps}")

    trainer = DanceGRPOTrainer(cfg)
    trainable = list(_named_trainable_parameters(trainer))
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}

    try:
        trainer.train()
        changed_tensors = 0
        changed_elements = 0
        max_abs_delta = 0.0
        squared_delta = 0.0
        for name, parameter in trainable:
            delta = parameter.detach().cpu().float() - before[name].float()
            tensor_max = delta.abs().max().item()
            if tensor_max > 0:
                changed_tensors += 1
                changed_elements += int(torch.count_nonzero(delta).item())
            max_abs_delta = max(max_abs_delta, tensor_max)
            squared_delta += delta.double().square().sum().item()

        summary = {
            "trainable_tensors": len(trainable),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "changed_tensors": changed_tensors,
            "changed_elements": changed_elements,
            "max_abs_delta": max_abs_delta,
            "delta_l2": math.sqrt(squared_delta),
        }
        print("PARAMETER_UPDATE " + json.dumps(summary, sort_keys=True))
        if changed_tensors == 0:
            raise RuntimeError("Optimizer step completed but no trainable parameter changed")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
