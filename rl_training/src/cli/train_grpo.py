"""Wan2.2 I2V DanceGRPO training entry point.

Usage:
    .venv/bin/torchrun --nproc_per_node=8 -m src.cli.train_grpo \
        --config configs/train_rl_a14b_rule.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

from src.cli.validate_grpo_runtime import validate_vbvr_evalkit_contract
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime
from src.trainer import DanceGRPOTrainer, RLConfig


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 I2V DanceGRPO Training")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    # CLI overrides (auto-generated from RLConfig fields)
    for name, field_info in RLConfig.model_fields.items():
        if field_info.annotation is bool:
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
        elif field_info.annotation is int:
            parser.add_argument(f"--{name}", type=int, default=None)
        elif field_info.annotation is float:
            parser.add_argument(f"--{name}", type=float, default=None)
        else:
            parser.add_argument(f"--{name}", type=str, default=None)
    args = parser.parse_args()

    # Build config: defaults -> YAML -> CLI
    cfg_dict = {}
    if args.config:
        cfg_dict = yaml.safe_load(Path(args.config).read_text()) or {}
    for name in RLConfig.model_fields:
        v = getattr(args, name, None)
        if v is not None:
            cfg_dict[name] = v

    cfg = RLConfig(**cfg_dict)
    if cfg.grpo_reward_fn == "vbvr_rule":
        evalkit_report = validate_vbvr_evalkit_contract(
            cfg.vbvr_reward_evalkit_dir,
            cfg.vbvr_reward_evalkit_source_sha256,
        )
        runtime_report = validate_vbvr_scorer_runtime(verify_imports=False)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "[preflight] VBVR scorer and external EvalKit passed before trainer initialization: "
                f"runtime_sha256={runtime_report['sha256']} evalkit_sha256={evalkit_report['sha256']}"
            )
    trainer = DanceGRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
