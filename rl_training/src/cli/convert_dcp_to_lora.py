"""Convert a DCP training checkpoint to PEFT-format LoRA safetensors.

Single-process, no distributed init required. Uses
``torch.distributed.checkpoint.format_utils.dcp_to_torch_save`` under the hood,
then filters LoRA tensors out of the resulting flat state-dict and writes them
in the layout that ``pipe.load_lora_weights(...)`` expects.

Usage::

    .venv/bin/python -m src.cli.convert_dcp_to_lora \\
        --config   configs/train_xxx.yaml \\
        --checkpoint storage/checkpoints/run/checkpoint-200 \\
        --output     storage/lora_exports/run/checkpoint-200

Output layout (flat checkpoint)::

    output/
      ├─ transformer/{adapter_model.safetensors, adapter_config.json}
      └─ transformer_2/{adapter_model.safetensors, adapter_config.json}

For an expert-parallel checkpoint (``high/`` and ``low/`` subdirs in the input)
each subdir is mapped to its single trained transformer:
``high → transformer``, ``low → transformer_2``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml
from loguru import logger
from peft import LoraConfig
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from src.models.wan_i2v import LoRATrainConfig
from src.trainer.config import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Training config YAML used for the run")
    p.add_argument("--checkpoint", type=Path, required=True, help="DCP checkpoint dir (flat or EP)")
    p.add_argument("--output", type=Path, required=True, help="Output directory for adapter files")
    p.add_argument("--adapter-name", default="default", help="PEFT adapter name (default: 'default')")
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Override target_modules (defaults to LoRATrainConfig defaults)",
    )
    return p.parse_args()


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    return TrainConfig(**raw)


def detect_dcp_layout(ckpt_dir: Path) -> dict[str, Path]:
    """Map ``transformer | transformer_2 -> dcp_dir`` for flat or EP checkpoints."""
    high, low = ckpt_dir / "high", ckpt_dir / "low"
    if (high / ".metadata").exists() or (low / ".metadata").exists():
        layout: dict[str, Path] = {}
        if (high / ".metadata").exists():
            layout["transformer"] = high
        if (low / ".metadata").exists():
            layout["transformer_2"] = low
        return layout
    if not (ckpt_dir / ".metadata").exists():
        raise FileNotFoundError(f"No DCP .metadata found at {ckpt_dir} or its high/ low/ subdirs")
    return {"transformer": ckpt_dir, "transformer_2": ckpt_dir}


def dcp_to_dict(dcp_dir: Path) -> dict[str, torch.Tensor]:
    """Materialize a DCP checkpoint to a flat ``{fqn: cpu_tensor}`` dict."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        dcp_to_torch_save(str(dcp_dir), str(tmp_path))
        sd = torch.load(tmp_path, map_location="cpu", weights_only=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    # dcp_to_torch_save returns the dict that DCP would build via the
    # Stateful protocol — for our TrainState that's a top-level "train_state"
    # mapping. Unwrap it; ignore everything else (ema, etc.).
    if isinstance(sd, dict) and "train_state" in sd:
        sd = sd["train_state"]
    return _flatten(sd)


def _flatten(d, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten nested dicts: ``{"a": {"b": t}} -> {"a.b": t}``."""
    out: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, torch.Tensor):
            out[key] = v
    return out


def extract_lora(
    flat_sd: dict[str, torch.Tensor],
    transformer_name: str,
    adapter_name: str,
) -> dict[str, torch.Tensor]:
    """Pull LoRA tensors for one transformer out of the flat DCP dict and rename
    them into the PEFT on-disk format ``base_model.model.<...>.lora_X.<adapter>.weight``.
    """
    prefix = f"{transformer_name}."
    out: dict[str, torch.Tensor] = {}
    for fqn, tensor in flat_sd.items():
        if not fqn.startswith(prefix):
            continue
        local = fqn[len(prefix) :]
        # Keep only LoRA-specific tensors (lora_A, lora_B, lora_magnitude_vector, ...).
        if ".lora_" not in local and not local.startswith("lora_"):
            continue
        # PEFT stores adapter tensors under "<...>.lora_A.<adapter_name>.weight";
        # the DCP dump preserves that structure, so we only need to add the
        # "base_model.model." prefix that PEFT prepends when wrapping the model.
        out[f"base_model.model.{local}"] = tensor.detach().contiguous().cpu()
    if not out:
        raise RuntimeError(
            f"No LoRA tensors found under prefix '{prefix}' — was this checkpoint trained with lora_rank>0? "
            f"(saw {len(flat_sd)} keys total)"
        )
    if not any(f".{adapter_name}." in k for k in out):
        adapters = sorted({k.rsplit(".", 2)[-2] for k in out if ".lora_" in k})
        raise RuntimeError(
            f"Adapter '{adapter_name}' not found in checkpoint keys for {transformer_name}. "
            f"Available adapters: {adapters}"
        )
    return out


def build_adapter_config(
    cfg: TrainConfig,
    base_model_path: Path,
    target_modules: list[str] | None,
) -> dict:
    lora_defaults = LoRATrainConfig()
    peft_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules or lora_defaults.target_modules,
        lora_dropout=cfg.lora_dropout,
    )
    cfg_dict = peft_cfg.to_dict()
    cfg_dict["base_model_name_or_path"] = str(base_model_path)
    return cfg_dict


def write_adapter(out_dir: Path, tensors: dict[str, torch.Tensor], adapter_cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    if cfg.lora_rank <= 0:
        raise ValueError(f"Config {args.config} has lora_rank={cfg.lora_rank}; nothing to convert.")

    layout = detect_dcp_layout(args.checkpoint)
    logger.info("Detected layout: {}", {k: str(v) for k, v in layout.items()})
    args.output.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(cfg.model_path)

    for transformer_name, dcp_dir in layout.items():
        logger.info("Reading DCP shards from {} ...", dcp_dir)
        flat_sd = dcp_to_dict(dcp_dir)
        tensors = extract_lora(flat_sd, transformer_name, args.adapter_name)

        adapter_cfg = build_adapter_config(
            cfg,
            base_model_path / transformer_name,
            args.target_modules,
        )

        out_dir = args.output / transformer_name
        write_adapter(out_dir, tensors, adapter_cfg)
        logger.info(
            "Wrote {} tensors to {} (base_model_name_or_path={})",
            len(tensors),
            out_dir,
            adapter_cfg["base_model_name_or_path"],
        )


if __name__ == "__main__":
    main()
