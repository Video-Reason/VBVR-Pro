"""Convert a DCP training checkpoint into a regular Diffusers pipeline directory.

This is a single-process export tool for inference/deployment. It loads the
base Wan I2V pipeline, applies a DCP checkpoint with the same code path used by
evaluation, optionally fuses LoRA adapters into the base weights, and writes a
plain ``save_pretrained`` Diffusers model.

Usage:
    .venv/bin/python -m src.cli.convert_dcp_to_diffusers
        --checkpoint storage/checkpoints/run/checkpoint-2000
        --output storage/models/run-checkpoint-2000-diffusers

    .venv/bin/python -m src.cli.convert_dcp_to_diffusers
        --checkpoint storage/checkpoints/run/checkpoint-2000 --output storage/models/run-2000
        --checkpoint storage/checkpoints/run/checkpoint-4000 --output storage/models/run-4000
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from loguru import logger

from src.trainer.checkpoint import load_dcp_into_pipeline

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="DCP checkpoint dir, flat or high/low layout. Repeat with --output for batch conversion.",
    )
    parser.add_argument(
        "--base_model",
        type=Path,
        default=Path("storage/models/Wan2.2-I2V-A14B-Diffusers"),
        help="Base Diffusers pipeline directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        required=True,
        help="Output Diffusers pipeline directory. Repeat once per --checkpoint.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="dtype used to load and save model weights",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used during conversion (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer EMA shadow weights when present",
    )
    parser.add_argument(
        "--merge_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse checkpoint LoRA adapters into plain base weights before saving",
    )
    parser.add_argument(
        "--safe_fusing",
        action="store_true",
        help="Ask Diffusers/PEFT to check for NaNs while fusing LoRA adapters",
    )
    parser.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save weights as safetensors",
    )
    parser.add_argument("--max_shard_size", default="10GB", help="Max shard size passed to save_pretrained")
    parser.add_argument(
        "--fastvideo_compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write configs/tokenizer files in a form current FastVideo versions can load",
    )
    return parser.parse_args()


def _resolve_device(value: str | None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_base_pipeline(args: argparse.Namespace, dtype: torch.dtype, device: torch.device) -> DiffusionPipeline:
    logger.info("Loading base pipeline from {} ({})", args.base_model, args.torch_dtype)
    # Place weights directly on the accelerator via device_map so the two 14B
    # transformers never accumulate in CPU RAM. Loading to CPU and then calling
    # pipe.to(device) peaks well above a typical container memory cgroup limit
    # and gets OOM-killed mid-load. device_map="cuda" (or "xpu"/etc.) is the
    # single-device strategy diffusers supports for this.
    device_map = device.type if device.type != "cpu" else None
    pipe = DiffusionPipeline.from_pretrained(
        str(args.base_model),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if device_map is None:
        pipe.to(device)
    gc.collect()
    return pipe


def _checkpoint_has_lora(checkpoint: Path) -> bool:
    return any(
        (checkpoint / rel / "adapter_config.json").exists()
        for rel in (
            "high/lora/transformer",
            "low/lora/transformer_2",
            "lora/transformer",
            "lora/transformer_2",
        )
    )


def _checkpoint_fully_overwrites_transformers(checkpoint: Path) -> bool:
    high = checkpoint / "high" / ".metadata"
    low = checkpoint / "low" / ".metadata"
    if high.exists() or low.exists():
        return high.exists() and low.exists()

    # Legacy flat checkpoints usually contain every trained module, but we
    # cannot cheaply know whether both experts are present without reading the
    # DCP. Treat them as requiring a clean base before conversion.
    return False


def _requires_clean_base(checkpoint: Path, merge_lora: bool) -> bool:
    return (merge_lora and _checkpoint_has_lora(checkpoint)) or not _checkpoint_fully_overwrites_transformers(
        checkpoint
    )


def _merge_lora_into_plain_weights(pipe: DiffusionPipeline, safe_fusing: bool) -> int:
    merged = 0
    for name in ("transformer", "transformer_2"):
        model = getattr(pipe, name, None)
        if model is None or not getattr(model, "peft_config", None):
            continue
        logger.info("Fusing LoRA adapters into {}", name)
        model.fuse_lora(safe_fusing=safe_fusing)
        model.unload_lora()
        merged += 1
    return merged


def _assert_finite_module(module: torch.nn.Module, module_name: str) -> None:
    for name, param in module.named_parameters():
        if not torch.is_floating_point(param):
            continue
        tensor = param.detach()
        finite = torch.isfinite(tensor)
        if finite.all().item():
            continue
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise FloatingPointError(
            f"{module_name}.{name} contains non-finite values after checkpoint load (nan={nan_count}, inf={inf_count})"
        )


def _validate_pipeline_weights_finite(pipe: DiffusionPipeline) -> None:
    for name in ("transformer", "transformer_2"):
        module = getattr(pipe, name, None)
        if module is not None:
            _assert_finite_module(module, name)


def _validate_outputs(outputs: list[Path]) -> None:
    for output in outputs:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Output path already exists and is not an empty directory: {output}")


def _copy_base_tokenizer_for_fastvideo(output: Path, base_model: Path) -> None:
    src = base_model / "tokenizer"
    dst = output / "tokenizer"
    if not src.is_dir():
        logger.warning("Base tokenizer directory does not exist at {}; skipping FastVideo tokenizer cleanup", src)
        return

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(item, target)
    logger.info("Copied base tokenizer files from {} to {}", src, dst)


def _strip_config_keys(config_path: Path, keys: tuple[str, ...], description: str) -> None:
    if not config_path.exists():
        logger.warning("{} was not written at {}; skipping FastVideo cleanup", description, config_path)
        return

    config = json.loads(config_path.read_text())
    removed = False
    for key in keys:
        if key in config:
            config.pop(key)
            removed = True

    if removed:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        logger.info("Wrote FastVideo-compatible {} at {}", description, config_path)


def _write_fastvideo_compatible_component_configs(output: Path) -> None:
    _strip_config_keys(output / "text_encoder" / "config.json", ("is_decoder",), "text_encoder config")
    for module_name in ("vae", "transformer", "transformer_2"):
        _strip_config_keys(
            output / module_name / "config.json",
            ("_diffusers_version", "_name_or_path"),
            f"{module_name} config",
        )


def _write_fastvideo_compatible_configs(output: Path, base_model: Path) -> None:
    _copy_base_tokenizer_for_fastvideo(output, base_model)
    _write_fastvideo_compatible_component_configs(output)

    model_index_path = output / "model_index.json"
    if not model_index_path.exists():
        logger.warning("model_index.json was not written at {}; skipping FastVideo compatibility cleanup", output)
    else:
        model_index = json.loads(model_index_path.read_text())
        removed = False
        for key in ("_name_or_path",):
            if key in model_index:
                model_index.pop(key)
                removed = True

        if model_index.get("_class_name") == "WanPipeline" and model_index.get("expand_timesteps"):
            # Current FastVideo picks its Wan config from model_index._class_name.
            # The TI2V 5B checkpoint still needs the image-to-video Wan config
            # while lmms_eval overrides the executable pipeline class to WanPipeline.
            model_index["_class_name"] = "WanImageToVideoPipeline"
            model_index.setdefault("image_encoder", [None, None])
            model_index.setdefault("image_processor", [None, None])
            removed = True

        if removed:
            model_index_path.write_text(json.dumps(model_index, indent=2) + "\n")
            logger.info("Wrote FastVideo-compatible model_index.json at {}", model_index_path)

    scheduler_config_path = output / "scheduler" / "scheduler_config.json"
    if not scheduler_config_path.exists():
        logger.warning(
            "scheduler_config.json was not written at {}; skipping FastVideo scheduler compatibility cleanup",
            scheduler_config_path,
        )
        return

    scheduler_config = json.loads(scheduler_config_path.read_text())
    removed = False
    for key in ("shift_terminal", "sigma_min", "sigma_max"):
        if key in scheduler_config:
            scheduler_config.pop(key)
            removed = True

    if removed:
        scheduler_config_path.write_text(json.dumps(scheduler_config, indent=2) + "\n")
        logger.info("Wrote FastVideo-compatible scheduler_config.json at {}", scheduler_config_path)


def _convert_one(pipe: DiffusionPipeline, args: argparse.Namespace, checkpoint: Path, output: Path) -> None:
    logger.info("Loading DCP checkpoint from {} (ema={})", checkpoint, args.use_ema)
    load_dcp_into_pipeline(pipe, str(checkpoint), use_ema=args.use_ema)
    if args.merge_lora:
        merged = _merge_lora_into_plain_weights(pipe, safe_fusing=args.safe_fusing)
        logger.info("Merged LoRA adapters into {} transformer module(s)", merged)
    else:
        logger.warning("--no-merge_lora keeps PEFT adapter layers in the output; this is not a plain model export.")
    _validate_pipeline_weights_finite(pipe)

    output.mkdir(parents=True, exist_ok=True)
    logger.info("Saving Diffusers pipeline to {}", output)
    pipe.save_pretrained(
        str(output),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    if args.fastvideo_compat:
        _write_fastvideo_compatible_configs(output, args.base_model)
    logger.info("Converted model is at {}", output)


def main() -> int:
    args = parse_args()
    if len(args.checkpoint) != len(args.output):
        raise ValueError(f"Expected one --output per --checkpoint, got {len(args.checkpoint)} and {len(args.output)}")

    dtype = DTYPES[args.torch_dtype]
    device = _resolve_device(args.device)
    _validate_outputs(args.output)

    pipe = _load_base_pipeline(args, dtype, device)
    pipe_is_clean_base = True

    for idx, (checkpoint, output) in enumerate(zip(args.checkpoint, args.output, strict=True), start=1):
        needs_clean_base = _requires_clean_base(checkpoint, merge_lora=args.merge_lora)
        if needs_clean_base and not pipe_is_clean_base:
            logger.info(
                "Reloading base pipeline before {} because the checkpoint is LoRA or partial-layout",
                checkpoint,
            )
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe = _load_base_pipeline(args, dtype, device)
            pipe_is_clean_base = True

        logger.info("[{}/{}] Converting {} -> {}", idx, len(args.checkpoint), checkpoint, output)
        _convert_one(pipe, args, checkpoint, output)
        pipe_is_clean_base = False

    logger.info("Done. Converted {} checkpoint(s).", len(args.checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
