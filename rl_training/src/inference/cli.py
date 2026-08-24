"""Command-line entry point for the unified ODE / SDE / CPS inference runner.

Examples::

    # precomputed latent sample, deterministic ODE, full per-step gallery
    .venv/bin/python -m src.inference \
        --config configs/<reviewed-rl-config>.yaml \
        --checkpoint storage/checkpoints/.../checkpoint-epoch3 \
        --mode ode --num_sampling_steps 50 --sample_index 0 \
        --output_dir storage/outputs/inf_ode_s0

    # raw image + prompt, stochastic DanceGRPO SDE
    .venv/bin/python -m src.inference \
        --model_path storage/models/Wan2.2-TI2V-5B-Diffusers \
        --checkpoint storage/checkpoints/.../checkpoint-epoch3 \
        --image first_frame.png --prompt "..." --mode sde \
        --output_dir storage/outputs/inf_img_sde
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.trainer.config import RLConfig, SFTConfig

from .config import InferenceConfig
from .engine import InferenceEngine, build_model
from .inputs import prepare_input
from .outputs import write_outputs


def _load_training_config(path: str) -> Any:
    cfg_dict = yaml.safe_load(Path(path).read_text()) or {}
    if cfg_dict.get("trainer") == "dancegrpo":
        return RLConfig(**cfg_dict)
    return SFTConfig(**cfg_dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.inference",
        description="Run ODE / SDE / CPS inference for any checkpoint and render per-step previews.",
    )

    g = p.add_argument_group("model / checkpoint")
    g.add_argument("--config", default=None, help="Training YAML (RLConfig/SFTConfig) to seed defaults")
    g.add_argument("--model_path", default=None, help="Model dir; required if --config is not given")
    g.add_argument("--checkpoint", default=None, help="DCP checkpoint dir (flat or high/low); omit for base model")
    g.add_argument("--weights", choices=["raw", "ema"], default=None, help="Which shadow to load (default: raw)")
    g.add_argument("--transformer_dtype", choices=["bfloat16", "float32"], default=None)
    g.add_argument("--device", default=None, help="Default cuda:0")

    g = p.add_argument_group("input source (exactly one)")
    g.add_argument("--latent_webdataset_dir", default=None, help="Dir of shard-*.tar latent samples")
    g.add_argument("--sample_index", type=int, default=None, help="Ordinal sample index (latent source)")
    g.add_argument("--image", default=None, help="First-frame image path (raw source)")
    g.add_argument("--prompt", default=None, help="Text prompt (required with --image)")
    g.add_argument("--height", type=int, default=None, help="Raw-image height (default 384)")
    g.add_argument("--width", type=int, default=None, help="Raw-image width (default 384)")
    g.add_argument("--num_frames", type=int, default=None, help="Raw-image pixel frame count (default 161)")

    g = p.add_argument_group("sampling")
    g.add_argument("--mode", choices=["ode", "sde", "cps"], default=None, help="Sampler (default sde)")
    g.add_argument("--num_sampling_steps", type=int, default=None, help="Denoising steps T (default 50)")
    g.add_argument("--noise_scale", type=float, default=None, help="eta(sde)/noise_level(cps); omit for mode default")
    g.add_argument("--cfg_scale", type=float, default=None, help="Classifier-free guidance scale (default 1.0)")
    g.add_argument("--sigma_min", type=float, default=None, help="flowgrpo-only passthrough")
    g.add_argument("--sigma_max", type=float, default=None, help="flowgrpo-only passthrough")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--batch_size", type=int, default=None, help="Rollouts per run (default 1)")
    g.add_argument(
        "--share_init_noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Share x_T across batch members (default on)",
    )

    g = p.add_argument_group("output")
    g.add_argument("--output_dir", required=True, help="Directory to write videos + manifest")
    g.add_argument("--fps", type=int, default=None, help="Output FPS (default 16)")
    g.add_argument(
        "--save_steps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode the per-step z0 preview gallery (default on)",
    )
    g.add_argument(
        "--save_reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode reference video latents when present (default on)",
    )
    g.add_argument("--grid_cols", type=int, default=None)
    g.add_argument("--grid_thumb_width", type=int, default=None)
    g.add_argument("--force", action="store_true", help="Overwrite an existing run (ignore manifest.json)")

    return p.parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Collect only the CLI args the user explicitly set (everything else stays None)."""
    overrides: dict[str, Any] = {}

    def setif(key: str, value: Any) -> None:
        if value is not None:
            overrides[key] = value

    setif("model_path", args.model_path)
    setif("checkpoint", args.checkpoint)
    if args.weights is not None:
        overrides["use_ema"] = args.weights == "ema"
    setif("transformer_dtype", args.transformer_dtype)
    setif("device", args.device)
    setif("latent_webdataset_dir", args.latent_webdataset_dir)
    setif("sample_index", args.sample_index)
    setif("image", args.image)
    setif("prompt", args.prompt)
    setif("height", args.height)
    setif("width", args.width)
    setif("num_frames", args.num_frames)
    setif("mode", args.mode)
    setif("num_sampling_steps", args.num_sampling_steps)
    setif("noise_scale", args.noise_scale)
    setif("cfg_scale", args.cfg_scale)
    setif("sigma_min", args.sigma_min)
    setif("sigma_max", args.sigma_max)
    setif("seed", args.seed)
    setif("batch_size", args.batch_size)
    setif("share_init_noise", args.share_init_noise)
    setif("output_dir", args.output_dir)
    setif("fps", args.fps)
    setif("save_steps", args.save_steps)
    setif("save_reference", args.save_reference)
    setif("grid_cols", args.grid_cols)
    setif("grid_thumb_width", args.grid_thumb_width)
    if args.force:
        overrides["force"] = True
    return overrides


def build_config(args: argparse.Namespace) -> InferenceConfig:
    overrides = _overrides_from_args(args)
    if args.config:
        train_cfg = _load_training_config(args.config)
        return InferenceConfig.from_training_config(train_cfg, **overrides)
    if "model_path" not in overrides:
        raise SystemExit("error: either --config or --model_path is required.")
    return InferenceConfig(**overrides)


def run(cfg: InferenceConfig) -> dict[str, Any]:
    """Build the model, sample, and render outputs for one resolved config."""
    out_dir = Path(cfg.output_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not cfg.force:
        print(f"[skip] {manifest_path} already exists (use --force to overwrite)", flush=True)
        return json.loads(manifest_path.read_text())

    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    started = time.time()
    print(
        f"[config] mode={cfg.mode} formula={cfg.sde_formula} noise={cfg.effective_noise_scale} "
        f"steps={cfg.num_sampling_steps} cfg={cfg.cfg_scale} seed={cfg.seed} batch={cfg.batch_size}",
        flush=True,
    )

    model = build_model(cfg, need_text_encoder=cfg.from_image)
    ckpt_msg = f" + checkpoint {cfg.checkpoint}" if cfg.checkpoint else " (base weights)"
    print(f"[model] {cfg.model_path}{ckpt_msg}", flush=True)

    prepared = prepare_input(model, cfg, device)
    print(f"[input] {prepared.source} :: {prepared.summary}", flush=True)

    result = InferenceEngine(model, cfg).sample(prepared)
    print(f"[sample] {len(result.pred_x0)} step previews captured", flush=True)

    manifest = write_outputs(model, cfg, prepared, result, out_dir, started=started)
    print(f"[done] wrote outputs to {out_dir} in {manifest.get('elapsed_seconds', 0.0):.1f}s", flush=True)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "finals": manifest["outputs"]["finals"],
                "grid": manifest.get("grid"),
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
