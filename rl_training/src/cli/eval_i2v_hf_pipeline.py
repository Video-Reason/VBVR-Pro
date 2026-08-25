"""Generate VBVR-Pro videos with a reviewed Hugging Face custom pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import re
from pathlib import Path
from typing import Any

import torch

from src.cli import eval_i2v as _base

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _pipeline_sha256(value: str) -> str:
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError(f"pipeline SHA-256 must be 64 lowercase hex characters, got {value!r}")
    return normalized


def _cps_eta(value: str) -> float:
    eta = float(value)
    if not 0.0 <= eta <= 1.0:
        raise argparse.ArgumentTypeError(f"CPS coefficient must be in [0, 1], got {value!r}")
    return eta


def build_parser() -> argparse.ArgumentParser:
    parser = _base.build_parser()
    parser.description = __doc__
    parser.add_argument("--sampler", choices=("cps", "euler", "unipc"), required=True)
    parser.add_argument("--cps_eta", type=_cps_eta)
    parser.add_argument("--pipeline_sha256", type=_pipeline_sha256, required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sampler == "cps" and args.cps_eta is None:
        parser.error("--cps_eta is required with --sampler cps")
    if args.sampler != "cps" and args.cps_eta is not None:
        parser.error("--cps_eta only applies to --sampler cps")
    if args.checkpoint is not None or args.use_ema:
        parser.error("the Hugging Face custom-pipeline backend accepts a complete Diffusers snapshot, not DCP input")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pipeline_source(model_path: str | Path, expected_sha256: str) -> Path:
    root = Path(model_path).expanduser().resolve()
    pipeline_path = root / "pipeline.py"
    if not pipeline_path.is_file():
        raise ValueError(f"reviewed custom pipeline is missing: {pipeline_path}")
    actual_sha256 = _sha256_file(pipeline_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "reviewed pipeline SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256} path={pipeline_path}"
        )
    return pipeline_path


def _load_pipeline(args: argparse.Namespace, device: torch.device, rank: int) -> Any:
    from diffusers import AutoencoderKLWan, DiffusionPipeline

    model_path = Path(args.model_path).expanduser().resolve()
    pipeline_path = verify_pipeline_source(model_path, args.pipeline_sha256)
    if rank == 0:
        print(
            f"Loading reviewed Hugging Face pipeline from {model_path} (pipeline_sha256={args.pipeline_sha256}) ...",
            flush=True,
        )

    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        custom_pipeline=str(pipeline_path),
        trust_remote_code=True,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if not callable(getattr(pipe, "set_sampler", None)):
        raise TypeError(f"reviewed pipeline does not expose set_sampler(): {type(pipe).__name__}")
    pipe.set_progress_bar_config(disable=args.disable_progress_bar)
    pipe.to(device)
    gc.collect()
    return pipe


def _pipeline_call_kwargs(args: argparse.Namespace, generator: torch.Generator) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "generator": generator,
        "sampler": args.sampler,
    }
    if args.sampler == "cps":
        kwargs["cps_eta"] = args.cps_eta
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    original_loader = _base._load_pipeline
    original_call_kwargs = _base._pipeline_call_kwargs
    _base._load_pipeline = _load_pipeline
    _base._pipeline_call_kwargs = _pipeline_call_kwargs
    try:
        return _base.run(args)
    finally:
        _base._load_pipeline = original_loader
        _base._pipeline_call_kwargs = original_call_kwargs


if __name__ == "__main__":
    raise SystemExit(main())
