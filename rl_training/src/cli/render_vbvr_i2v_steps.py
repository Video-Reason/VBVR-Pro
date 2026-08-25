"""Render all clean-endpoint previews for formal VBVR I2V samples.

The CPS path calls the same training-time Flow-CPS engine as
``src.cli.eval_i2v_cps``.  Euler and UniPC call the same Diffusers pipeline as
``src.cli.eval_i2v_euler`` / ``src.cli.eval_i2v`` and intercept the scheduler
immediately before each real solver step.  Thus every preview has the same
meaning: the post-CFG flow clean endpoint ``x0 = x_t - sigma * v``.  For the
last displayed cell we decode the solver's actual final latent at sigma zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import decord
import numpy as np
import torch
from diffusers.utils import export_to_video
from PIL import Image

from src.cli import eval_i2v
from src.cli.eval_i2v_cps import _inference_config, _prepare_input
from src.cli.eval_i2v_euler import install_flowmatch_euler_scheduler
from src.inference.engine import InferenceEngine, build_model
from src.inference.inputs import PreparedInput
from src.inference.outputs import save_step_contact_sheet, save_step_grid_video, write_outputs


def _default_negative_prompt() -> str:
    return eval_i2v.parse_args(["--eval_json", "unused", "--output_dir", "unused"]).negative_prompt


def _noise_level(value: str) -> float:
    level = float(value)
    if not 0.0 <= level <= 1.0:
        raise argparse.ArgumentTypeError(f"CPS noise level must be in [0, 1], got {value}")
    return level


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--model_path", required=True, help="Converted Diffusers model directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sampler", required=True, choices=("cps", "euler", "unipc"))
    parser.add_argument("--noise_level", type=_noise_level, default=None, help="Required only for CPS")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample_index", type=int, default=None)
    selection.add_argument(
        "--all_samples",
        action="store_true",
        help="Render every eval sample while keeping the model loaded and resume complete sample directories",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit --all_samples to the first N eval items")
    parser.add_argument(
        "--sample_shard_count",
        type=int,
        default=1,
        help="With --all_samples, render indices where index %% count equals --sample_shard_index",
    )
    parser.add_argument("--sample_shard_index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", default=_default_negative_prompt())
    parser.add_argument(
        "--formal_final_video",
        default=None,
        help="Bind step T/final byte-for-byte to the video actually used by the quantitative evaluation",
    )
    parser.add_argument(
        "--formal_final_root",
        default=None,
        help="With --all_samples, derive each formal final MP4 beneath this quantitative generated-video root",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid_cols", type=int, default=6)
    parser.add_argument("--grid_thumb_width", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Strictly audit selected trajectory outputs and formal-final bindings without loading a model",
    )
    args = parser.parse_args(argv)
    if args.sampler == "cps" and args.noise_level is None:
        parser.error("--noise_level is required when --sampler=cps")
    if args.sampler != "cps" and args.noise_level is not None:
        parser.error("--noise_level is only valid when --sampler=cps")
    if args.all_samples:
        if args.formal_final_root is None:
            parser.error("--formal_final_root is required with --all_samples")
        if args.formal_final_video is not None:
            parser.error("--formal_final_video cannot be combined with --all_samples")
    else:
        args.sample_index = 0 if args.sample_index is None else args.sample_index
        if args.formal_final_root is not None:
            parser.error("--formal_final_root is only valid with --all_samples")
        if args.limit is not None:
            parser.error("--limit is only valid with --all_samples")
        if args.sample_shard_count != 1 or args.sample_shard_index != 0:
            parser.error("sample sharding is only valid with --all_samples")
    if args.sample_index is not None and args.sample_index < 0:
        parser.error("--sample_index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.sample_shard_count <= 0:
        parser.error("--sample_shard_count must be positive")
    if not 0 <= args.sample_shard_index < args.sample_shard_count:
        parser.error("--sample_shard_index must be in [0, --sample_shard_count)")
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive")
    return args


def _load_eval_samples(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    eval_args = SimpleNamespace(eval_json=args.eval_json, limit=args.limit if args.all_samples else None)
    _, base_dir, data = eval_i2v._load_eval_data(eval_args)
    return base_dir, data


def _prepare_sample(
    args: argparse.Namespace,
    *,
    data: list[dict[str, Any]],
    base_dir: Path,
    index: int,
) -> tuple[dict[str, Any], Image.Image, str, str]:
    if index >= len(data):
        raise IndexError(f"sample_index={index} is out of range for {len(data)} eval samples")
    item = data[index]
    name = eval_i2v._sample_name(item, index)

    raw_image = item.get("image")
    if isinstance(raw_image, list):
        raw_image = raw_image[0] if raw_image else None
    if raw_image:
        image = eval_i2v._load_image(eval_i2v._resolve_path(raw_image, base_dir))
    elif item.get("video"):
        image = eval_i2v._first_frame_from_video(eval_i2v._resolve_path(item["video"], base_dir))
    else:
        raise ValueError(f"Eval item {index} has no image or video")

    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Eval item {index} has no prompt")
    image = image.resize((args.width, args.height), Image.Resampling.LANCZOS)
    return item, image, prompt, name


def _load_sample(args: argparse.Namespace) -> tuple[dict[str, Any], Image.Image, str, str]:
    base_dir, data = _load_eval_samples(args)
    assert args.sample_index is not None
    return _prepare_sample(args, data=data, base_dir=base_dir, index=args.sample_index)


def _sample_output_dir(root: Path, name: str) -> Path:
    """Map the formal relative MP4 name to a same-shaped sample directory."""
    return eval_i2v._output_path(root, name).with_suffix("")


def _formal_final_path(root: Path, name: str) -> Path:
    return eval_i2v._output_path(root, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sampler_name(sampler: str) -> str:
    return {"cps": "flow_cps", "euler": "flowmatch_euler", "unipc": "unipc"}[sampler]


def _trajectory_validation_error(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    formal_final_video: Path | None,
    sample_index: int,
    sample_name: str,
) -> str | None:
    """Validate one resumable sample directory without decoding 30 videos."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return f"missing or empty manifest: {manifest_path}"
    required = [output_dir / f"step_{index:02d}.mp4" for index in range(args.num_inference_steps)]
    required += [
        output_dir / "final_00.mp4",
        output_dir / "steps_grid.mp4",
        output_dir / "step_contact_sheet.jpg",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return f"missing or empty artifacts: {missing[:5]}"
    expected_mp4_names = {
        *(f"step_{index:02d}.mp4" for index in range(args.num_inference_steps)),
        "final_00.mp4",
        "steps_grid.mp4",
    }
    actual_mp4_names = {path.name for path in output_dir.glob("*.mp4") if path.is_file()}
    if actual_mp4_names != expected_mp4_names:
        extra = sorted(actual_mp4_names - expected_mp4_names)
        return f"unexpected MP4 artifacts: {extra[:5]}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid manifest: {exc}"

    expected: dict[str, Any] = {
        "sample_index": sample_index,
        "sample_name": sample_name,
        "sampler": _expected_sampler_name(args.sampler),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed + sample_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if key == "num_inference_steps" and actual is None:
            actual = manifest.get("num_sampling_steps")
        if actual != value:
            return f"manifest {key}={actual!r}, expected {value!r}"
    if args.sampler == "cps":
        actual_noise_level = manifest.get("noise_level", manifest.get("noise_scale"))
        if actual_noise_level != args.noise_level:
            return f"manifest CPS noise level={actual_noise_level!r}, expected {args.noise_level!r}"
    try:
        recorded_model = Path(manifest["model_path"]).resolve()
    except Exception as exc:
        return f"invalid manifest model_path: {exc}"
    if recorded_model != Path(args.model_path).resolve():
        return f"manifest model_path={recorded_model}, expected {Path(args.model_path).resolve()}"
    previews = manifest.get("step_previews")
    if not isinstance(previews, list) or len(previews) != args.num_inference_steps:
        return f"manifest has {len(previews) if isinstance(previews, list) else 'invalid'} step previews"

    if formal_final_video is None:
        return None
    if not formal_final_video.is_file() or formal_final_video.stat().st_size == 0:
        return f"formal final is missing or empty: {formal_final_video}"
    binding = manifest.get("formal_final_binding")
    if not isinstance(binding, dict):
        return "formal_final_binding is missing"
    source = str(formal_final_video.resolve())
    if binding.get("source") != source:
        return f"formal binding source={binding.get('source')!r}, expected {source!r}"
    digest = _sha256(formal_final_video)
    if binding.get("sha256") != digest:
        return "formal binding digest does not match the quantitative MP4"
    for bound in (output_dir / "final_00.mp4", output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4"):
        if _sha256(bound) != digest:
            return f"bound output digest mismatch: {bound}"
    return None


def _cps_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        checkpoint=None,
        use_ema=False,
        output_dir=args.output_dir,
        noise_level=args.noise_level,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=args.seed,
    )


def _load_cps_model(args: argparse.Namespace, device: torch.device):
    assert args.sample_index is not None
    return build_model(
        _inference_config(_cps_args(args), device, seed=args.seed + args.sample_index),
        need_text_encoder=True,
    )


def _render_cps(
    args: argparse.Namespace,
    *,
    item: dict[str, Any],
    image: Image.Image,
    prompt: str,
    name: str,
    device: torch.device,
    model: Any | None = None,
) -> dict[str, Any]:
    assert args.sample_index is not None
    started = time.time()
    cps_args = _cps_args(args)
    sample_seed = args.seed + args.sample_index
    cfg = _inference_config(cps_args, device, seed=sample_seed).model_copy(
        update={
            "image": item.get("image") or item.get("video") or "eval-item",
            "prompt": prompt,
            "save_steps": True,
            "save_reference": False,
            "grid_cols": args.grid_cols,
            "grid_thumb_width": args.grid_thumb_width,
            "force": args.force,
        }
    )
    model = model if model is not None else build_model(cfg, need_text_encoder=True)
    prepared_base = _prepare_input(model, image, prompt, cps_args, device)
    prepared = PreparedInput(
        condition=prepared_base.condition,
        prompt_embeds=prepared_base.prompt_embeds,
        source=f"vbvr-eval-json:{name}",
        metadata={"eval_json": args.eval_json, "sample_index": args.sample_index, "name": name},
        summary={"name": name, "sample_index": args.sample_index, "prompt": prompt},
    )
    result = InferenceEngine(model, cfg).sample(prepared)
    if len(result.pred_x0) != args.num_inference_steps:
        raise RuntimeError(f"Captured {len(result.pred_x0)} CPS previews, expected {args.num_inference_steps}")
    formal_final_video = _formal_final_frames(args)
    manifest = write_outputs(
        model,
        cfg,
        prepared,
        result,
        Path(args.output_dir),
        started=started,
        final_video_override=formal_final_video,
    )
    formal_binding = _bind_formal_final(args, Path(args.output_dir))
    manifest.update(
        {
            "sampler": "flow_cps",
            "sampler_implementation": "src.cli.eval_i2v_cps / WanI2VForTraining.sde_generate",
            "eval_json": str(Path(args.eval_json).resolve()),
            "sample_index": args.sample_index,
            "sample_name": name,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": sample_seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "preview_contract": "post-CFG x0=x_t-sigma*v; final cell is actual solver latent at sigma=0",
            "formal_final_binding": formal_binding,
        }
    )
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class _OdeStepRecorder:
    """Intercept a Diffusers scheduler without changing its numerical step."""

    def __init__(self, pipe: Any):
        self.pipe = pipe
        self.pred_x0: list[torch.Tensor] = []
        self.sigmas: list[float] = []
        self.timesteps: list[float] = []
        self.condition: torch.Tensor | None = None
        self.first_frame_mask: torch.Tensor | None = None
        self._prepare_latents = pipe.prepare_latents
        self._scheduler_step = pipe.scheduler.step

    def install(self) -> None:
        def prepare_latents(*args: Any, **kwargs: Any):
            outputs = self._prepare_latents(*args, **kwargs)
            if len(outputs) == 3:
                # Keep the recorder's copies on CPU. Allocating a full x0 tensor
                # on CUDA before the real scheduler step changes allocator
                # pressure and can perturb otherwise bit-exact fused kernels.
                _, condition, first_frame_mask = outputs
                self.condition = condition.detach().cpu().to(torch.float32)
                self.first_frame_mask = first_frame_mask.detach().cpu().to(torch.float32)
            return outputs

        def scheduler_step(model_output: torch.Tensor, timestep: Any, sample: torch.Tensor, *args: Any, **kwargs: Any):
            scheduler = self.pipe.scheduler
            prediction_type = getattr(scheduler.config, "prediction_type", "flow_prediction")
            if prediction_type != "flow_prediction":
                raise ValueError(f"Step renderer requires flow_prediction, got {prediction_type!r}")

            # Execute the untouched solver first. Only after it has produced the
            # real next state do we copy its immutable inputs to CPU and form x0.
            # This makes the recorder observational with respect to the solver.
            output = self._scheduler_step(model_output, timestep, sample, *args, **kwargs)
            step_index = int(scheduler.step_index) - 1
            sigma = float(scheduler.sigmas[step_index].detach().cpu().item())
            pred_x0 = sample.detach().cpu().to(torch.float32) - sigma * model_output.detach().cpu().to(torch.float32)
            if self.condition is not None and self.first_frame_mask is not None:
                pred_x0 = (1.0 - self.first_frame_mask) * self.condition + self.first_frame_mask * pred_x0

            self.pred_x0.append(pred_x0)
            self.sigmas.append(sigma)
            self.timesteps.append(float(timestep.detach().cpu().item() if torch.is_tensor(timestep) else timestep))
            return output

        self.pipe.prepare_latents = prepare_latents
        self.pipe.scheduler.step = scheduler_step

    def restore(self) -> None:
        self.pipe.prepare_latents = self._prepare_latents
        self.pipe.scheduler.step = self._scheduler_step


@torch.no_grad()
def _decode_diffusers_latent(pipe: Any, latent: torch.Tensor, device: torch.device) -> np.ndarray:
    latents = latent.to(device=device, dtype=pipe.vae.dtype)
    mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device=latents.device, dtype=latents.dtype)
    )
    inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    decoded = pipe.vae.decode(latents / inv_std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(decoded, output_type="np")[0]


def _as_uint8(video: np.ndarray) -> np.ndarray:
    return (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)


def _formal_final_frames(args: argparse.Namespace) -> np.ndarray | None:
    if args.formal_final_video is None:
        return None
    path = Path(args.formal_final_video)
    reason = eval_i2v._video_validation_error(
        path,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
    )
    if reason is not None:
        raise RuntimeError(f"Formal final video is invalid ({reason}): {path}")
    reader = decord.VideoReader(str(path), num_threads=1)
    frames = reader.get_batch(range(args.num_frames))
    # decord2 imports under the legacy ``decord`` name but returns a torch-like
    # tensor exposing ``numpy()``; classic decord exposes ``asnumpy()``.
    return frames.asnumpy() if hasattr(frames, "asnumpy") else frames.numpy()


def _bind_formal_final(args: argparse.Namespace, output_dir: Path) -> dict[str, Any] | None:
    if args.formal_final_video is None:
        return None
    source = Path(args.formal_final_video).resolve()
    final_path = output_dir / "final_00.mp4"
    final_step_path = output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4"
    shutil.copyfile(source, final_path)
    shutil.copyfile(source, final_step_path)
    digest = _sha256(source)
    return {
        "source": str(source),
        "sha256": digest,
        "bound_outputs": [str(final_path), str(final_step_path)],
    }


def _render_ode(
    args: argparse.Namespace,
    *,
    image: Image.Image,
    prompt: str,
    name: str,
    device: torch.device,
    pipe: Any | None = None,
) -> dict[str, Any]:
    assert args.sample_index is not None
    if pipe is None:
        pipe = _load_ode_pipeline(args, device)

    recorder = _OdeStepRecorder(pipe)
    recorder.install()
    try:
        generator = torch.Generator(device=device).manual_seed(args.seed + args.sample_index)
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        )
    finally:
        recorder.restore()

    if len(recorder.pred_x0) != args.num_inference_steps:
        raise RuntimeError(f"Captured {len(recorder.pred_x0)} ODE previews, expected {args.num_inference_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_final_video = np.asarray(output.frames[0])
    formal_final_video = _formal_final_frames(args)
    final_display_video = formal_final_video if formal_final_video is not None else _as_uint8(generated_final_video)
    final_path = output_dir / "final_00.mp4"
    if formal_final_video is None:
        export_to_video(list(generated_final_video), str(final_path), fps=args.fps)
    else:
        shutil.copyfile(Path(args.formal_final_video), final_path)

    videos: list[np.ndarray] = []
    previews: list[dict[str, Any]] = []
    labels: list[str] = []
    for index, pred_x0 in enumerate(recorder.pred_x0):
        is_final = index == args.num_inference_steps - 1
        step_path = output_dir / f"step_{index:02d}.mp4"
        if is_final:
            video_uint8 = final_display_video
            if formal_final_video is None:
                export_to_video(list(generated_final_video), str(step_path), fps=args.fps)
            else:
                shutil.copyfile(Path(args.formal_final_video), step_path)
        else:
            video = _decode_diffusers_latent(pipe, pred_x0, device)
            export_to_video(list(video), str(step_path), fps=args.fps)
            video_uint8 = _as_uint8(video)
        videos.append(video_uint8)
        label = (
            f"{index + 1:02d}/{args.num_inference_steps:02d} final s=0"
            if is_final
            else f"{index + 1:02d}/{args.num_inference_steps:02d} x0 s={recorder.sigmas[index]:.3f}"
        )
        labels.append(label)
        previews.append(
            {
                "display_step": index + 1,
                "file_index": index,
                "kind": "final_latent" if is_final else "predicted_clean_x0",
                "source_sigma": recorder.sigmas[index],
                "source_timestep": recorder.timesteps[index],
                "output_sigma": 0.0,
                "file": str(step_path),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grid_path = output_dir / "steps_grid.mp4"
    contact_path = output_dir / "step_contact_sheet.jpg"
    save_step_grid_video(
        videos,
        grid_path,
        fps=args.fps,
        cols=args.grid_cols,
        thumb_width=args.grid_thumb_width,
        step_labels=labels,
    )
    save_step_contact_sheet(videos, contact_path, step_labels=labels)

    manifest: dict[str, Any] = {
        "model_path": str(Path(args.model_path).resolve()),
        "eval_json": str(Path(args.eval_json).resolve()),
        "sample_index": args.sample_index,
        "sample_name": name,
        "sampler": "flowmatch_euler" if args.sampler == "euler" else "unipc",
        "sampler_implementation": (
            "src.cli.eval_i2v_euler / FlowMatchEulerDiscreteScheduler"
            if args.sampler == "euler"
            else "src.cli.eval_i2v / UniPCMultistepScheduler"
        ),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed + args.sample_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "sigmas": [*recorder.sigmas, 0.0],
        "timesteps": recorder.timesteps,
        "preview_contract": "post-CFG x0=x_t-sigma*v; final cell is actual solver latent at sigma=0",
        "step_preview_semantics": (
            "Steps 1..T-1 decode scheduler clean endpoints before the real solver step; step T decodes the actual "
            "final solver latent. TI2V-5B frame zero is pinned with the pipeline condition/mask in every preview."
        ),
        "step_previews": previews,
        "outputs": {
            "steps": [entry["file"] for entry in previews],
            "finals": [str(final_path)],
        },
        "grid": str(grid_path),
        "contact_sheet": str(contact_path),
        "formal_final_binding": (
            {
                "source": str(Path(args.formal_final_video).resolve()),
                "sha256": hashlib.sha256(Path(args.formal_final_video).read_bytes()).hexdigest(),
                "bound_outputs": [str(final_path), str(output_dir / f"step_{args.num_inference_steps - 1:02d}.mp4")],
            }
            if args.formal_final_video is not None
            else None
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gc.collect()
    return manifest


def _load_ode_pipeline(args: argparse.Namespace, device: torch.device):
    load_args = SimpleNamespace(
        model_path=args.model_path,
        checkpoint=None,
        use_ema=False,
        disable_progress_bar=True,
    )
    pipe = eval_i2v._load_pipeline(load_args, device, rank=0)
    if args.sampler == "euler":
        install_flowmatch_euler_scheduler(pipe)
    return pipe


def _clean_render_artifacts(output_dir: Path) -> None:
    """Remove only artifacts owned by this renderer before regenerating a sample."""
    if not output_dir.is_dir():
        return
    owned = [*output_dir.glob("step_*.mp4"), *output_dir.glob("final_*.mp4")]
    owned += [
        output_dir / "steps_grid.mp4",
        output_dir / "step_contact_sheet.jpg",
        output_dir / "manifest.json",
        output_dir / "sample_metadata.json",
    ]
    for path in owned:
        path.unlink(missing_ok=True)


def _configure_device(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    return device


def _sample_args(
    args: argparse.Namespace,
    *,
    index: int,
    output_dir: Path,
    formal_final_video: Path | None,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "sample_index": index,
            "all_samples": False,
            "limit": None,
            "output_dir": str(output_dir),
            "formal_final_video": str(formal_final_video) if formal_final_video is not None else None,
            "formal_final_root": None,
            "validate_only": False,
        }
    )
    return argparse.Namespace(**values)


def _write_cell_manifest(
    args: argparse.Namespace,
    *,
    output_root: Path,
    sample_count: int,
    selected_count: int,
    completed_count: int,
    initial_completed_count: int,
    started_at_unix: float,
) -> None:
    common = {
        "model_path": str(Path(args.model_path).resolve()),
        "eval_json": str(Path(args.eval_json).resolve()),
        "formal_final_root": str(Path(args.formal_final_root).resolve()),
        "sampler": args.sampler,
        "noise_level": args.noise_level,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "files_per_sample": args.num_inference_steps + 4,
        "started_at_unix": started_at_unix,
        "updated_at_unix": time.time(),
    }
    if args.sample_shard_count == 1:
        payload = {
            **common,
            "state": "complete" if completed_count == selected_count else "in_progress",
            "sample_count": sample_count,
            "completed_count": completed_count,
            "initial_completed_count": initial_completed_count,
        }
        destination = output_root / "cell_manifest.json"
    else:
        payload = {
            **common,
            "state": "complete" if completed_count == selected_count else "in_progress",
            "global_sample_count": sample_count,
            "sample_shard_count": args.sample_shard_count,
            "sample_shard_index": args.sample_shard_index,
            "selected_sample_count": selected_count,
            "completed_selected_count": completed_count,
            "initial_completed_selected_count": initial_completed_count,
        }
        destination = output_root / (
            f"cell_manifest.shard-{args.sample_shard_index:03d}-of-{args.sample_shard_count:03d}.json"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{destination.name}.tmp-{time.time_ns()}.json"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _render_all_samples(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    formal_root = Path(args.formal_final_root)
    if not formal_root.is_dir():
        raise FileNotFoundError(f"Formal quantitative video root does not exist: {formal_root}")
    base_dir, data = _load_eval_samples(args)
    selected = [
        (index, item) for index, item in enumerate(data) if index % args.sample_shard_count == args.sample_shard_index
    ]
    tasks: list[tuple[int, dict[str, Any], str, Path, Path]] = []
    errors: list[str] = []
    completed = 0
    for index, item in selected:
        name = eval_i2v._sample_name(item, index)
        sample_dir = _sample_output_dir(output_root, name)
        formal = _formal_final_path(formal_root, name)
        error = _trajectory_validation_error(
            args,
            output_dir=sample_dir,
            formal_final_video=formal,
            sample_index=index,
            sample_name=name,
        )
        if error is None and not args.force:
            completed += 1
            continue
        if args.validate_only:
            errors.append(f"{index}:{name}: {error or 'forced regeneration requested'}")
        else:
            tasks.append((index, item, name, sample_dir, formal))

    missing_formals = [
        str(formal) for _, _, _, _, formal in tasks if not formal.is_file() or formal.stat().st_size == 0
    ]
    if missing_formals and not args.validate_only:
        raise RuntimeError(
            f"Formal quantitative outputs are incomplete ({len(missing_formals)} missing/empty); "
            f"first entries: {missing_formals[:5]}"
        )

    if args.validate_only:
        if errors:
            detail = "\n  - ".join(errors[:20])
            suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
            raise RuntimeError(
                f"Trajectory validation failed for {len(errors)}/{len(selected)} selected samples:\n"
                f"  - {detail}{suffix}"
            )
        cell_started = time.time()
        _write_cell_manifest(
            args,
            output_root=output_root,
            sample_count=len(data),
            selected_count=len(selected),
            completed_count=len(selected),
            initial_completed_count=len(selected),
            started_at_unix=cell_started,
        )
        print(
            f"Validated {len(selected)} complete 30-step trajectories for shard "
            f"{args.sample_shard_index}/{args.sample_shard_count} in {output_root}",
            flush=True,
        )
        return 0

    cell_started = time.time()
    initial_completed = completed
    print(
        f"[cell] sampler={args.sampler} samples={len(data)} shard="
        f"{args.sample_shard_index}/{args.sample_shard_count} selected={len(selected)} "
        f"complete={completed} pending={len(tasks)} output={output_root}",
        flush=True,
    )
    _write_cell_manifest(
        args,
        output_root=output_root,
        sample_count=len(data),
        selected_count=len(selected),
        completed_count=completed,
        initial_completed_count=initial_completed,
        started_at_unix=cell_started,
    )
    if not tasks:
        print(f"[done] all {len(selected)} selected trajectories already complete", flush=True)
        return 0

    device = _configure_device(args)
    first_index, _, _, first_output, first_formal = tasks[0]
    runtime_args = _sample_args(
        args,
        index=first_index,
        output_dir=first_output,
        formal_final_video=first_formal,
    )
    if args.sampler == "cps":
        model = _load_cps_model(runtime_args, device)
        pipe = None
    else:
        model = None
        pipe = _load_ode_pipeline(runtime_args, device)

    for ordinal, (index, item, name, sample_dir, formal) in enumerate(tasks, start=1):
        sample_args = _sample_args(
            args,
            index=index,
            output_dir=sample_dir,
            formal_final_video=formal,
        )
        _, image, prompt, prepared_name = _prepare_sample(
            sample_args,
            data=data,
            base_dir=base_dir,
            index=index,
        )
        assert prepared_name == name
        print(
            f"[render] [{ordinal}/{len(tasks)}] sample={index}:{name} sampler={args.sampler} seed={args.seed + index}",
            flush=True,
        )
        sample_started = time.time()
        _clean_render_artifacts(sample_dir)
        if args.sampler == "cps":
            _render_cps(
                sample_args,
                item=item,
                image=image,
                prompt=prompt,
                name=name,
                device=device,
                model=model,
            )
        else:
            _render_ode(
                sample_args,
                image=image,
                prompt=prompt,
                name=name,
                device=device,
                pipe=pipe,
            )
        error = _trajectory_validation_error(
            sample_args,
            output_dir=sample_dir,
            formal_final_video=formal,
            sample_index=index,
            sample_name=name,
        )
        if error is not None:
            raise RuntimeError(f"Rendered trajectory failed validation for {index}:{name}: {error}")
        completed += 1
        _write_cell_manifest(
            args,
            output_root=output_root,
            sample_count=len(data),
            selected_count=len(selected),
            completed_count=completed,
            initial_completed_count=initial_completed,
            started_at_unix=cell_started,
        )
        print(
            f"[saved] shard={args.sample_shard_index}/{args.sample_shard_count} "
            f"[{completed}/{len(selected)}] global_sample={index} {sample_dir} "
            f"elapsed={time.time() - sample_started:.2f}s",
            flush=True,
        )
        del image
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_cell_manifest(
        args,
        output_root=output_root,
        sample_count=len(data),
        selected_count=len(selected),
        completed_count=len(selected),
        initial_completed_count=initial_completed,
        started_at_unix=cell_started,
    )
    print(
        f"[done] rendered and validated shard {args.sample_shard_index}/{args.sample_shard_count}: "
        f"{len(selected)} trajectories in {time.time() - cell_started:.2f}s",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_samples:
        return _render_all_samples(args)

    output_dir = Path(args.output_dir)
    formal_final = Path(args.formal_final_video) if args.formal_final_video is not None else None
    item, image, prompt, name = _load_sample(args)
    assert args.sample_index is not None
    validation_error = _trajectory_validation_error(
        args,
        output_dir=output_dir,
        formal_final_video=formal_final,
        sample_index=args.sample_index,
        sample_name=name,
    )
    if args.validate_only:
        if validation_error is not None:
            raise RuntimeError(f"Trajectory validation failed: {validation_error}")
        print(f"Validated complete 30-step trajectory in {output_dir}", flush=True)
        return 0
    if validation_error is None and not args.force:
        print(f"[skip] validated complete trajectory: {output_dir}", flush=True)
        return 0

    device = _configure_device(args)
    _clean_render_artifacts(output_dir)
    print(
        f"[render] sampler={args.sampler} sample={args.sample_index}:{name} "
        f"steps={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed + args.sample_index}",
        flush=True,
    )
    started = time.time()
    if args.sampler == "cps":
        model = _load_cps_model(args, device)
        manifest = _render_cps(
            args,
            item=item,
            image=image,
            prompt=prompt,
            name=name,
            device=device,
            model=model,
        )
    else:
        pipe = _load_ode_pipeline(args, device)
        manifest = _render_ode(args, image=image, prompt=prompt, name=name, device=device, pipe=pipe)
    validation_error = _trajectory_validation_error(
        args,
        output_dir=output_dir,
        formal_final_video=formal_final,
        sample_index=args.sample_index,
        sample_name=name,
    )
    if validation_error is not None:
        raise RuntimeError(f"Rendered trajectory failed validation: {validation_error}")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "grid": manifest.get("grid"),
                "contact_sheet": manifest.get("contact_sheet"),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
