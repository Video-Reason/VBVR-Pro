#!/usr/bin/env python3
"""Unified single-example inference for every VBVR-Pro release.

The selected backend is inferred from ``--model_path`` by default.  Imports are
lazy so that each model family can run in its own compatible Python environment.
Run ``python example.py --list-models`` to see the supported model types.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    description: str
    backend: str
    needs_image: bool
    output_kind: str
    default_steps: int


MODEL_SPECS = {
    "bagel": ModelSpec("VBVR-Pro BAGEL", "BAGEL", True, "interleaved", 50),
    "thinkmorph": ModelSpec(
        "VBVR-Pro ThinkMorph", "ThinkMorph", True, "interleaved", 50
    ),
    "sensenova-u1": ModelSpec(
        "VBVR-Pro SenseNova-U1 / Neo-Unify", "Transformers", True, "images", 50
    ),
    "flux2": ModelSpec("VBVR-Pro FLUX.2-dev", "Diffusers", True, "image", 50),
    "qwen-image-edit": ModelSpec(
        "VBVR-Pro Qwen-Image-Edit", "Diffusers", True, "image", 40
    ),
    "ltx2.3": ModelSpec(
        "VBVR-Pro LTX-2.3 (merged)", "Diffusers", True, "audio-video", 40
    ),
    "wan2.1-i2v-14b": ModelSpec(
        "VBVR-Pro Wan2.1-I2V-14B", "Diffusers", True, "video", 50
    ),
    "wan2.2-i2v-a14b": ModelSpec(
        "VBVR-Pro Wan2.2-I2V-A14B", "Diffusers", True, "video", 50
    ),
    "wan2.2-ti2v-5b": ModelSpec(
        "VBVR-Pro Wan2.2-TI2V-5B", "Diffusers", False, "video", 50
    ),
    "wan2.2-ti2v-5b-qwen-judge-rl": ModelSpec(
        "VBVR-Pro Wan2.2-TI2V-5B Qwen-Judge-RL",
        "Diffusers",
        True,
        "video",
        30,
    ),
    "wan2.2-ti2v-5b-rule-rl": ModelSpec(
        "VBVR-Pro Wan2.2-TI2V-5B Rule-RL",
        "Diffusers",
        True,
        "video",
        30,
    ),
    "flux2-diffsynth": ModelSpec(
        "VBVR-Pro FLUX.2-dev DiffSynth LoRA", "DiffSynth", True, "image", 50
    ),
    "qwen-image-edit-diffsynth": ModelSpec(
        "VBVR-Pro Qwen-Image-Edit DiffSynth LoRA",
        "DiffSynth",
        True,
        "image",
        40,
    ),
    "ltx2.3-diffsynth": ModelSpec(
        "VBVR-Pro LTX-2.3 DiffSynth LoRA",
        "DiffSynth",
        True,
        "audio-video",
        40,
    ),
    "wan2.1-i2v-14b-diffsynth": ModelSpec(
        "VBVR-Pro Wan2.1-I2V-14B DiffSynth LoRA",
        "DiffSynth",
        True,
        "video",
        50,
    ),
    "wan2.2-i2v-a14b-diffsynth": ModelSpec(
        "VBVR-Pro Wan2.2-I2V-A14B DiffSynth LoRAs",
        "DiffSynth",
        True,
        "video",
        50,
    ),
    "wan2.2-ti2v-5b-diffsynth": ModelSpec(
        "VBVR-Pro Wan2.2-TI2V-5B DiffSynth LoRA",
        "DiffSynth",
        False,
        "video",
        50,
    ),
}


# Check longer/more-specific names first.
MODEL_NAME_MARKERS = (
    (
        "wan2.2-ti2v-5b-qwen-judge-rl",
        "wan2.2-ti2v-5b-qwen-judge-rl",
    ),
    ("wan2.2-ti2v-5b-rule-rl", "wan2.2-ti2v-5b-rule-rl"),
    ("wan2.2-i2v-a14b-diffsynth", "wan2.2-i2v-a14b-diffsynth"),
    ("wan2.2-ti2v-5b-diffsynth", "wan2.2-ti2v-5b-diffsynth"),
    ("wan2.1-i2v-14b-diffsynth", "wan2.1-i2v-14b-diffsynth"),
    ("qwen-image-edit-diffsynth", "qwen-image-edit-diffsynth"),
    ("flux2-dev-diffsynth", "flux2-diffsynth"),
    ("ltx2.3-diffsynth", "ltx2.3-diffsynth"),
    ("wan2.2-i2v-a14b", "wan2.2-i2v-a14b"),
    ("wan2.2-ti2v-5b", "wan2.2-ti2v-5b"),
    ("wan2.1-i2v-14b", "wan2.1-i2v-14b"),
    ("qwen-image-edit", "qwen-image-edit"),
    ("sensenova-u1", "sensenova-u1"),
    ("thinkmorph", "thinkmorph"),
    ("flux2-dev", "flux2"),
    ("ltx2.3", "ltx2.3"),
    ("bagel", "bagel"),
)

WAN_RL_MODEL_TYPES = {
    "wan2.2-ti2v-5b-qwen-judge-rl",
    "wan2.2-ti2v-5b-rule-rl",
}
WAN_RL_PAPER_SAMPLERS = (
    "cps-0.1",
    "cps-0.3",
    "cps-0.7",
    "cps-0.9",
    "euler",
    "unipc",
)


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, low quality, "
    "motion blur, distorted, artifacts"
)
LTX_NEGATIVE_PROMPT = "blurry, low quality, flickering, motion blur, distorted"
THINKMORPH_SYSTEM_PROMPT = (
    "Let's think step by step to answer the question. For text-based thinking, "
    "enclose the process within <think> </think>. For visual thinking, enclose "
    "the content within <image_start> </image_end>. Finally conclude with the "
    "final answer wrapped in <answer> </answer>"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--list-models", action="store_true", help="List model types and exit.")
    parser.add_argument(
        "--model_path",
        "--model-path",
        dest="model_path",
        help="Local model directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--model_type",
        "--model-type",
        dest="model_type",
        default="auto",
        choices=["auto", *MODEL_SPECS],
        help="Backend override; auto detects from the model path/repository name.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt", "--question", dest="prompt", help="Editing/generation instruction."
    )
    prompt_group.add_argument(
        "--prompt_file",
        "--prompt-file",
        dest="prompt_file",
        type=Path,
        help="UTF-8 text file containing the instruction.",
    )
    parser.add_argument(
        "--image_paths",
        "--image-paths",
        "--image",
        dest="image_paths",
        nargs="+",
        default=[],
        type=Path,
        help="Input image path(s); Qwen Image Edit accepts more than one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image/video path, or a directory for multiple images.",
    )
    parser.add_argument(
        "--negative_prompt",
        "--negative-prompt",
        dest="negative_prompt",
        help="Negative prompt for backends that support one.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Random seed (defaults to 1 for BAGEL, 0 for Wan2.2 TI2V RL, "
            "and 42 for all other models)."
        ),
    )
    parser.add_argument("--steps", type=int, help="Number of denoising steps.")
    parser.add_argument(
        "--guidance_scale",
        "--guidance-scale",
        dest="guidance_scale",
        type=float,
        help="Primary text/CFG guidance scale.",
    )
    parser.add_argument(
        "--image_guidance_scale",
        "--image-guidance-scale",
        dest="image_guidance_scale",
        type=float,
        help="Image CFG scale for BAGEL, ThinkMorph, and SenseNova-U1.",
    )
    parser.add_argument("--width", type=int, help="Output width.")
    parser.add_argument("--height", type=int, help="Output height.")
    parser.add_argument(
        "--num_images",
        "--num-images",
        dest="num_images",
        type=int,
        help=(
            "SenseNova output count; optionally limits saved BAGEL/ThinkMorph "
            "images. All interleaved images are saved when omitted."
        ),
    )
    parser.add_argument(
        "--num_frames",
        "--num-frames",
        dest="num_frames",
        type=int,
        help="Video frame count.",
    )
    parser.add_argument("--fps", type=int, help="Output video frame rate.")
    parser.add_argument(
        "--sampler",
        choices=[*WAN_RL_PAPER_SAMPLERS, "cps"],
        help=(
            "Sampler for the Wan2.2 TI2V RL checkpoints. Use 'cps' with "
            "--cps_eta for a custom Flow-CPS coefficient."
        ),
    )
    parser.add_argument(
        "--cps_eta",
        "--cps-eta",
        dest="cps_eta",
        type=float,
        help="Custom Flow-CPS coefficient in [0, 1]; implies --sampler cps.",
    )
    parser.add_argument(
        "--cps_seed",
        "--cps-seed",
        dest="cps_seed",
        type=int,
        help=(
            "Optional independent seed for Flow-CPS transition noise. The main "
            "--seed stream is reused when omitted."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cpu_offload",
        "--cpu-offload",
        dest="cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use model CPU offloading where the selected backend supports it.",
    )
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tiled VAE processing in DiffSynth video pipelines.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable reasoning (enabled by default for ThinkMorph only).",
    )
    parser.add_argument(
        "--understanding",
        action="store_true",
        help="Return text understanding instead of images (BAGEL/ThinkMorph only).",
    )
    parser.add_argument(
        "--max_think_tokens",
        "--max-think-tokens",
        dest="max_think_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max_rounds",
        "--max-rounds",
        dest="max_rounds",
        type=int,
        help="Maximum interleaved generation rounds for BAGEL/ThinkMorph.",
    )
    parser.add_argument(
        "--max_attempts",
        "--max-attempts",
        dest="max_attempts",
        type=int,
        default=3,
        help="ThinkMorph attempts when sampled text produces no image marker.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        "--max-memory-per-gpu",
        dest="max_memory_per_gpu",
        default="80GiB",
        help="Accelerate device-map memory limit for BAGEL/ThinkMorph.",
    )
    parser.add_argument(
        "--offload_dir",
        "--offload-dir",
        dest="offload_dir",
        type=Path,
        default=SCRIPT_DIR / "offload",
        help="Disk offload directory for BAGEL/ThinkMorph.",
    )
    parser.add_argument(
        "--vbvr_pro_models_dir",
        "--vbvr-pro-models-dir",
        dest="vbvr_pro_models_dir",
        type=Path,
        default=SCRIPT_DIR / "vbvr_pro_models",
        help="Directory containing the VBVR-Pro model runtime repositories.",
    )
    parser.add_argument(
        "--base_model",
        "--base-model",
        dest="base_model",
        help="Override the base model ID/path for a DiffSynth LoRA release.",
    )
    parser.add_argument(
        "--text_encoder_model",
        "--text-encoder-model",
        dest="text_encoder_model",
        help="Override the LTX Gemma text encoder ID/path.",
    )
    parser.add_argument(
        "--tokenizer_model",
        "--tokenizer-model",
        dest="tokenizer_model",
        help="Override the Wan tokenizer model ID/path.",
    )
    return parser


def list_models() -> None:
    width = max(len(key) for key in MODEL_SPECS)
    for key, spec in MODEL_SPECS.items():
        if spec.needs_image:
            image = "image required"
        else:
            image = "image optional"
        print(f"{key:<{width}}  {spec.backend:<12}  {image:<26}  {spec.description}")


def detect_model_type(model_path: str) -> str:
    normalized = model_path.rstrip("/").lower().replace("_", "-")
    for marker, model_type in MODEL_NAME_MARKERS:
        if marker in normalized:
            return model_type
    raise ValueError(
        f"Cannot infer a model type from {model_path!r}; pass --model_type explicitly."
    )


def prepare_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.list_models:
        list_models()
        raise SystemExit(0)
    if not args.model_path:
        parser.error("--model_path is required (or use --list-models)")
    if args.prompt_file:
        if not args.prompt_file.is_file():
            parser.error(f"prompt file does not exist: {args.prompt_file}")
        args.prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not args.prompt or not args.prompt.strip():
        parser.error("--prompt/--question or --prompt_file is required")
    args.prompt = args.prompt.strip()

    if args.model_type == "auto":
        try:
            args.model_type = detect_model_type(args.model_path)
        except ValueError as error:
            parser.error(str(error))
    spec = MODEL_SPECS[args.model_type]

    missing_images = [str(path) for path in args.image_paths if not path.is_file()]
    if missing_images:
        parser.error("input image(s) do not exist: " + ", ".join(missing_images))
    if spec.needs_image and not args.image_paths:
        parser.error(f"{args.model_type} requires --image_paths")
    if len(args.image_paths) > 1 and args.model_type not in {
        "qwen-image-edit",
        "qwen-image-edit-diffsynth",
    }:
        parser.error(f"{args.model_type} accepts one input image")
    if args.understanding and args.model_type not in {"bagel", "thinkmorph"}:
        parser.error("--understanding is supported only by BAGEL and ThinkMorph")
    if args.num_images is not None and args.num_images <= 0:
        parser.error("--num_images must be positive")
    if args.max_attempts <= 0:
        parser.error("--max_attempts must be positive")
    if args.num_frames is not None and args.num_frames <= 0:
        parser.error("--num_frames must be positive")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")

    is_wan_rl = args.model_type in WAN_RL_MODEL_TYPES
    if not is_wan_rl and any(
        value is not None
        for value in (args.sampler, args.cps_eta, args.cps_seed)
    ):
        parser.error(
            "--sampler, --cps_eta, and --cps_seed are supported only by the "
            "Wan2.2 TI2V RL checkpoints"
        )
    if is_wan_rl:
        if args.cps_eta is not None:
            if not 0.0 <= args.cps_eta <= 1.0:
                parser.error("--cps_eta must be in [0, 1]")
            if args.sampler is None:
                args.sampler = "cps"
            elif args.sampler != "cps":
                parser.error("--cps_eta can be used only with --sampler cps")
        args.sampler = args.sampler or "cps-0.7"
        if args.cps_seed is not None and not args.sampler.startswith("cps"):
            parser.error("--cps_seed can be used only with a Flow-CPS sampler")

    args.steps = args.steps or spec.default_steps
    if args.seed is None:
        if is_wan_rl:
            args.seed = 0
        else:
            args.seed = 1 if args.model_type == "bagel" else 42
    if args.think is None:
        args.think = args.model_type == "thinkmorph"
    if args.num_images is None and args.model_type == "sensenova-u1":
        args.num_images = 1
    if args.output is None:
        if args.understanding:
            args.output = Path("output.txt")
        elif spec.output_kind in {"video", "audio-video"}:
            args.output = Path("output.mp4")
        elif spec.output_kind in {"images", "interleaved"}:
            args.output = Path("outputs")
        else:
            args.output = Path("output.png")
    return args


def require_cuda(torch: Any, device: str) -> None:
    if not device.startswith("cuda"):
        raise ValueError("VBVR-Pro inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_images(paths: Sequence[Path]) -> list[Any]:
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def save_images(images: Sequence[Any], output: Path, prefix: str = "frame") -> list[Path]:
    if not images:
        return []
    written: list[Path] = []
    if output.suffix:
        output.parent.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images, start=1):
            path = output if index == 1 else output.with_name(f"{output.stem}_{index}{output.suffix}")
            image.save(path)
            written.append(path)
    else:
        output.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images, start=1):
            path = output / f"{prefix}_{index}.png"
            image.save(path)
            written.append(path)
    for path in written:
        print(f"Saved: {path}")
    return written


def save_text(parts: Sequence[str], output: Path) -> None:
    text = "\n".join(part for part in parts if part).strip()
    if text:
        print(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + ("\n" if text else ""), encoding="utf-8")
    print(f"Saved: {output}")


def resolve_local_repository(model_path: str) -> Path:
    local = Path(model_path).expanduser()
    if local.is_dir():
        return local.absolute()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
    )


def resolve_repository_file(model_path: str, filename: str) -> Path:
    local = Path(model_path).expanduser()
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(f"model file does not exist: {path}")
        # Keep the public filename when this is a Hugging Face cache symlink.
        # Some loaders select safetensors by suffix and cannot identify the
        # extensionless blob returned by Path.resolve().
        # Canonicalize the directory but not the file itself: Hugging Face
        # cache files can be symlinks to extensionless blobs.
        path = path.parent.resolve() / path.name
        if path.suffix == ".safetensors":
            validate_safetensors_file(path)
        return path
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=model_path, filename=filename))
    if path.suffix == ".safetensors":
        validate_safetensors_file(path)
    return path


def validate_local_diffusers_repository(model_path: str) -> None:
    """Reject an incomplete local Diffusers snapshot before allocating a model."""
    import json

    root = Path(model_path).expanduser()
    if not root.is_dir():
        return

    missing: list[Path] = []
    for index_path in root.rglob("*.safetensors.index.json"):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for filename in set(index.get("weight_map", {}).values()):
            shard = index_path.parent / filename
            if not shard.is_file():
                missing.append(shard)

    model_index_path = root / "model_index.json"
    if model_index_path.is_file():
        model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        for component, registration in model_index.items():
            if component.startswith("_") or registration is None:
                continue
            if registration == [None, None]:
                continue
            if isinstance(registration, list) and len(registration) == 2:
                component_path = root / component
                if not component_path.is_dir():
                    missing.append(component_path)
                    continue
                class_name = registration[1] or ""
                weightless_types = ("Scheduler", "Tokenizer", "Processor")
                has_weights = any(
                    next(component_path.rglob(pattern), None) is not None
                    for pattern in ("*.safetensors", "*.bin")
                )
                is_weightless = any(name in class_name for name in weightless_types)
                if not is_weightless and not has_weights:
                    missing.append(component_path / "<model weights>")

    if missing:
        details = "\n".join(f"  - {path}" for path in sorted(set(missing)))
        raise FileNotFoundError(
            "Incomplete local Diffusers repository; download the missing "
            f"component(s) before inference:\n{details}"
        )


def validate_safetensors_file(path: Path) -> None:
    """Fail early with a useful message for partial checkpoint downloads."""
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            handle.keys()
    except Exception as error:
        raise RuntimeError(
            f"Invalid or incomplete safetensors checkpoint: {path}. "
            "Delete the partial file and download the model repository again. "
            f"Loader error: {error}"
        ) from error


def prepend_python_path(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"required upstream repository is missing: {path}. See README.md for setup."
        )
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def run_bagel_family(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    from flash_attn_compat import install_flash_attn_fallback

    install_flash_attn_fallback()
    upstream_name = "BAGEL" if args.model_type == "bagel" else "ThinkMorph"
    prepend_python_path(args.vbvr_pro_models_dir / upstream_name)

    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    if args.model_type == "thinkmorph":
        from thinkmorph_compat import ThinkMorphInterleaveInferencer as InterleaveInferencer
    else:
        from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = resolve_local_repository(args.model_path)
    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    if args.model_type == "thinkmorph":
        # These are the transforms used by the released ThinkMorph evaluator.
        vae_transform = ImageTransform(512, 256, 16)
        vit_transform = ImageTransform(448, 224, 14)
    else:
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={
            index: args.max_memory_per_gpu for index in range(torch.cuda.device_count())
        },
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], args.device)
        if first_device in {"cpu", "disk"}:
            first_device = args.device
        for module in same_device_modules:
            device_map[module] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        if first_device is not None:
            for module in same_device_modules:
                if module in device_map:
                    device_map[module] = first_device

    checkpoint_name = "model.safetensors" if args.model_type == "bagel" else "ema.safetensors"
    checkpoint = model_path / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint),
        device_map=device_map,
        offload_buffers=True,
        offload_folder=str(args.offload_dir),
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
    # Keep the standalone BAGEL VAE on CPU in its released FP32 dtype, matching
    # the upstream evaluator. Accelerate moves the surrounding projection I/O.
    vae_model = vae_model.to(device="cpu", dtype=torch.float32).eval()
    # Torch/Accelerate combinations differ in whether autocast also converts
    # inputs for this custom VAE. Align them explicitly at the encode/decode
    # boundary so either the released FP32 weights or BF16-cast weights work.
    original_encode = vae_model.encode
    original_decode = vae_model.decode

    def encode_with_model_dtype(value: Any) -> Any:
        parameter = next(vae_model.encoder.parameters())
        with torch.autocast(device_type="cuda", enabled=False):
            return original_encode(value.to(device=parameter.device, dtype=parameter.dtype))

    def decode_with_model_dtype(value: Any) -> Any:
        parameter = next(vae_model.decoder.parameters())
        with torch.autocast(device_type="cuda", enabled=False):
            return original_decode(value.to(device=parameter.device, dtype=parameter.dtype))

    vae_model.encode = encode_with_model_dtype
    vae_model.decode = decode_with_model_dtype
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    input_image = open_images(args.image_paths)[0]
    guidance = args.guidance_scale if args.guidance_scale is not None else 4.0
    image_guidance = (
        args.image_guidance_scale if args.image_guidance_scale is not None else 2.0
    )
    inference_kwargs: dict[str, Any] = dict(
        think=args.think,
        understanding_output=args.understanding,
        max_think_token_n=args.max_think_tokens,
        do_sample=args.think,
        text_temperature=0.3,
        cfg_text_scale=guidance,
        cfg_img_scale=image_guidance,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.steps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    if args.model_type == "bagel":
        # BAGEL was trained/evaluated with text before the conditioning image,
        # and its public inferencer has no max_rounds argument.
        input_list = [args.prompt.replace("<image>", "").strip(), input_image]
    else:
        input_list = [input_image, args.prompt]
        inference_kwargs["max_rounds"] = args.max_rounds or 10
    attempts = args.max_attempts if args.model_type == "thinkmorph" else 1
    result: list[Any] = []
    images: list[Any] = []
    for attempt in range(attempts):
        set_seed(torch, args.seed + attempt)
        attempt_kwargs = dict(inference_kwargs)
        if args.model_type == "thinkmorph":
            attempt_kwargs["system_prompt"] = (
                THINKMORPH_SYSTEM_PROMPT if args.think else None
            )
            attempt_kwargs["text_temperature"] = min(1.0, 0.3 + 0.2 * attempt)
            if attempt:
                attempt_kwargs["max_think_token_n"] = min(args.max_think_tokens, 1024)
        result = inferencer.interleave_inference(input_list, **attempt_kwargs)
        images = [item for item in result if isinstance(item, Image.Image)]
        if images or args.understanding:
            break
        if attempt + 1 < attempts:
            print(
                "WARNING: ThinkMorph returned no image; retrying with "
                f"seed {args.seed + attempt + 1} and text temperature "
                f"{min(1.0, 0.3 + 0.2 * (attempt + 1)):g}"
            )

    texts = [item for item in result if isinstance(item, str)]
    if args.understanding:
        save_text(texts, args.output)
    else:
        if texts:
            print("\n".join(texts))
        if not images:
            raise RuntimeError("the model returned no generated images")
        selected_images = images if args.num_images is None else images[: args.num_images]
        save_images(selected_images, args.output, prefix="generated")
        if texts:
            response = "".join(
                item if isinstance(item, str) else "<image>"
                for item in result
                if isinstance(item, (str, Image.Image))
            )
            response_output = (
                args.output.with_suffix(".txt")
                if args.output.suffix
                else args.output / "response.txt"
            )
            save_text([response], response_output)


def tensor_to_image(torch: Any, frame: Any) -> Any:
    import numpy as np
    from PIL import Image

    if frame.ndim != 4 or frame.shape[0] != 1 or frame.shape[1] != 3:
        raise ValueError(f"unexpected generated tensor shape: {tuple(frame.shape)}")
    image = (frame.detach().float() * 0.5 + 0.5).clamp(0, 1)
    array = (image[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def run_sensenova_u1(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    width, height = args.width or 512, args.height or 512
    if width % 32 or height % 32:
        raise ValueError("SenseNova-U1 width and height must be multiples of 32")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device)
    model.eval()
    input_image = open_images(args.image_paths)[0]
    guidance = args.guidance_scale if args.guidance_scale is not None else 1.0
    image_guidance = (
        args.image_guidance_scale if args.image_guidance_scale is not None else 1.0
    )
    with torch.inference_mode():
        frames = model.interleave_gen_image_only(
            tokenizer,
            args.prompt.replace("<image>", "").strip(),
            gt_text="<image>" * args.num_images,
            images=[input_image],
            image_size=(width, height),
            max_images=args.num_images,
            num_steps=args.steps,
            cfg_scale=guidance,
            img_cfg_scale=image_guidance,
            timestep_shift=1.0,
        )
    if len(frames) != args.num_images:
        raise RuntimeError(f"model returned {len(frames)} frames; expected {args.num_images}")
    save_images([tensor_to_image(torch, frame) for frame in frames], args.output)


def place_diffusers_pipeline(pipe: Any, args: argparse.Namespace) -> Any:
    if not args.cpu_offload:
        return pipe.to(args.device)
    if args.device in {"cuda", "cuda:0"}:
        pipe.enable_model_cpu_offload()
        return pipe
    try:
        pipe.enable_model_cpu_offload(device=args.device)
    except TypeError:
        gpu_id = int(args.device.split(":", 1)[1])
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe


def run_flux2(args: argparse.Namespace) -> None:
    import torch
    from diffusers import Flux2Pipeline

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    validate_local_diffusers_repository(args.model_path)
    pipe = Flux2Pipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    place_diffusers_pipeline(pipe, args)
    kwargs: dict[str, Any] = {
        "image": open_images(args.image_paths)[0],
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else 2.5,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    if args.width is not None:
        kwargs["width"] = args.width
    if args.height is not None:
        kwargs["height"] = args.height
    image = pipe(**kwargs).images[0]
    save_images([image], args.output)


def run_qwen_image_edit(args: argparse.Namespace) -> None:
    import torch
    from diffusers import QwenImageEditPlusPipeline

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    validate_local_diffusers_repository(args.model_path)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    place_diffusers_pipeline(pipe, args)
    kwargs: dict[str, Any] = {
        "image": open_images(args.image_paths),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or " ",
        "num_inference_steps": args.steps,
        "true_cfg_scale": args.guidance_scale if args.guidance_scale is not None else 4.0,
        "guidance_scale": 1.0,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    if args.width is not None:
        kwargs["width"] = args.width
    if args.height is not None:
        kwargs["height"] = args.height
    image = pipe(**kwargs).images[0]
    save_images([image], args.output)


def run_ltx_diffusers(args: argparse.Namespace) -> None:
    import torch
    from diffusers import LTX2ImageToVideoPipeline
    from diffusers.utils import encode_video

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    validate_local_diffusers_repository(args.model_path)
    pipe = LTX2ImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    place_diffusers_pipeline(pipe, args)

    requested_frames = args.num_frames or 49
    output_fps = args.fps or 24
    video, audio = pipe(
        image=open_images(args.image_paths)[0],
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or LTX_NEGATIVE_PROMPT,
        height=args.height or 512,
        width=args.width or 768,
        num_frames=requested_frames,
        frame_rate=output_fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else 5.0,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        output_type="np",
        return_dict=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video[0][:requested_frames],
        fps=output_fps,
        output_path=str(args.output),
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    )
    print(f"Saved: {args.output}")


def run_wan_diffusers(args: argparse.Namespace) -> None:
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
    from diffusers.utils import export_to_video

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    validate_local_diffusers_repository(args.model_path)
    width, height = args.width or 832, args.height or 480
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32
    )
    if args.model_type == "wan2.2-ti2v-5b" and not args.image_paths:
        pipe = WanPipeline.from_pretrained(
            args.model_path, vae=vae, torch_dtype=torch.bfloat16
        )
    else:
        pipe = WanImageToVideoPipeline.from_pretrained(
            args.model_path, vae=vae, torch_dtype=torch.bfloat16
        )
    place_diffusers_pipeline(pipe, args)
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        "height": height,
        "width": width,
        "num_frames": args.num_frames or 81,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale if args.guidance_scale is not None else 5.0,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
    }
    if args.image_paths:
        kwargs["image"] = open_images(args.image_paths)[0].resize((width, height))
    frames = pipe(**kwargs).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=args.fps or 15)
    print(f"Saved: {args.output}")


def run_wan_rl_diffusers(args: argparse.Namespace) -> None:
    """Run the custom six-sampler pipeline bundled with the Wan TI2V RL models."""
    import torch
    from diffusers import AutoencoderKLWan, DiffusionPipeline
    from diffusers.utils import export_to_video

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    validate_local_diffusers_repository(args.model_path)

    local_model_path = Path(args.model_path).expanduser()
    if local_model_path.is_dir() and not (local_model_path / "pipeline.py").is_file():
        raise FileNotFoundError(
            "The Wan2.2 TI2V RL checkpoint requires its bundled pipeline.py; "
            f"the file is missing from {local_model_path}."
        )
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        custom_pipeline="pipeline",
        trust_remote_code=True,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    place_diffusers_pipeline(pipe, args)

    width, height = args.width or 512, args.height or 512
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    kwargs: dict[str, Any] = {
        "image": open_images(args.image_paths)[0],
        "prompt": args.prompt,
        "height": height,
        "width": width,
        "num_frames": args.num_frames or 81,
        "num_inference_steps": args.steps,
        "guidance_scale": (
            args.guidance_scale if args.guidance_scale is not None else 1.0
        ),
        "sampler": args.sampler,
        "generator": generator,
    }
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.cps_eta is not None:
        kwargs["cps_eta"] = args.cps_eta
    if args.cps_seed is not None:
        kwargs["cps_generator"] = torch.Generator(device=args.device).manual_seed(
            args.cps_seed
        )

    frames = pipe(**kwargs).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=args.fps or 16)
    print(f"Saved: {args.output}")


def use_diffsynth(args: argparse.Namespace) -> None:
    prepend_python_path(args.vbvr_pro_models_dir / "DiffSynth-Studio")


def diffsynth_vram_config(torch: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not args.cpu_offload:
        return {}
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": args.device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": args.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": args.device,
    }


def diffsynth_config(ModelConfig: Any, source: str, pattern: str, **kwargs: Any) -> Any:
    """Build a DiffSynth ModelConfig for either a local directory or a hub ID."""
    local = Path(source).expanduser()
    if local.is_dir():
        if pattern in {"", "./"}:
            path: str | list[str] = str(local.absolute())
        elif pattern.endswith("/"):
            path = str((local / pattern.rstrip("/")).absolute())
        else:
            matches = sorted(local.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"no files matching {pattern!r} under {local}")
            # Do not resolve Hugging Face cache symlinks: their targets are
            # extensionless blobs, while DiffSynth dispatches by file suffix.
            paths = [str(match.absolute()) for match in matches]
            path = paths[0] if len(paths) == 1 else paths
        return ModelConfig(path=path, **kwargs)
    return ModelConfig(
        model_id=source,
        origin_file_pattern=pattern,
        download_source="huggingface",
        **kwargs,
    )


def run_ltx_diffsynth(args: argparse.Namespace) -> None:
    import torch

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    use_diffsynth(args)
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
    from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

    vram = diffsynth_vram_config(torch, args)
    base = args.base_model or "DiffSynth-Studio/LTX-2.3-Repackage"
    text_encoder = (
        args.text_encoder_model or "google/gemma-3-12b-it-qat-q4_0-unquantized"
    )
    model_configs = [
        diffsynth_config(ModelConfig, text_encoder, "model-*.safetensors", **vram),
        diffsynth_config(ModelConfig, base, "transformer.safetensors", **vram),
        diffsynth_config(
            ModelConfig, base, "text_encoder_post_modules.safetensors", **vram
        ),
        diffsynth_config(ModelConfig, base, "video_vae_decoder.safetensors", **vram),
        diffsynth_config(ModelConfig, base, "audio_vae_decoder.safetensors", **vram),
        diffsynth_config(ModelConfig, base, "audio_vocoder.safetensors", **vram),
        diffsynth_config(ModelConfig, base, "video_vae_encoder.safetensors", **vram),
    ]
    tokenizer_config = diffsynth_config(ModelConfig, text_encoder, "")
    lora_path = resolve_repository_file(args.model_path, "lora.safetensors")

    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    pipe.load_lora(pipe.dit, str(lora_path), alpha=1)
    requested_frames = args.num_frames or 49
    output_fps = args.fps or 24
    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or LTX_NEGATIVE_PROMPT,
        input_images=open_images(args.image_paths),
        input_images_indexes=[0],
        input_images_strength=1.0,
        seed=args.seed,
        height=args.height or 512,
        width=args.width or 768,
        num_frames=requested_frames,
        frame_rate=output_fps,
        cfg_scale=args.guidance_scale if args.guidance_scale is not None else 3.0,
        num_inference_steps=args.steps,
        tiled=args.tiled,
    )
    video = video[:requested_frames]
    if audio is not None:
        audio_samples = round(
            requested_frames
            / output_fps
            * pipe.audio_vocoder.output_sampling_rate
        )
        audio = audio[..., :audio_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio_ltx2(
        video=video,
        audio=audio,
        output_path=str(args.output),
        fps=output_fps,
        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
    )
    print(f"Saved: {args.output}")


def run_flux2_lora(args: argparse.Namespace) -> None:
    import torch

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    use_diffsynth(args)
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    lora_path = resolve_repository_file(args.model_path, "lora.safetensors")
    base = args.base_model or "black-forest-labs/FLUX.2-dev"
    vram = diffsynth_vram_config(torch, args)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            diffsynth_config(ModelConfig, base, "text_encoder/*.safetensors", **vram),
            diffsynth_config(ModelConfig, base, "transformer/*.safetensors", **vram),
            diffsynth_config(
                ModelConfig, base, "vae/diffusion_pytorch_model.safetensors", **vram
            ),
        ],
        tokenizer_config=diffsynth_config(ModelConfig, base, "tokenizer/"),
    )
    pipe.load_lora(
        pipe.dit,
        str(lora_path),
        alpha=1,
    )
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        edit_image=open_images(args.image_paths),
        seed=args.seed,
        rand_device=args.device,
        num_inference_steps=args.steps,
        height=args.height or 512,
        width=args.width or 512,
        edit_image_auto_resize=True,
        embedded_guidance=args.guidance_scale if args.guidance_scale is not None else 1.0,
    )
    save_images([image], args.output)


def run_qwen_image_edit_lora(args: argparse.Namespace) -> None:
    import torch

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    use_diffsynth(args)
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    lora_path = resolve_repository_file(args.model_path, "lora.safetensors")
    base = args.base_model or "Qwen/Qwen-Image-Edit-2511"
    vram = diffsynth_vram_config(torch, args)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            diffsynth_config(
                ModelConfig,
                base,
                "transformer/diffusion_pytorch_model*.safetensors",
                **vram,
            ),
            diffsynth_config(
                ModelConfig, base, "text_encoder/model*.safetensors", **vram
            ),
            diffsynth_config(
                ModelConfig, base, "vae/diffusion_pytorch_model.safetensors", **vram
            ),
        ],
        tokenizer_config=diffsynth_config(ModelConfig, base, "tokenizer/"),
        processor_config=diffsynth_config(ModelConfig, base, "processor/"),
    )
    pipe.load_lora(
        pipe.dit,
        str(lora_path),
        alpha=1,
    )
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or "",
        cfg_scale=args.guidance_scale if args.guidance_scale is not None else 4.0,
        edit_image=open_images(args.image_paths),
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height or 512,
        width=args.width or 512,
        edit_image_auto_resize=True,
        zero_cond_t=True,
    )
    save_images([image], args.output)


def run_wan_diffsynth(args: argparse.Namespace) -> None:
    import torch

    require_cuda(torch, args.device)
    set_seed(torch, args.seed)
    use_diffsynth(args)
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    defaults = {
        "wan2.1-i2v-14b-diffsynth": "Wan-AI/Wan2.1-I2V-14B-720P",
        "wan2.2-i2v-a14b-diffsynth": "Wan-AI/Wan2.2-I2V-A14B",
        "wan2.2-ti2v-5b-diffsynth": "Wan-AI/Wan2.2-TI2V-5B",
    }
    if args.model_type == "wan2.2-i2v-a14b-diffsynth":
        lora_paths = (
            resolve_repository_file(args.model_path, "high_noise_lora.safetensors"),
            resolve_repository_file(args.model_path, "low_noise_lora.safetensors"),
        )
    else:
        lora_paths = (resolve_repository_file(args.model_path, "lora.safetensors"),)
    base = args.base_model or defaults[args.model_type]
    vram = diffsynth_vram_config(torch, args)

    if args.model_type == "wan2.1-i2v-14b-diffsynth":
        model_configs = [
            diffsynth_config(
                ModelConfig, base, "diffusion_pytorch_model*.safetensors", **vram
            ),
            diffsynth_config(
                ModelConfig, base, "models_t5_umt5-xxl-enc-bf16.pth", **vram
            ),
            diffsynth_config(ModelConfig, base, "Wan2.1_VAE.pth", **vram),
            diffsynth_config(
                ModelConfig,
                base,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                **vram,
            ),
        ]
    elif args.model_type == "wan2.2-i2v-a14b-diffsynth":
        model_configs = [
            diffsynth_config(
                ModelConfig,
                base,
                "high_noise_model/diffusion_pytorch_model*.safetensors",
                **vram,
            ),
            diffsynth_config(
                ModelConfig,
                base,
                "low_noise_model/diffusion_pytorch_model*.safetensors",
                **vram,
            ),
            diffsynth_config(
                ModelConfig, base, "models_t5_umt5-xxl-enc-bf16.pth", **vram
            ),
            diffsynth_config(ModelConfig, base, "Wan2.1_VAE.pth", **vram),
        ]
    else:
        model_configs = [
            diffsynth_config(
                ModelConfig, base, "diffusion_pytorch_model*.safetensors", **vram
            ),
            diffsynth_config(
                ModelConfig, base, "models_t5_umt5-xxl-enc-bf16.pth", **vram
            ),
            diffsynth_config(ModelConfig, base, "Wan2.2_VAE.pth", **vram),
        ]

    base_path = Path(base).expanduser()
    if args.tokenizer_model:
        tokenizer_source = args.tokenizer_model
        tokenizer_pattern = ""
    elif base_path.is_dir():
        tokenizer_source = base
        tokenizer_pattern = "google/umt5-xxl/"
    elif args.model_type == "wan2.1-i2v-14b-diffsynth":
        tokenizer_source = base
        tokenizer_pattern = "google/umt5-xxl/"
    else:
        tokenizer_source = "Wan-AI/Wan2.1-T2V-1.3B"
        tokenizer_pattern = "google/umt5-xxl/"

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=diffsynth_config(
            ModelConfig, tokenizer_source, tokenizer_pattern
        ),
    )

    if args.model_type == "wan2.2-i2v-a14b-diffsynth":
        pipe.load_lora(
            pipe.dit,
            str(lora_paths[0]),
            alpha=1,
        )
        pipe.load_lora(
            pipe.dit2,
            str(lora_paths[1]),
            alpha=1,
        )
    else:
        pipe.load_lora(
            pipe.dit,
            str(lora_paths[0]),
            alpha=1,
        )

    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        "height": args.height or 512,
        "width": args.width or 512,
        "num_frames": args.num_frames or 49,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "tiled": args.tiled,
    }
    if args.image_paths:
        kwargs["input_image"] = open_images(args.image_paths)[0]
    if args.model_type == "wan2.2-i2v-a14b-diffsynth":
        kwargs["switch_DiT_boundary"] = 0.9

    video = pipe(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        video[: (args.num_frames or 49)],
        str(args.output),
        fps=args.fps or 15,
        quality=5,
    )
    print(f"Saved: {args.output}")


RUNNERS = {
    "bagel": run_bagel_family,
    "thinkmorph": run_bagel_family,
    "sensenova-u1": run_sensenova_u1,
    "flux2": run_flux2,
    "qwen-image-edit": run_qwen_image_edit,
    "ltx2.3": run_ltx_diffusers,
    "wan2.1-i2v-14b": run_wan_diffusers,
    "wan2.2-i2v-a14b": run_wan_diffusers,
    "wan2.2-ti2v-5b": run_wan_diffusers,
    "wan2.2-ti2v-5b-qwen-judge-rl": run_wan_rl_diffusers,
    "wan2.2-ti2v-5b-rule-rl": run_wan_rl_diffusers,
    "flux2-diffsynth": run_flux2_lora,
    "qwen-image-edit-diffsynth": run_qwen_image_edit_lora,
    "ltx2.3-diffsynth": run_ltx_diffsynth,
    "wan2.1-i2v-14b-diffsynth": run_wan_diffsynth,
    "wan2.2-i2v-a14b-diffsynth": run_wan_diffsynth,
    "wan2.2-ti2v-5b-diffsynth": run_wan_diffsynth,
}


def main() -> int:
    parser = build_parser()
    args = prepare_args(parser, parser.parse_args())
    spec = MODEL_SPECS[args.model_type]
    print(f"Model type: {args.model_type} ({spec.description})")
    print(f"Model path: {args.model_path}")
    RUNNERS[args.model_type](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
