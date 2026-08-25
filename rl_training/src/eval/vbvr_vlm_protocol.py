"""Pure task-specific VBVR VLM protocol shared by training and evaluation."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# SHA-256 of the byte-identical source copied from
# storage/codes/eval_prompts.py on 2026-08-05.
EVAL_PROMPTS_SOURCE_SHA256 = "4d3159232590bd4b99266c9e82df445a3a54ada50a7af30051cf505057574202"
TASK_VLM_JUDGE_OUTPUT_REMINDER = (
    'Follow the "Output format (exactly these lines)" instruction in the task prompt. '
    "Verify that all numeric *_weight values add up arithmetically to exactly 100. "
    "Return only those lines, in that order, with no analysis, headings, bullets, Markdown, or code fences."
)
TASK_VLM_JUDGE_REPAIR_PROMPT = (
    "Your previous response failed semantic validation. Correct the reported error, recalculate the *_weight values, "
    "and verify that their arithmetic sum is exactly 100. Return the complete output again in the exact field order "
    "required by the task rubric, with no extra text."
)


def load_pinned_eval_prompts() -> dict[str, str]:
    """Load the pinned prompt literal without importing the eager reward package."""
    path = Path(__file__).parents[1] / "trainer/rewards/vbvr_vlm_eval_prompts.py"
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != EVAL_PROMPTS_SOURCE_SHA256:
        raise RuntimeError(
            f"Pinned VLM prompt source hash mismatch: expected {EVAL_PROMPTS_SOURCE_SHA256}, got {actual_sha256}"
        )
    tree = ast.parse(source, filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "EVAL_PROMPTS" for target in statement.targets):
            value = ast.literal_eval(statement.value)
            if not isinstance(value, dict) or len(value) != 100:
                raise RuntimeError(f"Pinned VLM prompt source has an invalid EVAL_PROMPTS mapping: {type(value)}")
            if not all(isinstance(key, str) and isinstance(prompt, str) for key, prompt in value.items()):
                raise RuntimeError("Pinned VLM prompt mapping must contain only string keys and prompts")
            return value
    raise RuntimeError(f"Pinned VLM prompt source has no EVAL_PROMPTS assignment: {path}")


def vllm_video_sampling_overrides(num_frames: int) -> dict[str, Any]:
    """Force one uniform vLLM sampling pass and disable HF resampling."""
    return {
        "media_io_kwargs": {
            "video": {
                "video_backend": "opencv",
                "num_frames": num_frames,
                "fps": -1,
            }
        },
        "mm_processor_kwargs": {"do_sample_frames": False},
    }


def encode_vlm_image_data_url(image: Image.Image, *, max_edge: int, jpeg_quality: int) -> str:
    """Encode one RGB reference image with the reward's downscale-only contract."""
    image = image.convert("RGB")
    if max(image.size) > max_edge:
        image = image.copy()
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def encode_vlm_video_data_url(video: np.ndarray, *, fps: int, max_edge: int) -> str:
    """Encode a complete uint8 RGB rollout as an in-memory H.264 MP4 data URL."""
    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] <= 0:
        raise ValueError(f"Expected nonempty video shaped (T,H,W,3), got {video.shape}")
    if video.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video, got {video.dtype}")

    height, width = map(int, video.shape[1:3])
    if max(height, width) > max_edge:
        scale = max_edge / max(height, width)
        target_width = max(2, int(round(width * scale)))
        target_height = max(2, int(round(height * scale)))
        # H.264 yuv420p requires even dimensions. Wan's native outputs are
        # already divisible by 16; this only matters for oversized custom
        # inputs that pass through the downscale safety bound.
        target_width -= target_width % 2
        target_height -= target_height % 2
        video = np.stack(
            [
                np.asarray(
                    Image.fromarray(frame, mode="RGB").resize(
                        (target_width, target_height),
                        Image.Resampling.LANCZOS,
                    )
                )
                for frame in video
            ]
        )
    elif height % 2 or width % 2:
        raise ValueError(f"VLM MP4 dimensions must be even for yuv420p, got {width}x{height}")

    import imageio.v3 as iio

    encoded_video = iio.imwrite(
        "<bytes>",
        np.ascontiguousarray(video),
        extension=".mp4",
        plugin="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    if not isinstance(encoded_video, bytes) or not encoded_video:
        raise RuntimeError("VLM video encoder returned no MP4 bytes")
    encoded = base64.b64encode(encoded_video).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def build_task_vlm_judge_messages(
    *,
    task_prompt: str,
    first_frame_data_url: str,
    generated_video_data_url: str,
    source_frame_count: int,
    video_fps: int,
) -> list[dict[str, Any]]:
    """Build the exact task-specific message shared by training and offline evaluation."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": task_prompt.strip()},
        {"type": "text", "text": "First frame (input):"},
        {"type": "image_url", "image_url": {"url": first_frame_data_url}},
        {
            "type": "text",
            "text": (
                f"Generated video ({source_frame_count} source frames encoded at {video_fps} FPS, chronological order):"
            ),
        },
        {"type": "video_url", "video_url": {"url": generated_video_data_url}},
        {"type": "text", "text": TASK_VLM_JUDGE_OUTPUT_REMINDER},
    ]
    return [{"role": "user", "content": content}]


def build_task_vlm_judge_payload(
    *,
    model_name: str,
    messages: list[dict[str, Any]],
    task_prompt: str,
    max_tokens: int,
    video_num_frames: int,
) -> dict[str, Any]:
    """Build the exact task-specific OpenAI-compatible request payload."""
    return {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        **vllm_video_sampling_overrides(video_num_frames),
        "structured_outputs": {"regex": task_vlm_judge_output_regex(task_prompt)},
    }


def _task_prompt_output_fields(task_prompt: str) -> list[str]:
    marker = "Output format (exactly these lines):"
    if marker not in task_prompt:
        raise ValueError("Task-specific VLM prompt has no output-format section")
    output_section = task_prompt.split(marker, maxsplit=1)[1]
    fields = re.findall(r"^([a-z][a-z0-9_]*):", output_section, flags=re.MULTILINE)
    if not fields or fields[-2:] != ["total_score", "reason"]:
        raise ValueError(f"Task-specific VLM prompt has an invalid output contract: {fields}")
    if len(fields) != len(set(fields)):
        raise ValueError(f"Task-specific VLM prompt repeats output fields: {fields}")
    return fields


def task_vlm_judge_output_regex(task_prompt: str) -> str:
    """Build a vLLM regex constraint for one prompt's exact line schema."""
    number = r"(?:100(?:\.0+)?|(?:[0-9]|[1-9][0-9])(?:\.[0-9]+)?)"
    lines = []
    for field in _task_prompt_output_fields(task_prompt):
        if field == "reason":
            lines.append(r"reason: [^\n]+")
        else:
            lines.append(rf"{re.escape(field)}: {number}")
    return "\n".join(lines)


def parse_task_vlm_judge_score(
    response: str,
    *,
    task_prompt: str,
    score_max: float = 100.0,
) -> tuple[float, str]:
    """Parse and validate the exact per-task line-oriented judge contract."""
    if not math.isfinite(score_max) or score_max <= 0:
        raise ValueError(f"score_max must be finite and > 0, got {score_max}")

    expected_fields = _task_prompt_output_fields(task_prompt)
    values: dict[str, float] = {}
    reason = ""
    for field in expected_fields:
        matches = re.findall(
            rf"^\s*{re.escape(field)}\s*:\s*(.*?)\s*$",
            response,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise ValueError(f"VLM response must contain exactly one {field!r} line: {response[:1000]!r}")
        if field == "reason":
            reason = matches[0].strip()
            if not reason:
                raise ValueError("VLM response reason must not be empty")
            continue
        try:
            value = float(matches[0])
        except ValueError as exc:
            raise ValueError(f"VLM response field {field!r} is not numeric: {matches[0]!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= score_max:
            raise ValueError(f"VLM response field {field!r} is outside [0, {score_max}]: {value}")
        values[field] = value

    weights = [value for field, value in values.items() if field.endswith("_weight")]
    if not weights or not math.isclose(sum(weights), 100.0, abs_tol=0.5):
        raise ValueError(f"VLM response weights must sum to 100, got {sum(weights):.6g}")
    return values["total_score"] / score_max, reason
