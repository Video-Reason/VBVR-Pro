"""OpenAI-compatible VLM judge reward for VBVR GRPO training.

Generated latents are decoded by the training process, encoded as an in-memory
MP4, and sent to a separately hosted multimodal model. The service owns video
frame sampling, judge weights, and KV cache; training ranks only retain the HTTP
payload and the decoded rollout while a request is pending.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from loguru import logger
from PIL import Image

from src.eval.vbvr_vlm_protocol import (
    EVAL_PROMPTS_SOURCE_SHA256,
    TASK_VLM_JUDGE_OUTPUT_REMINDER,  # noqa: F401 - compatibility re-export
    TASK_VLM_JUDGE_REPAIR_PROMPT,
    build_task_vlm_judge_messages,
    build_task_vlm_judge_payload,
    encode_vlm_image_data_url,
    encode_vlm_video_data_url,
    parse_task_vlm_judge_score,
    task_vlm_judge_output_regex,  # noqa: F401 - compatibility re-export
    vllm_video_sampling_overrides,
)
from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward
from src.trainer.rewards.vbvr_vlm_eval_prompts import EVAL_PROMPTS

# This remains available as an explicit custom-prompt mode. The default reward
# uses the task-specific EVAL_PROMPTS contract above.
DEFAULT_VLM_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of video-based visual reasoning.

You will receive an instruction, a starting frame, an expected final frame, and a generated video. Judge whether the
generated video actually performs the requested task. Use the expected final frame
as a semantic reference, not as a demand for pixel-perfect copying. Check object identity and count, spatial relations,
state changes, task progress, temporal consistency, and the final outcome. Do not reward visual quality when the
requested reasoning or action is wrong.

Return only one JSON object with this schema:
{"score": <number from 0 to 100>, "reasoning": "<one concise sentence>"}

Scoring anchors:
- 100: clearly and completely solves the task; the final state matches the intended outcome.
- 75: mostly correct, with only a minor omission or deviation.
- 50: meaningful partial progress, but a key step or relation is missing or wrong.
- 25: attempts the task, but the central action or result fails.
- 0: irrelevant, broken, static when motion is required, or shows no useful task progress.

Use intermediate numeric scores when appropriate so that similar candidates can still be distinguished. Do not include
markdown fences or any text outside the JSON object."""


def parse_vlm_judge_score(response: str, *, score_max: float = 100.0) -> tuple[float, str]:
    """Parse a judge response into a normalized score and short reasoning."""
    if not math.isfinite(score_max) or score_max <= 0:
        raise ValueError(f"score_max must be finite and > 0, got {score_max}")

    total_match = re.search(
        r"^\s*total_score\s*:\s*(-?\d+(?:\.\d+)?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if total_match:
        raw_score = float(total_match.group(1))
        if math.isfinite(raw_score) and 0.0 <= raw_score <= score_max:
            reason_match = re.search(r"^\s*reason\s*:\s*(.*?)\s*$", response, flags=re.IGNORECASE | re.MULTILINE)
            return raw_score / score_max, reason_match.group(1).strip() if reason_match else ""

    candidates = re.findall(r"\{[^{}]*\}", response, flags=re.DOTALL)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            raw_score = float(payload["total_score"] if "total_score" in payload else payload["score"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not math.isfinite(raw_score) or not 0.0 <= raw_score <= score_max:
            continue
        return raw_score / score_max, str(payload.get("reason", payload.get("reasoning", ""))).strip()

    # Keep a small compatibility fallback for services that prepend prose even
    # after constrained decoding has been disabled.
    match = re.search(
        r"^\s*score\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        raw_score = float(match.group(1))
        if math.isfinite(raw_score) and 0.0 <= raw_score <= score_max:
            return raw_score / score_max, ""
    raise ValueError(f"VLM response does not contain a score in [0, {score_max}]: {response[:500]!r}")


@dataclass(slots=True)
class _PendingVLMReward:
    """VLM requests that may still be running on the external service."""

    batch_size: int
    device: torch.device
    futures: list[tuple[int, Future[float]]]

    def result(self) -> torch.Tensor:
        values = [0.0] * self.batch_size
        try:
            for index, future in self.futures:
                values[index] = future.result()
        except BaseException:
            for _, future in self.futures:
                future.cancel()
            raise
        return torch.tensor(values, device=self.device, dtype=torch.float32)


@register_reward("vbvr_vlm")
class VBVRVLMReward(BaseReward):
    """Reward generated videos through a separately hosted multimodal judge."""

    requires_vae: ClassVar[bool] = True

    def __init__(self, trainer, cfg) -> None:
        super().__init__(trainer, cfg)
        self._base_url = os.environ.get("WAN_TRAINER_VLM_BASE_URL", cfg.vlm_reward_base_url).rstrip("/")
        self._model_name = os.environ.get("WAN_TRAINER_VLM_MODEL", cfg.vlm_reward_model)
        self._api_key = os.environ.get("WAN_TRAINER_VLM_API_KEY", cfg.vlm_reward_api_key)
        # Inherited proxies must never intercept loopback judge traffic.
        self._url_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._prompt_mode = str(cfg.vlm_reward_prompt_mode)
        self._system_prompt = self._load_system_prompt(cfg) if self._prompt_mode == "custom" else None
        # The task-specific source has a different line-oriented schema for
        # every task, so the fixed generic JSON schema is custom-mode only.
        self._structured_output = self._prompt_mode == "custom" and bool(cfg.vlm_reward_use_structured_output)
        self._decode_batch_size = int(cfg.vlm_reward_decode_batch_size)
        self._concurrency = int(cfg.vlm_reward_concurrency)
        configured_pending = int(cfg.vlm_reward_max_pending_jobs)
        self._max_pending_jobs = configured_pending or max(self._decode_batch_size, 2 * self._concurrency)
        tensor_parallel_enabled = bool(getattr(trainer, "tensor_parallel_enabled", False))
        tensor_parallel_rank = int(getattr(trainer, "tp_rank", 0))
        self._active_score_rank = not tensor_parallel_enabled or tensor_parallel_rank == 0

        self._executor: ThreadPoolExecutor | None = None
        self._pending_slots: threading.BoundedSemaphore | None = None
        if self._active_score_rank:
            self._executor = ThreadPoolExecutor(
                max_workers=self._concurrency,
                thread_name_prefix=f"vbvr-vlm-rank{trainer.rank}",
            )
            self._pending_slots = threading.BoundedSemaphore(self._max_pending_jobs)
            atexit.register(self.close)

        self._error_lock = threading.Lock()
        self._error_count = 0
        self._response_log_lock = threading.Lock()
        self._response_log_count = 0

        if self._active_score_rank and bool(cfg.vlm_reward_validate_service):
            self._validate_service()
        if trainer.rank == 0:
            logger.info(
                "VBVR VLM reward: endpoint={} model={} prompt_mode={} media=video video_fps={} sampled_frames={} "
                "image_max_edge={} include_start={} "
                "decode_batch={} requests/rank={} max_pending={} structured_output={} prompt_sha256={}",
                self._base_url,
                self._model_name,
                self._prompt_mode,
                cfg.vlm_reward_video_fps,
                cfg.vlm_reward_video_num_frames,
                cfg.vlm_reward_image_max_edge,
                cfg.vlm_reward_include_gt_first_frame,
                self._decode_batch_size,
                self._concurrency,
                self._max_pending_jobs,
                self._structured_output,
                EVAL_PROMPTS_SOURCE_SHA256 if self._prompt_mode == "task_specific" else "custom",
            )

    @staticmethod
    def _load_system_prompt(cfg) -> str:
        prompt_path = cfg.vlm_reward_system_prompt_path
        if prompt_path:
            path = Path(prompt_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"VLM judge system prompt does not exist: {path}")
            prompt = path.read_text().strip()
        else:
            prompt = str(cfg.vlm_reward_system_prompt or DEFAULT_VLM_JUDGE_SYSTEM_PROMPT).strip()
        if not prompt:
            raise ValueError("VLM judge system prompt must not be empty")
        return prompt

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    @torch.no_grad()
    def __call__(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return self.submit(
            generated_latents,
            gt_video_latents,
            condition,
            prompt_embeds,
            indices=indices,
            expert_filter=expert_filter,
            meta=meta,
        ).result()

    @torch.no_grad()
    def submit(
        self,
        generated_latents: torch.Tensor,
        gt_video_latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        indices: torch.Tensor | None = None,
        expert_filter: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> _PendingVLMReward:
        """Decode previews now, then queue HTTP judge calls in sample order."""
        del condition, prompt_embeds, indices

        batch_size = int(generated_latents.shape[0])
        device = generated_latents.device
        if expert_filter == "high":
            return _PendingVLMReward(batch_size=batch_size, device=device, futures=[])
        if not self._active_score_rank:
            raise RuntimeError("VLM reward was called on a non-scoring tensor-parallel rank")
        if meta is None:
            raise RuntimeError("VBVRVLMReward requires task and reference-frame metadata")
        if self._prompt_mode == "custom" and "sample_prompt" not in meta:
            raise RuntimeError("Custom VBVRVLMReward prompt mode requires sample_prompt metadata")

        futures: list[tuple[int, Future[float]]] = []
        try:
            for start in range(0, batch_size, self._decode_batch_size):
                batch_indices = list(range(start, min(start + self._decode_batch_size, batch_size)))
                latent_indices = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
                first_paths = [self._meta_path(meta, "sample_gt_first_frame", index) for index in batch_indices]
                final_paths = [self._meta_path(meta, "sample_gt_final_frame", index) for index in batch_indices]
                if self._prompt_mode == "task_specific":
                    # The exact EvalKit prompt contract includes only the input
                    # first frame and generated video, never the GT final frame.
                    needs_gt_decode = [first_path is None for first_path in first_paths]
                else:
                    needs_gt_decode = [
                        final_path is None or (bool(self.cfg.vlm_reward_include_gt_first_frame) and first_path is None)
                        for first_path, final_path in zip(first_paths, final_paths, strict=True)
                    ]

                gen_latents = generated_latents.index_select(0, latent_indices)
                if any(needs_gt_decode):
                    decoded = self.trainer.model.decode_latents(
                        torch.cat([gen_latents, gt_video_latents.index_select(0, latent_indices)], dim=0)
                    )
                else:
                    decoded = self.trainer.model.decode_latents(gen_latents)
                videos = self._to_uint8_videos(decoded)
                count = len(batch_indices)
                generated_videos = videos[:count]
                gt_videos = videos[count:] if any(needs_gt_decode) else [None] * count

                for local_index, index in enumerate(batch_indices):
                    task_name = self._task_name(meta, index)
                    future = self._submit_score_job(
                        task_name=task_name,
                        prompt=str(self._meta_item(meta, "sample_prompt", index, default="") or ""),
                        generated_video=generated_videos[local_index],
                        gt_video=gt_videos[local_index],
                        gt_first_frame_path=first_paths[local_index],
                        gt_final_frame_path=final_paths[local_index],
                    )
                    futures.append((index, future))
                del decoded
        except BaseException:
            for _, future in futures:
                future.cancel()
            raise

        return _PendingVLMReward(batch_size=batch_size, device=device, futures=futures)

    def _submit_score_job(self, **kwargs: Any) -> Future[float]:
        executor = self._executor
        pending_slots = self._pending_slots
        if executor is None or pending_slots is None:
            raise RuntimeError("VLM reward request queue is not initialized on this rank")
        pending_slots.acquire()
        try:
            future = executor.submit(self._score_video, **kwargs)
        except BaseException:
            pending_slots.release()
            raise
        future.add_done_callback(lambda _future: pending_slots.release())
        return future

    def _score_video(
        self,
        *,
        task_name: str,
        prompt: str,
        generated_video: np.ndarray,
        gt_video: np.ndarray | None,
        gt_first_frame_path: str | None,
        gt_final_frame_path: str | None,
    ) -> float:
        try:
            task_prompt: str | None = None
            if self._prompt_mode == "task_specific":
                task_prompt = self._task_prompt(task_name)
                first_frame = self._load_reference_frame(gt_first_frame_path, gt_video, frame_index=0)
                messages = self._build_task_messages(
                    task_prompt=task_prompt,
                    first_frame=first_frame,
                    generated_video=generated_video,
                )
            else:
                first_frame = (
                    self._load_reference_frame(gt_first_frame_path, gt_video, frame_index=0)
                    if bool(self.cfg.vlm_reward_include_gt_first_frame)
                    else None
                )
                final_frame = self._load_reference_frame(gt_final_frame_path, gt_video, frame_index=-1)
                messages = self._build_custom_messages(
                    task_name=task_name,
                    prompt=prompt,
                    first_frame=first_frame,
                    final_frame=final_frame,
                    generated_video=generated_video,
                )
            if self._prompt_mode == "task_specific":
                assert task_prompt is not None
                score, reasoning, response = self._task_completion_with_retries(
                    messages=messages,
                    task_prompt=task_prompt,
                    task_name=task_name,
                )
            else:
                response = self._chat_completion(messages)
                score, reasoning = parse_vlm_judge_score(response, score_max=float(self.cfg.vlm_reward_score_max))
            self._log_response(task_name, score, reasoning, response)
            return score
        except Exception as exc:
            with self._error_lock:
                self._error_count += 1
                error_count = self._error_count
            fail_on_error = bool(self.cfg.vlm_reward_fail_on_error)
            error_score = float(self.cfg.vlm_reward_error_score)
            if error_count <= 10:
                action = "raising" if fail_on_error else f"using fallback score {error_score:.4f}"
                logger.warning(
                    "VBVR VLM reward failed: rank={} task={} action={} error={}",
                    self.trainer.rank,
                    task_name or "<unknown>",
                    action,
                    exc,
                )
            if fail_on_error:
                raise RuntimeError(f"VBVR VLM reward failed for task {task_name or '<unknown>'}: {exc}") from exc
            return error_score

    def _task_completion_with_retries(
        self,
        *,
        messages: list[dict[str, Any]],
        task_prompt: str,
        task_name: str,
    ) -> tuple[float, str, str]:
        """Request and semantically validate a task response, repairing invalid outputs."""
        request_messages = messages
        retries = int(self.cfg.vlm_reward_max_retries)
        for attempt in range(retries + 1):
            response = self._chat_completion(request_messages, task_prompt=task_prompt)
            try:
                score, reasoning = parse_task_vlm_judge_score(
                    response,
                    task_prompt=task_prompt,
                    score_max=float(self.cfg.vlm_reward_score_max),
                )
                return score, reasoning, response
            except ValueError as exc:
                if attempt >= retries:
                    raise
                logger.warning(
                    "Retrying semantically invalid VLM response: rank={} task={} retry={}/{} error={}",
                    self.trainer.rank,
                    task_name or "<unknown>",
                    attempt + 1,
                    retries,
                    exc,
                )
                request_messages = [
                    *request_messages,
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": f"{TASK_VLM_JUDGE_REPAIR_PROMPT}\nValidation error: {str(exc)[:300]}",
                    },
                ]
                time.sleep(float(self.cfg.vlm_reward_retry_backoff_seconds) * (2**attempt))
        raise AssertionError("unreachable")

    @staticmethod
    def _task_prompt(task_name: str) -> str:
        try:
            return EVAL_PROMPTS[task_name]
        except KeyError as exc:
            raise KeyError(
                f"No task-specific VLM judge prompt for {task_name or '<unknown>'!r}; "
                f"the pinned prompt set covers {len(EVAL_PROMPTS)} tasks"
            ) from exc

    def _build_task_messages(
        self,
        *,
        task_prompt: str,
        first_frame: Image.Image,
        generated_video: np.ndarray,
    ) -> list[dict[str, Any]]:
        first_frame_data_url = self._image_content(first_frame)["image_url"]["url"]
        generated_video_data_url = self._video_content(generated_video)["video_url"]["url"]
        return build_task_vlm_judge_messages(
            task_prompt=task_prompt,
            first_frame_data_url=first_frame_data_url,
            generated_video_data_url=generated_video_data_url,
            source_frame_count=len(generated_video),
            video_fps=int(self.cfg.vlm_reward_video_fps),
        )

    def _build_custom_messages(
        self,
        *,
        task_name: str,
        prompt: str,
        first_frame: Image.Image | None,
        final_frame: Image.Image,
        generated_video: np.ndarray,
    ) -> list[dict[str, Any]]:
        task_prefix = f"Task identifier: {task_name}\n" if task_name else ""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"{task_prefix}Task description:\n{prompt}"},
        ]
        if bool(self.cfg.vlm_reward_include_gt_first_frame):
            if first_frame is None:
                raise ValueError("VLM judge is configured to include the starting frame, but none was provided")
            content.extend(
                [
                    {"type": "text", "text": "Starting frame (the input condition):"},
                    self._image_content(first_frame),
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Expected final frame (semantic ground-truth outcome):"},
                self._image_content(final_frame),
                {
                    "type": "text",
                    "text": (
                        f"Generated video ({len(generated_video)} source frames encoded at "
                        f"{int(self.cfg.vlm_reward_video_fps)} FPS, chronological order):"
                    ),
                },
                self._video_content(generated_video),
            ]
        )
        content.append({"type": "text", "text": "Return the JSON verdict now."})
        if self._system_prompt is None:
            raise RuntimeError("Custom VLM prompt mode was initialized without a system prompt")
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]

    def _image_content(self, image: Image.Image) -> dict[str, Any]:
        max_edge = int(self.cfg.vlm_reward_image_max_edge)
        # Preserve native judge pixels whenever they already fit the configured
        # safety bound. PIL.thumbnail is deliberately downscale-only: this path
        # never expands a 384/512 frame to EvalKit's separate 1024 preparation.
        data_url = encode_vlm_image_data_url(
            image,
            max_edge=max_edge,
            jpeg_quality=int(self.cfg.vlm_reward_jpeg_quality),
        )
        return {"type": "image_url", "image_url": {"url": data_url}}

    def _video_content(self, video: np.ndarray) -> dict[str, Any]:
        data_url = encode_vlm_video_data_url(
            video,
            fps=int(self.cfg.vlm_reward_video_fps),
            max_edge=int(self.cfg.vlm_reward_image_max_edge),
        )
        return {"type": "video_url", "video_url": {"url": data_url}}

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        task_prompt: str | None = None,
    ) -> str:
        payload: dict[str, Any]
        if self._prompt_mode == "task_specific":
            if task_prompt is None:
                raise ValueError("Task-specific VLM chat completion requires its task prompt")
            # A per-task regex preserves the source's exact line-oriented
            # contract and prevents verbose analysis from consuming the token
            # budget before the required fields.
            payload = build_task_vlm_judge_payload(
                model_name=self._model_name,
                messages=messages,
                task_prompt=task_prompt,
                max_tokens=int(self.cfg.vlm_reward_max_new_tokens),
                video_num_frames=int(self.cfg.vlm_reward_video_num_frames),
            )
        else:
            payload = {
                "model": self._model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": int(self.cfg.vlm_reward_max_new_tokens),
                "seed": 0,
                # Qwen3.6 enables thinking by default. A scalar judge response is
                # faster and easier to constrain when hidden reasoning is disabled.
                "chat_template_kwargs": {"enable_thinking": False},
                **vllm_video_sampling_overrides(int(self.cfg.vlm_reward_video_num_frames)),
            }
        if self._prompt_mode != "task_specific" and self._structured_output:
            score_max = float(self.cfg.vlm_reward_score_max)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vbvr_video_judgment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "minimum": 0.0, "maximum": score_max},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["score", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            }
        response = self._request_json("POST", "/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed VLM chat-completion response: {response!r}") from exc
        if isinstance(content, list):
            content = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"VLM returned empty chat content: {response!r}")
        return content.strip()

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        timeout = float(self.cfg.vlm_reward_request_timeout_seconds)
        retries = int(self.cfg.vlm_reward_max_retries)
        for attempt in range(retries + 1):
            request = urllib.request.Request(f"{self._base_url}{path}", data=body, headers=headers, method=method)
            try:
                with self._url_opener.open(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                error = RuntimeError(f"VLM service HTTP {exc.code}: {detail[:1000]}")
                if not retryable or attempt >= retries:
                    raise error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = RuntimeError(f"VLM service request failed: {exc}")
                if attempt >= retries:
                    raise error from exc
            time.sleep(float(self.cfg.vlm_reward_retry_backoff_seconds) * (2**attempt))
        raise AssertionError("unreachable")

    def _validate_service(self) -> None:
        response = self._request_json("GET", "/models")
        try:
            model_ids = {str(item["id"]) for item in response["data"]}
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Malformed VLM /models response: {response!r}") from exc
        if self._model_name not in model_ids:
            raise RuntimeError(
                f"VLM model {self._model_name!r} is not served by {self._base_url}; available={sorted(model_ids)}"
            )

    def _log_response(self, task_name: str, score: float, reasoning: str, response: str) -> None:
        limit = int(self.cfg.vlm_reward_log_first_n)
        if limit <= 0 or self.trainer.rank != 0:
            return
        with self._response_log_lock:
            if self._response_log_count >= limit:
                return
            self._response_log_count += 1
            count = self._response_log_count
        logger.info(
            "VLM judge sample {}/{} task={} score={:.4f} reasoning={!r} raw={!r}",
            count,
            limit,
            task_name,
            score,
            reasoning,
            response[:1000],
        )

    @staticmethod
    def _load_reference_frame(path: str | None, gt_video: np.ndarray | None, *, frame_index: int) -> Image.Image:
        if path:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        if gt_video is None:
            raise FileNotFoundError("Reference frame path is missing and no decoded GT video is available")
        return Image.fromarray(gt_video[frame_index], mode="RGB")

    @staticmethod
    def _to_uint8_videos(decoded: torch.Tensor) -> np.ndarray:
        from src.inference.outputs import uint8_from_decoded

        return uint8_from_decoded(decoded)

    @staticmethod
    def _meta_item(meta: dict[str, Any], key: str, index: int, *, default: Any = None) -> Any:
        value = meta.get(key, default)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            item = value[index]
            return item.item() if item.ndim == 0 else item
        if isinstance(value, list | tuple):
            return value[index]
        return value

    def _meta_path(self, meta: dict[str, Any], key: str, index: int) -> str | None:
        value = self._meta_item(meta, key, index, default=None)
        if not value:
            return None
        path = Path(str(value)).expanduser()
        return str(path.resolve()) if path.is_file() else None

    def _task_name(self, meta: dict[str, Any], index: int) -> str:
        for key in ("sample_task_name", "task_name"):
            value = str(self._meta_item(meta, key, index, default="") or "")
            if value:
                return value
        tar_name = str(self._meta_item(meta, "sample_tar", index, default="") or "")
        return Path(tar_name).stem if tar_name else ""
