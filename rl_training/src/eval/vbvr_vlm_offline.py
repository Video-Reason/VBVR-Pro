"""Resumable task-specific Qwen judging for already-generated VBVR-Pro videos.

The online DanceGRPO reward and this evaluator share their message/payload
builders, rubric parser, and pinned prompt set.  Offline evaluation sends the
existing native MP4 directly, avoiding a lossy second video transcode.
"""

from __future__ import annotations

import base64
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from PIL import Image

from src.eval.vbvr_vlm_protocol import (
    EVAL_PROMPTS_SOURCE_SHA256,
    TASK_VLM_JUDGE_REPAIR_PROMPT,
    build_task_vlm_judge_messages,
    build_task_vlm_judge_payload,
    encode_vlm_image_data_url,
    load_pinned_eval_prompts,
    parse_task_vlm_judge_score,
)

SCHEMA_VERSION = 1
_DOMAIN_DIRS = {"In_Domain": "In-Domain_50", "Out_of_Domain": "Out-of-Domain_50"}
EVAL_PROMPTS = load_pinned_eval_prompts()
_PROTOCOL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).with_name("vbvr_vlm_protocol.py").read_bytes()).hexdigest()
_EVALUATOR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@dataclass(frozen=True, slots=True)
class OfflineJudgeConfig:
    base_url: str = "http://127.0.0.1:18080/v1"
    model: str = "qwen3.6-27b"
    api_key: str = "EMPTY"
    model_revision: str = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    vllm_version: str = "0.26.0"
    video_fps: int = 16
    source_frame_count: int = 81
    sampled_video_frames: int = 32
    image_max_edge: int = 512
    jpeg_quality: int = 85
    max_tokens: int = 1024
    score_max: float = 100.0
    request_timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    error_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("VLM base URL and model must be nonempty")
        for name in ("video_fps", "source_frame_count", "sampled_video_frames", "image_max_edge", "max_tokens"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.max_retries < 0 or self.retry_backoff_seconds < 0 or self.request_timeout_seconds <= 0:
            raise ValueError("retry and timeout settings are invalid")
        if not math.isfinite(self.score_max) or self.score_max != 100.0:
            raise ValueError("Task-specific Qwen rubrics require score_max=100")
        if not math.isfinite(self.error_score) or not 0.0 <= self.error_score <= 1.0:
            raise ValueError("error_score must be finite and in [0, 1]")

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    def contract(self) -> dict[str, Any]:
        value = asdict(self)
        # Authentication does not alter judge semantics and must not be written
        # into shared result metadata.
        value.pop("api_key")
        value["base_url"] = self.normalized_base_url
        value.update(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt_mode": "task_specific",
                "prompt_source_sha256": EVAL_PROMPTS_SOURCE_SHA256,
                "protocol_source_sha256": _PROTOCOL_SOURCE_SHA256,
                "offline_evaluator_source_sha256": _EVALUATOR_SOURCE_SHA256,
                "media_input": "existing_native_mp4_data_url",
                "include_gt_first_frame": True,
                "include_gt_final_frame": False,
                "enable_thinking": False,
                "temperature": 0.0,
                "seed": 0,
                "structured_output": "per_task_regex",
            }
        )
        return value

    @property
    def contract_sha256(self) -> str:
        return _canonical_sha256(self.contract())


@dataclass(frozen=True, slots=True)
class OfflineSample:
    name: str
    task_name: str
    video_idx: str
    domain: str
    prompt: str
    first_frame_path: Path
    video_path: Path


@dataclass(frozen=True, slots=True)
class EvalCell:
    name: str
    source_dir: Path
    video_root: Path
    samples: tuple[OfflineSample, ...]
    eval_samples_sha256: str
    generated_tree_sha256: str
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    score: float
    reasoning: str
    response: str | None
    error: str | None
    request_attempts: int
    semantic_attempts: int
    elapsed_seconds: float


def discover_eval_cell_dirs(input_root: Path, patterns: tuple[str, ...] = ()) -> list[Path]:
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"VLM evaluation input root does not exist: {input_root}")
    cells = [
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and (path / "eval_samples.json").is_file()
        and (path / "generation-provenance.json").is_file()
        and (not patterns or any(fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns))
    ]
    cells.sort(key=lambda path: _cell_sort_key(path.name))
    if not cells:
        suffix = f" matching {patterns}" if patterns else ""
        raise ValueError(f"No generated VBVR-Pro evaluation cells found beneath {input_root}{suffix}")
    return cells


def _generated_video_root(cell_dir: Path, generation: dict[str, Any]) -> tuple[Path, str]:
    try:
        media = generation["media_trees"]["generated_videos"]
        recorded_path = Path(str(media["path"]))
        tree_sha256 = str(media["sha256"])
        state = str(generation["values"]["state"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{cell_dir}: generation provenance lacks the generated-video contract") from exc
    if state != "complete":
        raise ValueError(f"{cell_dir}: generation provenance is not complete (state={state!r})")
    if not re.fullmatch(r"[0-9a-f]{64}", tree_sha256):
        raise ValueError(f"{cell_dir}: generated-video tree SHA-256 is invalid")
    local_path = cell_dir / recorded_path.name
    video_root = local_path if local_path.is_dir() else recorded_path.expanduser()
    if not video_root.is_dir():
        raise FileNotFoundError(f"{cell_dir}: generated-video tree is missing: {video_root}")
    return video_root.resolve(), tree_sha256


def load_eval_cell(
    cell_dir: Path,
    *,
    expected_samples: int,
    sample_limit: int | None = None,
    verify_media_count: bool = True,
) -> EvalCell:
    cell_dir = cell_dir.expanduser().resolve()
    eval_path = cell_dir / "eval_samples.json"
    generation_path = cell_dir / "generation-provenance.json"
    eval_bytes = eval_path.read_bytes()
    try:
        raw_samples = json.loads(eval_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{eval_path} is not valid JSON: {exc}") from exc
    if not isinstance(raw_samples, list) or len(raw_samples) != expected_samples:
        actual = len(raw_samples) if isinstance(raw_samples, list) else type(raw_samples).__name__
        raise ValueError(f"{cell_dir}: expected {expected_samples} eval samples, got {actual}")

    generation = _read_json(generation_path)
    if not isinstance(generation, dict):
        raise ValueError(f"{generation_path} must contain an object")
    video_root, generated_tree_sha256 = _generated_video_root(cell_dir, generation)

    samples: list[OfflineSample] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"{eval_path}: sample {index} is not an object")
        try:
            name = str(raw["name"])
            task_name = str(raw["task_name"])
            video_idx = str(raw["video_idx"])
            domain = str(raw["domain"])
            first_frame_path = Path(str(raw["image"])).expanduser()
        except KeyError as exc:
            raise ValueError(f"{eval_path}: sample {index} is missing {exc.args[0]!r}") from exc
        name_path = PurePosixPath(name)
        expected_domain_dir = _DOMAIN_DIRS.get(domain)
        if (
            expected_domain_dir is None
            or len(name_path.parts) != 3
            or name_path.parts != (expected_domain_dir, task_name, video_idx)
            or name_path.is_absolute()
            or ".." in name_path.parts
        ):
            raise ValueError(f"{eval_path}: inconsistent sample identity at index {index}: {name!r}")
        if name in seen:
            raise ValueError(f"{eval_path}: duplicate sample name {name!r}")
        seen.add(name)
        if task_name not in EVAL_PROMPTS:
            raise ValueError(f"{eval_path}: no pinned task-specific rubric for {task_name!r}")
        if not first_frame_path.is_absolute():
            first_frame_path = cell_dir / first_frame_path
        if not first_frame_path.is_file():
            raise FileNotFoundError(f"{eval_path}: input first frame is missing: {first_frame_path}")
        video_path = video_root.joinpath(*name_path.parts).with_suffix(".mp4")
        if not video_path.is_file():
            raise FileNotFoundError(f"{cell_dir}: generated video is missing: {video_path}")
        samples.append(
            OfflineSample(
                name=name,
                task_name=task_name,
                video_idx=video_idx,
                domain=domain,
                prompt=str(raw.get("prompt", "")),
                first_frame_path=first_frame_path.resolve(),
                video_path=video_path.resolve(),
            )
        )

    if verify_media_count:
        actual_media = sum(1 for path in video_root.rglob("*.mp4") if path.is_file())
        if actual_media != expected_samples:
            raise ValueError(f"{cell_dir}: expected exactly {expected_samples} generated MP4s, found {actual_media}")

    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        samples = samples[:sample_limit]
    eval_sha256 = _sha256_bytes(eval_bytes)
    source_contract = {
        "cell_name": cell_dir.name,
        "eval_samples_sha256": eval_sha256,
        "generated_tree_sha256": generated_tree_sha256,
        "selected_sample_names": [sample.name for sample in samples],
    }
    return EvalCell(
        name=cell_dir.name,
        source_dir=cell_dir,
        video_root=video_root,
        samples=tuple(samples),
        eval_samples_sha256=eval_sha256,
        generated_tree_sha256=generated_tree_sha256,
        source_fingerprint=_canonical_sha256(source_contract),
    )


class OfflineTaskVLMJudge:
    """Thread-safe OpenAI-compatible client matching ``VBVRVLMReward`` task mode."""

    def __init__(self, config: OfflineJudgeConfig) -> None:
        self.config = config
        # Never route loopback data-URL payloads through an inherited proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def validate_service(self) -> None:
        response, _ = self._request_json("GET", "/models")
        try:
            model_ids = {str(item["id"]) for item in response["data"]}
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Malformed VLM /models response: {response!r}") from exc
        if self.config.model not in model_ids:
            raise RuntimeError(
                f"VLM model {self.config.model!r} is not served by {self.config.normalized_base_url}; "
                f"available={sorted(model_ids)}"
            )

    def score(self, sample: OfflineSample) -> JudgeOutcome:
        started = time.monotonic()
        response_content: str | None = None
        request_attempts = 0
        semantic_attempts = 0
        try:
            task_prompt = EVAL_PROMPTS[sample.task_name]
            with Image.open(sample.first_frame_path) as image:
                first_frame_data_url = encode_vlm_image_data_url(
                    image,
                    max_edge=self.config.image_max_edge,
                    jpeg_quality=self.config.jpeg_quality,
                )
            video_bytes = sample.video_path.read_bytes()
            if not video_bytes:
                raise ValueError(f"Generated MP4 is empty: {sample.video_path}")
            video_data_url = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")
            messages = build_task_vlm_judge_messages(
                task_prompt=task_prompt,
                first_frame_data_url=first_frame_data_url,
                generated_video_data_url=video_data_url,
                source_frame_count=self.config.source_frame_count,
                video_fps=self.config.video_fps,
            )

            request_messages = messages
            for semantic_index in range(self.config.max_retries + 1):
                semantic_attempts = semantic_index + 1
                payload = build_task_vlm_judge_payload(
                    model_name=self.config.model,
                    messages=request_messages,
                    task_prompt=task_prompt,
                    max_tokens=self.config.max_tokens,
                    video_num_frames=self.config.sampled_video_frames,
                )
                response, attempts = self._request_json("POST", "/chat/completions", payload)
                request_attempts += attempts
                response_content = self._response_content(response)
                try:
                    score, reasoning = parse_task_vlm_judge_score(
                        response_content,
                        task_prompt=task_prompt,
                        score_max=self.config.score_max,
                    )
                    return JudgeOutcome(
                        score=score,
                        reasoning=reasoning,
                        response=response_content,
                        error=None,
                        request_attempts=request_attempts,
                        semantic_attempts=semantic_attempts,
                        elapsed_seconds=time.monotonic() - started,
                    )
                except ValueError as exc:
                    if semantic_index >= self.config.max_retries:
                        raise
                    request_messages = [
                        *request_messages,
                        {"role": "assistant", "content": response_content},
                        {
                            "role": "user",
                            "content": f"{TASK_VLM_JUDGE_REPAIR_PROMPT}\nValidation error: {str(exc)[:300]}",
                        },
                    ]
                    time.sleep(self.config.retry_backoff_seconds * (2**semantic_index))
            raise AssertionError("unreachable")
        except Exception as exc:
            return JudgeOutcome(
                score=self.config.error_score,
                reasoning="",
                response=response_content,
                error=f"{type(exc).__name__}: {exc}",
                request_attempts=request_attempts,
                semantic_attempts=semantic_attempts,
                elapsed_seconds=time.monotonic() - started,
            )

    @staticmethod
    def _response_content(response: dict[str, Any]) -> str:
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

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(
                f"{self.config.normalized_base_url}{path}",
                data=body,
                headers=headers,
                method=method,
            )
            try:
                with self._opener.open(request, timeout=self.config.request_timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise RuntimeError(f"VLM service returned non-object JSON: {type(decoded).__name__}")
                return decoded, attempt + 1
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                error = RuntimeError(f"VLM service HTTP {exc.code}: {detail[:1000]}")
                if not retryable or attempt >= self.config.max_retries:
                    raise error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                error = RuntimeError(f"VLM service request failed: {exc}")
                if attempt >= self.config.max_retries:
                    raise error from exc
            time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise AssertionError("unreachable")


def _output_metadata(cell: EvalCell, config: OfflineJudgeConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_name": cell.name,
        "source_dir": str(cell.source_dir),
        "video_root": str(cell.video_root),
        "eval_samples_sha256": cell.eval_samples_sha256,
        "generated_tree_sha256": cell.generated_tree_sha256,
        "source_fingerprint": cell.source_fingerprint,
        "expected_samples": len(cell.samples),
        "judge_contract": config.contract(),
        "judge_contract_sha256": config.contract_sha256,
    }


def _metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        return _read_json(path) == expected
    except ValueError:
        return False


def _load_score_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                name = str(record["name"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Ignoring malformed VLM score record {}:{}: {}", path, line_number, exc)
                continue
            records[name] = record
    return records


def _summary_stats(records: list[dict[str, Any]], domain: str | None) -> dict[str, Any]:
    selected = [record for record in records if domain is None or record["domain"] == domain]
    by_task: dict[str, list[float]] = defaultdict(list)
    for record in selected:
        by_task[str(record["task_name"])].append(float(record["score"]))
    return {
        "mean_score": sum(float(record["score"]) for record in selected) / len(selected) if selected else 0.0,
        "num_samples": len(selected),
        "num_errors": sum(record.get("error") is not None for record in selected),
        "by_task": {task: sum(values) / len(values) for task, values in sorted(by_task.items())},
    }


def _build_cell_summary(
    cell: EvalCell,
    records_by_name: dict[str, dict[str, Any]],
    config: OfflineJudgeConfig,
) -> dict[str, Any]:
    selected = [records_by_name[sample.name] for sample in cell.samples if sample.name in records_by_name]
    errors = sum(record.get("error") is not None for record in selected)
    complete = len(selected) == len(cell.samples) and errors == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "complete" if complete else "incomplete",
        "cell_name": cell.name,
        "timestamp": _utc_now(),
        "source_fingerprint": cell.source_fingerprint,
        "judge_contract_sha256": config.contract_sha256,
        "expected_samples": len(cell.samples),
        "completed_samples": len(selected),
        "error_samples": errors,
        "summary": {
            "overall": _summary_stats(selected, None),
            "In_Domain": _summary_stats(selected, "In_Domain"),
            "Out_of_Domain": _summary_stats(selected, "Out_of_Domain"),
        },
    }


def cell_is_complete(cell: EvalCell, output_root: Path, config: OfflineJudgeConfig) -> bool:
    output_dir = output_root / cell.name
    metadata = _output_metadata(cell, config)
    if not _metadata_matches(output_dir / "metadata.json", metadata):
        return False
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = _read_json(summary_path)
    except ValueError:
        return False
    return (
        summary.get("state") == "complete"
        and summary.get("source_fingerprint") == cell.source_fingerprint
        and summary.get("judge_contract_sha256") == config.contract_sha256
        and summary.get("completed_samples") == len(cell.samples)
        and summary.get("error_samples") == 0
    )


@contextmanager
def _cell_lock(output_dir: Path, cell: EvalCell, rank: int) -> Iterator[None]:
    lock_dir = output_dir / ".score.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        owner_path = lock_dir / "owner.json"
        owner = owner_path.read_text(errors="replace")[:1000] if owner_path.is_file() else "<missing owner metadata>"
        raise RuntimeError(f"Evaluation cell is already locked: {cell.name}; owner={owner}") from exc
    try:
        _atomic_write_json(
            lock_dir / "owner.json",
            {"pid": os.getpid(), "rank": rank, "hostname": os.uname().nodename, "started_at": _utc_now()},
        )
        yield
    finally:
        owner_path = lock_dir / "owner.json"
        if owner_path.exists():
            owner_path.unlink()
        with suppress(FileNotFoundError):
            lock_dir.rmdir()


def _sample_record(
    sample: OfflineSample,
    outcome: JudgeOutcome,
    *,
    cell: EvalCell,
    config: OfflineJudgeConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": _utc_now(),
        "cell_name": cell.name,
        "name": sample.name,
        "task_name": sample.task_name,
        "video_idx": sample.video_idx,
        "domain": sample.domain,
        "video_path": str(sample.video_path),
        "first_frame_path": str(sample.first_frame_path),
        "score": outcome.score,
        "reasoning": outcome.reasoning,
        "judge_response": outcome.response,
        "error": outcome.error,
        "request_attempts": outcome.request_attempts,
        "semantic_attempts": outcome.semantic_attempts,
        "elapsed_seconds": outcome.elapsed_seconds,
        "source_fingerprint": cell.source_fingerprint,
        "judge_contract_sha256": config.contract_sha256,
    }


def score_eval_cell(
    cell: EvalCell,
    *,
    output_root: Path,
    judge: OfflineTaskVLMJudge,
    concurrency: int,
    rank: int,
    progress_interval_seconds: float,
    fsync_every: int = 25,
) -> dict[str, Any]:
    if concurrency <= 0 or fsync_every <= 0 or progress_interval_seconds <= 0:
        raise ValueError("concurrency, fsync_every, and progress interval must be positive")
    output_dir = output_root / cell.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _output_metadata(cell, judge.config)
    metadata_path = output_dir / "metadata.json"
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    with _cell_lock(output_dir, cell, rank):
        if metadata_path.exists() and not _metadata_matches(metadata_path, metadata):
            raise ValueError(
                f"{output_dir}: existing VLM metadata does not match the current source/judge contract; "
                "use a fresh output root"
            )
        if not metadata_path.exists():
            _atomic_write_json(metadata_path, metadata)

        existing = _load_score_records(records_path)
        remaining = [
            sample
            for sample in cell.samples
            if sample.name not in existing or existing[sample.name].get("error") is not None
        ]
        if not remaining:
            summary = _build_cell_summary(cell, existing, judge.config)
            _atomic_write_json(summary_path, summary)
            logger.info("[vlm-judge] skip complete cell={} samples={}", cell.name, len(cell.samples))
            return summary

        started = time.monotonic()
        completed_now = 0
        last_progress = started
        logger.info(
            "[vlm-judge] start cell={} existing_success={} remaining={} concurrency={}",
            cell.name,
            len(cell.samples) - len(remaining),
            len(remaining),
            concurrency,
        )
        with records_path.open("a", encoding="utf-8") as output_handle:
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="vbvr-vlm-offline") as executor:
                future_to_sample = {executor.submit(judge.score, sample): sample for sample in remaining}
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    outcome = future.result()
                    record = _sample_record(sample, outcome, cell=cell, config=judge.config)
                    output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output_handle.flush()
                    completed_now += 1
                    if completed_now % fsync_every == 0:
                        os.fsync(output_handle.fileno())
                    existing[sample.name] = record
                    now = time.monotonic()
                    if outcome.error is not None:
                        logger.warning(
                            "[vlm-judge] sample error cell={} sample={} error={}",
                            cell.name,
                            sample.name,
                            outcome.error,
                        )
                    if now - last_progress >= progress_interval_seconds or completed_now == len(remaining):
                        rate = completed_now / max(now - started, 1e-9)
                        logger.info(
                            "[vlm-judge] progress cell={} new={}/{} total={}/{} rate={:.3f} req/s eta={:.1f}s",
                            cell.name,
                            completed_now,
                            len(remaining),
                            len(cell.samples) - len(remaining) + completed_now,
                            len(cell.samples),
                            rate,
                            (len(remaining) - completed_now) / max(rate, 1e-9),
                        )
                        last_progress = now
            os.fsync(output_handle.fileno())

        summary = _build_cell_summary(cell, existing, judge.config)
        _atomic_write_json(summary_path, summary)
        logger.info(
            "[vlm-judge] finish cell={} state={} score={:.6f} errors={} elapsed={:.1f}s",
            cell.name,
            summary["state"],
            summary["summary"]["overall"]["mean_score"],
            summary["error_samples"],
            time.monotonic() - started,
        )
        return summary


def assigned_cell_dirs(cell_dirs: list[Path], *, world_size: int, rank: int) -> list[Path]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid machine rank {rank} for world size {world_size}")
    return cell_dirs[rank::world_size]


def wait_for_complete_cells(
    cells: list[EvalCell],
    *,
    output_root: Path,
    config: OfflineJudgeConfig,
    timeout_seconds: float,
    poll_seconds: float,
) -> None:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("wait timeout and polling interval must be positive")
    deadline = time.monotonic() + timeout_seconds
    previous = -1
    while True:
        completed = sum(cell_is_complete(cell, output_root, config) for cell in cells)
        if completed != previous:
            logger.info("[vlm-judge] distributed completion {}/{} cells", completed, len(cells))
            previous = completed
        if completed == len(cells):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for VLM judge cells: {completed}/{len(cells)} complete")
        time.sleep(poll_seconds)


def _cell_attributes(name: str) -> dict[str, Any]:
    checkpoint = re.search(r"checkpoint-([1-9][0-9]*)", name)
    step = int(checkpoint.group(1)) if checkpoint else 0
    if match := re.search(r"cps-noise-(0(?:\.[0-9]+)?)", name):
        noise = float(match.group(1))
        sampler = f"CPS {noise:g}"
        sampler_order = int(round(noise * 10))
    elif "cps0p7" in name:
        sampler = "CPS 0.7"
        sampler_order = 7
    elif name == "euler" or "euler-ode" in name:
        sampler = "Euler ODE"
        sampler_order = 100
    elif name == "unipc" or "unipc-ode" in name:
        sampler = "UniPC ODE"
        sampler_order = 101
    else:
        sampler = "unknown"
        sampler_order = 999
    return {
        "step": step,
        "model": "baseline" if step == 0 else f"checkpoint-{step}",
        "sampler": sampler,
        "sampler_order": sampler_order,
    }


def _cell_sort_key(name: str) -> tuple[int, int, str]:
    attributes = _cell_attributes(name)
    return int(attributes["step"]), int(attributes["sampler_order"]), name


def aggregate_complete_cells(
    cells: list[EvalCell],
    *,
    output_root: Path,
    config: OfflineJudgeConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not cell_is_complete(cell, output_root, config):
            raise ValueError(f"Cannot aggregate incomplete or mismatched VLM judge cell: {cell.name}")
        cell_summary = _read_json(output_root / cell.name / "summary.json")
        attributes = _cell_attributes(cell.name)
        summary = cell_summary["summary"]
        rows.append(
            {
                "cell_name": cell.name,
                **{key: value for key, value in attributes.items() if key != "sampler_order"},
                "num_samples": int(summary["overall"]["num_samples"]),
                "overall": float(summary["overall"]["mean_score"]),
                "in_domain": float(summary["In_Domain"]["mean_score"]),
                "out_of_domain": float(summary["Out_of_Domain"]["mean_score"]),
                "errors": int(summary["overall"]["num_errors"]),
            }
        )
    rows.sort(key=lambda row: _cell_sort_key(str(row["cell_name"])))
    total_samples = sum(int(row["num_samples"]) for row in rows)
    weighted_mean = (
        sum(float(row["overall"]) * int(row["num_samples"]) for row in rows) / total_samples if total_samples else 0.0
    )
    by_sampler: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sampler[str(row["sampler"])].append(row)
    sampler_summary = {
        sampler: {
            "num_cells": len(values),
            "mean_over_cells": sum(float(value["overall"]) for value in values) / len(values),
            "best_cell": max(values, key=lambda value: float(value["overall"])),
            "latest_cell": max(values, key=lambda value: int(value["step"])),
        }
        for sampler, values in sorted(by_sampler.items())
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "timestamp": _utc_now(),
        "judge_contract": config.contract(),
        "judge_contract_sha256": config.contract_sha256,
        "num_cells": len(rows),
        "num_sample_judgments": total_samples,
        "mean_over_all_judgments": weighted_mean,
        "best_cell": max(rows, key=lambda row: float(row["overall"])) if rows else None,
        "by_sampler": sampler_summary,
        "cells": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_root / "summary.json", result)

    csv_buffer = io.StringIO()
    fieldnames = [
        "step",
        "model",
        "sampler",
        "overall",
        "in_domain",
        "out_of_domain",
        "num_samples",
        "errors",
        "cell_name",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fieldnames} for row in rows)
    _atomic_write_text(output_root / "summary.csv", csv_buffer.getvalue())

    markdown = [
        "# Qwen3.6-27B VBVR-Pro VLM judge results",
        "",
        f"- Cells: {len(rows)}",
        f"- Sample judgments: {total_samples}",
        f"- Mean over all judgments: {weighted_mean:.6f}",
        "",
        "| Step | Sampler | Overall | In-Domain | Out-of-Domain | Samples |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    markdown.extend(
        f"| {row['step']} | {row['sampler']} | {row['overall']:.6f} | {row['in_domain']:.6f} | "
        f"{row['out_of_domain']:.6f} | {row['num_samples']} |"
        for row in rows
    )
    _atomic_write_text(output_root / "summary.md", "\n".join(markdown) + "\n")
    return result
