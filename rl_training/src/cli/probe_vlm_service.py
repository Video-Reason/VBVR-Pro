"""Wait for an OpenAI-compatible VLM endpoint and optionally run a vision request."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _request_json(
    opener: urllib.request.OpenerDirector,
    *,
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _image_data_url(color: str, *, size: int = 224) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _multimodal_smoke(
    opener: urllib.request.OpenerDirector,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at these two images. Which one is brighter?"},
                    {"type": "text", "text": "Image 1:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url("black")}},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image_url", "image_url": {"url": _image_data_url("white")}},
                    {"type": "text", "text": "Return only JSON."},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 32,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vision_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"brighter_image": {"type": "integer", "enum": [1, 2]}},
                    "required": ["brighter_image"],
                    "additionalProperties": False,
                },
            },
        },
    }
    response = _request_json(
        opener,
        base_url=base_url,
        api_key=api_key,
        method="POST",
        path="/chat/completions",
        timeout=timeout,
        payload=payload,
    )
    try:
        content = str(response["choices"][0]["message"]["content"])
        verdict = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Malformed multimodal smoke response: {response!r}") from exc
    if verdict.get("brighter_image") != 2:
        raise RuntimeError(f"VLM vision probe returned the wrong answer: {content!r}")
    return content


def _task_prompt_payload(
    *,
    model: str,
    task_name: str,
    frame_count: int,
    image_size: int,
    video_fps: int = 16,
    video_num_frames: int = 32,
) -> tuple[dict[str, Any], str]:
    """Build one exact task-rubric request and return its source prompt."""
    from src.eval.vbvr_vlm_protocol import (
        build_task_vlm_judge_messages,
        build_task_vlm_judge_payload,
        encode_vlm_video_data_url,
        load_pinned_eval_prompts,
    )

    task_prompt = load_pinned_eval_prompts()[task_name]
    video = np.full((frame_count, image_size, image_size, 3), 224, dtype=np.uint8)
    square_size = max(4, image_size // 8)
    for index, frame in enumerate(video):
        left = round((image_size - square_size) * index / max(1, frame_count - 1))
        frame[image_size // 2 - square_size // 2 : image_size // 2 + square_size // 2, left : left + square_size] = 32
    video_data_url = encode_vlm_video_data_url(video, fps=video_fps, max_edge=image_size)
    messages = build_task_vlm_judge_messages(
        task_prompt=task_prompt,
        first_frame_data_url=_image_data_url("white", size=image_size),
        generated_video_data_url=video_data_url,
        source_frame_count=frame_count,
        video_fps=video_fps,
    )
    return (
        build_task_vlm_judge_payload(
            model_name=model,
            messages=messages,
            task_prompt=task_prompt,
            max_tokens=1024,
            video_num_frames=video_num_frames,
        ),
        task_prompt,
    )


def _task_prompt_request(
    opener: urllib.request.OpenerDirector,
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    payload: dict[str, Any],
    task_prompt: str,
) -> tuple[float, str]:
    from src.eval.vbvr_vlm_protocol import parse_task_vlm_judge_score

    response = _request_json(
        opener,
        base_url=base_url,
        api_key=api_key,
        method="POST",
        path="/chat/completions",
        timeout=timeout,
        payload=payload,
    )
    try:
        response_content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed task-prompt smoke response: {response!r}") from exc
    if not isinstance(response_content, str):
        raise RuntimeError(f"Task-prompt smoke returned non-text content: {response_content!r}")
    return parse_task_vlm_judge_score(response_content, task_prompt=task_prompt)


def _task_prompt_smoke(
    opener: urllib.request.OpenerDirector,
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> tuple[float, str]:
    """Exercise the exact dynamic line schema used by the default reward."""
    payload, task_prompt = _task_prompt_payload(
        model=model,
        task_name="G-21_multiple_occlusions_vertical_data-generator",
        frame_count=81,
        image_size=224,
        video_fps=16,
        video_num_frames=32,
    )
    return _task_prompt_request(
        opener,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        payload=payload,
        task_prompt=task_prompt,
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _benchmark_task_prompts(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    request_count: int,
    concurrency: int,
    warmup_count: int,
) -> dict[str, float | int]:
    """Benchmark mixed exact rubrics with production-like image count/size."""
    from src.eval.vbvr_vlm_protocol import load_pinned_eval_prompts

    task_names = sorted(load_pinned_eval_prompts())
    requests = [
        _task_prompt_payload(
            model=model,
            task_name=task_names[index % len(task_names)],
            frame_count=81,
            image_size=384,
            video_fps=16,
            video_num_frames=32,
        )
        for index in range(max(request_count, warmup_count))
    ]

    def run_one(request_index: int) -> float:
        payload, task_prompt = requests[request_index % len(requests)]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start = time.perf_counter()
        _task_prompt_request(
            opener,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            payload=payload,
            task_prompt=task_prompt,
        )
        return time.perf_counter() - start

    for index in range(warmup_count):
        run_one(index)

    wall_start = time.perf_counter()
    latencies: list[float] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="vlm-benchmark") as executor:
        futures = [executor.submit(run_one, index) for index in range(request_count)]
        for future in as_completed(futures):
            latencies.append(future.result())
    wall_seconds = time.perf_counter() - wall_start
    return {
        "requests": request_count,
        "concurrency": concurrency,
        "wall_seconds": wall_seconds,
        "requests_per_second": request_count / wall_seconds,
        "mean_seconds": sum(latencies) / len(latencies),
        "p50_seconds": _percentile(latencies, 0.50),
        "p95_seconds": _percentile(latencies, 0.95),
        "p99_seconds": _percentile(latencies, 0.99),
        "max_seconds": max(latencies),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("WAN_TRAINER_VLM_BASE_URL", "http://127.0.0.1:18080/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("WAN_TRAINER_VLM_MODEL", "qwen3.6-27b"))
    parser.add_argument("--api-key", default=os.environ.get("WAN_TRAINER_VLM_API_KEY", "EMPTY"))
    parser.add_argument("--wait-seconds", type=float, default=900.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--multimodal-smoke", action="store_true")
    parser.add_argument(
        "--task-prompt-smoke",
        action="store_true",
        help="Exercise one pinned VBVR task prompt and its dynamic line parser",
    )
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=0,
        help="Run this many mixed exact-rubric requests after readiness probes",
    )
    parser.add_argument("--benchmark-concurrency", type=int, default=16)
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--server-pid", type=int, default=None, help="Fail early if this local service process exits")
    parser.add_argument("--server-log", type=Path, default=None, help="Print the tail when startup fails")
    return parser.parse_args(argv)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _print_log_tail(path: Path | None, *, lines: int = 80) -> None:
    if path is None or not path.is_file():
        return
    tail = path.read_text(errors="replace").splitlines()[-lines:]
    print(f"[error] Last {len(tail)} lines from {path}:", file=sys.stderr)
    print("\n".join(tail), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.wait_seconds <= 0 or args.request_timeout_seconds <= 0 or args.poll_seconds <= 0:
        print("[error] wait/request-timeout/poll durations must all be > 0", file=sys.stderr)
        return 2
    if args.benchmark_requests < 0 or args.benchmark_concurrency <= 0 or args.benchmark_warmup < 0:
        print("[error] benchmark requests/warmup must be >= 0 and concurrency must be > 0", file=sys.stderr)
        return 2

    # Bypass inherited proxies for loopback and private service endpoints.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.wait_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if args.server_pid is not None and not _pid_is_alive(args.server_pid):
            last_error = RuntimeError(f"VLM server process {args.server_pid} exited during startup")
            break
        try:
            response = _request_json(
                opener,
                base_url=args.base_url,
                api_key=args.api_key,
                method="GET",
                path="/models",
                timeout=args.request_timeout_seconds,
            )
            model_ids = {str(item["id"]) for item in response["data"]}
            if args.model not in model_ids:
                raise RuntimeError(f"expected model {args.model!r}; endpoint serves {sorted(model_ids)}")
            print(f"[preflight] VLM service ready: endpoint={args.base_url} model={args.model}")
            if args.multimodal_smoke:
                try:
                    content = _multimodal_smoke(
                        opener,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        timeout=max(args.request_timeout_seconds, 180.0),
                    )
                except Exception as exc:
                    print(f"[error] VLM multimodal structured-output probe failed: {exc}", file=sys.stderr)
                    _print_log_tail(args.server_log)
                    return 1
                print(f"[preflight] VLM multimodal structured-output probe passed: {content}")
            if args.task_prompt_smoke:
                try:
                    score, reasoning = _task_prompt_smoke(
                        opener,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        timeout=max(args.request_timeout_seconds, 180.0),
                    )
                except Exception as exc:
                    print(f"[error] VLM task-prompt probe failed: {exc}", file=sys.stderr)
                    _print_log_tail(args.server_log)
                    return 1
                print(f"[preflight] VLM task-prompt probe passed: score={score:.4f} reasoning={reasoning!r}")
            if args.benchmark_requests:
                try:
                    metrics = _benchmark_task_prompts(
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model=args.model,
                        timeout=max(args.request_timeout_seconds, 300.0),
                        request_count=args.benchmark_requests,
                        concurrency=min(args.benchmark_concurrency, args.benchmark_requests),
                        warmup_count=args.benchmark_warmup,
                    )
                except Exception as exc:
                    print(f"[error] VLM task-prompt benchmark failed: {exc}", file=sys.stderr)
                    _print_log_tail(args.server_log)
                    return 1
                print("[benchmark] " + json.dumps(metrics, sort_keys=True))
            return 0
        except (OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))

    print(f"[error] VLM service did not become ready at {args.base_url}: {last_error}", file=sys.stderr)
    _print_log_tail(args.server_log)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
