"""Score generated VBVR-Pro MP4 trees with the training-time Qwen3.6 judge."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from src.eval.vbvr_vlm_final_scores import write_cell_final_scores
from src.eval.vbvr_vlm_offline import (
    OfflineJudgeConfig,
    OfflineTaskVLMJudge,
    aggregate_complete_cells,
    assigned_cell_dirs,
    cell_is_complete,
    discover_eval_cell_dirs,
    load_eval_cell,
    score_eval_cell,
    wait_for_complete_cells,
)


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Formal VBVR-Pro result root containing run cells",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Independent resumable VLM-judge result root")
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Optional shell-style cell-name filter; repeat for multiple patterns",
    )
    parser.add_argument("--limit-cells", type=int, help="Deterministic debug limit after cell sorting")
    parser.add_argument("--expected-samples-per-cell", type=int, default=500)
    parser.add_argument("--sample-limit", type=int, help="Deterministic debug limit within every selected cell")
    parser.add_argument(
        "--verify-media-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly the expected number of MP4s in each generated tree",
    )


def _add_judge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=os.environ.get("WAN_TRAINER_VLM_BASE_URL", "http://127.0.0.1:18080/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("WAN_TRAINER_VLM_MODEL", "qwen3.6-27b"))
    parser.add_argument("--api-key", default=os.environ.get("WAN_TRAINER_VLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--model-revision",
        default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        help="Recorded immutable Qwen snapshot revision",
    )
    parser.add_argument("--vllm-version", default="0.26.0", help="Recorded serving runtime version")
    parser.add_argument("--video-fps", type=int, default=16)
    parser.add_argument("--source-frame-count", type=int, default=81)
    parser.add_argument("--sampled-video-frames", type=int, default=32)
    parser.add_argument("--image-max-edge", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--error-score", type=float, default=0.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing native VBVR-Pro MP4s with the exact task-specific Qwen request contract used by "
            "DanceGRPO training"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score this machine's deterministic cell shard and optionally merge")
    _add_source_arguments(score)
    _add_judge_arguments(score)
    score.add_argument(
        "--world-size",
        type=int,
        default=_environment_int("WORLD_SIZE", 1),
        help="Evaluation machine count",
    )
    score.add_argument(
        "--rank",
        type=int,
        default=_environment_int("RANK", 0),
        help="Zero-based evaluation machine rank",
    )
    score.add_argument("--concurrency", type=int, default=16, help="Concurrent HTTP judge requests on this machine")
    score.add_argument("--progress-interval-seconds", type=float, default=30.0)
    score.add_argument("--fsync-every", type=int, default=25, help="Durably sync the append-only JSONL every N results")
    score.add_argument(
        "--validate-service",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify the configured served model before submitting any pending work",
    )
    score.add_argument("--assignment-only", action="store_true", help="Read-only assignment and resume audit")
    score.add_argument(
        "--quick-assignment-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    score.add_argument(
        "--wait-for-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On rank 0, wait for every machine's cells and publish the aggregate",
    )
    score.add_argument("--wait-timeout-seconds", type=float, default=172800.0)
    score.add_argument("--wait-poll-seconds", type=float, default=30.0)

    summarize = subparsers.add_parser("summarize", help="Strictly audit and aggregate already-complete cells")
    _add_source_arguments(summarize)
    _add_judge_arguments(summarize)
    return parser


def _judge_config(args: argparse.Namespace) -> OfflineJudgeConfig:
    return OfflineJudgeConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        model_revision=args.model_revision,
        vllm_version=args.vllm_version,
        video_fps=args.video_fps,
        source_frame_count=args.source_frame_count,
        sampled_video_frames=args.sampled_video_frames,
        image_max_edge=args.image_max_edge,
        jpeg_quality=args.jpeg_quality,
        max_tokens=args.max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        error_score=args.error_score,
    )


def _selected_dirs(args: argparse.Namespace) -> list[Path]:
    cell_dirs = discover_eval_cell_dirs(args.input_root, tuple(args.cell))
    if args.limit_cells is not None:
        if args.limit_cells <= 0:
            raise ValueError("--limit-cells must be positive")
        cell_dirs = cell_dirs[: args.limit_cells]
    return cell_dirs


def _load_cell(path: Path, args: argparse.Namespace):
    if args.expected_samples_per_cell <= 0:
        raise ValueError("--expected-samples-per-cell must be positive")
    return load_eval_cell(
        path,
        expected_samples=args.expected_samples_per_cell,
        sample_limit=args.sample_limit,
        verify_media_count=args.verify_media_count,
    )


def _print_assignments(
    cell_dirs: list[Path],
    *,
    world_size: int,
    output_root: Path,
    config: OfflineJudgeConfig,
    args: argparse.Namespace,
    current_cells: list,
) -> None:
    for node_rank in range(world_size):
        directories = assigned_cell_dirs(cell_dirs, world_size=world_size, rank=node_rank)
        if node_rank == args.rank:
            pending = sum(not cell_is_complete(cell, output_root, config) for cell in current_cells)
            suffix = f" pending={pending}"
        else:
            suffix = ""
        names = ",".join(path.name for path in directories) or "<none>"
        print(f"[assignment] node={node_rank}/{world_size} cells={len(directories)}{suffix}: {names}", flush=True)


def _run_score(args: argparse.Namespace) -> int:
    config = _judge_config(args)
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cell_dirs = _selected_dirs(args)
    assigned_dirs = assigned_cell_dirs(cell_dirs, world_size=args.world_size, rank=args.rank)
    if args.quick_assignment_only:
        for node_rank in range(args.world_size):
            directories = assigned_cell_dirs(cell_dirs, world_size=args.world_size, rank=node_rank)
            suffix = f" pending={len(directories)}" if node_rank == args.rank else ""
            names = ",".join(path.name for path in directories) or "<none>"
            print(
                f"[assignment] node={node_rank}/{args.world_size} cells={len(directories)}{suffix}: {names}",
                flush=True,
            )
        return 0
    assigned_cells = [_load_cell(path, args) for path in assigned_dirs]
    _print_assignments(
        cell_dirs,
        world_size=args.world_size,
        output_root=output_root,
        config=config,
        args=args,
        current_cells=assigned_cells,
    )
    if args.assignment_only:
        return 0

    pending_cells = [cell for cell in assigned_cells if not cell_is_complete(cell, output_root, config)]
    logger.info(
        "[vlm-judge] input={} output={} node={}/{} assigned={} pending={} contract={}",
        input_root,
        output_root,
        args.rank,
        args.world_size,
        len(assigned_cells),
        len(pending_cells),
        config.contract_sha256,
    )
    judge = OfflineTaskVLMJudge(config)
    if pending_cells and args.validate_service:
        judge.validate_service()
        logger.info("[vlm-judge] service ready endpoint={} model={}", config.normalized_base_url, config.model)

    incomplete: list[str] = []
    for cell in assigned_cells:
        if cell_is_complete(cell, output_root, config):
            write_cell_final_scores(output_root / cell.name)
            logger.info("[vlm-judge] skip already-complete cell={}", cell.name)
            continue
        summary = score_eval_cell(
            cell,
            output_root=output_root,
            judge=judge,
            concurrency=args.concurrency,
            rank=args.rank,
            progress_interval_seconds=args.progress_interval_seconds,
            fsync_every=args.fsync_every,
        )
        if summary["state"] == "complete":
            write_cell_final_scores(output_root / cell.name, summary)
        else:
            incomplete.append(cell.name)
    if incomplete:
        logger.error(
            "[vlm-judge] {} assigned cells remain incomplete after exhausted judge retries: {}",
            len(incomplete),
            ", ".join(incomplete),
        )
        return 2

    if args.rank != 0:
        return 0

    if args.world_size == 1:
        all_cells = assigned_cells
    else:
        loaded_by_name = {cell.name: cell for cell in assigned_cells}
        all_cells = [loaded_by_name.get(path.name) or _load_cell(path, args) for path in cell_dirs]
    if args.wait_for_all:
        wait_for_complete_cells(
            all_cells,
            output_root=output_root,
            config=config,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.wait_poll_seconds,
        )
        for cell in all_cells:
            write_cell_final_scores(output_root / cell.name)
        result = aggregate_complete_cells(all_cells, output_root=output_root, config=config)
        logger.info(
            "[vlm-judge] aggregate complete cells={} judgments={} mean={:.6f}",
            result["num_cells"],
            result["num_sample_judgments"],
            result["mean_over_all_judgments"],
        )
    elif all(cell_is_complete(cell, output_root, config) for cell in all_cells):
        aggregate_complete_cells(all_cells, output_root=output_root, config=config)
    else:
        logger.info("[vlm-judge] rank 0 finished its shard; aggregate deferred because --no-wait-for-all was set")
    return 0


def _run_summarize(args: argparse.Namespace) -> int:
    config = _judge_config(args)
    cells = [_load_cell(path, args) for path in _selected_dirs(args)]
    result = aggregate_complete_cells(cells, output_root=args.output_root.expanduser().resolve(), config=config)
    for cell in cells:
        write_cell_final_scores(args.output_root / cell.name)
    print(
        f"[vlm-judge] aggregate complete cells={result['num_cells']} "
        f"judgments={result['num_sample_judgments']} mean={result['mean_over_all_judgments']:.6f}",
        flush=True,
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "score":
            return _run_score(args)
        if args.command == "summarize":
            return _run_summarize(args)
        raise AssertionError(f"Unhandled command: {args.command}")
    except (FileNotFoundError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
