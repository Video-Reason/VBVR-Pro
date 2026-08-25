#!/usr/bin/env python3
"""Watch GPU utilization and launch a command after a sustained idle window.

Default behavior:
  - poll every 2 minutes
  - require all GPUs to stay below 10% utilization
  - require 30 consecutive minutes of low utilization
  - print a rolling 30-minute summary
  - then launch:
      .venv/bin/torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_sft_vbvr_5e-6.yaml

Examples:
  .venv/bin/python scripts/dev/watch_gpu_idle.py
  .venv/bin/python scripts/dev/watch_gpu_idle.py --dry-run
  .venv/bin/python scripts/dev/watch_gpu_idle.py --interval-seconds 300 --report-window-minutes 30 --dry-run
  .venv/bin/python scripts/dev/watch_gpu_idle.py --command "echo triggered"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

DEFAULT_COMMAND = ".venv/bin/torchrun --nproc_per_node=8 -m src.cli.train_i2v --config configs/train_sft_vbvr_5e-6.yaml"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_file: Path | None = None) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _fmt_percent(values: list[float]) -> str:
    return ", ".join(f"{value:.0f}%" for value in values)


def _fmt_gib(values: list[float]) -> str:
    return ", ".join(f"{value:.1f}GiB" for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a command after all GPUs remain below utilization and memory thresholds."
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_COMMAND,
        help="Command to execute once the idle condition is satisfied.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=120,
        help="Polling interval in seconds. Default: 120.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="All GPUs must stay strictly below this utilization percentage. Default: 10.",
    )
    parser.add_argument(
        "--memory-threshold-gb",
        type=float,
        default=None,
        help="Optional per-GPU memory-used threshold in GiB. If set, memory must also stay below this value.",
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=30.0,
        help="Required consecutive low-utilization duration in minutes. Default: 30.",
    )
    parser.add_argument(
        "--report-window-minutes",
        type=float,
        default=30.0,
        help="Print a rolling-window summary at this cadence. Default: 30.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional file to append all status lines and window summaries.",
    )
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parents[2]),
        help="Working directory used when launching the command. Default: repository root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute the command; only report when it would have been launched.",
    )
    return parser.parse_args()


def sample_gpu_status() -> dict:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found in PATH")

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    indices: list[int] = []
    utils: list[float] = []
    mem_used_gb: list[float] = []
    mem_total_gb: list[float] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output line: {line!r}")
        try:
            indices.append(int(parts[0]))
            utils.append(float(parts[1]))
            mem_used_gb.append(float(parts[2]) / 1024.0)
            mem_total_gb.append(float(parts[3]) / 1024.0)
        except ValueError as exc:
            raise RuntimeError(f"Unexpected nvidia-smi output line: {line!r}") from exc

    if not utils:
        raise RuntimeError("No GPUs reported by nvidia-smi")

    return {
        "indices": indices,
        "utils": utils,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "wall_time": time.monotonic(),
    }


def query_gpu_utilizations() -> list[float]:
    """Compatibility helper for callers that only need utilization."""
    return sample_gpu_status()["utils"]


def all_below_thresholds(sample: dict, threshold: float, memory_threshold_gb: float | None) -> bool:
    util_ok = all(util < threshold for util in sample["utils"])
    mem_ok = memory_threshold_gb is None or all(mem < memory_threshold_gb for mem in sample["mem_used_gb"])
    return util_ok and mem_ok


def window_summary(samples: list[dict], threshold: float, memory_threshold_gb: float | None) -> str:
    n_gpus = len(samples[0]["utils"])
    max_util = [max(sample["utils"][i] for sample in samples) for i in range(n_gpus)]
    avg_util = [sum(sample["utils"][i] for sample in samples) / len(samples) for i in range(n_gpus)]
    max_mem = [max(sample["mem_used_gb"][i] for sample in samples) for i in range(n_gpus)]
    low_count = sum(1 for sample in samples if all_below_thresholds(sample, threshold, memory_threshold_gb))
    span_minutes = (samples[-1]["wall_time"] - samples[0]["wall_time"]) / 60.0 if len(samples) > 1 else 0.0
    return (
        f"Window summary: samples={len(samples)}, span={span_minutes:.1f}min, "
        f"all-low={low_count}/{len(samples)}, "
        f"max_util=[{_fmt_percent(max_util)}], avg_util=[{_fmt_percent(avg_util)}], "
        f"max_mem=[{_fmt_gib(max_mem)}]"
    )


def launch_command(command: str, workdir: str, dry_run: bool, log_file: Path | None) -> int | None:
    if dry_run:
        log(f"[dry-run] Idle window satisfied. Would run: {command}", log_file)
        return None

    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=workdir,
        start_new_session=True,
    )
    log(f"Launched command with PID {proc.pid}: {command}", log_file)
    return proc.pid


def main() -> int:
    args = parse_args()

    interval = args.interval_seconds
    if interval <= 0:
        raise SystemExit("--interval-seconds must be > 0")
    if args.idle_minutes <= 0:
        raise SystemExit("--idle-minutes must be > 0")
    if args.report_window_minutes <= 0:
        raise SystemExit("--report-window-minutes must be > 0")
    if args.memory_threshold_gb is not None and args.memory_threshold_gb <= 0:
        raise SystemExit("--memory-threshold-gb must be > 0 when set")

    idle_required_seconds = args.idle_minutes * 60.0
    report_window_seconds = args.report_window_minutes * 60.0
    workdir = str(Path(args.workdir).resolve())
    log_file = Path(args.log_file) if args.log_file else None
    low_since: float | None = None
    next_report_at: float | None = None
    window_samples: list[dict] = []

    memory_text = "disabled" if args.memory_threshold_gb is None else f"{args.memory_threshold_gb:.1f}GiB"
    log(
        "Starting GPU idle watchdog: "
        f"threshold<{args.threshold:.1f}%, "
        f"memory_threshold={memory_text}, "
        f"interval={interval}s, "
        f"idle_window={args.idle_minutes:.1f}min, "
        f"report_window={args.report_window_minutes:.1f}min, "
        f"workdir={workdir}",
        log_file,
    )
    log(f"Target command: {args.command}", log_file)

    while True:
        try:
            sample = sample_gpu_status()
        except Exception as exc:
            log(f"GPU query failed: {exc}. Retrying in {interval}s.", log_file)
            time.sleep(interval)
            continue

        now = sample["wall_time"]
        all_low = all_below_thresholds(sample, args.threshold, args.memory_threshold_gb)
        window_samples.append(sample)
        cutoff = now - report_window_seconds
        window_samples = [old_sample for old_sample in window_samples if old_sample["wall_time"] >= cutoff]
        if next_report_at is None:
            next_report_at = now + report_window_seconds

        if all_low:
            if low_since is None:
                low_since = now
                elapsed = 0.0
            else:
                elapsed = now - low_since

            remaining = max(idle_required_seconds - elapsed, 0.0)
            log(
                "All GPUs below thresholds. "
                f"util=[{_fmt_percent(sample['utils'])}], "
                f"mem=[{_fmt_gib(sample['mem_used_gb'])}], "
                f"idle_streak={elapsed / 60.0:.1f}min, remaining={remaining / 60.0:.1f}min.",
                log_file,
            )

            if elapsed >= idle_required_seconds:
                launch_command(args.command, workdir, args.dry_run, log_file)
                return 0
        else:
            if low_since is not None:
                elapsed = now - low_since
                log(
                    f"Idle streak reset after {elapsed / 60.0:.1f}min. "
                    f"util=[{_fmt_percent(sample['utils'])}], mem=[{_fmt_gib(sample['mem_used_gb'])}].",
                    log_file,
                )
            else:
                log(
                    f"GPUs are active. util=[{_fmt_percent(sample['utils'])}], "
                    f"mem=[{_fmt_gib(sample['mem_used_gb'])}].",
                    log_file,
                )
            low_since = None

        if now >= next_report_at:
            log(window_summary(window_samples, args.threshold, args.memory_threshold_gb), log_file)
            next_report_at = now + report_window_seconds

        time.sleep(interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130) from None
