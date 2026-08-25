"""Report live progress for the 60-cell VBVR trajectory matrix.

The renderer publishes one small atomic manifest per sample shard. This watcher
reads only those manifests (never the million-file media tree), so it is safe to
poll on shared storage every few seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_MODELS = ("baseline", *(str(step) for step in range(100, 1000, 100)))
DEFAULT_SAMPLERS = ("cps0p1", "cps0p3", "cps0p7", "cps0p9", "euler", "unipc")


@dataclass(frozen=True)
class CellProgress:
    model: str
    sampler: str
    completed: int
    total: int
    initial_completed: int
    started_at: float | None
    updated_at: float | None
    state: str


def _cell(value: str) -> tuple[str, str]:
    parts = value.replace("/", ",").split(",")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"Cell must be MODEL,SAMPLER, got {value!r}")
    return parts[0], parts[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--cell", action="append", type=_cell, default=[])
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--samples-per-cell", type=int, default=500)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Print totals only")
    args = parser.parse_args(argv)
    if args.shard_count is not None and args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if args.samples_per_cell <= 0:
        parser.error("--samples-per-cell must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if not args.cell:
        args.cell = [(model, sampler) for sampler in DEFAULT_SAMPLERS for model in DEFAULT_MODELS]
    return args


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _valid_shard_set(cell_root: Path, shard_count: int) -> list[dict[str, Any]] | None:
    payloads: list[dict[str, Any]] = []
    for index in range(shard_count):
        path = cell_root / f"cell_manifest.shard-{index:03d}-of-{shard_count:03d}.json"
        payload = _read_json(path)
        if (
            payload is None
            or payload.get("sample_shard_count") != shard_count
            or payload.get("sample_shard_index") != index
        ):
            return None
        payloads.append(payload)
    return payloads


def _auto_shards(cell_root: Path) -> list[dict[str, Any]] | None:
    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    counts: set[int] = set()
    for path in cell_root.glob("cell_manifest.shard-*-of-*.json"):
        payload = _read_json(path)
        count = payload.get("sample_shard_count") if payload else None
        if isinstance(count, int) and count > 1:
            counts.add(count)
    for count in counts:
        payloads = _valid_shard_set(cell_root, count)
        if payloads is not None:
            candidates.append((max(float(item.get("updated_at_unix", 0.0)) for item in payloads), payloads))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def read_cell_progress(
    trajectory_root: Path,
    model: str,
    sampler: str,
    *,
    shard_count: int | None,
    samples_per_cell: int,
) -> CellProgress:
    cell_root = trajectory_root / f"{model}-{sampler}"
    canonical = _read_json(cell_root / "cell_manifest.json")
    if canonical is not None and canonical.get("state") == "complete":
        total = int(canonical.get("sample_count", samples_per_cell))
        completed = int(canonical.get("completed_count", 0))
        if completed == total:
            return CellProgress(
                model=model,
                sampler=sampler,
                completed=completed,
                total=total,
                initial_completed=int(canonical.get("initial_completed_count", completed)),
                started_at=canonical.get("started_at_unix"),
                updated_at=canonical.get("updated_at_unix"),
                state="complete",
            )

    shards = _valid_shard_set(cell_root, shard_count) if shard_count is not None else _auto_shards(cell_root)
    if shards:
        total = int(shards[0].get("global_sample_count", samples_per_cell))
        completed = sum(int(item.get("completed_selected_count", 0)) for item in shards)
        initial = sum(int(item.get("initial_completed_selected_count", 0)) for item in shards)
        starts = [float(item["started_at_unix"]) for item in shards if item.get("started_at_unix") is not None]
        updates = [float(item["updated_at_unix"]) for item in shards if item.get("updated_at_unix") is not None]
        return CellProgress(
            model=model,
            sampler=sampler,
            completed=min(completed, total),
            total=total,
            initial_completed=min(initial, total),
            started_at=min(starts) if starts else None,
            updated_at=max(updates) if updates else None,
            state="complete" if all(item.get("state") == "complete" for item in shards) else "in_progress",
        )

    if canonical is not None:
        total = int(canonical.get("sample_count", samples_per_cell))
        completed = min(int(canonical.get("completed_count", 0)), total)
        return CellProgress(
            model=model,
            sampler=sampler,
            completed=completed,
            total=total,
            initial_completed=int(canonical.get("initial_completed_count", completed)),
            started_at=canonical.get("started_at_unix"),
            updated_at=canonical.get("updated_at_unix"),
            state=str(canonical.get("state", "pending")),
        )

    return CellProgress(model, sampler, 0, samples_per_cell, 0, None, None, "pending")


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def report(args: argparse.Namespace) -> bool:
    rows = [
        read_cell_progress(
            args.trajectory_root,
            model,
            sampler,
            shard_count=args.shard_count,
            samples_per_cell=args.samples_per_cell,
        )
        for model, sampler in args.cell
    ]
    now = time.time()
    completed = sum(row.completed for row in rows)
    total = sum(row.total for row in rows)
    generated = sum(max(0, row.completed - row.initial_completed) for row in rows)
    starts = [row.started_at for row in rows if row.started_at is not None]
    elapsed = now - min(starts) if starts else 0.0
    rate = generated / elapsed if generated > 0 and elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else None
    complete_cells = sum(row.completed == row.total for row in rows)
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(
        f"[progress] {stamp} cells={complete_cells}/{len(rows)} trajectories={completed}/{total} "
        f"({100.0 * completed / total:.2f}%) run_rate={rate * 60.0:.2f}/min ETA={_duration(eta)}",
        flush=True,
    )
    if not args.compact:
        labels = [f"{row.model}/{row.sampler}={row.completed}/{row.total}" for row in rows]
        for start in range(0, len(labels), 4):
            print("[progress-cells] " + " ".join(labels[start : start + 4]), flush=True)
    return completed == total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        if report(args) or not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
