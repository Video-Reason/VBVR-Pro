"""Plot one audited offline Qwen-judge checkpoint series with a matched baseline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.cli.plot_vbvr_checkpoint_trends import (
    _atomic_write_text,
    _series_payload,
    _write_scores_csv,
    load_vlm_judge_series,
    plot_series,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-judge-root", required=True, type=Path)
    parser.add_argument(
        "--vlm-baseline-root",
        required=True,
        type=Path,
        help="Complete VLM-judge root containing the six matched DiffSynth baselines.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--label",
        default="Qwen-judge-RL model · Qwen3.6-27B task judge",
        help="Plot title.",
    )
    parser.add_argument("--epoch-one-end", type=int, default=1546)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.epoch_one_end is not None and args.epoch_one_end <= 0:
        raise SystemExit("--epoch-one-end must be positive")

    series = load_vlm_judge_series(
        args.vlm_judge_root,
        baseline_root=args.vlm_baseline_root,
        epoch_one_end=args.epoch_one_end,
        key="qwen_judge_rl_qwen_judge",
        label=args.label,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_trends"
    png, svg = plot_series(series, stem, dpi=args.dpi)
    scores_csv = output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_scores.csv"
    _write_scores_csv((series,), scores_csv)
    summary_path = output_dir / "qwen_judge_rl_qwen_judge_sampler_checkpoint_trend_summary.json"
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "series": _series_payload(series),
        "artifacts": {
            "png": str(png),
            "svg": str(svg),
            "scores_csv": str(scores_csv),
        },
    }
    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "series": _series_payload(series),
                "outputs": {
                    "png": str(png),
                    "svg": str(svg),
                    "scores_csv": str(scores_csv),
                    "summary_json": str(summary_path),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
