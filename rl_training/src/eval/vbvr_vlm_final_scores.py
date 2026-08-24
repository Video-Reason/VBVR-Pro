"""EvalKit-compatible text summaries for offline VBVR VLM judge cells."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def _score(section: dict[str, Any], label: str) -> float:
    try:
        value = float(section["mean_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"VLM cell summary has no valid {label} mean score") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"VLM cell summary {label} mean score must be in [0, 1], got {value!r}")
    return value


def _count(section: dict[str, Any], label: str) -> int:
    try:
        value = int(section["num_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"VLM cell summary has no valid {label} sample count") from exc
    if value < 0 or value != section["num_samples"]:
        raise ValueError(f"VLM cell summary {label} sample count must be a nonnegative integer")
    return value


def render_final_scores(cell_summary: dict[str, Any]) -> str:
    """Render the compact ``final_scores.txt`` format used by VBVR EvalKit."""

    if cell_summary.get("state") != "complete":
        raise ValueError("Cannot export final_scores.txt from an incomplete VLM cell summary")
    try:
        summary = cell_summary["summary"]
        overall = summary["overall"]
        in_domain = summary["In_Domain"]
        out_of_domain = summary["Out_of_Domain"]
    except (KeyError, TypeError) as exc:
        raise ValueError("VLM cell summary is missing overall/domain statistics") from exc
    if not all(isinstance(section, dict) for section in (overall, in_domain, out_of_domain)):
        raise ValueError("VLM cell summary statistics must be objects")

    overall_score = _score(overall, "overall")
    in_domain_score = _score(in_domain, "in-domain")
    out_of_domain_score = _score(out_of_domain, "out-of-domain")
    sample_count = _count(overall, "overall")
    in_domain_count = _count(in_domain, "in-domain")
    out_of_domain_count = _count(out_of_domain, "out-of-domain")
    if sample_count != in_domain_count + out_of_domain_count:
        raise ValueError(
            f"VLM cell summary domain counts do not add up: {sample_count} != {in_domain_count} + {out_of_domain_count}"
        )
    by_task = overall.get("by_task")
    if not isinstance(by_task, dict):
        raise ValueError("VLM cell summary is missing per-task statistics")

    return (
        f"Overall:        {overall_score:.6f}\n"
        f"In-Domain:      {in_domain_score:.6f}\n"
        f"Out-of-Domain:  {out_of_domain_score:.6f}\n"
        "\n"
        f"Samples: {sample_count} ({in_domain_count} in-domain + {out_of_domain_count} out-of-domain)\n"
        f"Tasks:   {len(by_task)}\n"
    )


def write_cell_final_scores(output_dir: Path, cell_summary: dict[str, Any] | None = None) -> Path:
    """Atomically write ``final_scores.txt`` beside a complete cell summary."""

    output_dir = output_dir.expanduser().resolve()
    summary_path = output_dir / "summary.json"
    if cell_summary is None:
        try:
            cell_summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read valid VLM cell summary from {summary_path}: {exc}") from exc
    text = render_final_scores(cell_summary)
    output_path = output_dir / "final_scores.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".{output_path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path
