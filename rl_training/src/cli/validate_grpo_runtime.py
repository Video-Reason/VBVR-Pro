"""Cheap, config-aware runtime preflight for GRPO launchers."""

from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime


@dataclass(frozen=True)
class _RuntimeSelection:
    reward: str | None
    attention: str | None
    evalkit_dir: str | None
    evalkit_source_sha256: str | None


def _runtime_selection(argv: Sequence[str] | None = None) -> _RuntimeSelection:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--grpo_reward_fn")
    parser.add_argument("--attention_backend")
    parser.add_argument("--vbvr_reward_evalkit_dir")
    parser.add_argument("--vbvr_reward_evalkit_source_sha256")
    args, _ = parser.parse_known_args(argv)

    config: dict = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(config, dict):
            raise TypeError(f"GRPO config must contain a mapping: {args.config}")
    return _RuntimeSelection(
        reward=args.grpo_reward_fn or config.get("grpo_reward_fn"),
        attention=args.attention_backend or config.get("attention_backend"),
        evalkit_dir=args.vbvr_reward_evalkit_dir or config.get("vbvr_reward_evalkit_dir"),
        evalkit_source_sha256=(
            args.vbvr_reward_evalkit_source_sha256 or config.get("vbvr_reward_evalkit_source_sha256")
        ),
    )


def _selected_runtime(argv: Sequence[str] | None = None) -> tuple[str | None, str | None]:
    selection = _runtime_selection(argv)
    return selection.reward, selection.attention


def _selected_reward(argv: Sequence[str] | None = None) -> str | None:
    """Backward-compatible reward-only view used by callers and tests."""
    return _selected_runtime(argv)[0]


def validate_vbvr_evalkit_contract(evalkit_dir: str | None, expected_source_sha256: str | None) -> dict[str, str]:
    """Validate the explicit external evaluator path and source fingerprint."""
    if not evalkit_dir:
        raise ValueError("vbvr_rule requires vbvr_reward_evalkit_dir")
    if not expected_source_sha256:
        raise ValueError("vbvr_rule requires vbvr_reward_evalkit_source_sha256")
    expected = expected_source_sha256.lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("vbvr_reward_evalkit_source_sha256 must be a 64-character hexadecimal digest")
    directory = Path(evalkit_dir).expanduser().resolve()
    actual = evalkit_source_sha256(directory)
    if actual != expected:
        raise RuntimeError(
            f"EvalKit source fingerprint mismatch: expected={expected}, actual={actual}, path={directory}"
        )
    return {"path": str(directory), "sha256": actual}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        selection = _runtime_selection(argv)
    except Exception as exc:
        print(f"[error] Could not resolve the GRPO runtime contract: {exc}", file=sys.stderr)
        return 1
    reward = selection.reward
    attention = selection.attention
    if reward == "vbvr_rule":
        try:
            evalkit_report = validate_vbvr_evalkit_contract(
                selection.evalkit_dir,
                selection.evalkit_source_sha256,
            )
            report = validate_vbvr_scorer_runtime()
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        key_versions = ", ".join(
            f"{name}={report['distributions'][name]}"
            for name in ("opencv-python", "opencv-python-headless", "numpy", "scipy", "scikit-image")
        )
        print(
            "[preflight] VBVR scorer runtime passed on "
            f"{socket.gethostname()}: sha256={report['sha256']} "
            f"cv2={report['modules']['cv2']} HoughLinesP={report['probes']['hough_lines_p_layout']}"
        )
        print(
            "[preflight] External EvalKit source passed: "
            f"sha256={evalkit_report['sha256']} path={evalkit_report['path']}"
        )
        print(f"[preflight] VBVR scorer key distributions: {key_versions}")
    else:
        print(f"[preflight] No VBVR scorer runtime required for grpo_reward_fn={reward!r}")

    if attention is not None:
        try:
            from src.trainer.utils import prepare_diffusers_attention_backend

            prepare_diffusers_attention_backend(attention)
        except Exception as exc:
            print(f"[error] Could not prepare attention backend {attention!r}: {exc}", file=sys.stderr)
            return 1
        print(f"[preflight] Attention backend ready on {socket.gethostname()}: {attention}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
