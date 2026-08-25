#!/usr/bin/env python3
"""Validate that the training VBVR reward equals final main_v2 evaluation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from diffusers.utils import export_to_video
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cli.prepare_vbvr_eval_videos import prepare_video  # noqa: E402
from src.eval.vbvr_run_evaluation_parallel import (  # noqa: E402
    _init_worker,
    _score_one,
    evalkit_source_sha256,
)
from src.trainer.rewards.vbvr_rule import VBVRRuleReward  # noqa: E402


def _read_rgb_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    return np.stack(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evalkit-dir", type=Path, required=True)
    parser.add_argument("--easyocr-module-path", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-fps", type=int, default=16)
    parser.add_argument("--prepared-width", type=int, default=1024)
    parser.add_argument("--prepared-height", type=int, default=1024)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evalkit = args.evalkit_dir.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    workdir = args.work_dir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source_sha256 = evalkit_source_sha256(evalkit)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256:
        raise RuntimeError(
            f"EvalKit source fingerprint mismatch: expected={args.expected_source_sha256}, actual={source_sha256}"
        )

    prompt_path = gt_dir / "prompt.txt"
    metadata_path = gt_dir / "metadata.json"
    cfg = SimpleNamespace(
        vbvr_reward_evalkit_dir=str(evalkit),
        vbvr_reward_evalkit_source_sha256=source_sha256,
        vbvr_reward_easyocr_module_path=(
            str(args.easyocr_module_path.expanduser().resolve()) if args.easyocr_module_path else None
        ),
        vbvr_reward_device="cpu",
        vbvr_reward_fps=args.source_fps,
        vbvr_reward_prepared_width=args.prepared_width,
        vbvr_reward_prepared_height=args.prepared_height,
        vbvr_reward_max_duration_seconds=args.max_duration,
        vbvr_reward_prepare_crf=args.crf,
        vbvr_reward_decode_batch_size=1,
        vbvr_reward_cpu_workers=1,
        vbvr_reward_cpu_threads_per_worker=args.threads_per_worker,
        vbvr_reward_use_process_pool=True,
        vbvr_reward_task_specific_only=True,
        vbvr_reward_fail_on_error=True,
        vbvr_reward_tmp_dir=str(workdir / "tmp"),
        vbvr_reward_keep_tmp=True,
        vbvr_reward_unsupported_score=0.0,
    )
    source_frames = _read_rgb_video(args.video.expanduser().resolve())
    trainer = SimpleNamespace(rank=0, tensor_parallel_enabled=False, tp_rank=0)
    reward = VBVRRuleReward(trainer, cfg)
    try:
        training_score = reward._score_in_dir(
            workdir,
            args.task_name,
            prompt_path.read_text().strip(),
            source_frames,
            None,
            gt_video_path=str(gt_dir / "ground_truth.mp4"),
            gt_first_frame_path=str(gt_dir / "first_frame.png"),
            gt_final_frame_path=str(gt_dir / "final_frame.png"),
            gt_metadata_path=str(metadata_path),
            gt_source_dir=str(gt_dir),
        )
    finally:
        reward.close()

    final_raw_path = workdir / "final_eval_raw.mp4"
    final_prepared_path = workdir / "final_eval_prepared.mp4"
    export_to_video(
        [Image.fromarray(frame) for frame in source_frames],
        str(final_raw_path),
        fps=args.source_fps,
    )
    prepare_video(
        final_raw_path,
        final_prepared_path,
        width=args.prepared_width,
        height=args.prepared_height,
        max_duration=args.max_duration,
        crf=args.crf,
        force=True,
    )

    gt_info = {
        "gt_video_path": str(gt_dir / "ground_truth.mp4"),
        "gt_first_frame": str(gt_dir / "first_frame.png"),
        "gt_final_frame": str(gt_dir / "final_frame.png"),
        "prompt": prompt_path.read_text().strip(),
        "gt_path": str(gt_dir),
        "metafile_path": [str(metadata_path)],
    }
    final_task = {
        "video_path": str(final_prepared_path),
        "video_file": "alignment-check.mp4",
        "task_name": args.task_name,
        "split": "alignment-check",
        "category": "alignment-check",
        "folder": "alignment-check",
        "gt_info": gt_info,
        "device": "cpu",
    }
    easyocr_module_path = str(args.easyocr_module_path.expanduser().resolve()) if args.easyocr_module_path else None
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(evalkit), args.threads_per_worker, easyocr_module_path, True),
    ) as pool:
        final_result = pool.submit(_score_one, final_task).result()
    if final_result.get("error"):
        raise RuntimeError(f"Final evaluator failed: {final_result['error']}")
    final_score = float(final_result["score"])
    difference = abs(training_score - final_score)
    training_raw_frames = _read_rgb_video(workdir / "generated_raw.mp4")
    final_raw_frames = _read_rgb_video(final_raw_path)
    training_prepared_frames = _read_rgb_video(workdir / "generated.mp4")
    final_prepared_frames = _read_rgb_video(final_prepared_path)
    raw_frames_equal = np.array_equal(training_raw_frames, final_raw_frames)
    prepared_frames_equal = np.array_equal(training_prepared_frames, final_prepared_frames)

    capture = cv2.VideoCapture(str(workdir / "generated.mp4"))
    prepared = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
    }
    capture.release()
    output = {
        "task_name": args.task_name,
        "evalkit_source_sha256": source_sha256,
        "training_reward": training_score,
        "final_eval": final_score,
        "absolute_difference": difference,
        "raw_frames_equal": raw_frames_equal,
        "prepared_frames_equal": prepared_frames_equal,
        "prepared_video": prepared,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if not raw_frames_equal:
        raise RuntimeError("Training/final source video frames differ")
    if not prepared_frames_equal:
        raise RuntimeError("Training/final prepared video frames differ")
    if difference > 1e-12:
        raise RuntimeError(f"Training/final score mismatch: {difference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
