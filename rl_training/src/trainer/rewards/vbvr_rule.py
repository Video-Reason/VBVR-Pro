"""VBVR EvalKit rule reward for GRPO.

The reward intentionally uses the same main_v2 entrypoint and generated-video
preparation contract as final VBVR-Pro reporting. Generated latents are decoded
to a temporary source MP4, resized/padded/retimed through the shared evaluator
preparer, and scored in isolated CPU processes with CUDA hidden.
"""

from __future__ import annotations

import atexit
import importlib.util
import math
import multiprocessing
import tempfile
import threading
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from loguru import logger

from src.cli.prepare_vbvr_eval_videos import prepare_video
from src.eval.vbvr_run_evaluation_parallel import (
    _init_worker,
    _load_evalkit,
    _normalize_evalkit_result,
    _score_one,
    evalkit_source_sha256,
)
from src.eval.vbvr_run_evaluation_parallel import (
    evalkit_supported_task_names as _evalkit_supported_task_names,
)
from src.eval.vbvr_runtime import validate_vbvr_scorer_runtime
from src.trainer.rewards.base import BaseReward
from src.trainer.rewards.registry import register_reward


@dataclass(slots=True)
class _PendingVBVRReward:
    """Decoded VBVR samples queued for CPU preparation and scoring."""

    batch_size: int
    device: torch.device
    unsupported_score: float
    futures: list[tuple[int, Future[float]]]

    def result(self) -> torch.Tensor:
        values = [self.unsupported_score] * self.batch_size
        try:
            for index, future in self.futures:
                values[index] = future.result()
        except BaseException:
            for _, future in self.futures:
                future.cancel()
            raise
        return torch.tensor(values, device=self.device, dtype=torch.float32)


def _resolve_evalkit_path(evalkit_dir: str | None = None) -> Path:
    if not evalkit_dir:
        raise ValueError(
            "vbvr_rule requires an explicit external EvalKit checkout; "
            "set vbvr_reward_evalkit_dir and vbvr_reward_evalkit_source_sha256"
        )
    return Path(evalkit_dir).expanduser().resolve()


def _ensure_evalkit_path(evalkit_dir: str | None = None) -> Path:
    evalkit = _resolve_evalkit_path(evalkit_dir)
    if not (evalkit / "run_evaluation.py").is_file():
        raise FileNotFoundError(f"VBVR EvalKit path does not contain run_evaluation.py: {evalkit}")
    if not (evalkit / "vbvr_bench").is_dir():
        raise FileNotFoundError(f"VBVR EvalKit path does not contain vbvr_bench: {evalkit}")
    return evalkit


def evalkit_supported_task_names(evalkit_dir: str | None = None) -> frozenset[str]:
    return _evalkit_supported_task_names(_ensure_evalkit_path(evalkit_dir))


def _evalkit_uses_easyocr(evalkit: Path) -> bool:
    return any("easyocr.Reader" in path.read_text(errors="ignore") for path in (evalkit / "vbvr_bench").rglob("*.py"))


def _ensure_easyocr_runtime() -> None:
    """Fail before training when EasyOCR or its packaged character data is incomplete."""
    spec = importlib.util.find_spec("easyocr")
    if spec is None:
        raise ModuleNotFoundError(
            "main_v2 requires EasyOCR, but the `easyocr` package is not installed; "
            "run `uv sync --frozen` before launching training"
        )

    package_paths = [Path(path) for path in spec.submodule_search_locations or ()]
    if not package_paths and spec.origin:
        package_paths.append(Path(spec.origin).parent)
    character_paths = [path / "character" / "en_char.txt" for path in package_paths]
    if not any(path.is_file() for path in character_paths):
        expected = ", ".join(str(path) for path in character_paths) or "<easyocr package>/character/en_char.txt"
        raise FileNotFoundError(
            "EasyOCR is installed incompletely: required English character table is missing "
            f"(expected {expected}); run `uv sync --frozen` before launching training"
        )


@register_reward("vbvr_rule")
class VBVRRuleReward(BaseReward):
    """Reward = final VBVR-Pro main_v2 task-specific rule score."""

    requires_vae: ClassVar[bool] = True

    def __init__(self, trainer, cfg) -> None:
        super().__init__(trainer, cfg)
        self._runtime_report = validate_vbvr_scorer_runtime()
        self._evalkit = _ensure_evalkit_path(cfg.vbvr_reward_evalkit_dir)
        if str(cfg.vbvr_reward_device).lower() != "cpu":
            raise ValueError("Aligned VBVR-Pro reward requires vbvr_reward_device='cpu'")
        if not bool(cfg.vbvr_reward_task_specific_only):
            raise ValueError("main_v2 final evaluation always uses task_specific_only=True")

        actual_source_sha256 = evalkit_source_sha256(self._evalkit)
        expected_source_sha256 = cfg.vbvr_reward_evalkit_source_sha256
        if not expected_source_sha256:
            raise ValueError(
                "vbvr_rule requires vbvr_reward_evalkit_source_sha256; "
                "an unpinned EvalKit can silently change the training reward"
            )
        if actual_source_sha256 != str(expected_source_sha256).lower():
            raise RuntimeError(
                "VBVR EvalKit source fingerprint mismatch: "
                f"expected={str(expected_source_sha256).lower()}, "
                f"actual={actual_source_sha256}, path={self._evalkit}"
            )
        self._evalkit_source_sha256 = actual_source_sha256
        self._task_evaluator_map = evalkit_supported_task_names(str(self._evalkit))

        easyocr_module_path = cfg.vbvr_reward_easyocr_module_path
        self._easyocr_module_path = (
            str(Path(easyocr_module_path).expanduser().resolve()) if easyocr_module_path else None
        )
        if self._easyocr_module_path and not Path(self._easyocr_module_path).is_dir():
            raise FileNotFoundError(f"EasyOCR module path does not exist: {self._easyocr_module_path}")
        if _evalkit_uses_easyocr(self._evalkit):
            _ensure_easyocr_runtime()
            if not (self._evalkit / "easyocr_models").exists():
                raise FileNotFoundError(
                    f"main_v2 requires {self._evalkit / 'easyocr_models'} to exist (a symlink is allowed)"
                )

        self._error_lock = threading.Lock()
        self._score_workers = max(1, int(cfg.vbvr_reward_cpu_workers))
        self._score_threads_per_worker = max(1, int(cfg.vbvr_reward_cpu_threads_per_worker))
        self._decode_batch_size = max(1, int(cfg.vbvr_reward_decode_batch_size))
        configured_pending_jobs = int(getattr(cfg, "vbvr_reward_max_pending_jobs", 0))
        self._max_pending_jobs = configured_pending_jobs or max(self._decode_batch_size, 2 * self._score_workers)
        tensor_parallel_enabled = bool(getattr(trainer, "tensor_parallel_enabled", False))
        tensor_parallel_rank = int(getattr(trainer, "tp_rank", 0))
        self._delayed_min_pending_jobs = 0
        if bool(getattr(cfg, "grpo_delayed_replay", False)) and bool(getattr(cfg, "grpo_shared_prompt_batch", False)):
            logical_world_size = int(
                getattr(trainer, "dp_size", 0) if tensor_parallel_enabled else getattr(trainer, "world_size", 0)
            )
            if logical_world_size > 0:
                local_rollouts = math.ceil(int(cfg.batch_size) * int(cfg.grpo_group_size) / logical_world_size)
                # One full future optimizer step can coexist with the pending
                # replay slot. A smaller semaphore simply moves reward waiting
                # into the next rollout's submit path and defeats the overlap.
                self._delayed_min_pending_jobs = local_rollouts
                self._max_pending_jobs = max(self._max_pending_jobs, local_rollouts)
        self._active_score_rank = not tensor_parallel_enabled or tensor_parallel_rank == 0

        self._process_executor: ProcessPoolExecutor | None = None
        self._local_evalkit = None
        if self._active_score_rank and bool(cfg.vbvr_reward_use_process_pool):
            self._process_executor = ProcessPoolExecutor(
                max_workers=self._score_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    str(self._evalkit),
                    self._score_threads_per_worker,
                    self._easyocr_module_path,
                    True,
                ),
            )
            atexit.register(self.close)
        elif self._active_score_rank:
            # Intended for focused unit tests only. Production uses isolated
            # processes so main_v2/EasyOCR cannot allocate on training GPUs.
            self._local_evalkit = _load_evalkit(self._evalkit)

        self._score_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=self._score_workers,
                thread_name_prefix=f"vbvr-rule-rank{trainer.rank}",
            )
            if self._active_score_rank
            else None
        )
        self._pending_slots = threading.BoundedSemaphore(self._max_pending_jobs) if self._active_score_rank else None
        if self._active_score_rank and self._process_executor is None:
            atexit.register(self.close)
        self._warned_unsupported: set[str] = set()
        self._error_count = 0
        self._tmp_root = Path(cfg.vbvr_reward_tmp_dir) / f"rank{trainer.rank}"
        self._tmp_root.mkdir(parents=True, exist_ok=True)

        if trainer.rank == 0:
            logger.info(
                "VBVR reward aligned with final eval: evalkit={} source_sha256={} "
                "runtime_sha256={} "
                "prepared={}x{} max_duration={}s scorer_processes={} threads/process={} "
                "max_pending_jobs={} delayed_min_pending_jobs={}",
                self._evalkit,
                self._evalkit_source_sha256,
                self._runtime_report["sha256"],
                cfg.vbvr_reward_prepared_width,
                cfg.vbvr_reward_prepared_height,
                cfg.vbvr_reward_max_duration_seconds,
                self._score_workers,
                self._score_threads_per_worker,
                self._max_pending_jobs,
                self._delayed_min_pending_jobs,
            )

    def close(self) -> None:
        score_executor = self._score_executor
        self._score_executor = None
        if score_executor is not None:
            score_executor.shutdown(wait=True, cancel_futures=True)
        process_executor = self._process_executor
        self._process_executor = None
        if process_executor is not None:
            process_executor.shutdown(wait=True, cancel_futures=True)

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
    ) -> _PendingVBVRReward:
        """Decode on the caller thread and immediately queue each decoded batch.

        CPU video preparation and EvalKit scoring continue in the background,
        so the training thread can generate the next rollout chunk before
        resolving the returned handle.
        """
        del condition, prompt_embeds, indices

        batch_size = generated_latents.shape[0]
        device = generated_latents.device

        # EP double-count guard: the expert-parallel GRPO path sums high+low rewards.
        if expert_filter == "high":
            return _PendingVBVRReward(
                batch_size=batch_size,
                device=device,
                unsupported_score=0.0,
                futures=[],
            )
        if not self._active_score_rank:
            raise RuntimeError("VBVR reward was called on a non-scoring tensor-parallel rank")
        if meta is None or not any(key in meta for key in ("sample_task_name", "task_name", "sample_tar")):
            raise RuntimeError(
                "VBVRRuleReward requires task metadata. Expected sample_task_name "
                "or sample_tar plus the original GT paths."
            )

        task_names: list[str] = []
        supported_indices: list[int] = []
        for index in range(batch_size):
            task_name = self._task_name(meta, index)
            task_names.append(task_name)
            if self._is_supported(task_name):
                supported_indices.append(index)

        score_futures: list[tuple[int, Future[float]]] = []
        try:
            for start in range(0, len(supported_indices), self._decode_batch_size):
                batch_indices = supported_indices[start : start + self._decode_batch_size]
                latent_indices = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
                gt_video_paths = [self._meta_path(meta, "sample_gt_video_path", i) for i in batch_indices]
                gt_first_frame_paths = [self._meta_path(meta, "sample_gt_first_frame", i) for i in batch_indices]
                gt_final_frame_paths = [self._meta_path(meta, "sample_gt_final_frame", i) for i in batch_indices]
                gt_metadata_paths = [self._meta_path(meta, "sample_metadata_path", i) for i in batch_indices]
                gt_source_dirs = [self._meta_path(meta, "sample_source_dir", i) for i in batch_indices]
                needs_gt_decode = [
                    video_path is None or first_path is None or final_path is None
                    for video_path, first_path, final_path in zip(
                        gt_video_paths,
                        gt_first_frame_paths,
                        gt_final_frame_paths,
                        strict=True,
                    )
                ]
                gen_latents = generated_latents.index_select(0, latent_indices)
                if any(needs_gt_decode):
                    decoded = self.trainer.model.decode_latents(
                        torch.cat(
                            [
                                gen_latents,
                                gt_video_latents.index_select(0, latent_indices),
                            ],
                            dim=0,
                        )
                    )
                else:
                    decoded = self.trainer.model.decode_latents(gen_latents)
                videos = self._to_uint8_videos(decoded)
                count = len(batch_indices)
                gen_videos = videos[:count]
                gt_videos = videos[count:] if any(needs_gt_decode) else [None] * count

                for local_index, index in enumerate(batch_indices):
                    future = self._submit_score_job(
                        task_name=task_names[index],
                        prompt=str(self._meta_item(meta, "sample_prompt", index, default="") or ""),
                        gen_video=gen_videos[local_index],
                        gt_video=gt_videos[local_index],
                        gt_video_path=gt_video_paths[local_index],
                        gt_first_frame_path=gt_first_frame_paths[local_index],
                        gt_final_frame_path=gt_final_frame_paths[local_index],
                        gt_metadata_path=gt_metadata_paths[local_index],
                        gt_source_dir=gt_source_dirs[local_index],
                        sample_id=self._sample_id(meta, index),
                    )
                    score_futures.append((index, future))

                del decoded
        except BaseException:
            for _, future in score_futures:
                future.cancel()
            raise

        return _PendingVBVRReward(
            batch_size=batch_size,
            device=device,
            unsupported_score=float(self.cfg.vbvr_reward_unsupported_score),
            futures=score_futures,
        )

    def _is_supported(self, task_name: str) -> bool:
        if task_name in self._task_evaluator_map:
            return True
        if task_name not in self._warned_unsupported:
            self._warned_unsupported.add(task_name)
            if self.trainer.rank == 0:
                logger.warning(
                    "VBVR rule reward: task '{}' has no EvalKit evaluator; using unsupported_score={}",
                    task_name,
                    self.cfg.vbvr_reward_unsupported_score,
                )
        return False

    def _submit_score_job(self, **kwargs: Any) -> Future[float]:
        executor = self._score_executor
        pending_slots = self._pending_slots
        if executor is None or pending_slots is None:
            raise RuntimeError("VBVR scorer queue is not initialized on this rank")
        pending_slots.acquire()
        try:
            future = executor.submit(self._score_video_pair, **kwargs)
        except BaseException:
            pending_slots.release()
            raise
        future.add_done_callback(lambda _future: pending_slots.release())
        return future

    def _score_video_pair(
        self,
        *,
        task_name: str,
        prompt: str,
        gen_video,
        gt_video,
        gt_video_path: str | None,
        gt_first_frame_path: str | None,
        gt_final_frame_path: str | None,
        gt_metadata_path: str | None,
        gt_source_dir: str | None,
        sample_id: str,
    ) -> float:
        try:
            if self.cfg.vbvr_reward_keep_tmp:
                workdir = self._tmp_root / f"{sample_id}-{uuid.uuid4().hex[:8]}"
                workdir.mkdir(parents=True, exist_ok=False)
                return self._score_in_dir(
                    workdir,
                    task_name,
                    prompt,
                    gen_video,
                    gt_video,
                    gt_video_path=gt_video_path,
                    gt_first_frame_path=gt_first_frame_path,
                    gt_final_frame_path=gt_final_frame_path,
                    gt_metadata_path=gt_metadata_path,
                    gt_source_dir=gt_source_dir,
                )

            with tempfile.TemporaryDirectory(prefix=f"{sample_id}-", dir=self._tmp_root) as tmp:
                return self._score_in_dir(
                    Path(tmp),
                    task_name,
                    prompt,
                    gen_video,
                    gt_video,
                    gt_video_path=gt_video_path,
                    gt_first_frame_path=gt_first_frame_path,
                    gt_final_frame_path=gt_final_frame_path,
                    gt_metadata_path=gt_metadata_path,
                    gt_source_dir=gt_source_dir,
                )
        except Exception as exc:
            with self._error_lock:
                self._error_count += 1
                error_count = self._error_count
            if self.trainer.rank == 0 and error_count <= 10:
                logger.warning("VBVR rule reward failed for task '{}': {}", task_name, exc)
            if bool(self.cfg.vbvr_reward_fail_on_error):
                raise RuntimeError(f"VBVR main_v2 reward failed for task {task_name}: {exc}") from exc
            return float(self.cfg.vbvr_reward_unsupported_score)

    def _score_in_dir(
        self,
        workdir: Path,
        task_name: str,
        prompt: str,
        gen_video,
        gt_video,
        *,
        gt_video_path: str | None,
        gt_first_frame_path: str | None,
        gt_final_frame_path: str | None,
        gt_metadata_path: str | None,
        gt_source_dir: str | None,
    ) -> float:
        raw_gen_path = workdir / "generated_raw.mp4"
        prepared_gen_path = workdir / "generated.mp4"
        fallback_gt_path = workdir / "ground_truth.mp4"
        fallback_first_path = workdir / "first_frame.png"
        fallback_final_path = workdir / "final_frame.png"

        self._write_video(raw_gen_path, gen_video)
        prepare_video(
            raw_gen_path,
            prepared_gen_path,
            width=int(self.cfg.vbvr_reward_prepared_width),
            height=int(self.cfg.vbvr_reward_prepared_height),
            max_duration=float(self.cfg.vbvr_reward_max_duration_seconds),
            crf=int(self.cfg.vbvr_reward_prepare_crf),
            force=True,
        )
        if gt_video_path is None:
            if gt_video is None:
                raise RuntimeError("Cannot write fallback GT video without decoded GT frames")
            self._write_video(fallback_gt_path, gt_video)
            gt_video_path = str(fallback_gt_path)
        if gt_first_frame_path is None:
            if gt_video is None:
                raise RuntimeError("Cannot write fallback GT first frame without decoded GT frames")
            self._write_image(fallback_first_path, gt_video[0])
            gt_first_frame_path = str(fallback_first_path)
        if gt_final_frame_path is None:
            if gt_video is None:
                raise RuntimeError("Cannot write fallback GT final frame without decoded GT frames")
            self._write_image(fallback_final_path, gt_video[-1])
            gt_final_frame_path = str(fallback_final_path)

        gt_info: dict[str, Any] = {
            "gt_video_path": gt_video_path,
            "gt_first_frame": gt_first_frame_path,
            "gt_final_frame": gt_final_frame_path,
            "prompt": prompt,
        }
        if gt_source_dir:
            gt_info["gt_path"] = gt_source_dir
        if gt_metadata_path:
            # main_v2 accepts an ordered list so v2 metadata can precede legacy
            # annotation fallbacks. Raw VBVR-Pro supplies the v2 file directly.
            gt_info["metafile_path"] = [gt_metadata_path]

        task = {
            "video_path": str(prepared_gen_path),
            "video_file": prepared_gen_path.name,
            "task_name": task_name,
            "split": "",
            "category": "",
            "folder": "",
            "gt_info": gt_info,
            "device": "cpu",
        }
        result = self._evaluate_task(task)
        error = result.get("error")
        if error:
            raise RuntimeError(str(error))
        score = float(result["score"])
        if not math.isfinite(score):
            raise RuntimeError(f"EvalKit returned non-finite score: {score}")
        if not 0.0 <= score <= 1.0:
            raise RuntimeError(f"EvalKit returned score outside [0, 1]: {score}")
        return score

    def _evaluate_task(self, task: dict[str, Any]) -> dict[str, Any]:
        if self._process_executor is not None:
            return self._process_executor.submit(_score_one, task).result()
        if self._local_evalkit is None:
            raise RuntimeError("VBVR scorer is not initialized on this rank")
        result = self._local_evalkit.evaluate_single_video(
            task["video_path"],
            task["task_name"],
            task["gt_info"],
            task["device"],
        )
        return _normalize_evalkit_result(result)

    def _write_video(self, path: Path, video) -> None:
        from diffusers.utils import export_to_video
        from PIL import Image

        if video.ndim != 4:
            raise ValueError(f"Expected video shape (T,H,W,C), got {video.shape}")
        frames = [Image.fromarray(frame) for frame in video]
        export_to_video(frames, str(path), fps=int(self.cfg.vbvr_reward_fps))

    @staticmethod
    def _write_image(path: Path, frame) -> None:
        import cv2

        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write image {path}")

    @staticmethod
    def _to_uint8_videos(decoded: torch.Tensor):
        from src.inference.outputs import uint8_from_decoded

        return uint8_from_decoded(decoded)

    def _task_name(self, meta: dict[str, Any], index: int) -> str:
        for key in ("sample_task_name", "task_name"):
            task_name = str(self._meta_item(meta, key, index, default="") or "")
            if task_name:
                return task_name
        tar_name = str(self._meta_item(meta, "sample_tar", index, default=""))
        task_name = Path(tar_name).stem
        if not task_name:
            raise RuntimeError("VBVRRuleReward could not infer task_name from sample metadata")
        return task_name

    def _sample_id(self, meta: dict[str, Any], index: int) -> str:
        task = self._task_name(meta, index)
        explicit = self._meta_item(meta, "sample_id", index, default=None)
        if explicit:
            return f"{task}-{explicit}"
        sample_index = self._meta_item(meta, "sample_index_in_tar", index, default=index)
        return f"{task}-{sample_index}"

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
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        if not str(value) or not path.exists():
            return None
        # Scorer processes chdir into the pinned EvalKit checkout so its
        # relative annotations resolve exactly as in final evaluation.  A
        # dataset path that is valid relative to the trainer's cwd would
        # otherwise become missing in the worker and many evaluators quietly
        # return a legitimate-looking zero instead of an error.
        return str(path.resolve())
