"""I2V training dataset — parquet-native.

Config JSON points to one or more parquet files:

    Single dataset:
    {"data_path": "/path/to/train.parquet", "root": "/path/to/video/root"}

    Multi-dataset (list):
    [
        {"data_path": "/path/to/a/train.parquet", "root": "/path/to/a/"},
        {"data_path": "/path/to/b/train.parquet", "root": "/path/to/b/"}
    ]

Parquet schema:
    - video: string         — target video path
      OR videos: list<string> — compatibility form; only the final entry is used
    - prompt: string
    - image:  string        — optional reference image (uses the target's first frame if absent)

Per-dataset overrides (optional keys in the config dict):
    num_frames, max_area, height, width, fps
"""

import bisect
import faulthandler
import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import Dataset

from src.data.remote_io import is_remote_path, localize_media_path, resolve_media_path

logger = logging.getLogger(__name__)

try:
    import decord  # type: ignore
except Exception:
    decord = None

_HAS_DECORD = decord is not None and hasattr(decord, "VideoReader")
if _HAS_DECORD and hasattr(decord, "bridge") and hasattr(decord.bridge, "set_bridge"):
    decord.bridge.set_bridge("torch")

# Height/width must be divisible by vae_scale_factor_spatial * patch_size.
# For Wan2.2: 8 * 2 = 16.
_MOD_VALUE = 16


class _ItemConfig(BaseModel):
    num_frames: int
    max_area: int
    fixed_height: int | None = None
    fixed_width: int | None = None
    fps: int


@dataclass(frozen=True)
class _VBVRProSample:
    task_name: str
    sample_id: str
    sample_index: int
    sample_dir: Path
    video_path: Path
    image_path: Path
    final_frame_path: Path | None
    metadata_path: Path
    prompt: str | None = None


def compute_hw(max_area: int, aspect_ratio: float) -> tuple[int, int]:
    """Compute (height, width) from a pixel budget and aspect ratio (h/w)."""
    height = round(np.sqrt(max_area * aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    width = round(np.sqrt(max_area / aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    height = max(height, _MOD_VALUE)
    width = max(width, _MOD_VALUE)
    return height, width


def _decord_num_threads() -> int:
    value = os.environ.get("WAN_TRAINER_DECORD_NUM_THREADS", "1")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning("Invalid WAN_TRAINER_DECORD_NUM_THREADS=%r; using 1", value)
        return 1


class _SlowItemWatchdog:
    def __init__(self, seconds: int, message: str):
        self._seconds = seconds
        self._message = message
        self._timer: threading.Timer | None = None

    def start(self) -> "_SlowItemWatchdog":
        if self._seconds <= 0:
            return self
        self._timer = threading.Timer(self._seconds, self._dump)
        self._timer.daemon = True
        self._timer.start()
        return self

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _dump(self) -> None:
        print(self._message, file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class I2VDataset(Dataset):
    """Parquet-backed video dataset. Rows are read directly from Arrow tables."""

    def __init__(
        self,
        json_path: str,
        num_frames: int | None = None,
        max_area: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        shuffle_indices: bool = False,
        shuffle_seed: int = 42,
        remote_prefetch_lookahead: int = 0,
        remote_prefetch_workers: int = 1,
        remote_prefetch_stride: int = 1,
        item_trace_seconds: int = 0,
    ):
        config_path = Path(json_path)
        parent_dir = config_path.parent
        raw = json.loads(config_path.read_text())

        if isinstance(raw, dict):
            entries = [raw]
        elif isinstance(raw, list):
            entries = raw
        else:
            raise ValueError(f"Config JSON must be a dict or list of dicts: {config_path}")

        self._sources: list[dict[str, Any]] = []
        self._cumulative: list[int] = []

        total = 0
        for entry in entries:
            cfg = _ItemConfig(
                num_frames=num_frames if num_frames is not None else entry.get("num_frames", 81),
                max_area=max_area if max_area is not None else entry.get("max_area", 480 * 832),
                fixed_height=height if height is not None else entry.get("height"),
                fixed_width=width if width is not None else entry.get("width"),
                fps=fps if fps is not None else entry.get("fps", 16),
            )

            fmt = str(entry.get("format", entry.get("type", "parquet"))).lower().replace("-", "_")
            if fmt == "parquet":
                data_path = Path(entry["data_path"])
                if not data_path.is_absolute():
                    data_path = parent_dir / data_path

                table = pq.read_table(data_path)
                n = table.num_rows

                if "root" in entry:
                    root = Path(entry["root"])
                    if not root.is_absolute():
                        root = parent_dir / root
                else:
                    root = data_path.parent

                self._sources.append({"format": "parquet", "table": table, "root": root, "cfg": cfg})
                logger.info("Loaded %d rows from %s (root=%s)", n, data_path, root)
            elif fmt == "vbvr_pro":
                samples = self._load_vbvr_pro_manifest(entry, parent_dir)
                n = len(samples)
                self._sources.append({"format": "vbvr_pro", "samples": samples, "cfg": cfg})
                logger.info("Loaded %d VBVR-Pro %s samples", n, entry.get("split", "train"))
            else:
                raise ValueError(f"Unsupported I2V dataset entry format: {fmt!r}")

            total += n
            self._cumulative.append(total)

        self._len = total
        self._index_map: np.ndarray | None = None
        self._remote_prefetch_lookahead = max(0, int(remote_prefetch_lookahead))
        self._remote_prefetch_workers = max(1, int(remote_prefetch_workers))
        self._remote_prefetch_stride = max(1, int(remote_prefetch_stride))
        self._item_trace_seconds = max(0, int(item_trace_seconds))
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: dict[int, Future] = {}
        if shuffle_indices:
            self._index_map = np.arange(total, dtype=np.int64)
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self._index_map)
            logger.info("Shuffled %d raw dataset indices with seed=%d", total, shuffle_seed)

    # ------------------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------------------

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map global index -> (table_index, local_row)."""
        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    def _source_index(self, idx: int) -> int:
        """Map logical shuffled index -> physical source row index."""
        idx = int(idx)
        if self._index_map is None:
            return idx
        return int(self._index_map[idx])

    # ------------------------------------------------------------------
    # Row reading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_row(table, row: int) -> tuple[str, str, str | None]:
        """Read a single row. Returns (target_video_path, prompt, image_path)."""
        cols = table.column_names

        video_path = table.column("video")[row].as_py() if "video" in cols else None
        if not video_path and "videos" in cols:
            video_paths = table.column("videos")[row].as_py()
            if not video_paths:
                raise ValueError("'videos' column contains an empty list")
            video_path = video_paths[-1]
        if not video_path:
            raise ValueError("Table row has no target in 'video' or 'videos'")

        prompt = table.column("prompt")[row].as_py() if "prompt" in cols else ""
        image = table.column("image")[row].as_py() if "image" in cols else None

        return video_path, prompt, image

    @staticmethod
    def _resolve_config_path(path: str | os.PathLike[str], parent_dir: Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = parent_dir / resolved
        return resolved

    @classmethod
    def _load_vbvr_pro_manifest(cls, entry: dict, parent_dir: Path) -> list[_VBVRProSample]:
        manifest_path = cls._resolve_config_path(entry["split_manifest"], parent_dir)
        split = str(entry.get("split", "train"))
        records = json.loads(manifest_path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"VBVR-Pro split manifest must be a list: {manifest_path}")

        roots_raw = entry.get("data_roots")
        if roots_raw is None:
            root = entry.get("root")
            roots_raw = [root] if root is not None else []
        data_roots = [cls._resolve_config_path(root, parent_dir) for root in roots_raw]
        allowed_task_names = cls._load_allowed_task_names(entry, parent_dir)
        allowed_task_splits_raw = entry.get("allowed_task_splits")
        if isinstance(allowed_task_splits_raw, str):
            allowed_task_splits = {allowed_task_splits_raw}
        elif allowed_task_splits_raw is None:
            allowed_task_splits = None
        else:
            allowed_task_splits = {str(name) for name in allowed_task_splits_raw}
        exclude_sample_ids_from = entry.get("exclude_sample_ids_from_splits", [])
        if isinstance(exclude_sample_ids_from, str):
            exclude_sample_ids_from = [exclude_sample_ids_from]
        exclude_sample_ids_from = [str(name) for name in exclude_sample_ids_from]

        skip_missing = bool(entry.get("skip_missing", False))
        check_files = bool(entry.get("check_files", True))
        samples: list[_VBVRProSample] = []
        missing: list[str] = []
        limit_per_task = entry.get("limit_per_task")
        for record in records:
            if split not in record:
                raise ValueError(f"VBVR-Pro record for {record.get('task')} has no split {split!r}")
            task_name = str(record.get("task") or Path(str(record["source"])).parent.name)
            task_split = str(record.get("split", ""))
            if allowed_task_splits is not None and task_split not in allowed_task_splits:
                continue
            if allowed_task_names is not None and task_name not in allowed_task_names:
                continue
            source = Path(record["source"])
            rel_source = cls._relative_vbvr_pro_source(source, data_roots)
            sample_ids = list(record[split])
            excluded_sample_ids: set[str] = set()
            for excluded_split in exclude_sample_ids_from:
                if excluded_split not in record:
                    raise ValueError(f"VBVR-Pro record for {task_name} has no exclusion split {excluded_split!r}")
                excluded_sample_ids.update(str(sample_id) for sample_id in record[excluded_split])
            if excluded_sample_ids:
                sample_ids = [sample_id for sample_id in sample_ids if str(sample_id) not in excluded_sample_ids]
            if limit_per_task is not None:
                sample_ids = sample_ids[: int(limit_per_task)]

            source_dir: Path | None = None
            if not check_files:
                source_candidates = [root / rel_source for root in data_roots]
                source_candidates.append(source)
                source_dir = next((path for path in source_candidates if path.exists()), source_candidates[0])

            for sample_index, sample_id in enumerate(sample_ids):
                if source_dir is None:
                    candidates = [root / rel_source / sample_id for root in data_roots]
                    candidates.append(source / sample_id)
                    sample_dir = next((path for path in candidates if path.exists()), None)
                    if sample_dir is None:
                        missing.append(str(candidates[0] if candidates else source / sample_id))
                        continue
                else:
                    sample_dir = source_dir / str(sample_id)
                try:
                    samples.append(
                        cls._build_vbvr_pro_sample(
                            task_name,
                            str(sample_id),
                            sample_index,
                            sample_dir,
                            check_files=check_files,
                        )
                    )
                except FileNotFoundError:
                    if not skip_missing:
                        raise
                    missing.append(str(sample_dir))

        if missing and not skip_missing:
            preview = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(f"Missing VBVR-Pro samples for split {split!r}: {preview}{more}")
        if missing:
            logger.warning("Skipped %d missing VBVR-Pro samples for split %s", len(missing), split)
        if not samples:
            raise ValueError(f"No VBVR-Pro samples loaded from {manifest_path} split={split!r}")
        return samples

    @classmethod
    def _load_allowed_task_names(cls, entry: dict, parent_dir: Path) -> frozenset[str] | None:
        allowed = entry.get("allowed_task_names")
        allowed_from_evalkit = entry.get("allowed_task_names_from_evalkit")
        if allowed is None and allowed_from_evalkit is None:
            return None
        task_names = set(str(name) for name in (allowed or []))
        if allowed_from_evalkit is not None:
            evalkit_path = cls._resolve_config_path(allowed_from_evalkit, parent_dir)
            from src.eval.vbvr_run_evaluation_parallel import evalkit_supported_task_names

            task_names.update(evalkit_supported_task_names(evalkit_path))
        return frozenset(task_names)

    @staticmethod
    def _relative_vbvr_pro_source(source: Path, data_roots: list[Path]) -> Path:
        for root in data_roots:
            try:
                return source.relative_to(root)
            except ValueError:
                pass
        for marker in ("VBVR-Pro", "VBVR-Pro_revise"):
            if marker in source.parts:
                return Path(*source.parts[source.parts.index(marker) + 1 :])
        return Path(source.name)

    @classmethod
    def _build_vbvr_pro_sample(
        cls,
        task_name: str,
        sample_id: str,
        sample_index: int,
        sample_dir: Path,
        *,
        check_files: bool,
    ) -> _VBVRProSample:
        video_path = sample_dir / "video" / "ground_truth.mp4"
        image_path = sample_dir / "first_frame.png"
        final_frame_path = sample_dir / "video" / "final_frame.png"
        metadata_path = sample_dir / "metadata.json"
        if check_files:
            for required in (video_path, image_path, metadata_path):
                if not required.exists():
                    raise FileNotFoundError(f"Missing VBVR-Pro sample file: {required}")
        return _VBVRProSample(
            task_name=task_name,
            sample_id=sample_id,
            sample_index=sample_index,
            sample_dir=sample_dir,
            video_path=video_path,
            image_path=image_path,
            final_frame_path=final_frame_path if not check_files or final_frame_path.exists() else None,
            metadata_path=metadata_path,
        )

    @staticmethod
    def _read_vbvr_pro_prompt(sample_dir: Path, metadata_path: Path) -> str:
        for prompt_path in (sample_dir / "video" / "prompt.txt", sample_dir / "image" / "prompt.txt"):
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8").strip()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        render = metadata.get("generic_declarative_render")
        if isinstance(render, dict) and isinstance(render.get("prompt"), str):
            return render["prompt"]
        return ""

    # ------------------------------------------------------------------
    # Path / media helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(path: str, root: Path) -> str:
        return resolve_media_path(path, root)

    @staticmethod
    def _get_video_hw(video_path: str, cfg: _ItemConfig) -> tuple[int, int]:
        if cfg.fixed_height is not None and cfg.fixed_width is not None:
            return cfg.fixed_height, cfg.fixed_width
        video_path = localize_media_path(video_path)
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, num_threads=_decord_num_threads())
            orig_h, orig_w = vr[0].shape[:2]
        else:
            meta = iio.immeta(video_path)
            if "source_size" in meta:
                orig_w, orig_h = meta["source_size"]
            elif "size" in meta:
                orig_w, orig_h = meta["size"]
            else:
                _, orig_h, orig_w, _ = iio.improps(video_path).shape
        return compute_hw(cfg.max_area, orig_h / orig_w)

    @staticmethod
    def _load_video(video_path: str, height: int, width: int, cfg: _ItemConfig) -> torch.Tensor:
        """Load video frames as uint8. Returns (C, T, H, W)."""
        video_path = localize_media_path(video_path)
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, width=width, height=height, num_threads=_decord_num_threads())
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int).tolist()
            frames = vr.get_batch(indices)  # (T, H, W, C)
            return frames.permute(3, 0, 1, 2).contiguous()

        frames = iio.imread(video_path)
        total_frames = int(frames.shape[0])
        indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int)
        frames = frames[indices]
        if frames.shape[1] != height or frames.shape[2] != width:
            tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
            frames = tensor.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()

    @staticmethod
    def _load_image(path: str, height: int, width: int) -> torch.Tensor:
        """Load a single image as uint8. Returns (C, H, W)."""
        path = localize_media_path(path)
        with Image.open(path) as img:
            img = img.convert("RGB").resize((width, height), Image.LANCZOS)
            array = np.array(img, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self._len

    _MAX_RETRIES = 10

    def __getitem__(self, idx):
        self._schedule_remote_prefetch(int(idx))
        for attempt in range(self._MAX_RETRIES):
            watchdog = self._start_item_watchdog(idx, attempt + 1)
            try:
                return self._load_item(idx)
            except Exception:
                logger.warning(
                    "Failed to load item %d (attempt %d/%d), trying another sample.",
                    idx,
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc_info=True,
                )
                idx = random.randint(0, self._len - 1)
            finally:
                watchdog.cancel()
        watchdog = self._start_item_watchdog(idx, self._MAX_RETRIES + 1)
        try:
            return self._load_item(idx)
        finally:
            watchdog.cancel()

    def _start_item_watchdog(self, idx: int, attempt: int) -> _SlowItemWatchdog:
        if self._item_trace_seconds <= 0:
            return _SlowItemWatchdog(0, "")
        return _SlowItemWatchdog(
            self._item_trace_seconds,
            self._format_item_watchdog_message(idx, attempt),
        ).start()

    def _format_item_watchdog_message(self, idx: int, attempt: int) -> str:
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        try:
            source_idx = self._source_index(idx)
            si, row = self._locate(source_idx)
            source = self._sources[si]
            if source["format"] == "parquet":
                video_path, prompt, image_path = self._read_row(source["table"], row)
                root = source["root"]
                paths = [self._resolve(video_path, root)]
                if image_path is not None:
                    paths.append(self._resolve(image_path, root))
            else:
                sample = source["samples"][row]
                prompt = sample.prompt or self._read_vbvr_pro_prompt(sample.sample_dir, sample.metadata_path)
                paths = [str(sample.video_path), str(sample.image_path), str(sample.metadata_path)]
            if len(paths) > 4:
                paths = paths[:4] + [f"... +{len(paths) - 4} more"]
            prompt = prompt.replace("\n", " ")[:160]
            detail = (
                f"logical_idx={idx} source_idx={source_idx} source={si} row={row} "
                f"attempt={attempt}/{self._MAX_RETRIES + 1} paths={paths!r} prompt={prompt!r}"
            )
        except Exception as exc:  # noqa: BLE001
            detail = f"logical_idx={idx} attempt={attempt}/{self._MAX_RETRIES + 1} context_error={exc!r}"
        return (
            f"[I2VDataset watchdog] rank={rank} local_rank={local_rank} pid={os.getpid()} "
            f"item load exceeded {self._item_trace_seconds}s: {detail}"
        )

    def _ensure_prefetch_executor(self) -> ThreadPoolExecutor:
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self._remote_prefetch_workers,
                thread_name_prefix="i2v-remote-prefetch",
            )
        return self._prefetch_executor

    def _prune_prefetch_futures(self) -> None:
        if not self._prefetch_futures:
            return
        done = [idx for idx, future in self._prefetch_futures.items() if future.done()]
        for idx in done:
            future = self._prefetch_futures.pop(idx)
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Remote prefetch failed for logical index %d: %r", idx, exc)

    def _schedule_remote_prefetch(self, idx: int) -> None:
        if self._remote_prefetch_lookahead <= 0:
            return
        self._prune_prefetch_futures()
        max_pending = self._remote_prefetch_lookahead * self._remote_prefetch_workers
        if len(self._prefetch_futures) >= max_pending:
            return

        executor = self._ensure_prefetch_executor()
        for offset in range(1, self._remote_prefetch_lookahead + 1):
            next_idx = idx + offset * self._remote_prefetch_stride
            if next_idx >= self._len:
                break
            if next_idx in self._prefetch_futures:
                continue
            self._prefetch_futures[next_idx] = executor.submit(self._prefetch_remote_media_for_index, next_idx)
            if len(self._prefetch_futures) >= max_pending:
                break

    def _prefetch_remote_media_for_index(self, idx: int) -> None:
        source_idx = self._source_index(idx)
        si, row = self._locate(source_idx)
        source = self._sources[si]
        if source["format"] == "parquet":
            video_path, _prompt, image_path = self._read_row(source["table"], row)
            root = source["root"]
            paths = [self._resolve(video_path, root)]
            if image_path is not None:
                paths.append(self._resolve(image_path, root))
        else:
            sample = source["samples"][row]
            paths = [str(sample.video_path), str(sample.image_path)]
        for path in paths:
            if is_remote_path(path):
                localize_media_path(path)

    def _load_item(self, idx):
        source_idx = self._source_index(idx)
        si, row = self._locate(source_idx)
        source = self._sources[si]
        if source["format"] == "vbvr_pro":
            return self._load_vbvr_pro_item(source_idx, source["samples"][row], source["cfg"])

        video_path, prompt, image_path = self._read_row(source["table"], row)
        cfg = source["cfg"]
        root = source["root"]

        target_video_path = self._resolve(video_path, root)
        height, width = self._get_video_hw(target_video_path, cfg)
        video = self._load_video(target_video_path, height, width, cfg)

        # Reference image: explicit column, or first frame of the target video.
        if image_path is not None:
            image = self._load_image(self._resolve(image_path, root), height, width)
        else:
            image = video[:, 0].clone()

        return {
            "index": source_idx,
            "videos": [video],
            "image": image,
            "prompt": prompt,
        }

    def _load_vbvr_pro_item(self, source_idx: int, sample: _VBVRProSample, cfg: _ItemConfig):
        height, width = self._get_video_hw(str(sample.video_path), cfg)
        video = self._load_video(str(sample.video_path), height, width, cfg)
        image = self._load_image(str(sample.image_path), height, width)
        prompt = sample.prompt or self._read_vbvr_pro_prompt(sample.sample_dir, sample.metadata_path)
        return {
            "index": source_idx,
            "videos": [video],
            "image": image,
            "prompt": prompt,
            "sample_prompt": prompt,
            "sample_task_name": sample.task_name,
            "sample_tar": f"{sample.task_name}.tar",
            "sample_index_in_tar": sample.sample_index,
            "sample_id": sample.sample_id,
            "sample_gt_video_path": str(sample.video_path),
            "sample_gt_first_frame": str(sample.image_path),
            "sample_gt_final_frame": str(sample.final_frame_path or ""),
            "sample_metadata_path": str(sample.metadata_path),
            "sample_source_dir": str(sample.sample_dir),
        }
