"""Precompute latent WebDataset shards from a standard I2V training config.

This turns a regular parquet-backed dataset (the same format used by
``configs/precompute_diffsynth_mix_260603_raw_384.yaml``) into the latent
tar-shard layout consumed by ``VBVRLatentDataset``:

  output_dir/shard-000000.tar
    {key}.safetensors  - prompt_embeds, latents, condition
    {key}.json         - prompt, tar, index_in_tar, seq_len

Example:
    .venv/bin/torchrun --nproc_per_node=8 \
        -m src.precompute.i2v_latent_webdataset \
        --config configs/precompute_diffsynth_mix_260603_raw_384.yaml \
        --output_dir storage/latents/example/webdataset \
        --batch_size 4 \
        --samples_per_shard 1000
"""

import argparse
import io
import json
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from loguru import logger
from PIL import Image
from safetensors.torch import save as st_save
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.remote_io import localize_media_path, resolve_media_path
from src.precompute.vbvr_prompt_embeds import encode_text, load_text_encoder
from src.precompute.vbvr_vae_latents import encode_video, load_vae, prepare_condition


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _init_distributed() -> None:
    if not _is_distributed():
        return
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def parse_args():
    p = argparse.ArgumentParser(description="Precompute latent WebDataset from an I2V train config")
    p.add_argument("--config", required=True, help="Path to a non-latent YAML train config")
    p.add_argument("--output_dir", required=True, help="Directory to write shard-*.tar files")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for text/VAE encoding")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for video loading")
    p.add_argument("--samples_per_shard", type=int, default=1000, help="Samples per output tar shard")
    p.add_argument("--shuffle_seed", type=int, default=42, help="Shuffle seed before sharding across ranks")
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap for smoke tests")
    p.add_argument("--no_shuffle", action="store_true", help="Keep dataset order instead of shuffling")
    p.add_argument("--compile_text_encoder", action="store_true", help="Use torch.compile on the text encoder")
    p.add_argument("--compile_vae", action="store_true", help="Use torch.compile on the VAE encoder")
    return p.parse_args()


def _add_tar_entry(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def collate(batch):
    """Collate function for I2V-style batches."""
    collated = {}
    sample = batch[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            collated[key] = torch.stack([x[key] for x in batch])
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            collated[key] = [torch.stack([x[key][i] for x in batch]) for i in range(len(value))]
    if "prompt" in sample:
        collated["prompt"] = [x["prompt"] for x in batch]
    if "index" in sample:
        collated["index"] = torch.tensor([x["index"] for x in batch], dtype=torch.long)
    return collated


def to_model_pixels(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move uint8 pixels to GPU and normalize to [-1, 1] in bf16."""
    return tensor.to(device=device, dtype=torch.bfloat16, non_blocking=True).div(127.5).sub(1.0)


_MOD_VALUE = 16


def compute_hw(max_area: int, aspect_ratio: float) -> tuple[int, int]:
    """Compute (height, width) from a pixel budget and aspect ratio (h/w)."""
    height = round(np.sqrt(max_area * aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    width = round(np.sqrt(max_area / aspect_ratio)) // _MOD_VALUE * _MOD_VALUE
    height = max(height, _MOD_VALUE)
    width = max(width, _MOD_VALUE)
    return height, width


@dataclass
class _ItemConfig:
    num_frames: int
    max_area: int
    fixed_height: int | None = None
    fixed_width: int | None = None
    fps: int = 16


class ParquetI2VDataset:
    """Minimal parquet-backed I2V dataset without decord dependencies."""

    _MAX_RETRIES = 10

    def __init__(
        self,
        json_path: str,
        num_frames: int | None = None,
        max_area: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
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

        self._tables = []
        self._roots: list[Path] = []
        self._configs: list[_ItemConfig] = []
        self._cumulative: list[int] = []

        total = 0
        for entry in entries:
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

            cfg = _ItemConfig(
                num_frames=num_frames if num_frames is not None else entry.get("num_frames", 81),
                max_area=max_area if max_area is not None else entry.get("max_area", 480 * 832),
                fixed_height=height if height is not None else entry.get("height"),
                fixed_width=width if width is not None else entry.get("width"),
                fps=fps if fps is not None else entry.get("fps", 16),
            )

            self._tables.append(table)
            self._roots.append(root)
            self._configs.append(cfg)
            total += n
            self._cumulative.append(total)
            logger.info("Loaded {} rows from {} (root={})", n, data_path, root)

        self._len = total

    def __len__(self) -> int:
        return self._len

    def _locate(self, idx: int) -> tuple[int, int]:
        import bisect

        ti = bisect.bisect_right(self._cumulative, idx)
        local = idx if ti == 0 else idx - self._cumulative[ti - 1]
        return ti, local

    @staticmethod
    def _read_row(table, row: int) -> tuple[str, str, str | None]:
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
    def _resolve(path: str, root: Path) -> str:
        return resolve_media_path(path, root)

    @staticmethod
    def _get_video_hw(video_path: str, cfg: _ItemConfig) -> tuple[int, int]:
        video_path = localize_media_path(video_path)
        if cfg.fixed_height is not None and cfg.fixed_width is not None:
            return cfg.fixed_height, cfg.fixed_width

        meta = iio.immeta(video_path)
        if "source_size" in meta:
            orig_w, orig_h = meta["source_size"]
        elif "size" in meta:
            orig_w, orig_h = meta["size"]
        else:
            props = iio.improps(video_path)
            _, orig_h, orig_w, _ = props.shape

        return compute_hw(cfg.max_area, orig_h / orig_w)

    @staticmethod
    def _resize_video(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        return tensor.permute(1, 0, 2, 3).contiguous()

    @staticmethod
    def _load_video(video_path: str, height: int, width: int, cfg: _ItemConfig) -> torch.Tensor:
        video_path = localize_media_path(video_path)
        frames = iio.imread(video_path)
        total_frames = int(frames.shape[0])
        indices = np.linspace(0, total_frames - 1, cfg.num_frames).round().astype(int)
        sampled = frames[indices]
        if sampled.shape[1] != height or sampled.shape[2] != width:
            return ParquetI2VDataset._resize_video(sampled, height, width)
        return torch.from_numpy(sampled).permute(3, 0, 1, 2).contiguous()

    @staticmethod
    def _load_image(path: str, height: int, width: int) -> torch.Tensor:
        path = localize_media_path(path)
        with Image.open(path) as img:
            img = img.convert("RGB").resize((width, height), Image.LANCZOS)
            array = np.array(img, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int):
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._load_item(idx)
            except Exception:
                logger.warning(
                    "Failed to load item {} (attempt {}/{}), trying another sample.",
                    idx,
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc_info=True,
                )
                idx = random.randint(0, self._len - 1)
        return self._load_item(idx)

    def _load_item(self, idx: int):
        ti, row = self._locate(idx)
        video_path, prompt, image_path = self._read_row(self._tables[ti], row)
        cfg = self._configs[ti]
        root = self._roots[ti]

        target_video_path = self._resolve(video_path, root)
        height, width = self._get_video_hw(target_video_path, cfg)
        video = self._load_video(target_video_path, height, width, cfg)

        if image_path is not None:
            image = self._load_image(self._resolve(image_path, root), height, width)
        else:
            image = video[:, 0].clone()

        return {
            "index": idx,
            "videos": [video],
            "image": image,
            "prompt": prompt,
        }


class TarShardWriter:
    """Rank-local tar shard writer with globally unique shard ids."""

    def __init__(self, output_dir: Path, rank: int, world_size: int, samples_per_shard: int):
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.samples_per_shard = samples_per_shard

        self._local_shard_idx = 0
        self._samples_in_shard = 0
        self._current_tar: tarfile.TarFile | None = None
        self._tmp_path: Path | None = None
        self._final_path: Path | None = None

        self.samples_written = 0
        self.shards_written = 0

    def _next_global_shard_id(self) -> int:
        return self._local_shard_idx * self.world_size + self.rank

    def _open_next_shard(self) -> None:
        shard_id = self._next_global_shard_id()
        self._final_path = self.output_dir / f"shard-{shard_id:06d}.tar"
        self._tmp_path = self.output_dir / f".shard-{shard_id:06d}.rank{self.rank}.tmp"
        if self._final_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing shard: {self._final_path}")
        self._current_tar = tarfile.open(self._tmp_path, "w")  # noqa: SIM115
        self._samples_in_shard = 0

    def _close_current_shard(self) -> None:
        if self._current_tar is None:
            return
        self._current_tar.close()
        assert self._tmp_path is not None
        assert self._final_path is not None
        self._tmp_path.rename(self._final_path)
        self._current_tar = None
        self._tmp_path = None
        self._final_path = None
        self._local_shard_idx += 1
        self.shards_written += 1

    def write(self, key: str, tensors: dict[str, torch.Tensor], metadata: dict) -> None:
        if self._current_tar is None or self._samples_in_shard >= self.samples_per_shard:
            self._close_current_shard()
            self._open_next_shard()

        st_bytes = st_save(tensors)
        meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

        assert self._current_tar is not None
        _add_tar_entry(self._current_tar, f"{key}.safetensors", st_bytes)
        _add_tar_entry(self._current_tar, f"{key}.json", meta_bytes)

        self._samples_in_shard += 1
        self.samples_written += 1

    def close(self) -> None:
        self._close_current_shard()


@dataclass
class SourceTrainConfig:
    model_path: str
    dataset_json: str
    num_frames: int | None = None
    max_area: int | None = None
    height: int | None = None
    width: int | None = None
    fps: int | None = None


def load_train_config(config_path: Path) -> tuple[dict, SourceTrainConfig]:
    raw_cfg = yaml.safe_load(config_path.read_text()) or {}
    if "dataset_json" not in raw_cfg:
        raise ValueError(
            f"{config_path} does not define dataset_json. "
            "Pass the original non-latent train config, not a latent-only one."
        )
    if "model_path" not in raw_cfg:
        raise ValueError(f"{config_path} does not define model_path")
    return raw_cfg, SourceTrainConfig(
        model_path=raw_cfg["model_path"],
        dataset_json=raw_cfg["dataset_json"],
        num_frames=raw_cfg.get("num_frames"),
        max_area=raw_cfg.get("max_area"),
        height=raw_cfg.get("height"),
        width=raw_cfg.get("width"),
        fps=raw_cfg.get("fps"),
    )


def build_index_list(total: int, args) -> list[int]:
    indices = list(range(total))
    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(indices)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]
    return indices


def main():
    args = parse_args()
    _init_distributed()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latent precompute")

    rank = _get_rank()
    world_size = _get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    config_path = Path(args.config)
    _raw_cfg, cfg = load_train_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_shards = 0
    if rank == 0:
        existing_shards = 1 if any(output_dir.glob("shard-*.tar")) else 0
    if _is_distributed():
        flag = torch.tensor([existing_shards], device=device, dtype=torch.int32)
        dist.broadcast(flag, src=0)
        existing_shards = int(flag.item())
    if existing_shards:
        raise FileExistsError(f"{output_dir} already contains shard-*.tar files")

    logger.info("[rank {}] Loading dataset from {}", rank, cfg.dataset_json)
    dataset = ParquetI2VDataset(
        cfg.dataset_json,
        num_frames=cfg.num_frames,
        max_area=cfg.max_area,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
    )

    all_indices = build_index_list(len(dataset), args)
    my_indices = all_indices[rank::world_size]
    subset = Subset(dataset, my_indices)

    logger.info(
        "[rank {}] Assigned {} / {} samples (batch_size={}, workers={})",
        rank,
        len(my_indices),
        len(all_indices),
        args.batch_size,
        args.num_workers,
    )

    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
        drop_last=False,
    )

    logger.info("[rank {}] Loading text encoder from {}", rank, cfg.model_path)
    text_components = load_text_encoder(cfg.model_path, device)
    if args.compile_text_encoder:
        logger.info("[rank {}] Compiling text encoder", rank)
        text_components["text_encoder"] = torch.compile(text_components["text_encoder"])

    logger.info("[rank {}] Loading VAE from {}", rank, cfg.model_path)
    vae_components = load_vae(cfg.model_path, device)
    if args.compile_vae:
        logger.info("[rank {}] Compiling VAE encoder", rank)
        vae_components["vae"].encoder = torch.compile(vae_components["vae"].encoder)

    dataset_tag = Path(cfg.dataset_json).stem
    writer = TarShardWriter(output_dir, rank, world_size, args.samples_per_shard)
    pbar = tqdm(total=len(my_indices), desc=f"[rank {rank}]", position=local_rank, leave=True, unit="sample")

    try:
        for batch in dataloader:
            prompts = batch["prompt"]
            prompt_embeds = encode_text(text_components, prompts, device)

            target_video = batch["videos"][0]
            image = batch["image"]

            target_video_pixels = to_model_pixels(target_video, torch.device(device))
            image_pixels = to_model_pixels(image, torch.device(device))

            num_frames = int(target_video_pixels.shape[2])
            height = int(target_video_pixels.shape[3])
            width = int(target_video_pixels.shape[4])

            video_latents = encode_video(vae_components, target_video_pixels)
            condition = prepare_condition(vae_components, image_pixels, num_frames, height, width)

            for i, prompt in enumerate(prompts):
                global_idx = int(batch["index"][i].item())
                key = f"{global_idx:09d}"
                pe = prompt_embeds[i].contiguous().cpu()
                sample_condition = condition[i].contiguous().cpu()
                tensors = {
                    "prompt_embeds": pe,
                    "condition": sample_condition,
                    "latents": video_latents[i].contiguous().cpu(),
                }

                writer.write(
                    key=key,
                    tensors=tensors,
                    metadata={
                        "prompt": prompt,
                        "tar": dataset_tag,
                        "index_in_tar": global_idx,
                        "seq_len": int(pe.shape[0]),
                    },
                )

            pbar.update(len(prompts))
    finally:
        pbar.close()
        writer.close()

    logger.info(
        "[rank {}] Finished: {} samples across {} shards",
        rank,
        writer.samples_written,
        writer.shards_written,
    )

    if _is_distributed():
        dist.barrier()

    if rank == 0:
        info = {
            "source_config": str(config_path),
            "source_dataset_json": cfg.dataset_json,
            "model_path": cfg.model_path,
            "dataset_size": len(all_indices),
            "samples_per_shard": args.samples_per_shard,
            "world_size": world_size,
            "recommended_train_config": {
                "latent_webdataset_dir": str(output_dir),
                "dataset_size": len(all_indices),
            },
        }
        (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
        logger.info("Wrote {}", output_dir / "dataset_info.json")
        logger.info(
            "Recommended training fields: latent_webdataset_dir={}, dataset_size={}",
            output_dir,
            len(all_indices),
        )

    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
