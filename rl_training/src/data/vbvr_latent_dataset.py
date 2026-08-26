"""Dataset for VBVR precomputed latents in WebDataset tar-shard format.

Each tar shard contains pairs of files per sample:
  {key}.safetensors  — prompt_embeds, latents, condition
  {key}.json         — prompt, tar, index_in_tar, seq_len

Usage:
    dataset = VBVRLatentDataset("data/vbvr/latents/webdataset")
"""

import hashlib
import json
import logging
import math
import re
import tarfile
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
import webdataset as wds
from safetensors.torch import load as st_load
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)
_LATENTS_KEY_RE = re.compile(r"latents_(\d+)$")


_RESERVED_TENSOR_KEYS = frozenset({"prompt_embeds", "condition", "latents"})


def _json_metadata(sample: dict) -> dict | None:
    metadata = sample.get("json")
    if isinstance(metadata, (bytes, bytearray)):
        try:
            metadata = json.loads(metadata.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return metadata if isinstance(metadata, dict) else None


def _task_name_from_metadata(metadata: dict | None) -> str:
    if metadata is None:
        return ""
    if "task_name" in metadata:
        return str(metadata["task_name"])
    tar_name = metadata.get("tar", "")
    return Path(str(tar_name)).stem if tar_name else ""


def _task_allowed(sample: dict, allowed_task_names: frozenset[str] | None) -> bool:
    if allowed_task_names is None:
        return True
    task_name = _task_name_from_metadata(_json_metadata(sample))
    return task_name in allowed_task_names


def _metadata_int(metadata: dict | None, key: str) -> int | None:
    if metadata is None or key not in metadata:
        return None
    try:
        return int(metadata[key])
    except (TypeError, ValueError):
        return None


def _stable_index_from_parts(metadata: dict | None, key: str, url: str = "") -> int:
    for name in ("split_index", "global_index"):
        value = _metadata_int(metadata, name)
        if value is not None:
            return value

    key_str = str(key)
    if key_str.isdigit():
        return int(key_str)

    identity = f"{url}:{key_str}"
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _stable_sample_index(sample: dict) -> int:
    key = sample.get("__key__", "")
    if isinstance(key, bytes | bytearray):
        key = key.decode("utf-8", errors="replace")
    return _stable_index_from_parts(_json_metadata(sample), str(key), str(sample.get("__url__", "")))


def _build_group_sample_index(urls: list[str]) -> dict[int, int]:
    """Map stable sample ids in ``urls`` to contiguous positions within a shard group."""
    group_index: dict[int, int] = {}
    next_pos = 0
    for url in urls:
        with tarfile.open(url, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                metadata = _json_metadata({"json": f.read()})
                key = Path(member.name).stem
                stable_index = _stable_index_from_parts(metadata, key, url)
                group_index[stable_index] = next_pos
                next_pos += 1
    return group_index


def _sample_rank_allowed(
    sample: dict,
    rank: int,
    world_size: int,
    group_index: dict[int, int] | None = None,
) -> bool:
    if world_size <= 1:
        return True
    stable_index = _stable_sample_index(sample)
    if group_index is not None:
        group_position = group_index.get(stable_index)
        if group_position is None:
            return False
        stable_index = group_position
    return stable_index % world_size == rank


class _ShardSubsetSplitter:
    """Yield one deterministic shard subset: offset, offset+stride, ..."""

    def __init__(self, offset: int, stride: int, num_shards: int):
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not 0 <= offset < stride:
            raise ValueError(f"offset must be in [0, {stride}), got {offset}")
        self.offset = int(offset)
        self.stride = int(stride)
        self.num_shards = int(num_shards)

    @property
    def rank_shard_count(self) -> int:
        if self.offset >= self.num_shards:
            return 0
        return (self.num_shards + self.stride - 1 - self.offset) // self.stride

    def __call__(self, src, group=None):
        yield from islice(src, self.offset, None, self.stride)


class _RankShardSplitter:
    """Split WebDataset shards with an explicit rank/world-size pair."""

    def __init__(self, rank: int, world_size: int, num_shards: int | None = None):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if num_shards is not None and num_shards <= 0:
            raise ValueError(f"num_shards must be positive when provided, got {num_shards}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_shards = int(num_shards) if num_shards is not None else None

    @property
    def rank_shard_count(self) -> int | None:
        if self.num_shards is None:
            return None
        return (self.num_shards + self.world_size - 1 - self.rank) // self.world_size

    def __call__(self, src, group=None):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


def _decode_sample(sample: dict, max_text_len: int = 512) -> dict:
    """Decode a single webdataset sample into training tensors.

    Any non-reserved tensor in the safetensors blob (e.g. ``maze_*`` fields
    emitted by ``src.precompute.maze_webdataset``) is passed through verbatim
    so downstream rewards can consume it.  Callers that don't care simply
    ignore the extra keys.
    """
    tensors = st_load(sample["safetensors"])
    prompt_embeds = tensors["prompt_embeds"]

    seq_len = prompt_embeds.shape[0]
    if seq_len < max_text_len:
        prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_text_len - seq_len))
    elif seq_len > max_text_len:
        prompt_embeds = prompt_embeds[:max_text_len]

    decoded = {
        "prompt_embeds": prompt_embeds,
        "condition": tensors["condition"],
    }
    if "__key__" in sample:
        decoded["sample_key"] = sample["__key__"]
    if "__url__" in sample:
        decoded["sample_url"] = sample["__url__"]

    metadata = _json_metadata(sample)
    if isinstance(metadata, dict):
        for key in ("tar", "index_in_tar", "seq_len", "prompt"):
            if key in metadata:
                decoded[f"sample_{key}"] = metadata[key]
        maze = metadata.get("maze")
        if isinstance(maze, dict) and "path" in maze:
            decoded["maze_path"] = torch.tensor(maze["path"], dtype=torch.int16)

    legacy_latent_keys = sorted(key for key in tensors if _LATENTS_KEY_RE.fullmatch(key))
    if legacy_latent_keys:
        raise KeyError(
            "Numbered target latents are no longer supported; rebuild the sample with one 'latents' tensor "
            f"instead of {legacy_latent_keys}"
        )
    if "latents" not in tensors:
        raise KeyError("Expected a 'latents' tensor in latent sample")
    decoded["video_latents"] = tensors["latents"]

    # Pass through anything else (e.g. maze_* reward metadata).
    for key, value in tensors.items():
        if key in _RESERVED_TENSOR_KEYS or _LATENTS_KEY_RE.fullmatch(key):
            continue
        decoded[key] = value
    return decoded


class VBVRLatentDataset(IterableDataset):
    """WebDataset-based loader for precomputed VAE latents + T5 prompt embeddings.

    Composes a ``wds.WebDataset`` pipeline with shard discovery, distributed
    splitting, shuffle, and decoding.  This is an IterableDataset — set
    ``dataset_size`` in TrainConfig for LR scheduling.

    Args:
        webdataset_dir: Directory containing ``shard-NNNNNN.tar`` files.
        max_text_len: Pad/truncate prompt embeddings to this length.
        shuffle_buffer: Buffer size for sample-level shuffle (0 to disable).
        epoch_length: If set, cap each rank to exactly this many samples per
            epoch.  The cap is split across DataLoader workers so multi-worker
            loading does not multiply the effective epoch length.
        seed: Deterministic shard/sample shuffle seed.
        allowed_task_names: If set, drop samples whose JSON metadata task name
            is not in this allowlist before tensor decoding.
        node_rank: Explicit distributed rank for shard splitting.  Expert
            parallel duplicate mode passes the rank within an expert group so
            the high and low groups see the same shard stream.
        node_world_size: Explicit distributed world size for shard splitting.
    """

    def __init__(
        self,
        webdataset_dir: str,
        max_text_len: int = 512,
        shuffle_buffer: int = 50000,
        epoch_length: int | None = None,
        seed: int | None = None,
        allowed_task_names: set[str] | frozenset[str] | None = None,
        node_rank: int | None = None,
        node_world_size: int | None = None,
    ):
        shard_dir = Path(webdataset_dir)
        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")

        urls = [str(p) for p in shard_paths]
        logger.info("VBVRLatentDataset: %d shards in %s", len(shard_paths), shard_dir)

        if (node_rank is None) != (node_world_size is None):
            raise ValueError("node_rank and node_world_size must be provided together")
        rank_shard_count = None
        sample_rank_split: tuple[int, int, dict[int, int] | None] | None = None
        if node_rank is None:
            nodesplitter = wds.split_by_node
            logger.info("VBVRLatentDataset: shard splitting by global PyTorch rank")
        elif len(urls) < node_world_size:
            shard_group_count = math.gcd(len(urls), node_world_size)
            shard_group = node_rank % shard_group_count
            sample_group_rank = node_rank // shard_group_count
            sample_group_world_size = node_world_size // shard_group_count
            shard_group_urls = urls[shard_group::shard_group_count]
            nodesplitter = _ShardSubsetSplitter(shard_group, shard_group_count, len(urls))
            rank_shard_count = nodesplitter.rank_shard_count
            group_index = _build_group_sample_index(shard_group_urls)
            sample_rank_split = (sample_group_rank, sample_group_world_size, group_index)
            if node_rank == 0:
                logger.warning(
                    "VBVRLatentDataset: only %d shards for %d ranks; using %d shard groups and "
                    "sample-level splitting across %d ranks per group. Each rank reads about %d shards. "
                    "Reshard to at least the training world size for more sequential I/O.",
                    len(urls),
                    node_world_size,
                    shard_group_count,
                    sample_group_world_size,
                    rank_shard_count,
                )
        else:
            nodesplitter = _RankShardSplitter(node_rank, node_world_size, num_shards=len(urls))
            rank_shard_count = nodesplitter.rank_shard_count
            logger.info(
                "VBVRLatentDataset: shard splitting rank=%d/%d (%d shards)",
                node_rank,
                node_world_size,
                rank_shard_count,
            )

        pipeline = wds.WebDataset(
            urls,
            nodesplitter=nodesplitter,
            workersplitter=wds.split_by_worker,
            shardshuffle=len(urls),
            seed=seed,
            empty_check=False,
        )
        if sample_rank_split is not None:
            rank, world_size, group_index = sample_rank_split
            pipeline = pipeline.select(lambda sample: _sample_rank_allowed(sample, rank, world_size, group_index))
        allowed_tasks = frozenset(allowed_task_names) if allowed_task_names is not None else None
        if allowed_tasks is not None:
            logger.info("VBVRLatentDataset: filtering to %d allowed task names", len(allowed_tasks))
            pipeline = pipeline.select(lambda sample: _task_allowed(sample, allowed_tasks))
        if shuffle_buffer > 0:
            pipeline = pipeline.shuffle(shuffle_buffer, seed=seed)
        pipeline = pipeline.map(lambda s: _decode_sample(s, max_text_len))
        self._pipeline = pipeline
        self._epoch_length = epoch_length
        self._rank_shard_count = rank_shard_count
        if epoch_length is not None:
            logger.info("VBVRLatentDataset: epoch_length=%d samples per rank", epoch_length)

    def _worker_epoch_length(self) -> int | None:
        """Return this DataLoader worker's share of the per-rank epoch length."""
        if self._epoch_length is None:
            return None

        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            return self._epoch_length

        active_workers = worker.num_workers
        if self._rank_shard_count is not None:
            active_workers = min(active_workers, self._rank_shard_count)
        active_workers = min(active_workers, self._epoch_length)
        if active_workers <= 0 or worker.id >= active_workers:
            return 0

        base, extra = divmod(self._epoch_length, active_workers)
        return base + int(worker.id < extra)

    def __iter__(self):
        limit = self._worker_epoch_length()
        if limit is None:
            yield from self._pipeline
            return

        yielded = 0
        while yielded < limit:
            produced = 0
            for sample in self._pipeline:
                yield sample
                yielded += 1
                produced += 1
                if yielded >= limit:
                    break
            if produced == 0:
                break
