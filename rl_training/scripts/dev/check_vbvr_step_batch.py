"""Inspect VBVR WebDataset samples consumed around a training step.

This is a read-only debugging helper for latent SFT runs.  It reconstructs the
same WebDataset shard split used by expert-parallel SFT and checks the tensors
in the target DataLoader batch for non-finite values and unusual magnitudes.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import webdataset as wds
from safetensors.torch import load as st_load
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class RankShardSplitter:
    def __init__(self, rank: int, world_size: int):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.rank = rank
        self.world_size = world_size

    def __call__(self, src, group=None):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


def decode_sample(sample: dict[str, Any], max_text_len: int = 512) -> dict[str, Any]:
    tensors = st_load(sample["safetensors"])
    prompt_embeds = tensors["prompt_embeds"]
    seq_len = prompt_embeds.shape[0]
    if seq_len < max_text_len:
        prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, max_text_len - seq_len))
    elif seq_len > max_text_len:
        prompt_embeds = prompt_embeds[:max_text_len]

    meta = json.loads(sample["json"].decode("utf-8"))
    decoded = {
        "__key__": sample.get("__key__", ""),
        "meta": meta,
        "prompt_embeds": prompt_embeds,
        "condition": tensors["condition"],
        "video_latents": tensors["latents"],
    }
    return decoded


class DebugVBVRLatentDataset(IterableDataset):
    def __init__(
        self,
        webdataset_dir: str,
        *,
        node_rank: int,
        node_world_size: int,
        epoch_length: int,
        seed: int,
        shuffle_buffer: int,
    ):
        shard_dir = Path(webdataset_dir)
        shard_paths = sorted(shard_dir.glob("shard-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard-*.tar files found in {shard_dir}")

        urls = [str(p) for p in shard_paths]
        self._rank_shard_count = (len(urls) + node_world_size - 1 - node_rank) // node_world_size
        pipeline = wds.WebDataset(
            urls,
            nodesplitter=RankShardSplitter(node_rank, node_world_size),
            shardshuffle=len(urls),
            seed=seed,
            empty_check=False,
        )
        if shuffle_buffer > 0:
            pipeline = pipeline.shuffle(shuffle_buffer, seed=seed)
        self._pipeline = pipeline.map(decode_sample)
        self._epoch_length = epoch_length

    def _worker_epoch_length(self) -> int:
        worker = get_worker_info()
        if worker is None or worker.num_workers <= 1:
            return self._epoch_length

        active_workers = min(worker.num_workers, self._rank_shard_count, self._epoch_length)
        if active_workers <= 0 or worker.id >= active_workers:
            return 0
        base, extra = divmod(self._epoch_length, active_workers)
        return base + int(worker.id < extra)

    def __iter__(self):
        limit = self._worker_epoch_length()
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


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    local = t.detach()
    if not torch.is_floating_point(local):
        return {"shape": list(local.shape), "dtype": str(local.dtype)}
    f = local.float()
    finite = torch.isfinite(f)
    finite_count = int(finite.sum().item())
    total = f.numel()
    out: dict[str, Any] = {
        "shape": list(local.shape),
        "dtype": str(local.dtype),
        "finite": finite_count == total,
        "nan": int(torch.isnan(f).sum().item()),
        "inf": int(torch.isinf(f).sum().item()),
    }
    if finite_count:
        ff = f[finite]
        out.update(
            {
                "min": float(ff.min().item()),
                "max": float(ff.max().item()),
                "mean": float(ff.mean().item()),
                "std": float(ff.std(unbiased=False).item()),
                "absmax": float(ff.abs().max().item()),
            }
        )
    return out


def inspect_batch(batch: list[dict[str, Any]], *, data_rank: int, batch_idx: int, global_step_label: int) -> None:
    print(f"\n=== data_rank={data_rank} batch_idx={batch_idx} global_step_label={global_step_label} ===")
    for i, sample in enumerate(batch):
        meta = sample["meta"]
        summaries = {
            "prompt_embeds": tensor_summary(sample["prompt_embeds"]),
            "condition": tensor_summary(sample["condition"]),
            "video_latents": tensor_summary(sample["video_latents"]),
        }
        bad = [
            name
            for name, summary in summaries.items()
            if summary.get("finite") is False or float(summary.get("absmax", 0.0)) > 1.0e4
        ]
        status = "BAD" if bad else "ok"
        print(
            json.dumps(
                {
                    "status": status,
                    "local_sample": i,
                    "key": sample["__key__"],
                    "tar": meta.get("tar"),
                    "index_in_tar": meta.get("index_in_tar"),
                    "seq_len": meta.get("seq_len"),
                    "prompt": meta.get("prompt"),
                    "summaries": summaries,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--webdataset-dir", default="data/vbvr/latents/vbvr_384x384x81/webdataset/sft")
    parser.add_argument("--dataset-size", type=int, default=800000)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument(
        "--step",
        type=int,
        default=289,
        help=(
            "Step label from the trainer error/log. The script checks both batch_idx=step "
            "and batch_idx=step-1 because errors are reported before global_step increments."
        ),
    )
    parser.add_argument(
        "--data-rank",
        type=int,
        default=None,
        help="Rank within the data-parallel stream. For EP duplicate mode this is global_rank %% 16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_world_size = args.world_size // 2 if args.expert_parallel else args.world_size
    if args.world_size % 2 != 0 and args.expert_parallel:
        raise ValueError("--world-size must be even when --expert-parallel is set")
    epoch_length = args.dataset_size // data_world_size
    ranks = [args.data_rank] if args.data_rank is not None else list(range(data_world_size))
    target_batch_indices = sorted({max(args.step - 1, 0), args.step})

    for data_rank in ranks:
        if data_rank is None or not 0 <= data_rank < data_world_size:
            raise ValueError(f"--data-rank must be in [0, {data_world_size}), got {data_rank}")
        dataset = DebugVBVRLatentDataset(
            args.webdataset_dir,
            node_rank=data_rank,
            node_world_size=data_world_size,
            epoch_length=epoch_length,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=lambda batch: batch,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        for batch_idx, batch in enumerate(loader):
            if batch_idx in target_batch_indices:
                inspect_batch(batch, data_rank=data_rank, batch_idx=batch_idx, global_step_label=args.step)
            if batch_idx >= max(target_batch_indices):
                break


if __name__ == "__main__":
    main()
