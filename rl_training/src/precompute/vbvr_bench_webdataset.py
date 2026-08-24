"""Precompute VBVR-Bench eval dataset as WebDataset tar shards.

For each (task, video_idx) under <gt_base>/{In-Domain_50, Out-of-Domain_50}/
this script:

* encodes the prompt with T5 -> prompt_embeds
* VAE-encodes first_frame with first-frame-only mask -> condition
* bundles the raw first_frame.png / final_frame.png / ground_truth.mp4 so
  downstream scorers (VLM judge, VBVR-EvalKit rule-based) can read them
  directly from the tars without going back to the source GT tree

Output per sample (key = "{split}_{task_name}_{video_idx}"):
    {key}.safetensors   prompt_embeds, condition (bf16)
    {key}.json          task_name, video_idx, split, domain, prompt, h, w, num_frames
    {key}.first.png     raw first frame
    {key}.final.png     raw final frame
    {key}.gt.mp4        raw ground-truth video

Launch (single node, 8 GPUs):
    .venv/bin/torchrun --nproc_per_node=8 \\
        -m src.precompute.vbvr_bench_webdataset \\
        --gt_base data/vbvr/VBVR-Bench \\
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \\
        --output_dir data/vbvr/VBVR-Bench-wds \\
        --samples_per_shard 100
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from PIL import Image
from safetensors.torch import save as st_save
from tqdm import tqdm

from src.precompute.i2v_latent_webdataset import compute_hw
from src.precompute.vbvr_prompt_embeds import encode_text, load_text_encoder
from src.precompute.vbvr_vae_latents import load_vae, prepare_condition

Split = Literal["Open_60", "Hidden_40"]
Domain = Literal["In_Domain", "Out_of_Domain"]

_DOMAIN_DIRS: dict[str, Domain] = {
    "In-Domain_50": "In_Domain",
    "Out-of-Domain_50": "Out_of_Domain",
}


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _init_distributed() -> tuple[int, int, int]:
    if _is_distributed():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_base", type=Path, required=True, help="Dir with In-Domain_50/ and Out-of-Domain_50/")
    p.add_argument("--model_path", type=str, required=True, help="Wan2.2-I2V-A14B-Diffusers directory")
    p.add_argument("--output_dir", type=Path, required=True, help="Where to write shard-*.tar")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--max_area", type=int, default=480 * 832)
    p.add_argument("--samples_per_shard", type=int, default=100)
    p.add_argument(
        "--split_policy",
        type=str,
        default="in_to_open",
        choices=["in_to_open", "all_open"],
        help=(
            "How to label split for each sample. "
            "'in_to_open': In-Domain_50 -> Open_60, Out-of-Domain_50 -> Hidden_40. "
            "'all_open': everything -> Open_60. "
            "Scoring is split-agnostic; this only controls the directory label."
        ),
    )
    p.add_argument("--skip_existing", action="store_true", help="Skip samples whose key appears in any existing shard")
    return p.parse_args()


class _ShardWriter:
    """Rank-local tar shard writer — globally unique shard ids via rank stripe."""

    def __init__(self, output_dir: Path, rank: int, world_size: int, samples_per_shard: int) -> None:
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.samples_per_shard = samples_per_shard
        self._local_idx = 0
        self._in_shard = 0
        self._tar: tarfile.TarFile | None = None
        self._tmp: Path | None = None
        self._final: Path | None = None

    def _open(self) -> None:
        shard_id = self._local_idx * self.world_size + self.rank
        self._final = self.output_dir / f"shard-{shard_id:06d}.tar"
        self._tmp = self.output_dir / f".shard-{shard_id:06d}.rank{self.rank}.tmp"
        self._tar = tarfile.open(self._tmp, "w")  # noqa: SIM115
        self._in_shard = 0

    def _close(self) -> None:
        if self._tar is None:
            return
        self._tar.close()
        assert self._tmp is not None and self._final is not None
        self._tmp.rename(self._final)
        self._tar = None
        self._tmp = None
        self._final = None
        self._local_idx += 1

    @staticmethod
    def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, tensors: dict, metadata: dict, media: dict[str, bytes]) -> None:
        if self._tar is None or self._in_shard >= self.samples_per_shard:
            self._close()
            self._open()
        assert self._tar is not None
        self._add(self._tar, f"{key}.safetensors", st_save(tensors))
        self._add(self._tar, f"{key}.json", json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        for suffix, blob in media.items():
            self._add(self._tar, f"{key}.{suffix}", blob)
        self._in_shard += 1

    def close(self) -> None:
        self._close()


def _split_for(domain_dir: str, policy: str) -> Split:
    if policy == "all_open":
        return "Open_60"
    return "Open_60" if domain_dir == "In-Domain_50" else "Hidden_40"


def _discover_samples(gt_base: Path, split_policy: str) -> list[dict]:
    samples: list[dict] = []
    for dir_name, domain in _DOMAIN_DIRS.items():
        domain_path = gt_base / dir_name
        if not domain_path.is_dir():
            logger.warning("missing {}; skipping", domain_path)
            continue
        split = _split_for(dir_name, split_policy)
        for task_dir in sorted(domain_path.iterdir()):
            if not task_dir.is_dir():
                continue
            for sample_dir in sorted(task_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                first = sample_dir / "first_frame.png"
                final = sample_dir / "final_frame.png"
                gt_video = sample_dir / "ground_truth.mp4"
                prompt_file = sample_dir / "prompt.txt"
                if not all(p.exists() for p in (first, final, gt_video, prompt_file)):
                    logger.warning("incomplete {}; skipping", sample_dir)
                    continue
                samples.append(
                    {
                        "task_name": task_dir.name,
                        "video_idx": sample_dir.name,
                        "split": split,
                        "domain": domain,
                        "first_frame": first,
                        "final_frame": final,
                        "gt_video": gt_video,
                        "prompt": prompt_file.read_text(encoding="utf-8").strip(),
                    }
                )
    return samples


def _scan_existing_keys(output_dir: Path) -> set[str]:
    """Keys already present in any existing shard — used for resume."""
    keys: set[str] = set()
    for tar_path in sorted(output_dir.glob("shard-*.tar")):
        try:
            with tarfile.open(tar_path, "r") as tar:
                for name in tar.getnames():
                    if name.endswith(".safetensors"):
                        keys.add(name[: -len(".safetensors")])
        except Exception as e:
            logger.warning("could not scan {}: {}", tar_path, e)
    return keys


@torch.no_grad()
def _encode_sample(
    sample: dict,
    num_frames: int,
    max_area: int,
    text_components: dict,
    vae_components: dict,
    device: str,
) -> tuple[dict, dict, dict]:
    image_pil = Image.open(sample["first_frame"]).convert("RGB")
    h, w = compute_hw(max_area, image_pil.height / image_pil.width)
    image_resized = image_pil.resize((w, h), Image.Resampling.LANCZOS)
    image_np = np.array(image_resized, dtype=np.uint8)

    image_t = (
        torch.from_numpy(image_np).permute(2, 0, 1).to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
    ).unsqueeze(0)  # (1, 3, H, W)

    prompt_embeds = encode_text(text_components, [sample["prompt"]], device)[0]
    condition = prepare_condition(vae_components, image_t, num_frames, h, w)[0]

    tensors = {
        "prompt_embeds": prompt_embeds.contiguous().cpu(),
        "condition": condition.contiguous().cpu(),
    }
    metadata = {
        "task_name": sample["task_name"],
        "video_idx": sample["video_idx"],
        "split": sample["split"],
        "domain": sample["domain"],
        "prompt": sample["prompt"],
        "height": h,
        "width": w,
        "num_frames": num_frames,
    }
    media = {
        "first.png": sample["first_frame"].read_bytes(),
        "final.png": sample["final_frame"].read_bytes(),
        "gt.mp4": sample["gt_video"].read_bytes(),
    }
    return tensors, metadata, media


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    all_samples = _discover_samples(args.gt_base, args.split_policy)
    if rank == 0:
        logger.info("discovered {} samples under {}", len(all_samples), args.gt_base)

    existing = _scan_existing_keys(args.output_dir) if args.skip_existing else set()
    if args.skip_existing and rank == 0:
        logger.info("resume: {} samples already encoded", len(existing))

    my_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]
    if args.skip_existing:
        my_samples = [s for s in my_samples if f"{s['split']}_{s['task_name']}_{s['video_idx']}" not in existing]
    logger.info("[rank {}] my slice: {} samples", rank, len(my_samples))

    if not my_samples:
        logger.info("[rank {}] nothing to do", rank)
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    logger.info("[rank {}] loading text encoder", rank)
    text_components = load_text_encoder(args.model_path, device)
    logger.info("[rank {}] loading VAE", rank)
    vae_components = load_vae(args.model_path, device)

    writer = _ShardWriter(args.output_dir, rank, world_size, args.samples_per_shard)
    try:
        for sample in tqdm(my_samples, desc=f"[rank {rank}]", position=rank):
            key = f"{sample['split']}_{sample['task_name']}_{sample['video_idx']}"
            tensors, metadata, media = _encode_sample(
                sample,
                args.num_frames,
                args.max_area,
                text_components,
                vae_components,
                device,
            )
            writer.write(key, tensors, metadata, media)
    finally:
        writer.close()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        shards = sorted(args.output_dir.glob("shard-*.tar"))
        logger.info("done — {} shards under {}", len(shards), args.output_dir)


if __name__ == "__main__":
    main()
