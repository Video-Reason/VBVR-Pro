"""Precompute VAE latents and T5 embeddings for VBVR-Dataset, saving to parquet.

The VBVR dataset stores samples inside tar files. This script reads directly
from the tar archives (no extraction needed), encodes videos/images through
the VAE and prompts through T5, and writes parquet files compatible with
LatentDataset.

Single-GPU:
    .venv/bin/python scripts/precompute/vbvr_latents.py \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir data/vbvr/latents \
        --batch_size 4

Single-node multi-GPU (8 GPUs on 1 machine):
    .venv/bin/torchrun --nproc_per_node=8 scripts/precompute/vbvr_latents.py \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir data/vbvr/latents \
        --batch_size 4

Multi-node multi-GPU (e.g. 8 nodes x 2 GPUs = 16 GPUs total):
    # Run on EVERY node (torchrun launches --nproc_per_node processes per node).
    # --output_dir must be on shared storage (NFS/GPFS/Lustre) visible to all nodes.
    .venv/bin/torchrun \
        --nnodes=$WORLD_SIZE --nproc_per_node=1 \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        scripts/precompute/vbvr_latents.py \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir /shared/vbvr/latents \
        --batch_size 4

    With SLURM (srun launches one process per GPU automatically):
    srun --nodes=8 --ntasks-per-node=2 --gpus-per-node=2 \
        .venv/bin/torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=2 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 \
        scripts/precompute/vbvr_latents.py \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir /shared/vbvr/latents \
        --batch_size 4

Resume after failure:
    Add --skip_existing to skip tars whose output parquet already exists.
    This makes it safe to restart without re-processing completed tars.

Tar files are distributed round-robin across all GPUs (global rank). Each GPU
processes its assigned tars and writes one parquet per tar. After all GPUs
finish, rank 0 writes the final JSON config for LatentDataset.

Output format (safetensors):
    Each tar produces one .safetensors file containing:
    - {i}.prompt_embeds    (bf16 tensor)
    - {i}.latents          (bf16 tensor)
    - {i}.condition        (bf16 tensor)
    Metadata header stores: count, prompts (JSON list)
"""

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from loguru import logger
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _init_distributed():
    if not _is_distributed():
        return
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def _barrier():
    if _is_distributed():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Precompute VBVR VAE + T5 embeddings to parquet")
    p.add_argument("--metadata", required=True, help="Path to VBVR metadata.parquet")
    p.add_argument("--tar_dir", required=True, help="Directory containing VBVR tar files")
    p.add_argument("--model_path", required=True, help="Path to Wan2.2 diffusers model")
    p.add_argument("--output_dir", required=True, help="Directory to write output parquet files")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    p.add_argument("--num_frames", type=int, default=81, help="Number of frames to sample")
    p.add_argument("--height", type=int, default=None, help="Target height (computed from max_area if not set)")
    p.add_argument("--width", type=int, default=None, help="Target width (computed from max_area if not set)")
    p.add_argument("--max_area", type=int, default=480 * 832, help="Max pixel area (used when height/width not set)")
    p.add_argument("--tars", nargs="*", default=None, help="Process only these tar files (basenames). Default: all.")
    p.add_argument(
        "--skip_existing", action="store_true", help="Skip tars whose output parquet already exists (for resuming)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_components(model_path: str, device: str):
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_dir = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    logger.info("Loaded tokenizer")

    text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device).eval().requires_grad_(False)
    logger.info("Loaded text encoder")

    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32)
    vae.to(device).eval().requires_grad_(False)
    logger.info("Loaded VAE")

    vae_cfg = vae.config
    latents_mean = torch.tensor(vae_cfg.latents_mean).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    latents_std_inv = (1.0 / torch.tensor(vae_cfg.latents_std)).view(1, vae_cfg.z_dim, 1, 1, 1).to(device)
    scale_spatial = vae_cfg.scale_factor_spatial
    scale_temporal = vae_cfg.scale_factor_temporal

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "latents_mean": latents_mean,
        "latents_std_inv": latents_std_inv,
        "scale_spatial": scale_spatial,
        "scale_temporal": scale_temporal,
    }


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_text(components, prompts: list[str], device: str) -> torch.Tensor:
    import html

    import ftfy
    import regex as re

    def clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    prompts = [clean(p) for p in prompts]
    tokens = components["tokenizer"](
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    embeds = components["text_encoder"](input_ids, mask).last_hidden_state
    embeds = embeds.masked_fill(~mask.bool().unsqueeze(-1), 0)
    return embeds.to(torch.bfloat16)


@torch.no_grad()
def encode_video(components, video: torch.Tensor) -> torch.Tensor:
    vae = components["vae"]
    latents = vae.encode(video.to(vae.dtype)).latent_dist.mode()
    mean = components["latents_mean"].to(dtype=latents.dtype)
    std_inv = components["latents_std_inv"].to(dtype=latents.dtype)
    return ((latents - mean) * std_inv).to(torch.bfloat16)


@torch.no_grad()
def prepare_condition(components, image: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
    vae = components["vae"]
    scale_spatial = components["scale_spatial"]
    scale_temporal = components["scale_temporal"]

    B = image.shape[0]
    cond_video = image.new_zeros((B, 3, num_frames, height, width))
    cond_video[:, :, 0] = image

    cond_latents = vae.encode(cond_video.to(vae.dtype)).latent_dist.mode()
    mean = components["latents_mean"].to(dtype=cond_latents.dtype)
    std_inv = components["latents_std_inv"].to(dtype=cond_latents.dtype)
    cond_latents = ((cond_latents - mean) * std_inv).to(torch.bfloat16)

    latent_h = height // scale_spatial
    latent_w = width // scale_spatial
    mask = torch.ones(1, 1, num_frames, latent_h, latent_w, device=image.device, dtype=cond_latents.dtype)
    mask[:, :, 1:] = 0
    first_frame_mask = mask[:, :, 0:1].repeat(1, 1, scale_temporal, 1, 1)
    mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)
    mask = mask.view(1, -1, scale_temporal, latent_h, latent_w).transpose(1, 2).contiguous()
    mask = mask.expand(B, -1, -1, -1, -1)

    return torch.cat([mask, cond_latents], dim=1)


# ---------------------------------------------------------------------------
# Video loading from tar
# ---------------------------------------------------------------------------


def load_video_from_tar(
    tar: tarfile.TarFile, member_path: str, height: int, width: int, num_frames: int
) -> torch.Tensor:
    """Load and resize a video from inside a tar archive. Returns uint8 (C, T, H, W)."""
    import decord

    decord.bridge.set_bridge("torch")

    f = tar.extractfile(member_path)
    if f is None:
        raise FileNotFoundError(f"Cannot extract {member_path} from tar (is it a directory?)")
    video_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        vr = decord.VideoReader(tmp_path, width=width, height=height)
        indices = np.linspace(0, len(vr) - 1, num_frames).round().astype(int).tolist()
        frames = vr.get_batch(indices)  # (T, H, W, C)
        return frames.permute(3, 0, 1, 2).contiguous()
    finally:
        os.unlink(tmp_path)


def load_image_from_tar(tar: tarfile.TarFile, member_path: str, height: int, width: int) -> torch.Tensor:
    """Load and resize an image from inside a tar archive. Returns uint8 (C, H, W)."""
    f = tar.extractfile(member_path)
    if f is None:
        raise FileNotFoundError(f"Cannot extract {member_path} from tar (is it a directory?)")
    img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Process one tar file
# ---------------------------------------------------------------------------


def process_tar(
    tar_path: Path,
    rows: list[dict],
    components: dict,
    device: str,
    args,
    output_dir: Path,
    skip_existing: bool = False,
) -> int:
    """Process all samples from a single tar file, saving each sample immediately.

    Each sample is saved as an individual safetensors file:
        {output_dir}/{tar_stem}_{idx}.safetensors
    containing prompt_embeds, latents, condition, and prompt in metadata.

    Returns the number of samples written.
    """
    from src.data.i2v_dataset import compute_hw

    n = len(rows)
    if n == 0:
        return 0

    # Determine target resolution
    if args.height is not None and args.width is not None:
        h, w = args.height, args.width
    else:
        h, w = compute_hw(args.max_area, 1.0)

    rank = _get_rank()
    tar_stem = tar_path.stem
    samples_done = 0
    samples_skipped = 0
    last_logged = 0
    log_every = 100

    with tarfile.open(tar_path, "r") as tar:
        for batch_start in tqdm(
            range(0, n, args.batch_size),
            desc=f"[rank {rank}] {tar_stem}",
        ):
            batch_end = min(batch_start + args.batch_size, n)
            batch_rows = rows[batch_start:batch_end]
            cur_bs = len(batch_rows)

            # Check which samples in this batch already exist
            if skip_existing:
                batch_indices = list(range(batch_start, batch_end))
                missing = [
                    (i, local_j)
                    for local_j, i in enumerate(batch_indices)
                    if not (output_dir / f"{tar_stem}_{i}.safetensors").exists()
                ]
                if not missing:
                    samples_skipped += cur_bs
                    samples_done += cur_bs
                    continue
                # If some exist and some don't, process the full batch but only save missing ones
                missing_global_indices = {i for i, _ in missing}
            else:
                missing_global_indices = None

            batch_prompts: list[str] = []
            batch_videos: list[torch.Tensor] = []
            batch_images: list[torch.Tensor] = []

            for row in batch_rows:
                batch_prompts.append(row["prompt"])
                video = load_video_from_tar(tar, row["ground_truth_video_path"], h, w, args.num_frames)
                batch_videos.append(video)
                image = load_image_from_tar(tar, row["first_frame_path"], h, w)
                batch_images.append(image)

            # Encode text
            prompt_emb = encode_text(components, batch_prompts, device)

            # Encode video
            video_batch = torch.stack(batch_videos).to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            latents = encode_video(components, video_batch)

            # Encode condition
            image_batch = torch.stack(batch_images).to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
            cond = prepare_condition(components, image_batch, args.num_frames, h, w)

            # Save each sample immediately
            for j in range(cur_bs):
                idx = batch_start + j
                if missing_global_indices is not None and idx not in missing_global_indices:
                    continue

                sample_tensors = {
                    "prompt_embeds": prompt_emb[j].contiguous().cpu(),
                    "latents": latents[j].contiguous().cpu(),
                    "condition": cond[j].contiguous().cpu(),
                }
                sample_meta = {
                    "prompt": batch_prompts[j],
                    "tar": tar_path.name,
                    "index_in_tar": str(idx),
                }
                sample_path = output_dir / f"{tar_stem}_{idx}.safetensors"
                tmp_path = sample_path.with_suffix(".safetensors.tmp")
                save_file(sample_tensors, tmp_path, metadata=sample_meta)
                tmp_path.rename(sample_path)

            samples_done += cur_bs
            if samples_done - last_logged >= log_every:
                logger.info(
                    "[rank {}] {} — {}/{} samples encoded ({:.1f}%)",
                    rank,
                    tar_stem,
                    samples_done,
                    n,
                    samples_done / n * 100,
                )
                last_logged = samples_done

    if samples_skipped > 0:
        logger.info("[rank {}] {} — skipped {} already-existing samples", rank, tar_stem, samples_skipped)
    return samples_done - samples_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    _init_distributed()

    rank = _get_rank()
    world_size = _get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_dir = Path(args.tar_dir)

    # ---- Load metadata and group by tar file ----
    metadata = pq.read_table(args.metadata)
    if rank == 0:
        logger.info("Loaded metadata: {} rows", metadata.num_rows)

    # Build a mapping: tar_basename -> list of row dicts
    tar_to_rows: dict[str, list[dict]] = {}
    for i in range(metadata.num_rows):
        tar_file = metadata.column("tar_file")[i].as_py()  # e.g. "tars/G-11_xxx.tar"
        tar_basename = Path(tar_file).name
        row = {
            "prompt": metadata.column("prompt")[i].as_py(),
            "first_frame_path": metadata.column("first_frame_path")[i].as_py(),
            "ground_truth_video_path": metadata.column("ground_truth_video_path")[i].as_py(),
        }
        tar_to_rows.setdefault(tar_basename, []).append(row)

    # Filter if --tars specified
    if args.tars:
        tar_to_rows = {k: v for k, v in tar_to_rows.items() if k in args.tars}

    tar_names = sorted(tar_to_rows.keys())
    if rank == 0:
        logger.info("Found {} tar files to process", len(tar_names))

    # ---- Skip fully-processed tars if requested ----
    if args.skip_existing:
        before = len(tar_names)
        remaining = []
        for t in tar_names:
            stem = Path(t).stem
            expected = len(tar_to_rows[t])
            existing = len(list(output_dir.glob(f"{stem}_*.safetensors")))
            if existing < expected:
                remaining.append(t)
        skipped = before - len(remaining)
        tar_names = remaining
        if rank == 0 and skipped > 0:
            logger.info("Skipped {} fully-processed tars", skipped)

    # ---- Distribute tars across GPUs (round-robin by global rank) ----
    my_tars = tar_names[rank::world_size]
    logger.info("[rank {}] Assigned {} / {} tars (world_size={})", rank, len(my_tars), len(tar_names), world_size)

    # ---- Load models ----
    logger.info("[rank {}] Loading model components from {}", rank, args.model_path)
    components = load_model_components(args.model_path, device)

    # ---- Process each assigned tar ----
    total_my_samples = sum(len(tar_to_rows[t]) for t in my_tars)
    global_done = 0
    for tar_idx, tar_name in enumerate(my_tars):
        tar_path = tar_dir / tar_name
        rows = tar_to_rows[tar_name]
        logger.info(
            "[rank {}] ({}/{}) Processing {} ({} samples, global {}/{})",
            rank,
            tar_idx + 1,
            len(my_tars),
            tar_name,
            len(rows),
            global_done,
            total_my_samples,
        )

        num_written = process_tar(
            tar_path,
            rows,
            components,
            device,
            args,
            output_dir,
            skip_existing=args.skip_existing,
        )
        global_done += num_written
        logger.info(
            "[rank {}] Done {} ({} samples, global {}/{} = {:.1f}%)",
            rank,
            tar_name,
            num_written,
            global_done,
            total_my_samples,
            global_done / total_my_samples * 100,
        )

    # ---- Gather output paths and write config (rank 0) ----
    _barrier()

    if rank == 0:
        all_shards = sorted(output_dir.glob("*.safetensors"))
        config_entries = [{"data_path": str(p)} for p in all_shards]
        config_path = output_dir / "vbvr_latents.json"
        config_path.write_text(json.dumps(config_entries, indent=2))
        logger.info("Config written to {} ({} entries)", config_path, len(config_entries))

    if _is_distributed():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
