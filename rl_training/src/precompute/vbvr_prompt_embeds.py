"""Precompute T5 prompt embeddings for VBVR-Dataset (text-only, no VAE).

Launch on 2 nodes x 8 GPUs (16 GPUs total):
    # Run on EVERY node:
    .venv/bin/torchrun \
        --nnodes=2 --nproc_per_node=8 \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        -m src.precompute.vbvr_prompt_embeds \
        --metadata data/vbvr/VBVR-Dataset/data/metadata.parquet \
        --tar_dir data/vbvr/VBVR-Dataset/tars \
        --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
        --output_dir data/vbvr/latents/prompt_embeds \
        --batch_size 64

Output: one safetensors per sample: {tar_stem}_{idx}.safetensors
    - prompt_embeds (bf16)
    - metadata: prompt, tar, index_in_tar
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import torch
from loguru import logger
from safetensors.torch import save_file
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Precompute T5 prompt embeddings for VBVR")
    p.add_argument("--metadata", required=True, help="Path to VBVR metadata.parquet")
    p.add_argument("--tar_dir", required=True, help="Directory containing VBVR tar files (used for grouping only)")
    p.add_argument("--model_path", required=True, help="Path to Wan2.2 diffusers model")
    p.add_argument("--output_dir", required=True, help="Directory to write output safetensors files")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for T5 encoding")
    p.add_argument(
        "--encode_micro_batch_size",
        type=int,
        default=0,
        help=(
            "Optional prompt count per T5 forward inside each saved batch. "
            "Use this to avoid OOM while keeping output batch filenames stable."
        ),
    )
    p.add_argument("--tars", nargs="*", default=None, help="Process only these tar files (basenames). Default: all.")
    p.add_argument("--skip_existing", action="store_true", help="Skip samples whose output already exists")
    p.add_argument("--compile", action="store_true", help="Use torch.compile on text encoder")
    p.add_argument(
        "--output_rank_offset",
        type=int,
        default=0,
        help="Add this offset to torchrun rank when naming rank*_batch*.safetensors outputs",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (T5 only)
# ---------------------------------------------------------------------------


def load_text_encoder(model_path: str, device: str):
    from transformers import AutoTokenizer, UMT5EncoderModel

    model_dir = Path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    logger.info("Loaded tokenizer")

    text_encoder = UMT5EncoderModel.from_pretrained(model_dir / "text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to(device).eval().requires_grad_(False)
    logger.info("Loaded text encoder")

    max_length = tokenizer.model_max_length
    if max_length is None or max_length > 10_000:
        max_length = 512
    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "max_length": max_length}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_text(
    components,
    prompts: list[str],
    device: str,
    micro_batch_size: int = 0,
) -> list[torch.Tensor]:
    import html

    import ftfy
    import regex as re

    max_length = components["max_length"]

    def clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    results = []
    prompts = [clean(p) for p in prompts]
    step = micro_batch_size if micro_batch_size and micro_batch_size > 0 else len(prompts)
    for start in range(0, len(prompts), step):
        chunk = prompts[start : start + step]
        # Dynamic padding: pad to longest in microbatch, not max_length.
        tokens = components["tokenizer"](
            chunk,
            padding="longest",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        embeds = components["text_encoder"](input_ids, mask).last_hidden_state
        embeds = embeds.to(torch.bfloat16)

        # Trim each sample to its actual token length (variable-length output).
        seq_lens = mask.sum(dim=1).tolist()
        for i, length in enumerate(seq_lens):
            results.append(embeds[i, :length].contiguous())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    _init_distributed()

    rank = _get_rank()
    world_size = _get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    output_rank = args.output_rank_offset + rank

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load metadata and group by tar file ----
    metadata = pq.read_table(args.metadata)
    if rank == 0:
        logger.info("Loaded metadata: {} rows", metadata.num_rows)

    tar_to_rows: dict[str, list[dict]] = {}
    for i in range(metadata.num_rows):
        tar_file = metadata.column("tar_file")[i].as_py()
        tar_basename = Path(tar_file).name
        row = {
            "prompt": metadata.column("prompt")[i].as_py(),
            "index": i,
        }
        tar_to_rows.setdefault(tar_basename, []).append(row)

    if args.tars:
        tar_to_rows = {k: v for k, v in tar_to_rows.items() if k in args.tars}

    # Flatten all samples with tar_stem info for round-robin distribution
    all_samples = []
    for tar_name in sorted(tar_to_rows.keys()):
        tar_stem = Path(tar_name).stem
        for idx, row in enumerate(tar_to_rows[tar_name]):
            all_samples.append(
                {
                    "prompt": row["prompt"],
                    "tar_name": tar_name,
                    "tar_stem": tar_stem,
                    "index_in_tar": idx,
                }
            )

    if rank == 0:
        logger.info("Total samples: {}", len(all_samples))

    # Distribute samples round-robin
    my_samples = all_samples[rank::world_size]
    logger.info("[rank {}] Assigned {} / {} samples", rank, len(my_samples), len(all_samples))

    # Batch-level resume happens below after output batch paths are known.
    # ---- Load model ----
    logger.info("[rank {}] Loading text encoder from {}", rank, args.model_path)
    components = load_text_encoder(args.model_path, device)

    if args.compile:
        logger.info("[rank {}] Compiling text encoder with torch.compile", rank)
        components["text_encoder"] = torch.compile(components["text_encoder"])

    # ---- Process in batches ----
    n = len(my_samples)
    done = 0
    num_batches = (n + args.batch_size - 1) // args.batch_size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    pbar = tqdm(total=num_batches, desc=f"[rank {rank}]", position=local_rank, leave=True)

    def _save_batch(batch_idx, batch, embeds_cpu):
        tensors = {str(j): embeds_cpu[j] for j in range(len(batch))}
        meta = {
            "count": str(len(batch)),
            "samples": json.dumps(
                [{"tar": s["tar_name"], "index_in_tar": s["index_in_tar"], "prompt": s["prompt"]} for s in batch]
            ),
        }
        out_path = output_dir / f"rank{output_rank}_batch{batch_idx}.safetensors"
        tmp_path = out_path.with_suffix(".safetensors.tmp")
        save_file(tensors, tmp_path, metadata=meta)
        tmp_path.rename(out_path)

    save_executor = ThreadPoolExecutor(max_workers=1)
    save_future = None

    for batch_idx, batch_start in enumerate(range(0, n, args.batch_size)):
        batch = my_samples[batch_start : batch_start + args.batch_size]
        out_path = output_dir / f"rank{output_rank}_batch{batch_idx}.safetensors"
        if args.skip_existing and out_path.exists():
            done += len(batch)
            pbar.update(1)
            continue

        prompts = [s["prompt"] for s in batch]

        embeds_list = encode_text(components, prompts, device, args.encode_micro_batch_size)
        embeds_cpu = [e.cpu() for e in embeds_list]

        if save_future is not None:
            save_future.result()

        save_future = save_executor.submit(_save_batch, batch_idx, batch, embeds_cpu)

        done += len(batch)
        pbar.update(1)

    if save_future is not None:
        save_future.result()
    save_executor.shutdown(wait=True)

    pbar.close()

    logger.info("[rank {}] Done — {} prompt embeddings saved", rank, done)

    if _is_distributed():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
