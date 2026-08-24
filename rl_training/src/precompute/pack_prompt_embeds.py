"""Pack variable-length prompt embeddings into sharded packed safetensors.

Reads:  input_dir/rank*_batch*.safetensors  (each with N tensor keys "0".."N-1")
Writes: output_dir/shard_0000.safetensors … shard_NNNN.safetensors
        output_dir/index.parquet

Each shard contains two tensors:
  - embeds:  [total_tokens, 4096]  bf16   — all embeddings concatenated
  - offsets: [num_samples + 1]     int32  — cumulative token counts

index.parquet columns:
  shard_id, local_idx, seq_len, prompt, tar, index_in_tar

Usage:
    .venv/bin/python -m src.precompute.pack_prompt_embeds \
        --input_dir  data/vbvr/latents/prompt_embeds \
        --output_dir data/vbvr/latents/prompt_embeds_packed
"""

import argparse
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def read_safetensors_header(path: str) -> dict:
    """Read only the JSON header (no tensor data)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header


def parse_args():
    p = argparse.ArgumentParser(description="Pack prompt embeddings into sharded packed format")
    p.add_argument("--input_dir", required=True, help="Directory with rank*_batch*.safetensors")
    p.add_argument("--output_dir", required=True, help="Output directory for packed shards")
    p.add_argument("--samples_per_shard", type=int, default=10000, help="Samples per shard")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob("*.safetensors"))
    logger.info("Found {} source files", len(source_files))

    # ------------------------------------------------------------------
    # Pass 1: scan headers only (fast — no tensor data loaded)
    # ------------------------------------------------------------------
    logger.info("Pass 1: Scanning headers …")
    all_samples: list[dict] = []

    for sf_path in tqdm(source_files, desc="Scanning"):
        header = read_safetensors_header(str(sf_path))
        meta_raw = header.pop("__metadata__", {})
        samples_meta = json.loads(meta_raw.get("samples", "[]"))

        for key, info in header.items():
            seq_len = info["shape"][0]
            idx = int(key)
            sm = samples_meta[idx] if idx < len(samples_meta) else {}
            all_samples.append(
                {
                    "source_file": str(sf_path),
                    "key": key,
                    "seq_len": seq_len,
                    "prompt": sm.get("prompt", ""),
                    "tar": sm.get("tar", ""),
                    "index_in_tar": sm.get("index_in_tar", -1),
                }
            )

    logger.info("Total samples: {}", len(all_samples))

    # ------------------------------------------------------------------
    # Sort by seq_len → length-grouped shards
    # ------------------------------------------------------------------
    all_samples.sort(key=lambda s: s["seq_len"])

    for i, s in enumerate(all_samples):
        s["shard_id"] = i // args.samples_per_shard
        s["local_idx"] = i % args.samples_per_shard

    num_shards = all_samples[-1]["shard_id"] + 1
    logger.info("Will create {} shards (~{} samples each)", num_shards, args.samples_per_shard)

    # Group by shard
    shards: dict[int, list[dict]] = {}
    for s in all_samples:
        shards.setdefault(s["shard_id"], []).append(s)

    # ------------------------------------------------------------------
    # Pass 2: pack each shard
    # ------------------------------------------------------------------
    logger.info("Pass 2: Packing shards …")

    file_handles: dict[str, safe_open] = {}

    def get_handle(path: str):
        if path not in file_handles:
            file_handles[path] = safe_open(path, framework="pt")
        return file_handles[path]

    total_tokens = 0

    for shard_id in tqdm(range(num_shards), desc="Packing"):
        shard_samples = shards[shard_id]

        tensors: list[torch.Tensor] = []
        offsets = [0]
        for s in shard_samples:
            t = get_handle(s["source_file"]).get_tensor(s["key"])
            tensors.append(t)
            offsets.append(offsets[-1] + t.shape[0])

        embeds = torch.cat(tensors, dim=0).contiguous()  # [total_tokens, 4096]
        offsets_t = torch.tensor(offsets, dtype=torch.int32)
        total_tokens += embeds.shape[0]

        save_file(
            {"embeds": embeds, "offsets": offsets_t},
            str(output_dir / f"shard_{shard_id:04d}.safetensors"),
        )

    file_handles.clear()

    # ------------------------------------------------------------------
    # Write index parquet
    # ------------------------------------------------------------------
    logger.info("Writing index.parquet …")
    table = pa.table(
        {
            "shard_id": pa.array([s["shard_id"] for s in all_samples], type=pa.int32()),
            "local_idx": pa.array([s["local_idx"] for s in all_samples], type=pa.int32()),
            "seq_len": pa.array([s["seq_len"] for s in all_samples], type=pa.int32()),
            "prompt": pa.array([s["prompt"] for s in all_samples], type=pa.string()),
            "tar": pa.array([s["tar"] for s in all_samples], type=pa.string()),
            "index_in_tar": pa.array([s["index_in_tar"] for s in all_samples], type=pa.int32()),
        }
    )
    pq.write_table(table, str(output_dir / "index.parquet"))

    embed_gb = total_tokens * 4096 * 2 / 1e9
    logger.info(
        "Done! {} shards, {} samples, {} total tokens, ~{:.1f} GB (embeds only)",
        num_shards,
        len(all_samples),
        total_tokens,
        embed_gb,
    )


if __name__ == "__main__":
    main()
