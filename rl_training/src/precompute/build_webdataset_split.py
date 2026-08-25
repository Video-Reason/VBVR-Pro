"""Build globally shuffled SFT/RL WebDataset splits from VBVR latents.

Reads:
  prompt_embeds_dir/rank*_batch*.safetensors
  vae_latents_dir/{tar_stem}_{idx}.safetensors

Writes:
  sft_output_dir/shard-NNNNNN.tar
  rl_output_dir/shard-NNNNNN.tar

The split is made after a single global shuffle of all complete samples.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import struct
import tarfile
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value
from pathlib import Path

from loguru import logger
from safetensors import safe_open
from safetensors.torch import save as st_save
from tqdm import tqdm

_counter = None


def _init_worker(counter: Value):
    global _counter
    _counter = counter


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def parse_args():
    p = argparse.ArgumentParser(description="Build globally shuffled SFT/RL WebDataset splits")
    p.add_argument("--prompt_embeds_dir", required=True)
    p.add_argument("--vae_latents_dir", required=True)
    p.add_argument("--sft_output_dir", required=True)
    p.add_argument("--rl_output_dir", required=True)
    p.add_argument("--sft_ratio", type=float, default=0.8)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--allow_existing",
        action="store_true",
        help="Allow writing into output dirs that already contain shard-*.tar files.",
    )
    return p.parse_args()


def _add_tar_entry(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_shard(
    shard_id: int,
    samples: list[dict],
    output_dir: str,
    global_offset: int,
) -> tuple[int, int]:
    pe_handles: dict[str, safe_open] = {}
    tar_path = Path(output_dir) / f"shard-{shard_id:06d}.tar"
    tmp_path = tar_path.with_suffix(".tar.tmp")
    total_bytes = 0

    with tarfile.open(tmp_path, "w") as tar_file:
        for local_idx, sample in enumerate(samples):
            src = sample["source_file"]
            if src not in pe_handles:
                pe_handles[src] = safe_open(src, framework="pt")
            prompt_embeds = pe_handles[src].get_tensor(sample["key"])

            with safe_open(sample["vae_path"], framework="pt") as vae_sf:
                latents = vae_sf.get_tensor("latents")
                condition = vae_sf.get_tensor("condition")

            st_bytes = st_save(
                {
                    "prompt_embeds": prompt_embeds,
                    "latents": latents,
                    "condition": condition,
                }
            )
            meta_bytes = json.dumps(
                {
                    "prompt": sample["prompt"],
                    "tar": sample["tar"],
                    "index_in_tar": sample["index_in_tar"],
                    "seq_len": sample["seq_len"],
                    "split": sample["split"],
                }
            ).encode()

            key = f"{global_offset + local_idx:07d}"
            _add_tar_entry(tar_file, f"{key}.safetensors", st_bytes)
            _add_tar_entry(tar_file, f"{key}.json", meta_bytes)
            total_bytes += len(st_bytes) + len(meta_bytes)

            if _counter is not None:
                with _counter.get_lock():
                    _counter.value += 1

    tmp_path.rename(tar_path)
    pe_handles.clear()
    return shard_id, total_bytes


def _progress_monitor(counter: Value, pbar: tqdm, done: threading.Event):
    last = 0
    while not done.wait(0.5):
        current = counter.value
        if current > last:
            pbar.update(current - last)
            last = current
    current = counter.value
    if current > last:
        pbar.update(current - last)


def _scan_prompt_samples(prompt_dir: Path) -> list[dict]:
    logger.info("Scanning prompt embed headers ...")
    source_files = sorted(prompt_dir.glob("*.safetensors"))
    all_samples: list[dict] = []

    for sf_path in tqdm(source_files, desc="Scanning prompt embeds"):
        header = read_safetensors_header(str(sf_path))
        meta_raw = header.pop("__metadata__", {})
        samples_meta = json.loads(meta_raw.get("samples", "[]"))

        for key, info in header.items():
            idx = int(key)
            sm = samples_meta[idx] if idx < len(samples_meta) else {}
            tar_name = sm.get("tar", "")
            all_samples.append(
                {
                    "source_file": str(sf_path),
                    "key": key,
                    "seq_len": info["shape"][0],
                    "prompt": sm.get("prompt", ""),
                    "tar": tar_name,
                    "tar_stem": Path(tar_name).stem if tar_name else "",
                    "index_in_tar": sm.get("index_in_tar", -1),
                }
            )

    logger.info("Total prompt embed samples: {}", len(all_samples))
    return all_samples


def _join_vae_latents(samples: list[dict], vae_dir: Path) -> list[dict]:
    logger.info("Scanning VAE latent directory ...")
    vae_available = {f.name for f in vae_dir.iterdir() if f.suffix == ".safetensors"}
    logger.info("Found {} VAE latent files", len(vae_available))

    complete: list[dict] = []
    for s in samples:
        vae_filename = f"{s['tar_stem']}_{s['index_in_tar']}.safetensors"
        if vae_filename in vae_available:
            s["vae_path"] = str(vae_dir / vae_filename)
            complete.append(s)

    missing = len(samples) - len(complete)
    logger.info(
        "Complete samples (have both): {} / {} (missing VAE: {})",
        len(complete),
        len(samples),
        missing,
    )
    if not complete:
        raise RuntimeError("No complete samples found")
    return complete


def _write_split(
    name: str,
    samples: list[dict],
    output_dir: Path,
    samples_per_shard: int,
    num_workers: int,
) -> tuple[int, int]:
    for sample in samples:
        sample["split"] = name

    shard_chunks: list[tuple[int, list[dict], int]] = []
    for i in range(0, len(samples), samples_per_shard):
        shard_id = i // samples_per_shard
        shard_chunks.append((shard_id, samples[i : i + samples_per_shard], i))

    num_shards = len(shard_chunks)
    workers = min(num_workers, num_shards)
    logger.info(
        "{}: writing {} shards (~{} samples each) with {} workers -> {}",
        name,
        num_shards,
        samples_per_shard,
        workers,
        output_dir,
    )

    counter = Value("i", 0)
    done_event = threading.Event()
    pbar = tqdm(total=len(samples), desc=f"Writing {name}", unit="sample")
    monitor = threading.Thread(target=_progress_monitor, args=(counter, pbar, done_event))
    monitor.start()

    total_bytes = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(counter,),
    ) as executor:
        futures = {
            executor.submit(_write_shard, shard_id, chunk, str(output_dir), offset): shard_id
            for shard_id, chunk, offset in shard_chunks
        }
        for future in as_completed(futures):
            _, nbytes = future.result()
            total_bytes += nbytes

    done_event.set()
    monitor.join()
    pbar.close()

    logger.info(
        "{} done: {} shards, {} samples, {:.1f} GB",
        name,
        num_shards,
        len(samples),
        total_bytes / 1e9,
    )
    return num_shards, total_bytes


def main():
    args = parse_args()
    if not 0.0 < args.sft_ratio < 1.0:
        raise ValueError(f"--sft_ratio must be in (0, 1), got {args.sft_ratio}")

    prompt_dir = Path(args.prompt_embeds_dir)
    vae_dir = Path(args.vae_latents_dir)
    sft_dir = Path(args.sft_output_dir)
    rl_dir = Path(args.rl_output_dir)

    for out_dir in (sft_dir, rl_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob("shard-*.tar"))
        if existing and not args.allow_existing:
            raise FileExistsError(
                f"{out_dir} already contains {len(existing)} shard-*.tar files; remove them or pass --allow_existing"
            )

    all_samples = _scan_prompt_samples(prompt_dir)
    complete = _join_vae_latents(all_samples, vae_dir)

    logger.info("Global shuffle with seed={}", args.seed)
    random.Random(args.seed).shuffle(complete)

    sft_count = int(len(complete) * args.sft_ratio)
    sft_samples = complete[:sft_count]
    rl_samples = complete[sft_count:]
    logger.info(
        "Split counts: sft={} ({:.1%}), rl={} ({:.1%})",
        len(sft_samples),
        len(sft_samples) / len(complete),
        len(rl_samples),
        len(rl_samples) / len(complete),
    )

    sft_shards, sft_bytes = _write_split(
        "sft",
        sft_samples,
        sft_dir,
        args.samples_per_shard,
        args.num_workers,
    )
    rl_shards, rl_bytes = _write_split(
        "rl",
        rl_samples,
        rl_dir,
        args.samples_per_shard,
        args.num_workers,
    )

    manifest = {
        "seed": args.seed,
        "sft_ratio": args.sft_ratio,
        "total_samples": len(complete),
        "sft_samples": len(sft_samples),
        "rl_samples": len(rl_samples),
        "samples_per_shard": args.samples_per_shard,
        "sft_shards": sft_shards,
        "rl_shards": rl_shards,
        "sft_bytes": sft_bytes,
        "rl_bytes": rl_bytes,
    }
    manifest_path = sft_dir.parent / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote manifest: {}", manifest_path)


if __name__ == "__main__":
    main()
