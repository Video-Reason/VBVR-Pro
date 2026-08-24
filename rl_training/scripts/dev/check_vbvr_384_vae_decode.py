#!/usr/bin/env python3
"""Decode 384x384x81 VBVR WebDataset latents and compare them to source videos.

This is intended as a GPU-idle sanity check for the precomputed SFT latents.
It reads samples from the shuffled WebDataset split, resolves each sample back
to VBVR metadata/tar media, decodes the normalized VAE latent, and writes
side-by-side comparison images plus a JSON metric summary.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

import decord
import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw
from safetensors.torch import load as load_safetensors

decord.bridge.set_bridge("native")


def parse_sample_id(text: str) -> tuple[int, int]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("sample ids must be formatted as SHARD:LOCAL, e.g. 0:0")
    shard, local = text.split(":", 1)
    try:
        return int(shard), int(local)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid SHARD:LOCAL sample id: {text!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--webdataset-dir",
        type=Path,
        default=Path("data/vbvr/latents/vbvr_384x384x81/webdataset/sft"),
        help="Directory with shard-*.tar files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/vbvr/VBVR-Dataset/data/metadata.parquet"),
        help="VBVR metadata parquet.",
    )
    parser.add_argument(
        "--tar-dir",
        type=Path,
        default=Path("data/vbvr/VBVR-Dataset/tars"),
        help="Directory containing source VBVR tar files.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("storage/models/Wan2.2-I2V-A14B-Diffusers"),
        help="Wan diffusers model directory; uses its vae subfolder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage/debug/vbvr_384_vae_decode"),
        help="Where decoded videos, comparison PNGs, and summary JSON are written.",
    )
    parser.add_argument(
        "--sample",
        dest="samples",
        type=parse_sample_id,
        action="append",
        default=None,
        help="Sample to inspect as SHARD:LOCAL. May be repeated. Default: 0:0, 400:0, 799:999.",
    )
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def load_webdataset_sample(webdataset_dir: Path, shard_id: int, local_idx: int, samples_per_shard: int) -> dict:
    key = f"{shard_id * samples_per_shard + local_idx:07d}"
    shard_path = webdataset_dir / f"shard-{shard_id:06d}.tar"
    if not shard_path.exists():
        raise FileNotFoundError(shard_path)
    with tarfile.open(shard_path, "r") as tar:
        meta_file = tar.extractfile(f"{key}.json")
        tensor_file = tar.extractfile(f"{key}.safetensors")
        if meta_file is None or tensor_file is None:
            raise FileNotFoundError(f"missing {key}.json/.safetensors in {shard_path}")
        meta = json.load(meta_file)
        tensors = load_safetensors(tensor_file.read())
    return {
        "key": key,
        "shard_id": shard_id,
        "local_idx": local_idx,
        "meta": meta,
        "tensors": tensors,
    }


def build_metadata_lookup(metadata_path: Path, tar_names: set[str]) -> dict[tuple[str, int], object]:
    df = pd.read_parquet(
        metadata_path,
        columns=["prompt", "first_frame_path", "ground_truth_video_path", "tar_file"],
    )
    df["tar_base"] = df["tar_file"].map(lambda path: Path(path).name)
    df = df[df["tar_base"].isin(tar_names)].copy()
    df["index_in_tar"] = df.groupby("tar_base", sort=False).cumcount()
    return {(row.tar_base, int(row.index_in_tar)): row for row in df.itertuples(index=False)}


def load_source_frames(
    tar_dir: Path,
    tar_name: str,
    first_frame_path: str,
    video_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    tar_path = tar_dir / tar_name
    with tarfile.open(tar_path, "r") as tar:
        first_file = tar.extractfile(first_frame_path)
        video_file = tar.extractfile(video_path)
        if first_file is None:
            raise FileNotFoundError(f"{first_frame_path} not found in {tar_path}")
        if video_file is None:
            raise FileNotFoundError(f"{video_path} not found in {tar_path}")
        first_image = Image.open(io.BytesIO(first_file.read())).convert("RGB")
        video_bytes = video_file.read()

    first_resized = np.array(first_image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        vr_raw = decord.VideoReader(tmp_path)
        raw_shape = tuple(vr_raw[0].asnumpy().shape)
        raw_frames = len(vr_raw)
        raw_fps = vr_raw.get_avg_fps()

        vr = decord.VideoReader(tmp_path, width=width, height=height)
        indices = np.linspace(0, len(vr) - 1, num_frames).round().astype(int).tolist()
        frames = vr.get_batch(indices).asnumpy()
    finally:
        os.unlink(tmp_path)

    info = {
        "raw_frames": raw_frames,
        "raw_fps": raw_fps,
        "raw_first_shape": raw_shape,
        "sampled_indices": indices,
    }
    return first_resized, frames, info


def decode_latents(vae: AutoencoderKLWan, latents: torch.Tensor, device: str, dtype: torch.dtype) -> np.ndarray:
    cfg = vae.config
    mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, cfg.z_dim, 1, 1, 1)
    std_inv = (1.0 / torch.tensor(cfg.latents_std, device=device, dtype=dtype)).view(1, cfg.z_dim, 1, 1, 1)
    norm_latents = latents.unsqueeze(0).to(device=device, dtype=dtype)
    raw_latents = norm_latents / std_inv + mean
    if not torch.isfinite(raw_latents).all():
        raise RuntimeError("raw VAE latents contain non-finite values before decode")
    with torch.no_grad():
        decoded = vae.decode(raw_latents.to(vae.dtype)).sample[0]
    if not torch.isfinite(decoded).all():
        raise RuntimeError("decoded video contains non-finite values")
    decoded = ((decoded.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
    return decoded.permute(1, 2, 3, 0).numpy()


def save_comparison(path: Path, source: np.ndarray, decoded: np.ndarray, title: str) -> None:
    frame_ids = [0, min(len(source), len(decoded)) // 2, min(len(source), len(decoded)) - 1]
    rows: list[Image.Image] = []
    for frame_id in frame_ids:
        src = source[frame_id]
        dec = decoded[frame_id]
        diff = np.abs(src.astype(np.int16) - dec.astype(np.int16)).astype(np.uint8)
        row = np.concatenate([src, dec, diff], axis=1)
        rows.append(Image.fromarray(row))

    label_h = 28
    sheet = Image.new("RGB", (rows[0].width, label_h + sum(row.height for row in rows)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), f"{title} | columns: source, decoded, absdiff", fill=(0, 0, 0))
    y = label_h
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(path)


def compute_metrics(source: np.ndarray, decoded: np.ndarray, first_frame: np.ndarray) -> dict:
    n = min(len(source), len(decoded))
    src = source[:n].astype(np.float32)
    dec = decoded[:n].astype(np.float32)
    mse = float(np.mean((src - dec) ** 2))
    psnr = float(10.0 * np.log10((255.0**2) / mse)) if mse > 0 else float("inf")
    first_diff = np.abs(source[0].astype(np.int16) - first_frame.astype(np.int16))
    return {
        "decoded_frames": int(len(decoded)),
        "compared_frames": int(n),
        "mean_abs_pixel_diff": float(np.mean(np.abs(src - dec))),
        "mse": mse,
        "psnr_db": psnr,
        "source_first_vs_first_png_mean_abs": float(first_diff.mean()),
        "source_first_vs_first_png_p95_abs": float(np.percentile(first_diff, 95)),
        "source_first_vs_first_png_max_abs": int(first_diff.max()),
    }


def main() -> int:
    args = parse_args()
    samples = args.samples or [(0, 0), (400, 0), (799, 999)]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = [
        load_webdataset_sample(args.webdataset_dir, shard_id, local_idx, args.samples_per_shard)
        for shard_id, local_idx in samples
    ]
    lookup = build_metadata_lookup(args.metadata, {sample["meta"]["tar"] for sample in loaded})

    print(f"Loading VAE from {args.model_path / 'vae'} on {args.device} ({args.dtype})")
    vae = AutoencoderKLWan.from_pretrained(args.model_path / "vae", torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(args.device)

    summary: list[dict] = []
    for sample in loaded:
        meta = sample["meta"]
        tar_name = meta["tar"]
        index_in_tar = int(meta["index_in_tar"])
        row = lookup[(tar_name, index_in_tar)]
        prompt_match = meta["prompt"] == row.prompt

        first_frame, source_frames, source_info = load_source_frames(
            args.tar_dir,
            tar_name,
            row.first_frame_path,
            row.ground_truth_video_path,
            args.height,
            args.width,
            args.num_frames,
        )
        decoded = decode_latents(vae, sample["tensors"]["latents"], args.device, dtype)
        metrics = compute_metrics(source_frames, decoded, first_frame)
        latent = sample["tensors"]["latents"].float()

        stem = f"shard{sample['shard_id']:06d}_local{sample['local_idx']:04d}_{sample['key']}"
        decoded_video = args.output_dir / f"{stem}_decoded.mp4"
        comparison_png = args.output_dir / f"{stem}_compare.png"
        export_to_video([Image.fromarray(frame) for frame in decoded], str(decoded_video), fps=args.fps)
        save_comparison(comparison_png, source_frames, decoded, f"{sample['key']} {tar_name} #{index_in_tar}")

        item = {
            "key": sample["key"],
            "shard_id": sample["shard_id"],
            "local_idx": sample["local_idx"],
            "tar": tar_name,
            "index_in_tar": index_in_tar,
            "prompt_match": prompt_match,
            "prompt": meta["prompt"],
            "source": source_info,
            "latent": {
                "shape": list(sample["tensors"]["latents"].shape),
                "dtype": str(sample["tensors"]["latents"].dtype),
                "mean": float(latent.mean()),
                "std": float(latent.std()),
                "min": float(latent.min()),
                "max": float(latent.max()),
            },
            "metrics": metrics,
            "decoded_video": str(decoded_video),
            "comparison_png": str(comparison_png),
        }
        summary.append(item)
        print(
            f"{sample['key']}: prompt_match={prompt_match}, "
            f"mean_abs={metrics['mean_abs_pixel_diff']:.3f}, "
            f"psnr={metrics['psnr_db']:.2f}dB, output={comparison_png}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
