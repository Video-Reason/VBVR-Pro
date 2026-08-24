"""VAE encode-decode roundtrip test.

Loads a video, encodes it with the Wan2.2 VAE, decodes it back,
and saves the reconstructed video for visual comparison.

Usage:
    .venv/bin/python -m src.cli.test_vae --video path/to/video.mp4
"""

import argparse
from pathlib import Path

import decord
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

decord.bridge.set_bridge("torch")


def parse_args():
    parser = argparse.ArgumentParser(description="VAE encode-decode roundtrip test")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model_path",
        type=str,
        default="storage/models/Wan2.2-I2V-A14B-Diffusers",
        help="Path to the model directory (uses the vae subfolder)",
    )
    parser.add_argument("--output", type=str, default="vae_roundtrip.mp4", help="Output video path")
    parser.add_argument(
        "--max_area",
        type=int,
        default=480 * 832,
        help="Max pixel area for resizing (default 480*832)",
    )
    parser.add_argument("--num_frames", type=int, default=None, help="Max frames to use (default: all available)")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS for frame sampling and output")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="VAE dtype")
    return parser.parse_args()


def load_video(video_path: str, height: int, width: int, num_frames: int | None, fps: int) -> tuple[torch.Tensor, int]:
    """Load video frames as uint8 tensor (C, T, H, W). Returns (video, actual_frame_count)."""
    vr = decord.VideoReader(video_path, width=width, height=height)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)

    stride = max(1, round(video_fps / fps))
    indices = list(range(0, total_frames, stride))
    if num_frames is not None:
        indices = indices[:num_frames]

    # Ensure 4k+1 frames for VAE temporal alignment
    n = len(indices)
    aligned = ((n - 1) // 4) * 4 + 1
    if aligned < 1:
        aligned = 1
    indices = indices[:aligned]

    frames = vr.get_batch(indices)  # (T, H, W, C) uint8
    return frames.permute(3, 0, 1, 2).contiguous(), len(indices)  # (C, T, H, W)


def main():
    args = parse_args()

    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Load VAE
    vae_path = Path(args.model_path) / "vae"
    print(f"Loading VAE from {vae_path} (dtype={args.dtype}) ...")
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    # Compute resolution from video aspect ratio
    vr = decord.VideoReader(args.video)
    orig_h, orig_w = vr[0].shape[:2]
    aspect_ratio = orig_h / orig_w
    mod_value = vae.config.scale_factor_spatial * 2  # spatial_scale * patch_size
    height = round(np.sqrt(args.max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(args.max_area / aspect_ratio)) // mod_value * mod_value
    height = max(height, mod_value)
    width = max(width, mod_value)
    print(f"Video resolution: {orig_w}x{orig_h} -> {width}x{height}")

    # Load video (no padding)
    video, n_frames = load_video(args.video, height, width, args.num_frames, args.fps)  # (C, T, H, W) uint8
    print(f"Loaded {n_frames} frames (aligned to 4k+1)")

    video_input = video.unsqueeze(0).to(device=device, dtype=dtype)  # (1, C, T, H, W)
    video_input = video_input / 127.5 - 1.0  # normalize to [-1, 1]
    print(f"Input range: [{video_input.min().item():.4f}, {video_input.max().item():.4f}]")

    # Encode
    print("Encoding ...")
    with torch.no_grad():
        latents = vae.encode(video_input).latent_dist.mode()
    print(f"Latent shape: {latents.shape}")
    print(f"Latent range: [{latents.min().item():.4f}, {latents.max().item():.4f}]")

    # Decode
    print("Decoding ...")
    with torch.no_grad():
        decoded = vae.decode(latents).sample  # (1, C, T, H, W) in [-1, 1]

    print(f"Decoded shape: {decoded.shape}")
    print(f"Decoded range: [{decoded.min().item():.4f}, {decoded.max().item():.4f}]")

    # Compute MSE / PSNR
    mse = ((video_input.float() - decoded.float()) ** 2).mean().item()
    psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float("inf")  # range [-1,1] so max range = 2, max^2 = 4
    print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    # Save comparison frames (first, middle, last)
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_uint8 = ((video_input.squeeze(0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()  # (C,T,H,W)
    output_uint8 = ((decoded.squeeze(0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()  # (C,T,H,W)
    n_out = output_uint8.shape[1]

    for label, t in [("first", 0), ("mid", n_out // 2), ("last", n_out - 1)]:
        if t >= input_uint8.shape[1]:
            continue
        inp_frame = input_uint8[:, t].permute(1, 2, 0).numpy()  # (H,W,C)
        out_frame = output_uint8[:, t].permute(1, 2, 0).numpy()
        # Side-by-side comparison
        comparison = np.concatenate([inp_frame, out_frame], axis=1)
        img = Image.fromarray(comparison)
        img.save(str(output_dir / f"vae_compare_{label}.png"))
        print(f"Saved comparison frame: vae_compare_{label}.png (left=input, right=reconstructed)")

    # Export reconstructed video
    frames = [output_uint8[:, i].permute(1, 2, 0).numpy() for i in range(n_out)]
    export_to_video(frames, str(output_path), fps=args.fps)
    print(f"Reconstructed video saved to {output_path} ({n_out} frames at {args.fps}fps)")


if __name__ == "__main__":
    main()
