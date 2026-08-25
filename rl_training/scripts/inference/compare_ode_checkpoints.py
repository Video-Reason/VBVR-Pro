"""Compare the ODE output of several checkpoints on the same samples.

Reads the already-rendered ``ode.mp4`` / ``reference.mp4`` per sample (no GPU,
no model reload) and produces, under ``<root>/compare_ode/``:

* ``all_samples.jpg`` — one big sheet: rows = sample, cols = [ref, <ckpt>...],
  each cell the ODE final (last) frame. The fastest cross-checkpoint overview.
* ``sample_NN.mp4``   — the same row in motion: ref | ckptA | ckptB | ... tiled
  side by side so trajectories can be compared frame by frame.

Usage:
    .venv/bin/python scripts/inference/compare_ode_checkpoints.py \
        --root storage/outputs/noise_coeff_sweep \
        --ckpts checkpoint_a checkpoint_b checkpoint_c
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

LABEL_W = 130
COL_H = 18
PAD = 3


def read_video(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.stack(iio.mimread(path, format="FFMPEG", memtest=False))  # (T, H, W, C) uint8


def label_strip(width: int, height: int, text: str) -> Image.Image:
    img = Image.new("RGB", (width, height), (30, 30, 30))
    ImageDraw.Draw(img).text((4, max(0, height // 2 - 6)), text, fill=(255, 255, 255))
    return img


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="storage/outputs/noise_coeff_sweep")
    p.add_argument("--ckpts", nargs="+", required=True, help="ckpt subdir names, in display order")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--thumb", type=int, default=150)
    p.add_argument("--fps", type=int, default=16)
    args = p.parse_args()

    root = Path(args.root)
    out = root / "compare_ode"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["ref"] + args.ckpts

    overview_rows: list[tuple[int, list[np.ndarray]]] = []
    for i in range(args.num_samples):
        # reference is identical across ckpts (dataset target); take the first that has it
        ref = None
        for ck in args.ckpts:
            ref = read_video(root / ck / f"sample_{i:02d}" / "reference.mp4")
            if ref is not None:
                break
        odes = [read_video(root / ck / f"sample_{i:02d}" / "ode.mp4") for ck in args.ckpts]
        cells = [ref] + odes
        if any(c is None for c in cells):
            print(f"sample {i:02d}: missing cells, skipping")
            continue
        overview_rows.append((i, cells))

        # side-by-side motion video for this sample: ref | ckptA | ckptB | ...
        tframes = min(c.shape[0] for c in cells)
        frames_list = [np.concatenate([c[t] for c in cells], axis=1) for t in range(tframes)]
        iio.mimwrite(out / f"sample_{i:02d}.mp4", frames_list, format="FFMPEG", fps=args.fps, codec="libx264")

    # overview sheet
    if overview_rows:
        thumb = args.thumb
        aspect = overview_rows[0][1][0].shape[1] / overview_rows[0][1][0].shape[2]
        th = max(1, int(round(thumb * aspect)))
        ncol = len(cols)
        W = LABEL_W + ncol * thumb + (ncol + 1) * PAD
        H = COL_H + len(overview_rows) * (th + PAD) + PAD
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        for c, name in enumerate(cols):
            x = LABEL_W + PAD + c * (thumb + PAD)
            draw.text((x + 2, 3), name, fill=(0, 0, 0))
        for r, (idx, cells) in enumerate(overview_rows):
            y = COL_H + r * (th + PAD)
            draw.text((2, y + th // 2 - 4), f"s{idx:02d}", fill=(0, 0, 0))
            for c, cell in enumerate(cells):
                x = LABEL_W + PAD + c * (thumb + PAD)
                img = Image.fromarray(cell[-1]).resize((thumb, th), Image.Resampling.BILINEAR)
                sheet.paste(img, (x, y))
        sheet.save(out / "all_samples.jpg", quality=92)
        print(
            f"wrote {out / 'all_samples.jpg'} ({len(overview_rows)} samples) + {len(overview_rows)} side-by-side mp4s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
