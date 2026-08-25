"""Stack the two per-checkpoint contact sheets for each sample into one image.

The sweep driver writes ``<root>/<ckpt>/sample_NN/contact.jpg`` per checkpoint.
This collates them: for every sample present under both checkpoints, it stacks
the two sheets vertically (checkpoint A on top, B below) under a labelled banner
and writes ``<root>/compare/sample_NN.jpg`` plus an ``index.md`` listing them.

Each contact sheet already labels its own rows (ref / ode / sde_nX / cps_nX), so
the combined image shows: for one maze sample, both checkpoints' outputs across
the full ODE -> SDE(small..large) -> CPS(small..large) noise sweep, with the
shared dataset reference as the first row of each block.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

BANNER_H = 22


def _banner(width: int, text: str) -> Image.Image:
    img = Image.new("RGB", (width, BANNER_H), (30, 30, 30))
    ImageDraw.Draw(img).text((6, 5), text, fill=(255, 255, 255))
    return img


def combine_sample(root: Path, ckpt_a: str, ckpt_b: str, sample_index: int, out_dir: Path) -> Path | None:
    a = root / ckpt_a / f"sample_{sample_index:02d}" / "contact.jpg"
    b = root / ckpt_b / f"sample_{sample_index:02d}" / "contact.jpg"
    if not a.exists() or not b.exists():
        return None
    img_a = Image.open(a).convert("RGB")
    img_b = Image.open(b).convert("RGB")

    # Title from the sample metadata (difficulty / prompt), read from either side.
    meta_path = root / ckpt_a / f"sample_{sample_index:02d}" / "manifest.json"
    subtitle = ""
    if meta_path.exists():
        summary = json.loads(meta_path.read_text()).get("summary", {})
        subtitle = f"  key={summary.get('key')} diff={summary.get('difficulty')} path_len={summary.get('path_len')}"

    width = max(img_a.width, img_b.width)
    banner_a = _banner(width, f"[A] {ckpt_a}   sample {sample_index:02d}{subtitle}")
    banner_b = _banner(width, f"[B] {ckpt_b}   sample {sample_index:02d}")
    total_h = banner_a.height + img_a.height + banner_b.height + img_b.height
    canvas = Image.new("RGB", (width, total_h), "white")
    y = 0
    for piece in (banner_a, img_a, banner_b, img_b):
        canvas.paste(piece, (0, y))
        y += piece.height

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{sample_index:02d}.jpg"
    canvas.save(out_path, quality=92)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="storage/outputs/noise_coeff_sweep")
    p.add_argument("--ckpt_a", required=True, help="First checkpoint subdirectory")
    p.add_argument("--ckpt_b", required=True, help="Second checkpoint subdirectory")
    p.add_argument("--num_samples", type=int, default=32)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root / "compare"
    made: list[Path] = []
    for i in range(args.num_samples):
        out = combine_sample(root, args.ckpt_a, args.ckpt_b, i, out_dir)
        if out is not None:
            made.append(out)
    lines = [f"# Checkpoint comparison ({args.ckpt_a} [A] vs {args.ckpt_b} [B])", ""]
    lines += [f"- [sample {p.stem.split('_')[1]}]({p.name})" for p in made]
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(made)} combined sheets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
