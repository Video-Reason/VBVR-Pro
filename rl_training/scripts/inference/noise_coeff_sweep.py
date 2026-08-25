"""Noise-coefficient sweep for one checkpoint over a set of latent samples.

For a single checkpoint and a single GPU, this loads the model exactly once and
then, for each requested latent sample, runs a list of (mode, noise) sampler
configs through the same ``InferenceEngine`` the unified runner uses (ODE / SDE /
CPS, sharing the DanceGRPO ``sde_generate`` loop). The first latent frame is the
frozen I2V conditioning image and is never noised — that is handled inside
``sde_generate`` for ``expand_timesteps`` (5B TI2V) models, so every config here
inherits it.

Stochastic samplers (SDE / CPS) are re-sampled ``--rounds`` times with different
seeds to expose the spread of the noise. ODE is deterministic, so it runs once.
Round ``r`` uses ``seed = base_seed + r``; round 0 therefore shares its initial
``x_T`` with the ODE baseline (clean "same start, ODE vs stochastic" compare),
while rounds 1+ are independent draws.

Outputs, per sample, under ``<output_root>/<ckpt_name>/sample_NN/``:

* ``ode.mp4`` / ``<mode>_n<coeff>_r<round>.mp4`` — final video per config/round
* ``reference.mp4``  — the dataset target (moving-ball final), decoded once
* ``grid.mp4``       — a config x round matrix video (rows = coefficient
                       small->large, cols = round); ODE / reference span one cell
* ``contact.jpg``    — the same matrix as a single image of each cell's *last*
                       frame: the trend (down the rows) and the stochastic spread
                       (across the columns) at a glance
* ``manifest.json``  — config list, per-run seconds, freeze status

The 8-GPU sweep is just N copies of this script, each pinned to one GPU with a
disjoint slice of ``--sample_indices`` (see the companion launcher).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.inference.config import InferenceConfig
from src.inference.engine import InferenceEngine, build_model
from src.inference.inputs import PreparedInput, prepare_from_latent
from src.inference.outputs import decode_batch_to_uint8, decode_latents_to_uint8, export_uint8_video

LABEL_W = 116
COL_H = 16
PAD = 4


# ----------------------------------------------------------------------
# config x round matrix renderers (rows = coefficient, cols = round)
# ----------------------------------------------------------------------
def _matrix_dims(rows: list[tuple[str, list[np.ndarray]]], n_cols: int, thumb_width: int) -> tuple[int, int, int]:
    aspect = rows[0][1][0].shape[1] / rows[0][1][0].shape[2]
    thumb_height = max(1, int(round(thumb_width * aspect)))
    grid_w = LABEL_W + n_cols * thumb_width + (n_cols + 1) * PAD
    grid_h = COL_H + len(rows) * (thumb_height + PAD) + PAD
    return thumb_height, grid_w, grid_h


def _paste_cell(canvas: Image.Image, frame: np.ndarray, x: int, y: int, tw: int, th: int) -> None:
    img = Image.fromarray(frame).resize((tw, th), Image.Resampling.BILINEAR)
    canvas.paste(img, (x, y))


def save_matrix_grid_video(
    rows: list[tuple[str, list[np.ndarray]]],
    n_cols: int,
    path: Path,
    *,
    fps: int,
    thumb_width: int = 180,
) -> None:
    if not rows:
        return
    from diffusers.utils import export_to_video

    th, grid_w, grid_h = _matrix_dims(rows, n_cols, thumb_width)
    frame_count = min(v.shape[0] for _, vids in rows for v in vids)

    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        for col in range(n_cols):
            x = LABEL_W + PAD + col * (thumb_width + PAD)
            draw.text((x + 2, 2), f"r{col}", fill=(0, 0, 0))
        for r, (label, vids) in enumerate(rows):
            y = COL_H + r * (th + PAD)
            draw.text((2, y + th // 2 - 4), label, fill=(0, 0, 0))
            for col, video in enumerate(vids[:n_cols]):
                x = LABEL_W + PAD + col * (thumb_width + PAD)
                _paste_cell(canvas, video[frame_idx], x, y, thumb_width, th)
        frames.append(canvas)

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def save_matrix_contact_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    n_cols: int,
    path: Path,
    *,
    thumb_width: int = 180,
) -> None:
    """One image: each cell is the *last* frame of that (coefficient, round) run."""
    if not rows:
        return
    th, grid_w, grid_h = _matrix_dims(rows, n_cols, thumb_width)
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    for col in range(n_cols):
        x = LABEL_W + PAD + col * (thumb_width + PAD)
        draw.text((x + 2, 2), f"r{col}", fill=(0, 0, 0))
    for r, (label, vids) in enumerate(rows):
        y = COL_H + r * (th + PAD)
        draw.text((2, y + th // 2 - 4), label, fill=(0, 0, 0))
        for col, video in enumerate(vids[:n_cols]):
            x = LABEL_W + PAD + col * (thumb_width + PAD)
            _paste_cell(canvas, video[-1], x, y, thumb_width, th)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


# ----------------------------------------------------------------------
# Config plumbing
# ----------------------------------------------------------------------
def parse_configs(spec: str) -> list[tuple[str, float | None]]:
    """``"ode:0,sde:0.3,cps:0.7"`` -> ``[("ode", None), ("sde", 0.3), ("cps", 0.7)]``."""
    out: list[tuple[str, float | None]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        mode, _, coeff = token.partition(":")
        mode = mode.strip().lower()
        if mode == "ode":
            out.append(("ode", None))
        else:
            out.append((mode, float(coeff.strip())))
    return out


def config_label(mode: str, noise: float | None) -> str:
    return "ode" if mode == "ode" else f"{mode}_n{noise:.2f}"


def make_cfg(args, sample_index: int, mode: str, noise: float | None, seed: int) -> InferenceConfig:
    return InferenceConfig(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        use_ema=args.use_ema,
        device=args.device,
        latent_webdataset_dir=args.latent_webdataset_dir,
        sample_index=sample_index,
        mode=mode,
        num_sampling_steps=args.num_sampling_steps,
        noise_scale=noise,
        cfg_scale=args.cfg_scale,
        seed=seed,
        batch_size=args.batch_size,
        share_init_noise=True,
        save_steps=False,  # finals only; the per-step gallery is too costly across the sweep
        save_reference=False,
        output_dir="/tmp/unused_noise_sweep",  # required field; this driver writes outputs itself
    )


# ----------------------------------------------------------------------
def run_sample(model, args, device, configs, sample_index, out_root) -> dict:
    prepared: PreparedInput = prepare_from_latent(make_cfg(args, sample_index, "ode", None, args.seed), device)
    sample_dir = out_root / args.ckpt_name / f"sample_{sample_index:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, list[np.ndarray]]] = []
    runs: list[dict] = []
    freeze_ok = None

    # Reference (dataset target = last video latent). Decode once per sample.
    if prepared.reference_latents:
        ref_video = decode_latents_to_uint8(model, prepared.reference_latents[-1].unsqueeze(0).to(device))
        export_uint8_video(ref_video, sample_dir / "reference.mp4", args.fps)
        rows.append(("ref", [ref_video]))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for mode, noise in configs:
        n_rounds = 1 if mode == "ode" else args.rounds
        label = config_label(mode, noise)
        round_videos: list[np.ndarray] = []
        for r in range(n_rounds):
            seed = args.seed + (0 if mode == "ode" else r)
            cfg = make_cfg(args, sample_index, mode, noise, seed)
            t0 = time.time()
            result = InferenceEngine(model, cfg).sample(prepared)

            if freeze_ok is None:
                cond0 = prepared.condition[:, : result.final_latent.shape[1], 0]
                freeze_ok = bool(torch.allclose(result.final_latent[:, :, 0].float(), cond0.float(), atol=1e-3))

            final = decode_batch_to_uint8(model, result.final_latent.to(device))[0]
            fname = "ode.mp4" if mode == "ode" else f"{label}_r{r}.mp4"
            export_uint8_video(final, sample_dir / fname, args.fps)
            round_videos.append(final)
            runs.append(
                {
                    "mode": mode,
                    "noise_scale": cfg.effective_noise_scale,
                    "label": label,
                    "round": r,
                    "seed": seed,
                    "file": fname,
                    "seconds": round(time.time() - t0, 2),
                }
            )
            print(
                f"[{args.ckpt_name} s{sample_index:02d}] {label:>10} r{r} "
                f"eta={cfg.effective_noise_scale:<5} {runs[-1]['seconds']:.1f}s",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append((label, round_videos))

    save_matrix_grid_video(rows, args.rounds, sample_dir / "grid.mp4", fps=args.fps)
    save_matrix_contact_sheet(rows, args.rounds, sample_dir / "contact.jpg")

    manifest = {
        "checkpoint": args.checkpoint,
        "ckpt_name": args.ckpt_name,
        "sample_index": sample_index,
        "source": prepared.source,
        "summary": prepared.summary,
        "num_sampling_steps": args.num_sampling_steps,
        "rounds": args.rounds,
        "first_frame_frozen": freeze_ok,
        "configs": runs,
    }
    (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if prepared.metadata:
        (sample_dir / "sample_metadata.json").write_text(json.dumps(prepared.metadata, indent=2))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise-coefficient sweep for one checkpoint on one GPU.")
    p.add_argument("--model_path", default="storage/models/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ckpt_name", required=True, help="Short label for the output subdirectory")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--latent_webdataset_dir", required=True)
    p.add_argument("--sample_indices", required=True, help="Comma-separated sample ordinals, e.g. 0,4,8")
    p.add_argument("--configs", required=True, help='unique coeffs, e.g. "ode:0,sde:0.3,sde:1.0,cps:0.7"')
    p.add_argument("--rounds", type=int, default=4, help="Stochastic rounds per SDE/CPS config (ODE always 1)")
    p.add_argument("--num_sampling_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--output_root", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_indices = [int(s) for s in args.sample_indices.split(",") if s.strip() != ""]
    configs = parse_configs(args.configs)
    device = torch.device(args.device)

    print(
        f"[{args.ckpt_name}] device={args.device} rounds={args.rounds} samples={sample_indices} configs={configs}",
        flush=True,
    )
    t_load = time.time()
    model = build_model(make_cfg(args, sample_indices[0], "ode", None, args.seed), need_text_encoder=False)
    print(f"[{args.ckpt_name}] model+checkpoint loaded in {time.time() - t_load:.1f}s", flush=True)

    out_root = Path(args.output_root)
    for sample_index in sample_indices:
        run_sample(model, args, device, configs, sample_index, out_root)

    print(f"[{args.ckpt_name}] DONE {len(sample_indices)} samples on {args.device}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
