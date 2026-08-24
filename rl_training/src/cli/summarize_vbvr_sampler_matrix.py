"""Build score tables and a browsable 30-step gallery for the VBVR sampler matrix."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from src.cli.audit_vbvr_i2v_trajectories import _relative_video_path

SAMPLERS = (
    ("cps0p1", "CPS 0.1", "cps-noise-0.1"),
    ("cps0p3", "CPS 0.3", "cps-noise-0.3"),
    ("cps0p7", "CPS 0.7", "cps-noise-0.7"),
    ("cps0p9", "CPS 0.9", "cps-noise-0.9"),
    ("euler", "Euler ODE", "euler-ode-30steps-cfg1"),
    ("unipc", "UniPC ODE", "unipc-ode-30steps-cfg1"),
)
CATEGORIES = ("Abstraction", "Perception", "Spatiality", "Transformation", "Knowledge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-base", required=True, type=Path)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument(
        "--trajectory-layout",
        choices=("auto", "fixed-sample", "all-samples"),
        default="auto",
    )
    return parser.parse_args(argv)


def _model_ids(eval_base: Path) -> list[str]:
    steps: set[int] = set()
    for path in eval_base.glob("dancegrpo_vbvr_pro_5b_checkpoint-*-cps-noise-0.7"):
        suffix = path.name.split("checkpoint-", 1)[1]
        try:
            steps.add(int(suffix.split("-", 1)[0]))
        except ValueError:
            continue
    return ["baseline", *(str(step) for step in sorted(steps))]


def _all_trajectory_model_ids(trajectory_root: Path) -> list[str]:
    values = [path.parent.name.removesuffix("-cps0p7") for path in trajectory_root.glob("*-cps0p7/cell_manifest.json")]
    if not values:
        return []
    if "baseline" not in values:
        raise RuntimeError(f"Trajectory root has CPS 0.7 cells but no baseline: {trajectory_root}")
    return ["baseline", *(str(step) for step in sorted(int(value) for value in values if value != "baseline"))]


def _eval_root(eval_base: Path, model_id: str, sampler_id: str, label: str) -> Path:
    if model_id == "baseline":
        if sampler_id == "cps0p7":
            return eval_base / "diffsynth_step35500-baseline-cps0p7-30steps-cfg1"
        return eval_base / f"diffsynth_step35500-baseline-{label}"
    return eval_base / f"dancegrpo_vbvr_pro_5b_checkpoint-{model_id}-{label}"


def _trajectory_paths(
    *,
    eval_root: Path,
    trajectory_root: Path,
    model_id: str,
    sampler_id: str,
    layout: str,
) -> tuple[Path, Path | None, str]:
    all_cell = trajectory_root / f"{model_id}-{sampler_id}"
    selected_layout = layout
    if selected_layout == "auto":
        selected_layout = "all-samples" if (all_cell / "cell_manifest.json").is_file() else "fixed-sample"
    if selected_layout == "fixed-sample":
        return trajectory_root / f"{model_id}-{sampler_id}-sample00000", None, selected_layout

    data = json.loads((eval_root / "eval_samples.json").read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No eval samples in {eval_root / 'eval_samples.json'}")
    item = data[0]
    name = next((str(item[key]) for key in ("name", "id") if item.get(key) is not None and str(item[key])), "0")
    sample = (all_cell / _relative_video_path(name)).with_suffix("")
    return sample, all_cell / "cell_manifest.json", selected_layout


def _load_rows(
    eval_base: Path,
    trajectory_root: Path,
    *,
    layout: str = "auto",
    model_ids: list[str] | None = None,
) -> list[dict]:
    use_all_model_ids = layout == "all-samples" or (
        layout == "auto" and (trajectory_root / "baseline-cps0p7/cell_manifest.json").is_file()
    )
    trajectory_models = _all_trajectory_model_ids(trajectory_root) if use_all_model_ids else []
    models = model_ids or trajectory_models or _model_ids(eval_base)
    if models == ["baseline"]:
        raise RuntimeError(f"No checkpoint CPS 0.7 runs found under {eval_base}")
    rows: list[dict] = []
    missing: list[str] = []
    for model_id in models:
        for sampler_id, sampler_name, label in SAMPLERS:
            eval_root = _eval_root(eval_base, model_id, sampler_id, label)
            result = eval_root / "scores/eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
            trajectory, cell_manifest_path, selected_layout = _trajectory_paths(
                eval_root=eval_root,
                trajectory_root=trajectory_root,
                model_id=model_id,
                sampler_id=sampler_id,
                layout=layout,
            )
            manifest = trajectory / "manifest.json"
            required = (result, manifest, trajectory / "steps_grid.mp4", trajectory / "step_contact_sheet.jpg")
            if cell_manifest_path is not None:
                required = (*required, cell_manifest_path)
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            score = json.loads(result.read_text())["summary"]
            categories = score["overall"]["by_category"]
            trajectory_data = json.loads(manifest.read_text())
            cell_sample_count = 1
            if cell_manifest_path is not None:
                cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
                if cell_manifest.get("state") != "complete":
                    missing.append(f"incomplete cell manifest: {cell_manifest_path}")
                    continue
                cell_sample_count = int(cell_manifest.get("completed_count", 0))
            binding = trajectory_data.get("formal_final_binding") or {}
            rows.append(
                {
                    "model_id": model_id,
                    "model": "DiffSynth step-35500 baseline" if model_id == "baseline" else f"checkpoint-{model_id}",
                    "sampler_id": sampler_id,
                    "sampler": sampler_name,
                    "trajectory_layout": selected_layout,
                    "trajectory_sample_count": cell_sample_count,
                    "overall": float(score["overall"]["mean_score"]),
                    "in_domain": float(score["In_Domain"]["mean_score"]),
                    "out_of_domain": float(score["Out_of_Domain"]["mean_score"]),
                    "categories": {name: float(categories[name]) for name in CATEGORIES},
                    "eval_root": str(eval_root.resolve()),
                    "result": str(result.resolve()),
                    "trajectory": str(trajectory.resolve()),
                    "grid": str((trajectory / "steps_grid.mp4").resolve()),
                    "contact_sheet": str((trajectory / "step_contact_sheet.jpg").resolve()),
                    "formal_final": binding.get("source"),
                    "formal_final_sha256": binding.get("sha256"),
                }
            )
    if missing:
        preview = "\n  - ".join(missing[:20])
        raise RuntimeError(f"Matrix is incomplete; missing {len(missing)} required artifacts:\n  - {preview}")
    expected = len(models) * len(SAMPLERS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} matrix rows, got {len(rows)}")
    return rows


def _markdown_matrix(rows: list[dict], metric: str, title: str) -> str:
    by_key = {(row["model_id"], row["sampler_id"]): row for row in rows}
    models = list(dict.fromkeys((row["model_id"], row["model"]) for row in rows))
    lines = [f"## {title}", "", "| Model | " + " | ".join(name for _, name, _ in SAMPLERS) + " |"]
    lines.append("|---|" + "---:|" * len(SAMPLERS))
    for model_id, model_name in models:
        values = [f"{by_key[(model_id, sampler_id)][metric]:.6f}" for sampler_id, _, _ in SAMPLERS]
        lines.append(f"| {model_name} | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_tsv(rows: list[dict], path: Path) -> None:
    header = [
        "model_id",
        "model",
        "sampler_id",
        "sampler",
        "trajectory_layout",
        "trajectory_sample_count",
        "overall",
        "in_domain",
        "out_of_domain",
        *(name.lower() for name in CATEGORIES),
        "result",
        "grid",
        "contact_sheet",
        "formal_final",
        "formal_final_sha256",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        values = [
            row["model_id"],
            row["model"],
            row["sampler_id"],
            row["sampler"],
            row["trajectory_layout"],
            str(row["trajectory_sample_count"]),
            f"{row['overall']:.9f}",
            f"{row['in_domain']:.9f}",
            f"{row['out_of_domain']:.9f}",
            *(f"{row['categories'][name]:.9f}" for name in CATEGORIES),
            row["result"],
            row["grid"],
            row["contact_sheet"],
            str(row["formal_final"]),
            str(row["formal_final_sha256"]),
        ]
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative(path: str, output_dir: Path) -> str:
    import os

    return Path(os.path.relpath(path, output_dir)).as_posix()


def _write_gallery(rows: list[dict], output_dir: Path) -> None:
    models = list(dict.fromkeys((row["model_id"], row["model"]) for row in rows))
    by_key = {(row["model_id"], row["sampler_id"]): row for row in rows}
    sections: list[str] = []
    for model_id, model_name in models:
        cards: list[str] = []
        for sampler_id, sampler_name, _ in SAMPLERS:
            row = by_key[(model_id, sampler_id)]
            grid = html.escape(_relative(row["grid"], output_dir))
            contact = html.escape(_relative(row["contact_sheet"], output_dir))
            final = html.escape(_relative(row["formal_final"], output_dir))
            result = html.escape(_relative(row["result"], output_dir))
            cards.append(
                f"""
                <article class="card">
                  <h3>{html.escape(sampler_name)}</h3>
                  <p class="score">Overall {row["overall"]:.6f}</p>
                  <p>ID {row["in_domain"]:.6f} · OOD {row["out_of_domain"]:.6f}</p>
                  <video controls preload="metadata" src="{grid}"></video>
                  <p><a href="{contact}">30-step contact sheet</a> ·
                     <a href="{final}">scored final video</a> ·
                     <a href="{result}">score JSON</a></p>
                </article>
                """
            )
        sections.append(
            f'<section><h2>{html.escape(model_name)}</h2><div class="grid">{"".join(cards)}</div></section>'
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VBVR-Pro 30-step sampler matrix</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#17191c}}
h1,h2{{margin-bottom:.4rem}} .note{{max-width:1100px;color:#42464d}} section{{margin:32px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}
.card{{background:white;border:1px solid #d8dce2;border-radius:10px;padding:14px;box-shadow:0 1px 3px #0001}}
video{{width:100%;background:#111}}
.score{{font-size:1.18rem;font-weight:650;margin:.2rem 0}} p{{margin:.45rem 0}} a{{color:#1457b8}}
</style></head><body>
<h1>VBVR-Pro matched 30-step sampler matrix</h1>
<p class="note">All cells use 512×512×81, 16 FPS, 30 inference steps, CFG 1.0, seed 0 and EvalKit e140.
In each grid, cells 1–29 are post-CFG clean-endpoint previews x0=x_t−σv; cell 30 is copied byte-for-byte
from the quantitative video actually scored by EvalKit.</p>
{"".join(sections)}
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_base = args.eval_output_base.resolve()
    trajectory_root = args.trajectory_root.resolve()
    output_dir = (args.output_dir or trajectory_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(
        eval_base,
        trajectory_root,
        layout=args.trajectory_layout,
        model_ids=args.model_ids,
    )
    (output_dir / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_tsv(rows, output_dir / "scores.tsv")
    layouts = ", ".join(sorted({row["trajectory_layout"] for row in rows}))
    sample_counts = sorted({row["trajectory_sample_count"] for row in rows})
    sample_count_text = ", ".join(str(value) for value in sample_counts)
    gallery_note = (
        "Open `index.html` for the compact 60-cell sample-0 gallery; the all-output browser is under `gallery/`."
        if any(row["trajectory_layout"] == "all-samples" for row in rows)
        else "Open `index.html` for the complete 60-cell 30-step video gallery."
    )
    report = [
        "# VBVR-Pro matched 30-step sampler matrix",
        "",
        "Contract: 512×512×81, 16 FPS, 30 steps, CFG 1.0, seed 0, EvalKit e140.",
        f"Trajectory layout: {layouts}; completed outputs per cell: {sample_count_text}.",
        "",
        _markdown_matrix(rows, "overall", "Overall"),
        _markdown_matrix(rows, "in_domain", "In-domain"),
        _markdown_matrix(rows, "out_of_domain", "Out-of-domain"),
        gallery_note,
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    _write_gallery(rows, output_dir)
    print(f"Wrote {len(rows)} matrix rows and gallery to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
