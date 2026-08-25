"""Build a lazy-loading browser for every VBVR sampler-matrix trajectory."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.cli.audit_vbvr_i2v_trajectories import _relative_video_path, audit
from src.cli.summarize_vbvr_sampler_matrix import SAMPLERS, _eval_root, _model_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-base", required=True, type=Path)
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", action="append", dest="model_ids", default=None)
    parser.add_argument(
        "--sampler-id",
        action="append",
        dest="sampler_ids",
        choices=tuple(item[0] for item in SAMPLERS),
        default=None,
    )
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument(
        "--skip-artifact-audit",
        action="store_true",
        help="Build from cell manifests without rechecking every step/final SHA binding.",
    )
    args = parser.parse_args(argv)
    if args.expected_samples <= 0:
        parser.error("--expected-samples must be positive")
    return args


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _renderer_sampler(sampler_id: str) -> tuple[str, float | None]:
    if sampler_id.startswith("cps0p"):
        return "cps", float(sampler_id.removeprefix("cps0p")) / 10.0
    return sampler_id, None


def _trajectory_model_ids(trajectory_root: Path) -> list[str]:
    model_ids = [
        path.parent.name.removesuffix("-cps0p7") for path in trajectory_root.glob("*-cps0p7/cell_manifest.json")
    ]
    if not model_ids:
        return []
    if "baseline" not in model_ids:
        raise RuntimeError(f"Trajectory root has CPS 0.7 cells but no baseline: {trajectory_root}")
    steps = sorted(int(model_id) for model_id in model_ids if model_id != "baseline")
    return ["baseline", *(str(step) for step in steps)]


def _load_score(eval_root: Path) -> dict[str, float]:
    path = eval_root / "scores/eval_1024x1024_81f_fps16_5p0625s_vbvr_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    return {
        "overall": float(summary["overall"]["mean_score"]),
        "in_domain": float(summary["In_Domain"]["mean_score"]),
        "out_of_domain": float(summary["Out_of_Domain"]["mean_score"]),
    }


def _sample_name(item: dict[str, Any], index: int) -> str:
    for key in ("name", "id"):
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def _cell_record(
    *,
    eval_base: Path,
    trajectory_root: Path,
    model_id: str,
    sampler_id: str,
    sampler_name: str,
    label: str,
    expected_samples: int,
    strict_audit: bool,
) -> dict[str, Any]:
    eval_root = _eval_root(eval_base, model_id, sampler_id, label)
    eval_json = eval_root / "eval_samples.json"
    formal_root = eval_root / "generated_512x512x81"
    cell_root = trajectory_root / f"{model_id}-{sampler_id}"
    cell_manifest_path = cell_root / "cell_manifest.json"
    cell_manifest = json.loads(cell_manifest_path.read_text(encoding="utf-8"))
    model_path = Path(cell_manifest["model_path"])
    mode, noise_level = _renderer_sampler(sampler_id)
    if strict_audit:
        audit(
            SimpleNamespace(
                eval_json=eval_json,
                model_path=model_path,
                output_dir=cell_root,
                formal_final_root=formal_root,
                sampler=mode,
                noise_level=noise_level,
                limit=None,
                height=512,
                width=512,
                num_frames=81,
                num_inference_steps=30,
                guidance_scale=1.0,
                fps=16,
                seed=0,
            )
        )
    if cell_manifest.get("state") != "complete":
        raise RuntimeError(f"Trajectory cell is not complete: {cell_manifest_path}")
    if (
        cell_manifest.get("sample_count") != expected_samples
        or cell_manifest.get("completed_count") != expected_samples
    ):
        raise RuntimeError(
            f"Expected {expected_samples} complete samples in {cell_manifest_path}, got "
            f"{cell_manifest.get('completed_count')}/{cell_manifest.get('sample_count')}"
        )
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != expected_samples:
        raise RuntimeError(f"Expected {expected_samples} eval samples in {eval_json}, got {len(data)}")
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        name = _sample_name(item, index)
        relative_video = _relative_video_path(name)
        sample_root = (cell_root / relative_video).with_suffix("")
        formal_final = formal_root / relative_video
        samples.append(
            {
                "index": index,
                "name": name,
                "task_name": item.get("task_name") or Path(name).parent.name,
                "domain": item.get("domain") or Path(name).parts[0],
                "root": sample_root,
                "grid": sample_root / "steps_grid.mp4",
                "contact_sheet": sample_root / "step_contact_sheet.jpg",
                "manifest": sample_root / "manifest.json",
                "formal_final": formal_final,
            }
        )
    return {
        "model_id": model_id,
        "model": "DiffSynth step-35500 baseline" if model_id == "baseline" else f"checkpoint-{model_id}",
        "sampler_id": sampler_id,
        "sampler": sampler_name,
        "eval_root": eval_root,
        "cell_root": cell_root,
        "score": _load_score(eval_root),
        "samples": samples,
    }


def _page_style() -> str:
    return """
:root{color-scheme:light}*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;background:#f4f6f9;color:#17191d}
header{position:sticky;top:0;z-index:2;padding:14px 22px;background:#fffffff2;border-bottom:1px solid #d8dde5}
header{backdrop-filter:blur(8px)}main{padding:22px}.note{max-width:1120px;color:#4a5059}
.matrix{border-collapse:collapse;background:white;width:100%;max-width:1400px}
th,td{border:1px solid #d8dde5;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}
.cards{display:grid;gap:18px}.card{background:white;border:1px solid #d8dde5;border-radius:10px;padding:14px}
.card{box-shadow:0 1px 3px #0001;content-visibility:auto;contain-intrinsic-size:900px}
.media{display:grid;grid-template-columns:minmax(380px,2fr) minmax(240px,1fr);gap:12px}
video,img{display:block;width:100%;background:#111}.meta{color:#4a5059;font-size:.92rem}
.steps{display:flex;flex-wrap:wrap;gap:5px}.steps a{font-variant-numeric:tabular-nums}
a{color:#1459b8}input{width:min(760px,95vw);padding:9px;border:1px solid #aeb6c2;border-radius:7px}
input{font-size:1rem}
@media(max-width:800px){.media{grid-template-columns:1fr}main{padding:12px}}
"""


def _lazy_script() -> str:
    return """
const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;
 const el=entry.target;if(el.dataset.src){el.src=el.dataset.src;delete el.dataset.src;
 if(el.tagName==='VIDEO')el.load();}
 observer.unobserve(el);}}, {rootMargin:'800px'});
document.querySelectorAll('[data-src]').forEach(el=>observer.observe(el));
const search=document.getElementById('search');if(search){search.addEventListener('input',()=>{
 const q=search.value.toLowerCase();
 document.querySelectorAll('.card').forEach(card=>card.hidden=!card.dataset.search.includes(q));});}
"""


def _cell_document(cell: dict[str, Any], *, page_path: Path, root_index: Path) -> str:
    cards: list[str] = []
    page_dir = page_path.parent
    for sample in cell["samples"]:
        grid = html.escape(_relative(sample["grid"], page_dir))
        contact = html.escape(_relative(sample["contact_sheet"], page_dir))
        final = html.escape(_relative(sample["formal_final"], page_dir))
        manifest = html.escape(_relative(sample["manifest"], page_dir))
        step_links = " ".join(
            (
                f'<a href="{html.escape(_relative(sample["root"] / f"step_{index:02d}.mp4", page_dir))}">'
                f"{index + 1:02d}</a>"
            )
            for index in range(30)
        )
        search = html.escape(
            f"{sample['index']} {sample['name']} {sample['task_name']} {sample['domain']}".lower(), quote=True
        )
        cards.append(
            f"""
<article class="card" data-search="{search}">
 <h2>#{sample["index"]:03d} · {html.escape(sample["name"])}</h2>
 <p class="meta">{html.escape(str(sample["domain"]))} · {html.escape(str(sample["task_name"]))}</p>
 <div class="media">
  <video controls preload="none" data-src="{grid}"></video>
  <a href="{contact}"><img loading="lazy" data-src="{contact}" alt="30-step contact sheet"></a>
 </div>
 <p><a href="{grid}">30-step grid MP4</a> · <a href="{contact}">contact sheet</a> ·
    <a href="{final}">exact scored final</a> · <a href="{manifest}">manifest</a></p>
 <details><summary>Individual step MP4s (01–30)</summary><div class="steps">{step_links}</div></details>
</article>"""
        )
    score = cell["score"]
    back = html.escape(_relative(root_index, page_dir))
    title = f"{html.escape(cell['model'])} · {html.escape(cell['sampler'])}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{_page_style()}</style></head><body><header><a href="{back}">← Matrix</a>
<h1>{html.escape(cell["model"])} · {html.escape(cell["sampler"])}</h1>
<p>Overall {score["overall"]:.6f} · ID {score["in_domain"]:.6f} · OOD {score["out_of_domain"]:.6f} ·
{len(cell["samples"])} outputs</p>
<input id="search" type="search" placeholder="Filter by sample index, path, task, or domain"></header><main>
<p class="note">Each grid contains all 30 post-CFG clean-endpoint displays. Cells 01–29 show x0=x_t−σv; cell 30 and
the scored-final link are bound to the exact quantitative MP4. Videos and contact sheets load only near the
viewport.</p>
<div class="cards">{"".join(cards)}</div></main><script>{_lazy_script()}</script></body></html>"""


def _root_document(cells: list[dict[str, Any]], *, root_index: Path, cell_pages: dict[tuple[str, str], Path]) -> str:
    model_order = list(dict.fromkeys((cell["model_id"], cell["model"]) for cell in cells))
    by_key = {(cell["model_id"], cell["sampler_id"]): cell for cell in cells}
    sampler_order = [
        (sampler_id, name)
        for sampler_id, name, _ in SAMPLERS
        if any(cell["sampler_id"] == sampler_id for cell in cells)
    ]
    rows: list[str] = []
    for model_id, model_name in model_order:
        values: list[str] = []
        for sampler_id, _ in sampler_order:
            cell = by_key[(model_id, sampler_id)]
            page = html.escape(_relative(cell_pages[(model_id, sampler_id)], root_index.parent))
            values.append(f'<td><a href="{page}">{cell["score"]["overall"]:.6f}</a><br><small>500 × 30</small></td>')
        rows.append(f"<tr><td>{html.escape(model_name)}</td>{''.join(values)}</tr>")
    headers = "".join(f"<th>{html.escape(name)}</th>" for _, name in sampler_order)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VBVR-Pro all-output 30-step gallery</title>
<style>{_page_style()}</style></head><body><header><h1>VBVR-Pro all-output 30-step gallery</h1></header><main>
<p class="note">{len(cells)} model/sampler cells · {sum(len(cell["samples"]) for cell in cells):,} model outputs ·
30 displayed denoising endpoints per output. No media is copied or archived: the pages link directly to the unpacked
trajectory tree.</p>
<table class="matrix"><thead><tr><th>Model</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>
</main></body></html>"""


def build_gallery(args: argparse.Namespace) -> dict[str, Any]:
    eval_base = args.eval_output_base.resolve()
    trajectory_root = args.trajectory_root.resolve()
    output_dir = (args.output_dir or trajectory_root / "gallery").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    model_ids = args.model_ids or _trajectory_model_ids(trajectory_root) or _model_ids(eval_base)
    selected_sampler_ids = set(args.sampler_ids or (item[0] for item in SAMPLERS))
    cells: list[dict[str, Any]] = []
    for model_id in model_ids:
        for sampler_id, sampler_name, label in SAMPLERS:
            if sampler_id not in selected_sampler_ids:
                continue
            cells.append(
                _cell_record(
                    eval_base=eval_base,
                    trajectory_root=trajectory_root,
                    model_id=model_id,
                    sampler_id=sampler_id,
                    sampler_name=sampler_name,
                    label=label,
                    expected_samples=args.expected_samples,
                    strict_audit=not args.skip_artifact_audit,
                )
            )
    root_index = output_dir / "index.html"
    cell_pages: dict[tuple[str, str], Path] = {}
    for cell in cells:
        page = cells_dir / f"{cell['model_id']}-{cell['sampler_id']}.html"
        cell_pages[(cell["model_id"], cell["sampler_id"])] = page
        page.write_text(_cell_document(cell, page_path=page, root_index=root_index), encoding="utf-8")
    root_index.write_text(_root_document(cells, root_index=root_index, cell_pages=cell_pages), encoding="utf-8")
    manifest = {
        "state": "complete",
        "eval_output_base": str(eval_base),
        "trajectory_root": str(trajectory_root),
        "index": str(root_index),
        "cells": len(cells),
        "outputs": sum(len(cell["samples"]) for cell in cells),
        "steps_per_output": 30,
        "cell_pages": [str(cell_pages[(cell["model_id"], cell["sampler_id"])]) for cell in cells],
    }
    (output_dir / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_gallery(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
