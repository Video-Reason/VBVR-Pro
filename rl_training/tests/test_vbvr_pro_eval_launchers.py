from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "eval" / "vbvr_pro"
REPRODUCE = SCRIPT_ROOT / "reproduce.fish"
RUN = SCRIPT_ROOT / "run.fish"
SWEEP = SCRIPT_ROOT / "sweep.fish"
SUMMARY = SCRIPT_ROOT / "summarize.fish"
VLM_JUDGE = SCRIPT_ROOT / "vlm_judge.fish"
PIPELINE = SCRIPT_ROOT / "lib" / "rule_pipeline.fish"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    clean_env = os.environ.copy()
    for name in (
        "CHECKPOINT",
        "CONVERTED_MODEL",
        "CPS_NOISE_LEVEL",
        "DRY_RUN",
        "GENERATION_MODE",
        "GENERATION_BACKEND",
        "HF_PIPELINE_SHA256",
        "ODE_SOLVER",
        "OUTPUT_ROOT",
        "PRECONVERTED_MODEL",
        "RANK",
        "WORLD_SIZE",
    ):
        clean_env.pop(name, None)
    if env:
        clean_env.update(env)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_surface_has_no_experiment_specific_fish_wrappers() -> None:
    actual = {path.relative_to(SCRIPT_ROOT).as_posix() for path in SCRIPT_ROOT.rglob("*.fish")}
    assert actual == {
        "lib/rule_pipeline.fish",
        "reproduce.fish",
        "run.fish",
        "summarize.fish",
        "sweep.fish",
        "vlm_judge.fish",
    }


@pytest.mark.parametrize("path", [REPRODUCE, RUN, SWEEP, SUMMARY, VLM_JUDGE, PIPELINE])
def test_fish_launchers_parse(path: Path) -> None:
    result = _run("fish", "-n", str(path))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("path", [REPRODUCE, RUN, SWEEP, SUMMARY, VLM_JUDGE])
def test_public_launchers_have_help(path: Path) -> None:
    result = _run("fish", str(path), "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize(
    ("sampler_args", "expected"),
    [
        (("--sampler", "unipc"), "ode_solver=unipc"),
        (("--sampler", "euler"), "ode_solver=euler"),
        (("--sampler", "cps", "--cps-noise", "0.7"), "cps_noise_level=0.7"),
    ],
)
def test_run_dry_run_resolves_sampler(sampler_args: tuple[str, ...], expected: str) -> None:
    result = _run(
        "fish",
        str(RUN),
        "--model",
        "storage/models/example",
        "--output-root",
        "storage/eval_out/example",
        *sampler_args,
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "model=storage/models/example (preconverted Diffusers)" in result.stdout
    assert expected in result.stdout


def test_run_dry_run_resolves_dcp_conversion() -> None:
    result = _run(
        "fish",
        str(RUN),
        "--checkpoint",
        "storage/checkpoints/example/checkpoint-100",
        "--converted-model",
        "storage/models/converted/example-100",
        "--output-root",
        "storage/eval_out/example-100",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checkpoint=storage/checkpoints/example/checkpoint-100" in result.stdout
    assert "converted_model=storage/models/converted/example-100" in result.stdout


def test_run_dry_run_resolves_reviewed_hf_pipeline() -> None:
    pipeline_sha256 = "9" * 64
    result = _run(
        "fish",
        str(RUN),
        "--model",
        "storage/models/hf/example",
        "--output-root",
        "storage/eval_out/example",
        "--generation-backend",
        "hf-pipeline",
        "--hf-pipeline-sha256",
        pipeline_sha256,
        "--sampler",
        "cps",
        "--cps-noise",
        "0.7",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"generation_backend=hf-pipeline pipeline_sha256={pipeline_sha256}" in result.stdout


def test_run_rejects_ambiguous_or_incomplete_sampler_input() -> None:
    both = _run(
        "fish",
        str(RUN),
        "--checkpoint",
        "checkpoint",
        "--model",
        "model",
        "--converted-model",
        "converted",
        "--output-root",
        "output",
        "--dry-run",
    )
    assert both.returncode != 0
    assert "exactly one" in both.stderr

    missing_noise = _run(
        "fish",
        str(RUN),
        "--model",
        "model",
        "--output-root",
        "output",
        "--sampler",
        "cps",
        "--dry-run",
    )
    assert missing_noise.returncode != 0
    assert "--cps-noise is required" in missing_noise.stderr


def test_sweep_dry_run_expands_sampler_cells() -> None:
    result = _run(
        "fish",
        str(SWEEP),
        "--output-base",
        "storage/eval_out/example-sweep",
        "--samplers",
        "unipc,euler,cps:0.3,cps:0.7",
        "--",
        "--model",
        "storage/models/example",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[assignment]") == 4
    assert "ode_solver=unipc" in result.stdout
    assert "ode_solver=euler" in result.stdout
    assert "cps_noise_level=0.3" in result.stdout
    assert "cps_noise_level=0.7" in result.stdout
    assert "cps-noise-0.7" in result.stdout


def test_sweep_assignment_is_deterministic_across_machines() -> None:
    result = _run(
        "fish",
        str(SWEEP),
        "--output-base",
        "storage/eval_out/example-sweep",
        "--samplers",
        "unipc,euler,cps:0.3,cps:0.7",
        "--world-size",
        "2",
        "--rank",
        "1",
        "--assignment-only",
        "--",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assignments = [line for line in result.stdout.splitlines() if line.startswith("[assignment]")]
    assert len(assignments) == 2
    assert "sampler=euler" in assignments[0]
    assert "sampler=cps:0.7" in assignments[1]


def test_release_reproduction_dry_run_expands_both_pinned_matrices() -> None:
    result = _run(
        "fish",
        str(REPRODUCE),
        "--output-base",
        "storage/eval_out/published-hf",
        "--models",
        "rule,qwen",
        "--dry-run",
        "--",
        "--num-gpus",
        "8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[assignment]") == 12
    assert "003373efcbc356e263f4c8d10b3dbb8f5cd7c6d0" in result.stdout
    assert "1282a14cf5379f97ff77326373285533a9e2387d" in result.stdout
    assert "cps-0.7=0.548" in result.stdout
    assert "cps-0.9=0.509" in result.stdout
    assert "rule-003373efcbc3" in result.stdout
    assert "qwen-1282a14cf537" in result.stdout
    assert result.stdout.count("generation_backend=hf-pipeline") == 12
    assert result.stdout.count("media=512x512x81 fps=16") == 12
    assert result.stdout.count("steps=30 cfg=1.0") == 12


def test_vlm_judge_requires_explicit_input_root() -> None:
    result = _run("fish", str(VLM_JUDGE), "--assignment-only", "--no-start-service")
    assert result.returncode != 0
    assert "--input-root requires a value" in result.stderr
