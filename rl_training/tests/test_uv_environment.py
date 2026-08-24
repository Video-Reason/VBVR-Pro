from __future__ import annotations

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_uv_is_the_only_project_environment_workflow():
    manifest = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv = manifest["tool"]["uv"]

    assert uv["required-version"] == "==0.11.33"
    assert uv["override-dependencies"] == ["numpy==2.4.4"]
    assert uv["sources"] == {
        "torch": {"index": "pytorch-cu126"},
        "torchvision": {"index": "pytorch-cu126"},
    }
    assert manifest["dependency-groups"]["dev"] == ["pytest>=9.1.1", "ruff==0.16.3"]
    assert (_REPO_ROOT / "uv.lock").is_file()
    assert not (_REPO_ROOT / "pixi.lock").exists()
    assert (_REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"

    vllm_lock = (_REPO_ROOT / "requirements" / "vllm.lock").read_text(encoding="utf-8")
    assert "numpy==2.3.5" in vllm_lock
    assert "torch==2.11.0+cu126" in vllm_lock
    assert "vllm==0.26.0" in vllm_lock


def test_release_sources_do_not_reference_legacy_environment_commands():
    forbidden = (
        "[tool.pixi",
        ".pixi/envs/",
        "pixi install",
        "pixi lock",
        "pixi run",
        "setup-pixi",
    )
    roots = (
        _REPO_ROOT / ".github",
        _REPO_ROOT / "configs",
        _REPO_ROOT / "docs",
        _REPO_ROOT / "scripts",
        _REPO_ROOT / "src",
    )
    paths = [
        _REPO_ROOT / "AGENTS.md",
        _REPO_ROOT / "CONTRIBUTING.md",
        _REPO_ROOT / "README.md",
        _REPO_ROOT / "pyproject.toml",
    ]
    paths.extend(path for root in roots for path in root.rglob("*") if path.is_file())

    violations: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {token}")

    assert not violations, f"legacy environment workflow references: {violations}"
