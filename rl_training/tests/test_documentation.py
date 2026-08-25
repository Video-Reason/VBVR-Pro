from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINK_RE = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)")


def _release_documents() -> list[Path]:
    return [
        *sorted(_REPO_ROOT.glob("*.md")),
        *sorted((_REPO_ROOT / "docs").rglob("*.md")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.md")),
    ]


@pytest.mark.parametrize("document", _release_documents(), ids=lambda path: str(path.relative_to(_REPO_ROOT)))
def test_release_document_local_links_resolve(document: Path):
    missing: list[str] = []
    for match in _LINK_RE.finditer(document.read_text(encoding="utf-8")):
        raw_target = match.group("target").removeprefix("<").removesuffix(">")
        parsed = urlsplit(raw_target)
        if parsed.scheme or raw_target.startswith("#"):
            continue
        relative = unquote(parsed.path)
        if relative and not (document.parent / relative).resolve().exists():
            missing.append(raw_target)
    assert not missing, f"broken local links in {document.relative_to(_REPO_ROOT)}: {missing}"


def test_release_documents_do_not_expose_environment_specific_details():
    forbidden = (
        "fujian",
        "福建",
        "malaysia",
        "马来",
        "aoss",
        "/mnt/umm/",
        "/mnt/aigc/",
    )
    violations: list[str] = []
    for document in _release_documents():
        text = document.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token.lower() in text:
                violations.append(f"{document.relative_to(_REPO_ROOT)}: {token}")
    assert not violations, f"environment-specific release documentation: {violations}"


def test_only_vbvr_pro_evaluation_surface_is_shipped():
    assert not (_REPO_ROOT / "third_party/VBVR-EvalKit").exists()
    assert not (_REPO_ROOT / "scripts/eval/vbvr").exists()
    assert not (_REPO_ROOT / "src/eval/vbvr").exists()
    assert not (_REPO_ROOT / "src/cli/eval_vbvr.py").exists()


def test_docs_contains_only_indexed_public_guides():
    docs_root = _REPO_ROOT / "docs"
    for relative in (
        "reports",
        "improvements",
        "maze_generation_100k.md",
        "vbvr_lmms_eval.md",
    ):
        assert not (docs_root / relative).exists()

    index = (docs_root / "README.md").read_text(encoding="utf-8")
    unlisted = [
        str(path.relative_to(docs_root))
        for path in sorted(docs_root.rglob("*.md"))
        if path.name != "README.md" and path.relative_to(docs_root).as_posix() not in index
    ]
    assert not unlisted, f"public guides missing from docs/README.md: {unlisted}"


def test_release_uses_the_official_vbvr_pro_rl_dataset():
    sources = [
        *_release_documents(),
        *sorted((_REPO_ROOT / "configs").glob("*.yaml")),
        _REPO_ROOT / "scripts/data/vbvr_pro_unpack_hf.py",
    ]
    legacy_reference = "pufanyi/vbvr-pro-rl-indomain-50k"
    assert all(legacy_reference not in path.read_text(encoding="utf-8") for path in sources)
    for relative in ("README.md", "docs/data.md", "docs/getting_started.md"):
        assert "Video-Reason/VBVR-Pro-RL" in (_REPO_ROOT / relative).read_text(encoding="utf-8")
    assert not (_REPO_ROOT / "scripts/data/vbvr_pro_pack_hf.py").exists()


def test_removed_training_modes_are_not_shipped():
    removed_paths = (
        "src/cli/train_cos.py",
        "src/cli/train_i2v_correction.py",
        "src/models/cos_path.py",
        "src/trainer/cos_trainer.py",
        "src/trainer/i2v_correction_trainer.py",
    )
    assert all(not (_REPO_ROOT / relative).exists() for relative in removed_paths)
    assert not list((_REPO_ROOT / "configs").glob("train_cos_*.yaml"))
    assert not list((_REPO_ROOT / "configs").glob("train_correction_*.yaml"))

    forbidden_tokens = (
        "Chain-of-Step",
        "COSTrainer",
        "CorrectionConfig",
        "I2VCorrectionTrainer",
        "compute_cos_loss",
        "compute_correction_loss",
        "cos_tau_sigma",
        "cos_chain_mode",
        "trainer: cos",
        "train_i2v_correction",
    )
    release_sources = [
        _REPO_ROOT / "README.md",
        _REPO_ROOT / "CHANGELOG.md",
        _REPO_ROOT / "AGENTS.md",
        *_release_documents(),
        *sorted((_REPO_ROOT / "src").rglob("*.py")),
        *sorted((_REPO_ROOT / "configs").glob("*.yaml")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.py")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.fish")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.bash")),
        *sorted((_REPO_ROOT / "scripts").rglob("*.sh")),
    ]
    violations: list[str] = []
    for path in dict.fromkeys(release_sources):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            if token.lower() in text:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {token}")
    assert not violations, f"removed training mode references: {violations}"


def test_training_launcher_surface_is_minimal():
    expected = {
        "grpo_multinode.fish",
        "grpo_vlm_eval_multinode.fish",
        "sft_multinode.fish",
    }
    actual = {path.name for path in (_REPO_ROOT / "scripts/train").iterdir()}
    assert actual == expected
