from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.eval.vbvr_restructure_to_evalkit import _load_split_classifier


def _write_classifier(evalkit: Path, *, prefix: str) -> None:
    evaluators = evalkit / "vbvr_bench/evaluators"
    evaluators.mkdir(parents=True)
    (evalkit / "vbvr_bench/__init__.py").write_text("", encoding="utf-8")
    (evaluators / "__init__.py").write_text(
        f"def is_out_of_domain(task):\n    return task.startswith({prefix!r})\n",
        encoding="utf-8",
    )


def test_restructure_uses_explicit_external_evalkit_classifier(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_classifier(evalkit, prefix="out-")

    model_out = tmp_path / "model"
    source = model_out / "Open_60"
    (source / "in-task").mkdir(parents=True)
    (source / "out-task").mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.eval.vbvr_restructure_to_evalkit",
            "--model_out",
            str(model_out),
            "--evalkit_dir",
            str(evalkit),
            "--source_split",
            "Open_60",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (model_out / "In-Domain_50/in-task").is_symlink()
    assert (model_out / "Out-of-Domain_50/out-task").is_symlink()


def test_classifier_switches_between_external_checkouts(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_classifier(first, prefix="first-")
    _write_classifier(second, prefix="second-")
    try:
        first_classifier = _load_split_classifier(first)
        second_classifier = _load_split_classifier(second)

        assert first_classifier("first-task")
        assert not first_classifier("second-task")
        assert second_classifier("second-task")
        assert not second_classifier("first-task")
    finally:
        for name in list(sys.modules):
            if name == "vbvr_bench" or name.startswith("vbvr_bench."):
                sys.modules.pop(name, None)
        for checkout in (first, second):
            while str(checkout.resolve()) in sys.path:
                sys.path.remove(str(checkout.resolve()))
