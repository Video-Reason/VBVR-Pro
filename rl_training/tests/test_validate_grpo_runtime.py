from pathlib import Path

import pytest

from src.cli.validate_grpo_runtime import _selected_runtime, validate_vbvr_evalkit_contract
from src.eval.vbvr_run_evaluation_parallel import evalkit_source_sha256


def _write_evalkit(path: Path) -> None:
    (path / "vbvr_bench/evaluators").mkdir(parents=True)
    (path / "run_evaluation.py").write_text("TASK_EVALUATOR_MAP = {}\n", encoding="utf-8")
    (path / "vbvr_bench/__init__.py").write_text("", encoding="utf-8")
    (path / "vbvr_bench/evaluators/__init__.py").write_text("", encoding="utf-8")


def test_selected_runtime_reads_config_and_cli_overrides(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("grpo_reward_fn: vbvr_rule\nattention_backend: _flash_3_hub\n")

    assert _selected_runtime(["--config", str(config)]) == ("vbvr_rule", "_flash_3_hub")
    assert _selected_runtime(
        [
            "--config",
            str(config),
            "--grpo_reward_fn",
            "neg_loss",
            "--attention_backend",
            "_native_flash",
        ]
    ) == ("neg_loss", "_native_flash")


def test_external_evalkit_contract_requires_matching_source(tmp_path: Path):
    evalkit = tmp_path / "evalkit"
    _write_evalkit(evalkit)
    expected = evalkit_source_sha256(evalkit)

    assert validate_vbvr_evalkit_contract(str(evalkit), expected) == {
        "path": str(evalkit.resolve()),
        "sha256": expected,
    }

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        validate_vbvr_evalkit_contract(str(evalkit), "0" * 64)


@pytest.mark.parametrize(
    ("evalkit_dir", "digest", "message"),
    [
        (None, "0" * 64, "requires vbvr_reward_evalkit_dir"),
        ("unused", None, "requires vbvr_reward_evalkit_source_sha256"),
        ("unused", "bad", "64-character hexadecimal"),
    ],
)
def test_external_evalkit_contract_rejects_incomplete_configuration(evalkit_dir, digest, message):
    with pytest.raises(ValueError, match=message):
        validate_vbvr_evalkit_contract(evalkit_dir, digest)
