from pathlib import Path

from src.cli.config_path import resolve_config_path


def test_config_path_prefers_trainer_relative_file(tmp_path: Path, monkeypatch) -> None:
    trainer_root = tmp_path / "rl_training"
    caller_root = tmp_path / "outer"
    trainer_config = trainer_root / "configs/train.yaml"
    caller_config = caller_root / "configs/train.yaml"
    trainer_config.parent.mkdir(parents=True)
    caller_config.parent.mkdir(parents=True)
    trainer_config.write_text("trainer: dancegrpo\n", encoding="utf-8")
    caller_config.write_text("trainer: other\n", encoding="utf-8")
    monkeypatch.chdir(trainer_root)
    monkeypatch.setenv("WAN_TRAINER_CALLER_CWD", str(caller_root))

    assert resolve_config_path("configs/train.yaml") == trainer_config


def test_config_path_accepts_outer_checkout_relative_file(tmp_path: Path, monkeypatch) -> None:
    checkout_root = tmp_path / "VBVR-Pro"
    trainer_root = checkout_root / "rl_training"
    config = trainer_root / "configs/train.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("trainer: dancegrpo\n", encoding="utf-8")
    monkeypatch.chdir(trainer_root)
    monkeypatch.setenv("WAN_TRAINER_CALLER_CWD", str(checkout_root))

    assert resolve_config_path("rl_training/configs/train.yaml") == config


def test_config_path_preserves_absolute_file(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("trainer: dancegrpo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "..")

    assert resolve_config_path(config) == config


def test_missing_config_reports_trainer_relative_location(tmp_path: Path, monkeypatch) -> None:
    trainer_root = tmp_path / "rl_training"
    trainer_root.mkdir()
    monkeypatch.chdir(trainer_root)
    monkeypatch.setenv("WAN_TRAINER_CALLER_CWD", str(tmp_path))

    assert resolve_config_path("configs/missing.yaml") == trainer_root / "configs/missing.yaml"
