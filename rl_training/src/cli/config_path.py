"""Resolve config arguments passed through repository-aware launchers."""

from __future__ import annotations

import os
from pathlib import Path

_LAUNCH_CWD_ENV = "WAN_TRAINER_CALLER_CWD"


def resolve_config_path(value: str | Path) -> Path:
    """Resolve a config from the trainer root or the launcher's caller cwd.

    Fish launchers change into ``rl_training`` before invoking Python so that
    paths stored inside YAML remain trainer-relative. Preserve that behavior
    while also accepting a config argument written relative to the outer
    VBVR-Pro checkout, such as ``rl_training/configs/train_rl_5b_cps.yaml``.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    trainer_relative = (Path.cwd() / path).resolve()
    if trainer_relative.is_file():
        return trainer_relative

    caller_cwd = os.environ.get(_LAUNCH_CWD_ENV)
    if caller_cwd:
        caller_relative = (Path(caller_cwd).expanduser() / path).resolve()
        if caller_relative.is_file():
            return caller_relative

    # Keep the normal trainer-relative error location when neither candidate
    # exists; read_text() will then raise a precise FileNotFoundError.
    return trainer_relative
