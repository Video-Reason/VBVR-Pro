#!/usr/bin/env python
"""Compatibility wrapper for ``python -m src.cli.convert_dcp_to_lora``."""

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

main = import_module("src.cli.convert_dcp_to_lora").main

if __name__ == "__main__":
    raise SystemExit(main())
