"""Prefetch a pinned Hub attention kernel into persistent user storage."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="_flash_3_hub")
    parser.add_argument("--cache-dir", default="~/.cache/wan-trainer/kernels")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    os.environ["KERNELS_CACHE"] = str(cache_dir)

    try:
        from src.trainer.utils import prefetch_diffusers_attention_backend

        variant_path = prefetch_diffusers_attention_backend(args.backend)
    except Exception as exc:
        print(f"[error] Could not prefetch attention backend {args.backend!r}: {exc}", file=sys.stderr)
        return 1

    print(f"[prefetch] Attention backend {args.backend} is ready in {variant_path}")
    print(f"[prefetch] Persistent kernel cache: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
