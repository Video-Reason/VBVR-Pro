#!/usr/bin/env fish
# Provision same-major/minor Python headers in the shared ignored storage tree
# and prove that Triton's CUDA driver helper compiles from a fresh cache.

set -l project_root (realpath (dirname (status filename))/../..)
cd $project_root; or exit 1

if not test -x .venv/bin/python
    echo "[error] missing .venv/bin/python under $project_root; run 'uv sync --frozen'" >&2
    exit 1
end
if not command -q uv
    echo "[error] uv is required to provision the shared Python toolchain" >&2
    exit 1
end

set -l python_abi (.venv/bin/python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
set -l install_dir $project_root/storage/toolchains/uv-python

echo "[bootstrap] Installing uv Python $python_abi headers under $install_dir"
uv python install --install-dir $install_dir --no-bin $python_abi; or exit 1

source scripts/lib/env.fish; or exit 1
if not test -f "$WAN_TRAINER_PYTHON_INCLUDE/Python.h"
    echo "[error] scripts/lib/env.fish did not resolve a usable Python.h" >&2
    exit 1
end

set -l fresh_cache (mktemp -d /tmp/wan-trainer-triton-bootstrap.XXXXXX); or exit 1
set -lx TRITON_CACHE_DIR $fresh_cache
.venv/bin/python -c \
    'from triton.runtime import driver; print(f"[bootstrap] Triton target: {driver.active.get_current_target()}")'
or exit 1

echo "[bootstrap] Fresh-cache Triton compilation passed with headers from $WAN_TRAINER_PYTHON_INCLUDE"
