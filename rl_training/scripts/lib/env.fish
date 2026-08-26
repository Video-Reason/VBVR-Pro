# Shared project environment setup for fish launchers.

# Python entry points use this to resolve a config argument relative to the
# outer checkout even after the launcher changes into rl_training.
set -q WAN_TRAINER_CALLER_CWD; or set -gx WAN_TRAINER_CALLER_CWD (realpath .)

set -gx WAN_TRAINER_ROOT (realpath (dirname (status filename))/../..)

cd $WAN_TRAINER_ROOT; or exit 1

if not test -f .venv/bin/activate.fish
    echo "[error] missing .venv/bin/activate.fish under $WAN_TRAINER_ROOT; run 'uv sync --locked'" >&2
    exit 1
end

source .venv/bin/activate.fish; or exit 1
set -gx PYTHONPATH $WAN_TRAINER_ROOT $PYTHONPATH

# Triton compiles a small CUDA driver extension lazily on the first compiled
# forward. Debian/Ubuntu virtualenvs inherit the system Python include path,
# which may exist in sysconfig even when Python.h was omitted from the image.
# Prefer an explicit/operator-provided header directory, then the interpreter's
# native include directory, an existing CPATH entry, the project-local shared
# uv toolchain, or an already-installed user-level uv Python with the same
# major.minor ABI. Never download at launch.
set -l wan_python_include (.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')
set -l wan_python_abi (.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
set -l wan_python_header_include

for candidate in $WAN_TRAINER_PYTHON_INCLUDE $wan_python_include $CPATH
    if test -n "$candidate"; and test -f "$candidate/Python.h"
        set wan_python_header_include $candidate
        break
    end
end

# Cluster images may provide only the Python runtime package, while all nodes
# share the repository's storage/ tree. `uv python install --no-bin` can
# provision this directory once before launch without requiring root access.
if test -z "$wan_python_header_include"
    set -l wan_project_python_root $WAN_TRAINER_ROOT/storage/toolchains/uv-python
    for candidate in \
            $wan_project_python_root/cpython-$wan_python_abi.*-*/include/python$wan_python_abi \
            $wan_project_python_root/cpython-$wan_python_abi-*/include/python$wan_python_abi
        if test -f "$candidate/Python.h"
            set wan_python_header_include $candidate
            break
        end
    end
end

if test -z "$wan_python_header_include"; and command -q uv
    set -l wan_managed_python (uv python find --managed-python --no-python-downloads $wan_python_abi 2>/dev/null)
    if test -n "$wan_managed_python"; and test -x "$wan_managed_python"
        set -l wan_managed_include ($wan_managed_python -c \
            'import sysconfig; print(sysconfig.get_path("include"))')
        if test -f "$wan_managed_include/Python.h"
            set wan_python_header_include $wan_managed_include
        end
    end
end

if test -n "$wan_python_header_include"
    if not contains -- $wan_python_header_include $CPATH
        set -gx CPATH $wan_python_header_include $CPATH
    end
    set -gx WAN_TRAINER_PYTHON_INCLUDE $wan_python_header_include

    if test "$wan_python_header_include" != "$wan_python_include"
        if test -f "$wan_python_include/Python.h"
            echo "[env] Using configured Python headers from $wan_python_header_include via CPATH"
        else
            echo "[env] Python headers missing under $wan_python_include; using $wan_python_header_include via CPATH"
        end
    end
else
    echo "[warning] Python.h for Python $wan_python_abi was not found; torch.compile/Triton will fail." >&2
    echo "[warning] Install python$wan_python_abi-dev or export WAN_TRAINER_PYTHON_INCLUDE/CPATH before launch." >&2
end
