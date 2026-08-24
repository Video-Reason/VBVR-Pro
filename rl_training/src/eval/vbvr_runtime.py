"""Pinned runtime contract for VBVR-Pro ``main_v2`` scoring.

VBVR's evaluator is not defined by its Python source alone. Several rules call
into OpenCV, SciPy, and scikit-image, so a dependency upgrade can change scores
or even change array layouts without changing EvalKit itself. Training reward
workers and offline evaluation both call this module before loading EvalKit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

VBVR_SCORER_RUNTIME_CONTRACT = "vbvr-main-v2-e140038f-runtime-v1"
EXPECTED_VBVR_SCORER_DISTRIBUTIONS: dict[str, str] = {
    "easyocr": "1.7.2",
    "imageio": "2.37.3",
    "imageio-ffmpeg": "0.6.0",
    "norfair": "2.3.0",
    "numpy": "2.4.4",
    "opencv-python": "4.13.0.92",
    "opencv-python-headless": "4.13.0.92",
    "pillow": "12.2.0",
    "scikit-image": "0.26.0",
    "scipy": "1.17.1",
    "tifffile": "2026.3.3",
}
EXPECTED_CV2_VERSION = "4.13.0"
_SCORER_IMPORTS = ("easyocr", "norfair", "scipy", "skimage")


class VBVRScorerRuntimeError(RuntimeError):
    """Raised when the process does not match the pinned scorer runtime."""


def _hough_lines_p_layout(cv2_module: Any, numpy_module: Any) -> tuple[str, tuple[int, ...] | None]:
    """Exercise the array layout relied on by O-18/O-19.

    OpenCV 4.13 returns ``(N, 1, 4)`` here. OpenCV 5.0 changed the Python
    binding to ``(N, 4)``, while EvalKit e140038f still indexes ``line[0][2]``.
    """
    mask = numpy_module.zeros((64, 64), dtype=numpy_module.uint8)
    cv2_module.line(mask, (8, 8), (56, 56), 255, 2)
    lines = cv2_module.HoughLinesP(
        mask,
        1,
        numpy_module.pi / 180,
        threshold=10,
        minLineLength=10,
        maxLineGap=2,
    )
    if lines is None:
        return "none", None
    shape = tuple(int(value) for value in lines.shape)
    if len(shape) == 3 and shape[0] > 0 and shape[1:] == (1, 4):
        return "N,1,4", shape
    return "unexpected", shape


def _runtime_fingerprint(report: Mapping[str, Any]) -> str:
    # Keep the fingerprint stable across cheap/full validation and across a
    # harmless change in the number of lines detected by the synthetic probe.
    contract_payload = {
        "contract": report["contract"],
        "distributions": report["distributions"],
        "modules": report["modules"],
        "probes": {"hough_lines_p_layout": report["probes"]["hough_lines_p_layout"]},
    }
    payload = json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_vbvr_scorer_runtime(
    *,
    expected_distributions: Mapping[str, str] = EXPECTED_VBVR_SCORER_DISTRIBUTIONS,
    expected_cv2_version: str = EXPECTED_CV2_VERSION,
    version_getter: Callable[[str], str] | None = None,
    cv2_module: Any | None = None,
    numpy_module: Any | None = None,
    verify_imports: bool = True,
) -> dict[str, Any]:
    """Validate and fingerprint the dependency behavior used by VBVR scoring."""
    if version_getter is None:
        version_getter = importlib.metadata.version

    errors: list[str] = []
    distributions: dict[str, str] = {}
    for name, expected in expected_distributions.items():
        try:
            actual = version_getter(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "<missing>"
        except Exception as exc:
            actual = f"<error: {exc}>"
        distributions[name] = actual
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")

    if cv2_module is None:
        try:
            cv2_module = importlib.import_module("cv2")
        except Exception as exc:
            errors.append(f"cv2 import failed: {exc}")
    if numpy_module is None:
        try:
            numpy_module = importlib.import_module("numpy")
        except Exception as exc:
            errors.append(f"numpy import failed: {exc}")

    cv2_version = getattr(cv2_module, "__version__", "<unavailable>")
    if cv2_version != expected_cv2_version:
        errors.append(f"cv2 module: expected {expected_cv2_version}, found {cv2_version}")

    hough_layout = "unavailable"
    hough_shape: tuple[int, ...] | None = None
    if cv2_module is not None and numpy_module is not None:
        try:
            hough_layout, hough_shape = _hough_lines_p_layout(cv2_module, numpy_module)
        except Exception as exc:
            errors.append(f"cv2.HoughLinesP compatibility probe failed: {exc}")
        else:
            if hough_layout != "N,1,4":
                errors.append(
                    f"cv2.HoughLinesP returned an EvalKit-incompatible layout: expected (N, 1, 4), found {hough_shape}"
                )

    imported_modules: dict[str, str] = {}
    if verify_imports:
        for module_name in _SCORER_IMPORTS:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                imported_modules[module_name] = f"<error: {exc}>"
                errors.append(f"{module_name} import failed: {exc}")
            else:
                imported_modules[module_name] = str(getattr(module, "__version__", "available"))

    report: dict[str, Any] = {
        "contract": VBVR_SCORER_RUNTIME_CONTRACT,
        "distributions": distributions,
        "modules": {"cv2": cv2_version},
        "probes": {
            "hough_lines_p_layout": hough_layout,
            "hough_lines_p_shape": list(hough_shape) if hough_shape is not None else None,
        },
    }
    if verify_imports:
        report["imports"] = imported_modules
    report["sha256"] = _runtime_fingerprint(report)

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise VBVRScorerRuntimeError(
            "VBVR scorer runtime contract failed:\n"
            f"{detail}\n"
            "Run `uv sync --frozen` on every node, then restart the whole training/evaluation job. "
            "Updating the uv environment cannot replace cv2 or other modules already loaded by scorer workers."
        )
    return report


def runtime_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable runtime report")
    parser.add_argument(
        "--skip-import-checks",
        action="store_true",
        help="Check pinned versions and OpenCV behavior without importing all scorer libraries",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_vbvr_scorer_runtime(verify_imports=not args.skip_import_checks)
    except VBVRScorerRuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(runtime_report_json(report))
    else:
        print(
            "[preflight] VBVR scorer runtime passed: "
            f"contract={report['contract']} sha256={report['sha256']} "
            f"cv2={report['modules']['cv2']} HoughLinesP={report['probes']['hough_lines_p_layout']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
