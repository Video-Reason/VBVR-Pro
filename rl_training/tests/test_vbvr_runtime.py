from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import numpy as np
import pytest

from src.eval.vbvr_runtime import (
    EXPECTED_CV2_VERSION,
    EXPECTED_VBVR_SCORER_DISTRIBUTIONS,
    VBVRScorerRuntimeError,
    validate_vbvr_scorer_runtime,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_dependencies_exactly_pin_vbvr_scorer_runtime():
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    for name, version in EXPECTED_VBVR_SCORER_DISTRIBUTIONS.items():
        assert f"{name}=={version}" in dependencies


def test_current_vbvr_scorer_runtime_contract_passes():
    report = validate_vbvr_scorer_runtime()
    cheap_report = validate_vbvr_scorer_runtime(verify_imports=False)

    assert report["modules"]["cv2"] == EXPECTED_CV2_VERSION
    assert report["probes"]["hough_lines_p_layout"] == "N,1,4"
    assert len(report["sha256"]) == 64
    assert report["sha256"] == cheap_report["sha256"]


def test_runtime_rejects_distribution_drift():
    actual = dict(EXPECTED_VBVR_SCORER_DISTRIBUTIONS)
    actual["opencv-python-headless"] = "5.0.0.93"

    with pytest.raises(VBVRScorerRuntimeError, match=r"opencv-python-headless: expected 4\.13\.0\.92"):
        validate_vbvr_scorer_runtime(
            version_getter=actual.__getitem__,
            verify_imports=False,
        )


class _OpenCV5Layout:
    __version__ = EXPECTED_CV2_VERSION

    @staticmethod
    def line(mask, start, end, color, thickness):
        return mask

    @staticmethod
    def HoughLinesP(mask, rho, theta, threshold, minLineLength, maxLineGap):
        return np.zeros((2, 4), dtype=np.int32)


def test_runtime_rejects_hough_lines_p_layout_drift():
    with pytest.raises(VBVRScorerRuntimeError, match=r"expected \(N, 1, 4\), found \(2, 4\)"):
        validate_vbvr_scorer_runtime(
            version_getter=importlib.metadata.version,
            cv2_module=_OpenCV5Layout,
            numpy_module=np,
            verify_imports=False,
        )
