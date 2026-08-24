from __future__ import annotations

import hashlib

import pytest
import torch

from src.cli.eval_i2v_hf_pipeline import _pipeline_call_kwargs, parse_args, verify_pipeline_source


def _required_args(pipeline_sha256: str) -> list[str]:
    return [
        "--eval_json",
        "eval.json",
        "--model_path",
        "model",
        "--output_dir",
        "output",
        "--sampler",
        "cps",
        "--cps_eta",
        "0.7",
        "--pipeline_sha256",
        pipeline_sha256,
    ]


def test_hf_pipeline_parser_requires_sampler_specific_cps_eta() -> None:
    digest = "a" * 64
    args = parse_args(_required_args(digest))
    assert args.sampler == "cps"
    assert args.cps_eta == pytest.approx(0.7)
    assert args.pipeline_sha256 == digest

    missing_eta = _required_args(digest)
    del missing_eta[missing_eta.index("--cps_eta") : missing_eta.index("--cps_eta") + 2]
    with pytest.raises(SystemExit):
        parse_args(missing_eta)


def test_hf_pipeline_source_must_match_reviewed_digest(tmp_path) -> None:
    source = b"reviewed pipeline\n"
    pipeline_path = tmp_path / "pipeline.py"
    pipeline_path.write_bytes(source)
    digest = hashlib.sha256(source).hexdigest()

    assert verify_pipeline_source(tmp_path, digest) == pipeline_path
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_pipeline_source(tmp_path, "0" * 64)


def test_hf_pipeline_call_routes_sampler_and_generator() -> None:
    args = parse_args(_required_args("b" * 64))
    generator = torch.Generator().manual_seed(7)
    assert _pipeline_call_kwargs(args, generator) == {
        "generator": generator,
        "sampler": "cps",
        "cps_eta": pytest.approx(0.7),
    }
