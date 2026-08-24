from __future__ import annotations

import hashlib
import json

import torch
from safetensors.torch import save_file

from src.cli.materialize_hf_diffusers_model import (
    _expected_metadata,
    _verify_materialized_model,
    _write_json_atomic,
    main,
)


def _write_minimal_model(root) -> str:
    (root / "transformer").mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "TestPipeline", "transformer": ["diffusers", "TestModel"]}),
        encoding="utf-8",
    )
    save_file({"weight": torch.ones(1)}, root / "transformer" / "diffusion_pytorch_model.safetensors")
    pipeline = b"reviewed pipeline\n"
    (root / "pipeline.py").write_bytes(pipeline)
    return hashlib.sha256(pipeline).hexdigest()


def test_materializer_reuses_exact_validated_snapshot_without_network(tmp_path) -> None:
    repo_id = "owner/model"
    revision = "a" * 40
    pipeline_sha256 = _write_minimal_model(tmp_path)
    summary, metadata = _verify_materialized_model(
        tmp_path,
        repo_id=repo_id,
        revision=revision,
        pipeline_sha256=pipeline_sha256,
    )
    assert metadata == _expected_metadata(
        repo_id=repo_id,
        revision=revision,
        pipeline_sha256=pipeline_sha256,
        model_summary=summary,
    )
    _write_json_atomic(tmp_path / "conversion_metadata.json", metadata)

    assert (
        main(
            [
                "--repo-id",
                repo_id,
                "--revision",
                revision,
                "--pipeline-sha256",
                pipeline_sha256,
                "--output",
                str(tmp_path),
                "--local-files-only",
            ]
        )
        == 0
    )
