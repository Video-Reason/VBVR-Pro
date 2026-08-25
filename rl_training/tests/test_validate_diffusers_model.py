import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.eval.validate_diffusers_model import ModelValidationError, validate_diffusers_model


def _write_model(root: Path) -> None:
    (root / "transformer").mkdir(parents=True)
    (root / "vae").mkdir()
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "TestPipeline",
                "transformer": ["diffusers", "Transformer"],
                "vae": ["diffusers", "VAE"],
                "optional": [None, None],
            }
        )
    )
    save_file({"weight": torch.ones(1)}, root / "transformer" / "model-00001-of-00001.safetensors")
    (root / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    save_file({"vae_weight": torch.ones(1)}, root / "vae" / "model.safetensors")


def test_validate_diffusers_model_checks_layout_indexes_and_headers(tmp_path: Path):
    _write_model(tmp_path)
    summary = validate_diffusers_model(tmp_path)
    assert summary == {"components": 2, "indexes": 1, "safetensors": 2, "tensors": 2}


def test_validate_diffusers_model_rejects_missing_indexed_shard(tmp_path: Path):
    _write_model(tmp_path)
    (tmp_path / "transformer" / "model-00001-of-00001.safetensors").unlink()
    try:
        validate_diffusers_model(tmp_path)
    except ModelValidationError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing shard was accepted")


def test_validate_diffusers_model_rejects_index_key_mismatch(tmp_path: Path):
    _write_model(tmp_path)
    index = tmp_path / "transformer" / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"different": "model-00001-of-00001.safetensors"}}))
    try:
        validate_diffusers_model(tmp_path)
    except ModelValidationError as exc:
        assert "index mismatch" in str(exc)
    else:
        raise AssertionError("index mismatch was accepted")
