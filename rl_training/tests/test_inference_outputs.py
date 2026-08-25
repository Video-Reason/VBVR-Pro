from types import SimpleNamespace

import numpy as np
import torch

from src.inference.engine import StepwiseResult
from src.inference.outputs import _step_preview_labels, write_outputs


def test_step_preview_labels_are_one_based_and_mark_actual_final():
    result = StepwiseResult(
        final_latent=torch.zeros(1, 1, 1, 1, 1),
        pred_x0=[torch.zeros(1, 1, 1, 1, 1) for _ in range(3)],
        sigmas=[1.0, 0.625, 0.25, 0.0],
        timesteps=[1000.0, 625.0, 250.0],
    )

    assert _step_preview_labels(result) == [
        "01/03 x0 s=1.000",
        "02/03 x0 s=0.625",
        "03/03 final s=0",
    ]


def test_write_outputs_uses_actual_final_latent_for_last_step(tmp_path, monkeypatch):
    pred_0 = torch.full((1, 1, 1, 1, 1), 1.0)
    pred_1 = torch.full((1, 1, 1, 1, 1), 2.0)
    final = torch.full((1, 1, 1, 1, 1), 3.0)
    result = StepwiseResult(
        final_latent=final,
        pred_x0=[pred_0, pred_1],
        sigmas=[1.0, 0.25, 0.0],
        timesteps=[1000.0, 250.0],
    )
    cfg = SimpleNamespace(
        device="cpu",
        save_reference=False,
        save_steps=True,
        fps=16,
        grid_cols=2,
        grid_thumb_width=32,
        model_path="model",
        checkpoint=None,
        use_ema=False,
        mode="cps",
        sde_formula="flowcps",
        effective_noise_scale=0.7,
        num_sampling_steps=2,
        cfg_scale=1.0,
        seed=0,
        batch_size=1,
        share_init_noise=True,
    )
    prepared = SimpleNamespace(reference_latents=[], source="test", summary={}, metadata={})
    decoded_step_latents = []
    gallery_labels = []

    def fake_decode_step(_model, latent):
        decoded_step_latents.append(latent.clone())
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def fake_decode_final(_model, latent):
        assert torch.equal(latent, final)
        return np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)

    def fake_export(_video, path, _fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    def fake_gallery(_videos, path, **kwargs):
        gallery_labels.append(kwargs["step_labels"])
        path.write_bytes(b"gallery")

    monkeypatch.setattr("src.inference.outputs.decode_latents_to_uint8", fake_decode_step)
    monkeypatch.setattr("src.inference.outputs.decode_batch_to_uint8", fake_decode_final)
    monkeypatch.setattr("src.inference.outputs.export_uint8_video", fake_export)
    monkeypatch.setattr("src.inference.outputs.save_step_grid_video", fake_gallery)
    monkeypatch.setattr("src.inference.outputs.save_step_contact_sheet", fake_gallery)

    manifest = write_outputs(object(), cfg, prepared, result, tmp_path)

    assert torch.equal(decoded_step_latents[0], pred_0)
    assert torch.equal(decoded_step_latents[1], final)
    assert gallery_labels == [
        ["01/02 x0 s=1.000", "02/02 final s=0"],
        ["01/02 x0 s=1.000", "02/02 final s=0"],
    ]
    assert manifest["step_previews"][0]["kind"] == "predicted_clean_x0"
    assert manifest["step_previews"][1]["kind"] == "final_latent"
    assert manifest["step_previews"][1]["output_sigma"] == 0.0


def test_write_outputs_uses_formal_video_override_in_last_gallery_cell(tmp_path, monkeypatch):
    pred = torch.full((1, 1, 1, 1, 1), 1.0)
    final = torch.full((1, 1, 1, 1, 1), 3.0)
    result = StepwiseResult(
        final_latent=final,
        pred_x0=[pred, final],
        sigmas=[1.0, 0.25, 0.0],
        timesteps=[1000.0, 250.0],
    )
    cfg = SimpleNamespace(
        device="cpu",
        save_reference=False,
        save_steps=True,
        fps=16,
        grid_cols=2,
        grid_thumb_width=32,
        model_path="model",
        checkpoint=None,
        use_ema=False,
        mode="cps",
        sde_formula="flowcps",
        effective_noise_scale=0.7,
        num_sampling_steps=2,
        cfg_scale=1.0,
        seed=0,
        batch_size=1,
        share_init_noise=True,
    )
    prepared = SimpleNamespace(reference_latents=[], source="test", summary={}, metadata={})
    formal = np.full((2, 4, 4, 3), 211, dtype=np.uint8)
    decoded = []
    galleries = []

    def fake_decode_step(_model, latent):
        decoded.append(latent.clone())
        return np.zeros((2, 4, 4, 3), dtype=np.uint8)

    def fail_decode_final(*_args):
        raise AssertionError("the formal override should avoid decoding the final latent again")

    def fake_export(_video, path, _fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    def fake_gallery(videos, path, **_kwargs):
        galleries.append(videos)
        path.write_bytes(b"gallery")

    monkeypatch.setattr("src.inference.outputs.decode_latents_to_uint8", fake_decode_step)
    monkeypatch.setattr("src.inference.outputs.decode_batch_to_uint8", fail_decode_final)
    monkeypatch.setattr("src.inference.outputs.export_uint8_video", fake_export)
    monkeypatch.setattr("src.inference.outputs.save_step_grid_video", fake_gallery)
    monkeypatch.setattr("src.inference.outputs.save_step_contact_sheet", fake_gallery)

    write_outputs(object(), cfg, prepared, result, tmp_path, final_video_override=formal)

    assert len(decoded) == 1
    assert torch.equal(decoded[0], pred)
    assert len(galleries) == 2
    assert all(np.array_equal(videos[-1], formal) for videos in galleries)
