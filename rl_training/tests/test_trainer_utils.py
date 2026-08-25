import kernels
import torch
from diffusers.models import attention_dispatch

from src.trainer.checkpoint import _extract_pipeline_weights
from src.trainer.rewards.maze import _as_batched_tensor
from src.trainer.rewards.maze_line import _goal_region_score, _soft_color_mask
from src.trainer.utils import (
    collate,
    prefetch_diffusers_attention_backend,
    prepare_diffusers_attention_backend,
)


class _PlainModule(torch.nn.Module):
    pass


class _LoraModule(torch.nn.Module):
    peft_config = {"default": object()}


def test_extract_pipeline_weights_overlays_lora_ema_on_raw_base():
    flat = {
        "train_state.transformer.blocks.0.attn.to_q.base_layer.weight": torch.tensor([1.0]),
        "train_state.transformer.blocks.0.attn.to_q.lora_A.default.weight": torch.tensor([2.0]),
        "ema.shadow.transformer.blocks.0.attn.to_q.lora_A.default.weight": torch.tensor([3.0]),
    }

    weights, source = _extract_pipeline_weights(flat, "transformer", _LoraModule(), use_ema=True)

    assert source == "raw+EMA"
    assert torch.equal(weights["blocks.0.attn.to_q.base_layer.weight"], torch.tensor([1.0]))
    assert torch.equal(weights["blocks.0.attn.to_q.lora_A.default.weight"], torch.tensor([3.0]))


def test_extract_pipeline_weights_plain_model_uses_ema_directly():
    flat = {
        "train_state.transformer.blocks.0.attn.to_q.weight": torch.tensor([1.0]),
        "ema.shadow.transformer.blocks.0.attn.to_q.weight": torch.tensor([2.0]),
    }

    weights, source = _extract_pipeline_weights(flat, "transformer", _PlainModule(), use_ema=True)

    assert source == "EMA"
    assert list(weights) == ["blocks.0.attn.to_q.weight"]
    assert torch.equal(weights["blocks.0.attn.to_q.weight"], torch.tensor([2.0]))


def test_collate_preserves_variable_shape_tensor_metadata():
    batch = [
        {
            "prompt_embeds": torch.zeros(4, 8),
            "condition": torch.zeros(5, 3, 16, 16),
            "maze_grid": torch.zeros(16, 16, dtype=torch.int8),
            "sample_key": "a",
        },
        {
            "prompt_embeds": torch.ones(4, 8),
            "condition": torch.ones(5, 3, 16, 16),
            "maze_grid": torch.ones(24, 24, dtype=torch.int8),
            "sample_key": "b",
        },
    ]

    out = collate(batch)

    assert out["prompt_embeds"].shape == (2, 4, 8)
    assert out["condition"].shape == (2, 5, 3, 16, 16)
    assert isinstance(out["maze_grid"], list)
    assert [tuple(grid.shape) for grid in out["maze_grid"]] == [(16, 16), (24, 24)]
    assert out["sample_key"] == ["a", "b"]


def test_as_batched_tensor_pads_variable_shape_grids():
    small = torch.zeros(2, 2, dtype=torch.int8)
    large = torch.zeros(3, 4, dtype=torch.int8)

    out = _as_batched_tensor([small, large], device=torch.device("cpu"), pad_value=1)

    assert out.shape == (2, 3, 4)
    assert torch.equal(out[0, :2, :2], small)
    assert torch.equal(out[1], large)
    assert torch.all(out[0, 2, :] == 1)
    assert torch.all(out[0, :, 2:] == 1)


def test_soft_color_mask_prefers_configured_line_color():
    pixels = torch.full((1, 3, 1, 2, 2), 240.0)
    pixels[:, :, 0, 0, 0] = torch.tensor([220.0, 40.0, 40.0])
    mask = _soft_color_mask(
        pixels,
        torch.tensor([[220.0, 40.0, 40.0]]),
        threshold=90.0,
        temperature=12.0,
    )

    assert mask[0, 0, 0, 0] > 0.99
    assert mask[0, 0, 1, 1] < 0.01


def test_goal_region_score_uses_goal_cell_window():
    final_mask = torch.zeros(1, 8, 8)
    final_mask[0, 5, 5] = 0.75

    score = _goal_region_score(
        final_mask,
        goal_ij=torch.tensor([[2, 2]]),
        cell_px=torch.tensor([2.0]),
        goal_cells=1.0,
    )

    assert torch.allclose(score, torch.tensor([0.75]))


def test_prepare_diffusers_attention_backend_initializes_lazy_backend(monkeypatch):
    entered = []
    kernel_calls = []

    class _BackendContext:
        def __enter__(self):
            kernels.get_kernel("kernels-community/flash-attn3", version=1)
            entered.append("enter")

        def __exit__(self, exc_type, exc, traceback):
            entered.append("exit")

    def _load_kernel(repo_id, *args, **kwargs):
        kernel_calls.append((repo_id, args, kwargs))

    monkeypatch.setattr(kernels, "load_kernel", _load_kernel)
    monkeypatch.setattr(attention_dispatch, "attention_backend", lambda backend: _BackendContext())

    assert prepare_diffusers_attention_backend("_flash_3_hub")
    assert entered == ["enter", "exit"]
    assert kernel_calls == [
        (
            "kernels-community/flash-attn3",
            (),
            {
                "lockfile": None,
                "revision": "43f0bd269777115d94ff826e0d113ce9c1c9087b",
                "backend": None,
            },
        )
    ]
    assert not prepare_diffusers_attention_backend(None)


def test_prepare_diffusers_attention_backend_rejects_unexpected_hub_repository(monkeypatch):
    class _BackendContext:
        def __enter__(self):
            kernels.get_kernel("unexpected/kernel", version=1)

        def __exit__(self, exc_type, exc, traceback):
            pass

    monkeypatch.setattr(attention_dispatch, "attention_backend", lambda backend: _BackendContext())

    try:
        prepare_diffusers_attention_backend("_flash_3_hub")
    except RuntimeError as error:
        assert "unexpected Hub kernel repository" in str(error)
    else:
        raise AssertionError("unexpected Hub kernel repository was trusted")


def test_prefetch_diffusers_attention_backend_downloads_pinned_revision(monkeypatch):
    calls = []

    def _install_kernel(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return "/persistent/kernel/variant"

    monkeypatch.setattr(kernels, "install_kernel", _install_kernel)

    assert prefetch_diffusers_attention_backend("_flash_3_hub") == "/persistent/kernel/variant"
    assert calls == [
        (
            "kernels-community/flash-attn3",
            {
                "revision": "43f0bd269777115d94ff826e0d113ce9c1c9087b",
                "validate_dependencies": True,
            },
        )
    ]
