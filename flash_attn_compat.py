"""Small PyTorch SDPA fallback for BAGEL's variable-length attention call."""

from __future__ import annotations

import importlib.machinery
import sys
import types


def install_flash_attn_fallback() -> None:
    """Provide ``flash_attn_varlen_func`` when FlashAttention is unavailable."""
    try:
        import flash_attn  # noqa: F401

        return
    except ImportError:
        pass

    import torch
    from torch.nn.functional import scaled_dot_product_attention

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        **_kwargs,
    ):
        del max_seqlen_q, max_seqlen_k
        if window_size not in {(-1, -1), None}:
            raise NotImplementedError(
                "the SDPA fallback does not implement local attention windows"
            )

        outputs = []
        sequence_count = cu_seqlens_q.numel() - 1
        for index in range(sequence_count):
            q_start, q_end = (int(value) for value in cu_seqlens_q[index : index + 2])
            k_start, k_end = (int(value) for value in cu_seqlens_k[index : index + 2])
            query = q[q_start:q_end].transpose(0, 1)
            key = k[k_start:k_end].transpose(0, 1)
            value = v[k_start:k_end].transpose(0, 1)

            attention_mask = None
            if causal:
                query_length, key_length = query.shape[-2], key.shape[-2]
                query_positions = torch.arange(
                    query_length, device=query.device
                ).unsqueeze(1)
                key_positions = torch.arange(
                    key_length, device=query.device
                ).unsqueeze(0)
                attention_mask = key_positions <= (
                    key_length - query_length + query_positions
                )

            output = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                scale=softmax_scale,
                enable_gqa=query.shape[0] != key.shape[0],
            )
            outputs.append(output.transpose(0, 1))
        return torch.cat(outputs, dim=0)

    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    module.__version__ = "0.0.sdpa-fallback"
    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = module
    print(
        "WARNING: flash-attn is unavailable; using the slower PyTorch SDPA fallback."
    )
