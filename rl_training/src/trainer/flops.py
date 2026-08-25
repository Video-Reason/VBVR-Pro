"""FLOPs estimation and MFU monitoring for Wan I2V training."""

import torch
from loguru import logger


def compute_wan_seq_len(
    num_frames: int,
    height: int,
    width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    vae_temporal_factor: int = 4,
    vae_spatial_factor: int = 8,
) -> int:
    """Visual token count after VAE compression + patching."""
    T = (num_frames - 1) // vae_temporal_factor + 1
    H = height // vae_spatial_factor
    W = width // vae_spatial_factor
    p_t, p_h, p_w = patch_size
    return (T // p_t) * (H // p_h) * (W // p_w)


def estimate_wan_forward_flops(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    ffn_dim: int,
    seq_len: int,
    text_seq_len: int = 512,
) -> int:
    """Forward-pass FLOPs for one sample through one Wan transformer expert.

    Counts matmuls in self-attn, cross-attn, and FFN per block.
    Norms, activations, biases, RoPE contribute <1% and are ignored.

    Per block:
      Self-Attn:  8*S*d^2 + 4*S^2*d     (QKV+out proj, QK^T, attn*V)
      Cross-Attn: 4*S*d^2 + 4*St*d^2 + 4*S*St*d  (Q+out, KV on text, scores)
      FFN:        4*S*d*d_ff              (up + down proj)
    """
    d = num_heads * head_dim
    S = seq_len
    S_t = text_seq_len

    per_block = 8 * S * d * d + 4 * S * S * d + 4 * S * d * d + 4 * S_t * d * d + 4 * S * S_t * d + 4 * S * d * ffn_dim
    return num_layers * per_block


# bf16 dense tensor core peak TFLOPS (no structured sparsity)
_GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    "A100": 312.0,
    "A800": 312.0,
    "H100": 989.4,
    "H800": 989.4,
    "H200": 989.4,
    "L40S": 362.0,
    "4090": 165.2,
}


def get_gpu_peak_flops_bf16() -> float | None:
    """Per-GPU bf16 dense peak FLOPS. Returns None if GPU not recognized."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0).upper()
    for key, tflops in _GPU_PEAK_TFLOPS_BF16.items():
        if key in name:
            return tflops * 1e12
    logger.warning("Unknown GPU {!r} — MFU disabled. Add entry to _GPU_PEAK_TFLOPS_BF16.", name)
    return None


class MFUMonitor:
    """Lightweight MFU tracker using CUDA events.

    Only ``flush()`` calls ``synchronize()``; ``step()`` is purely async.
    Call ``flush()`` only at log time to avoid any pipeline stalls.
    """

    def __init__(self, flops_per_step: int, gpu_peak_flops: float):
        self.flops_per_step = flops_per_step
        self.gpu_peak_flops = gpu_peak_flops
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._steps = 0
        self._start.record()

    def step(self):
        """Record one optimizer step (async, no sync)."""
        self._steps += 1

    def flush(self) -> float | None:
        """Return MFU since last flush. Syncs GPU — call only at log time."""
        if self._steps == 0:
            return None
        self._end.record()
        self._end.synchronize()
        elapsed_s = self._start.elapsed_time(self._end) / 1000.0
        total_flops = self.flops_per_step * self._steps
        mfu = total_flops / (elapsed_s * self.gpu_peak_flops) if elapsed_s > 0 else 0.0
        self._steps = 0
        self._start.record()
        return mfu
