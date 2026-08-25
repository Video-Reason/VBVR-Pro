"""Tensor-parallel helpers for Diffusers Wan transformers.

The implementation follows the usual attention/MLP tensor-parallel layout:

* Q/K/V and the first FFN projection are column sharded.
* The attention output and second FFN projection are row sharded.
* Wan's Q/K RMSNorm keeps its original across-all-heads semantics by reducing
  the squared-norm statistic over the TP process group.

Tensor parallelism is applied before FSDP2.  FSDP2 then adds the data-parallel
sharding dimension to the already TP-sharded parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


@dataclass(frozen=True)
class WanTensorParallelStats:
    blocks: int
    attentions: int
    linears: int
    rms_norms: int
    liger_rms_norms: int


class TensorParallelRMSNorm(nn.Module):
    """RMSNorm over a hidden dimension sharded across a 1D TP mesh.

    Diffusers Wan uses one RMSNorm over all query/key heads rather than an
    independent norm per head.  A local RMSNorm would therefore change the
    model.  This module computes the local sum of squares and performs an
    autograd-aware all-reduce so both its forward and backward match the
    unsharded operation.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        eps: float,
        tp_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        if tp_mesh.ndim != 1:
            raise ValueError(f"TensorParallelRMSNorm requires a 1D TP mesh, got ndim={tp_mesh.ndim}")
        if weight.ndim != 1:
            raise ValueError(f"TensorParallelRMSNorm requires a 1D weight, got shape={tuple(weight.shape)}")

        self.normalized_shape = (int(weight.numel()),)
        self.eps = float(eps)
        self.tp_size = int(tp_mesh.size())
        if self.normalized_shape[0] % self.tp_size != 0:
            raise ValueError(
                "RMSNorm hidden size must be divisible by TP size, got "
                f"hidden={self.normalized_shape[0]}, tp={self.tp_size}"
            )
        self.local_hidden_size = self.normalized_shape[0] // self.tp_size
        self._tp_group = tp_mesh.get_group()

        mesh_device = torch.device(tp_mesh.device_type, torch.cuda.current_device())
        full_weight = weight.detach().to(device=mesh_device)
        sharded_weight = distribute_tensor(full_weight, tp_mesh, [Shard(0)])
        self.weight = nn.Parameter(sharded_weight, requires_grad=weight.requires_grad)

    # Dynamo currently traces through distributed.nn's autograd Function and
    # drops the matching all-reduce from its backward graph. Keep this small
    # collective-aware norm eager while compiling the surrounding Wan block.
    @torch.compiler.disable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.local_hidden_size:
            raise RuntimeError(
                "TensorParallelRMSNorm received an unexpected local hidden size: "
                f"expected={self.local_hidden_size}, got={hidden_states.shape[-1]}"
            )

        local_square_sum = hidden_states.float().square().sum(dim=-1, keepdim=True)
        global_square_sum = dist_nn.all_reduce(
            local_square_sum,
            op=dist.ReduceOp.SUM,
            group=self._tp_group,
        )
        inv_rms = torch.rsqrt(global_square_sum / self.normalized_shape[0] + self.eps)
        normalized = (hidden_states.float() * inv_rms).to(dtype=hidden_states.dtype)
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return normalized * weight


def _validate_tp_linear(module: nn.Module, name: str) -> nn.Linear:
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"Wan tensor parallelism currently requires an nn.Linear at {name}; "
            f"got {type(module).__name__}. LoRA/fused projection wrappers are not supported."
        )
    return module


def _replace_attention_norm(
    attention: nn.Module,
    norm_name: str,
    *,
    tp_mesh: DeviceMesh,
) -> bool:
    norm = getattr(attention, norm_name)
    is_liger = False
    if isinstance(norm, nn.RMSNorm):
        eps = float(norm.eps)
    else:
        # Liger is installed before the TP mesh exists. Its RMSNorm kernel
        # normalizes only the local hidden shard, which would change Wan's
        # global-across-heads Q/K semantics. Accept its module and preserve
        # the weight while installing the collective-aware implementation.
        try:
            from liger_kernel.transformers import LigerRMSNorm
        except ImportError:
            LigerRMSNorm = None
        if LigerRMSNorm is None or not isinstance(norm, LigerRMSNorm):
            raise TypeError(
                f"Wan tensor parallelism requires torch.nn.RMSNorm or LigerRMSNorm at {norm_name}; "
                f"got {type(norm).__name__}"
            )
        if float(norm.offset) != 0.0:
            raise ValueError(f"Wan TP requires zero-offset LigerRMSNorm at {norm_name}, got {norm.offset}")
        eps = float(norm.variance_epsilon)
        is_liger = True
    if norm.weight is None:
        raise ValueError(f"Wan tensor parallelism requires affine RMSNorm at {norm_name}")
    if norm.weight.ndim != 1:
        raise TypeError(
            f"Wan tensor parallelism requires a 1D RMSNorm weight at {norm_name}; got shape={tuple(norm.weight.shape)}"
        )
    setattr(
        attention,
        norm_name,
        TensorParallelRMSNorm(norm.weight, eps=eps, tp_mesh=tp_mesh),
    )
    return is_liger


def parallelize_wan_transformer(
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
) -> WanTensorParallelStats:
    """Apply 1D tensor parallelism to a Diffusers ``WanTransformer3DModel``.

    This function must run before FSDP2 wrapping.  It intentionally validates
    the concrete module topology so a future Diffusers architectural change
    fails loudly instead of silently leaving large projections replicated.
    """

    if tp_mesh.ndim != 1:
        raise ValueError(f"Wan tensor parallelism requires a 1D TP mesh, got ndim={tp_mesh.ndim}")
    tp_size = int(tp_mesh.size())
    if tp_size <= 1:
        return WanTensorParallelStats(blocks=0, attentions=0, linears=0, rms_norms=0, liger_rms_norms=0)

    previous_tp_size = getattr(transformer, "_wan_tensor_parallel_size", None)
    if previous_tp_size is not None:
        if int(previous_tp_size) != tp_size:
            raise ValueError(f"Transformer was already parallelized with TP={previous_tp_size}, requested TP={tp_size}")
        raise RuntimeError(f"Transformer was already parallelized with TP={tp_size}")

    blocks = getattr(transformer, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise TypeError("Wan tensor parallelism expected transformer.blocks to be a non-empty module list")

    plan: dict[str, ColwiseParallel | RowwiseParallel] = {}
    attention_count = 0
    norm_count = 0
    liger_norm_count = 0
    linear_count = 0

    for block_idx, block in enumerate(blocks):
        for attention_name in ("attn1", "attn2"):
            attention = getattr(block, attention_name, None)
            if attention is None:
                raise TypeError(f"Wan block {block_idx} is missing {attention_name}")
            if getattr(attention, "fused_projections", False):
                raise ValueError("Wan tensor parallelism does not support fused attention projections")

            heads = int(attention.heads)
            if heads % tp_size != 0:
                raise ValueError(
                    f"Wan attention heads must be divisible by TP size, got heads={heads}, tp={tp_size}, "
                    f"module=blocks.{block_idx}.{attention_name}"
                )

            projection_names = ["to_q", "to_k", "to_v"]
            if getattr(attention, "add_k_proj", None) is not None:
                projection_names.extend(["add_k_proj", "add_v_proj"])
            for projection_name in projection_names:
                fqn = f"blocks.{block_idx}.{attention_name}.{projection_name}"
                projection = _validate_tp_linear(getattr(attention, projection_name), fqn)
                if projection.out_features % tp_size != 0:
                    raise ValueError(
                        f"Column-parallel output size must divide TP size at {fqn}: "
                        f"out_features={projection.out_features}, tp={tp_size}"
                    )
                plan[fqn] = ColwiseParallel()
                linear_count += 1

            out_fqn = f"blocks.{block_idx}.{attention_name}.to_out.0"
            out_projection = _validate_tp_linear(attention.to_out[0], out_fqn)
            if out_projection.in_features % tp_size != 0:
                raise ValueError(
                    f"Row-parallel input size must divide TP size at {out_fqn}: "
                    f"in_features={out_projection.in_features}, tp={tp_size}"
                )
            plan[out_fqn] = RowwiseParallel()
            linear_count += 1

            liger_norm_count += int(_replace_attention_norm(attention, "norm_q", tp_mesh=tp_mesh))
            liger_norm_count += int(_replace_attention_norm(attention, "norm_k", tp_mesh=tp_mesh))
            norm_count += 2
            if getattr(attention, "add_k_proj", None) is not None:
                liger_norm_count += int(_replace_attention_norm(attention, "norm_added_k", tp_mesh=tp_mesh))
                norm_count += 1

            attention.heads = heads // tp_size
            attention_count += 1

        ffn_in_fqn = f"blocks.{block_idx}.ffn.net.0.proj"
        ffn_out_fqn = f"blocks.{block_idx}.ffn.net.2"
        ffn_in = _validate_tp_linear(block.ffn.net[0].proj, ffn_in_fqn)
        ffn_out = _validate_tp_linear(block.ffn.net[2], ffn_out_fqn)
        if ffn_in.out_features % tp_size != 0 or ffn_out.in_features % tp_size != 0:
            raise ValueError(
                f"Wan FFN hidden size must be divisible by TP size in block {block_idx}: "
                f"up={ffn_in.out_features}, down={ffn_out.in_features}, tp={tp_size}"
            )
        plan[ffn_in_fqn] = ColwiseParallel()
        plan[ffn_out_fqn] = RowwiseParallel()
        linear_count += 2

    parallelize_module(transformer, tp_mesh, plan)
    transformer._wan_tensor_parallel_size = tp_size
    return WanTensorParallelStats(
        blocks=len(blocks),
        attentions=attention_count,
        linears=linear_count,
        rms_norms=norm_count,
        liger_rms_norms=liger_norm_count,
    )
