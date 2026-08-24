"""Unified ODE / SDE / CPS inference for any VBVR-RL checkpoint.

Loads a DCP checkpoint into ``WanI2VForTraining`` and runs one denoising
trajectory (ODE deterministic, SDE stochastic DanceGRPO, or flow-CPS),
capturing the per-step predicted-clean preview plus the final video — from
either a precomputed latent sample or a raw image + prompt.

Note: the ``WanI2VForTraining`` load path targets the 5B TI2V model. A14B
two-expert checkpoints load both ~14B experts on CPU first and can OOM under a
64GB cgroup; pre-convert those with ``src.cli.convert_dcp_to_diffusers`` and run
the diffusers pipeline instead.
"""

from .config import InferenceConfig, SamplingMode
from .engine import InferenceEngine, StepwiseResult, build_model
from .inputs import PreparedInput, prepare_input
from .outputs import write_outputs

__all__ = [
    "InferenceConfig",
    "SamplingMode",
    "InferenceEngine",
    "StepwiseResult",
    "build_model",
    "PreparedInput",
    "prepare_input",
    "write_outputs",
]


def run(cfg: "InferenceConfig") -> dict:
    """Build the model, sample, and render outputs for one resolved config."""
    from .cli import run as _run

    return _run(cfg)
