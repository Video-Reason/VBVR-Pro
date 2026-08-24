"""Configuration for the unified ODE / SDE / CPS inference runner.

A single :class:`InferenceConfig` fully describes one inference run: which
checkpoint to load, where the input comes from (a precomputed latent sample or
a raw image + prompt), how to sample (ODE / SDE / CPS, number of steps), and
what to write out.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

SamplingMode = Literal["ode", "sde", "cps"]

# mode -> the ``sde_formula`` passed to ``WanI2VForTraining.sde_generate``.
# ODE is the deterministic limit of DanceGRPO (eta forced to 0 -> rectified-flow Euler).
_MODE_FORMULA: dict[str, str] = {
    "ode": "dancegrpo",
    "sde": "dancegrpo",
    "cps": "flowcps",
}

# mode -> default exploration noise when ``--noise_scale`` is not provided.
_MODE_DEFAULT_NOISE: dict[str, float] = {
    "sde": 0.3,  # DanceGRPO eta (matches grpo_sde_noise_scale default)
    "cps": 0.7,  # flow-CPS noise_level, must stay in [0, 1]
}


class InferenceConfig(BaseModel):
    """Fully-resolved settings for one inference run."""

    # ``model_path`` lives in the ``model_`` namespace Pydantic protects by default.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # ---- model / checkpoint ----
    model_path: str
    checkpoint: str | None = None  # DCP dir (flat or high/low); None runs the base model
    use_ema: bool = False
    transformer_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    device: str = "cuda:0"

    # ---- input source (exactly one of latent / image) ----
    latent_webdataset_dir: str | None = None
    sample_index: int = 0
    image: str | None = None
    prompt: str | None = None
    height: int = 384
    width: int = 384
    num_frames: int = 161

    # ---- sampling ----
    mode: SamplingMode = "sde"
    num_sampling_steps: int = 50
    noise_scale: float | None = None  # eta (sde) / noise_level (cps); None -> mode default
    cfg_scale: float = 1.0
    sigma_min: float = 0.0  # flowgrpo-only knob; inert for dancegrpo/flowcps
    sigma_max: float = 1.0
    seed: int = 42
    batch_size: int = 1
    share_init_noise: bool = True

    # ---- output ----
    output_dir: str
    fps: int = 16
    save_steps: bool = True  # decode the per-step z0 preview gallery
    save_reference: bool = True
    grid_cols: int = 6
    grid_thumb_width: int = 208
    force: bool = False

    # ------------------------------------------------------------------
    # Derived sampling parameters
    # ------------------------------------------------------------------
    @property
    def sde_formula(self) -> str:
        return _MODE_FORMULA[self.mode]

    @property
    def effective_noise_scale(self) -> float:
        """Noise level actually fed to ``sde_generate`` for the chosen mode."""
        if self.mode == "ode":
            return 0.0
        if self.noise_scale is not None:
            return float(self.noise_scale)
        return _MODE_DEFAULT_NOISE[self.mode]

    @property
    def from_image(self) -> bool:
        """True when the input is a raw image + prompt (needs the text encoder)."""
        return self.image is not None

    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate(self) -> "InferenceConfig":
        has_latent = self.latent_webdataset_dir is not None
        has_image = self.image is not None
        if has_latent == has_image:
            raise ValueError(
                "Provide exactly one input source: --latent_webdataset_dir or "
                "--image (with --prompt), not both or neither."
            )
        if has_image and not self.prompt:
            raise ValueError("--image requires --prompt.")
        if self.mode == "ode" and self.noise_scale not in (None, 0.0):
            raise ValueError(f"mode=ode is deterministic; --noise_scale must be 0 or unset, got {self.noise_scale}.")
        if self.mode == "cps" and not (0.0 <= self.effective_noise_scale <= 1.0):
            raise ValueError(f"mode=cps requires noise_level in [0, 1], got {self.effective_noise_scale}.")
        if self.num_sampling_steps < 1:
            raise ValueError(f"num_sampling_steps must be >= 1, got {self.num_sampling_steps}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        return self

    # ------------------------------------------------------------------
    @classmethod
    def from_training_config(cls, train_cfg: Any, **overrides: Any) -> "InferenceConfig":
        """Build an :class:`InferenceConfig`, seeding shared fields from a training config.

        ``train_cfg`` is a loaded ``RLConfig`` / ``SFTConfig``. The model path,
        latent dataset dir, seed, CFG scale and sigma bounds are inherited from
        it; sampling choices (mode, steps, noise) keep the inference defaults.
        Any non-``None`` value in ``overrides`` (the CLI-provided args) wins.
        """
        seeded: dict[str, Any] = {
            "model_path": getattr(train_cfg, "model_path", None),
            "latent_webdataset_dir": getattr(train_cfg, "latent_webdataset_dir", None),
            "seed": getattr(train_cfg, "seed", None),
            "cfg_scale": getattr(train_cfg, "grpo_cfg_scale", None),
            "sigma_min": getattr(train_cfg, "grpo_sde_sigma_min", None),
            "sigma_max": getattr(train_cfg, "grpo_sde_sigma_max", None),
        }
        merged = {k: v for k, v in seeded.items() if v is not None}
        # An explicit --image override switches to the raw-image source, so drop the
        # latent dir seeded from the training config (the two sources are exclusive).
        if overrides.get("image"):
            merged.pop("latent_webdataset_dir", None)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**merged)
