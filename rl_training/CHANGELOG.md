# Changelog

All notable public changes to VBVR-RL are documented here. The project uses
[Semantic Versioning](https://semver.org/) for source releases.

## [Unreleased]

### Changed

- Standardized development, CI, training, and evaluation on the locked uv
  project environment. The optional vLLM service uses a separate uv-managed
  environment because its runtime dependency contract differs from training.

## [0.1.0] - 2026-08-19

Initial public source release.

### Included

- Wan2.2 TI2V-5B and I2V-A14B training with SFT and DanceGRPO-style
  reinforcement learning.
- Flow-CPS rollout and replay, LoRA/full fine-tuning, FSDP2/HSDP, expert
  parallelism, and RL tensor parallelism.
- Raw Parquet and latent WebDataset pipelines, plus a resumable materializer
  for the official 50,000-sample `Video-Reason/VBVR-Pro-RL` archives.
- Manifest-locked VBVR-Pro generation, media preparation, external rule
  scoring, VLM judging, and stage provenance.
- DCP checkpoint save/resume, Diffusers conversion, LoRA extraction, tests,
  launchers, and public operator documentation.
- A focused training configuration surface: one A14B SFT reference, plus
  TI2V-5B rule reward, TI2V-5B VLM reward, and A14B rule reward, with a
  code-derived one-GPU RL smoke profile.
- A parameterized VBVR-Pro evaluation surface for UniPC, Euler, and Flow-CPS,
  including deterministic sampler sweeps and provenance-driven summaries.

### Release boundaries

- VBVR-EvalKit is not vendored. Rule reward and evaluation require an explicit
  external checkout and exact source fingerprint.
- Model weights, datasets, OCR weights, generated media, checkpoints, and
  machine-specific infrastructure are not included.
- The old non-VBVR-Pro evaluation implementation and compatibility wrappers
  have been removed; `scripts/eval/vbvr_pro/` is the supported evaluation
  surface.

[0.1.0]: https://github.com/pufanyi/vbvr-rl/releases/tag/v0.1.0
