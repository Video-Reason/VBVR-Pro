# Project Memory

## Maintenance

- Keep this file as durable project memory. Record stable workflow rules,
  contracts, and recurring pitfalls; do not add one-off command transcripts,
  temporary artifact paths, or dated performance anecdotes.
- Prefer Jujutsu (`jj`) for version-control workflows when available. Use Git
  when a requested operation or external service specifically requires Git
  semantics.
- Keep `README.md`, `docs/`, runnable `configs/`, launchers, and this file in
  sync when data, checkpoint, training, reward, or evaluation contracts change.
- Long-running tasks must be monitored to a final exit status. Verify no
  required background session or service remains before reporting completion.
- Run the suite as `.venv/bin/python -m pytest tests`. Do not use root-wide
  bare `pytest`; generated artifacts and external trees can disrupt discovery.
- Before pushing, run the repository-wide Ruff checks from the parent checkout
  root: `.venv/bin/ruff check --output-format=github .` and
  `.venv/bin/ruff format --check .`. Ruff configuration, its exact pin, and CI
  live in the parent repository rather than this subproject.
- Use repository-relative examples. Never commit credentials, private
  endpoints, scheduler settings, machine-specific paths, or generated model,
  data, video, checkpoint, W&B, log, and cache artifacts.
- Keep `docs/` limited to durable public guides for released interfaces. Do
  not commit dated experiment reports, presentation or email drafts, internal
  roadmaps, one-off generation plans, generated media, or private runbooks.

## Release Shape

- VBVR-RL is a research training and evaluation stack for Wan2.2 I2V/TI2V
  Diffusers models on VBVR-Pro. It supports SFT, DanceGRPO, Flow-CPS, raw and
  latent data, DCP, LoRA, rule rewards, and VLM rewards.
- Start with `docs/README.md`. Source layout: `src/cli/` entrypoints,
  `src/models/` model wrapper/LoRA, `src/data/` loaders, `src/trainer/`
  training and rewards, `src/precompute/` builders, `src/eval/`
  scoring/provenance, `scripts/` launchers, `configs/` references, and
  `tests/` contracts.
- The release does not bundle model weights, datasets, evaluator source, OCR
  weights, or generated outputs. Store all such artifacts beneath ignored
  paths such as `storage/`, `wandb/`, `logs/`, and `tmp/`.
- Keep the checked-in DanceGRPO surface to exactly four production references:
  `configs/train_rl_5b_cps.yaml`, `configs/train_rl_5b_sde.yaml`,
  `configs/train_rl_5b_vlm.yaml`, and `configs/train_rl_a14b_rule.yaml`. The
  first two are the paired TI2V-5B rule-reward Flow-CPS/RF-SDE references. Derive
  bounded smoke parameters in the validator instead of adding
  experiment-specific RL YAMLs.
- Keep `configs/train_sft_vbvr_5e-6.yaml` as the only checked-in
  `train_sft_*.yaml` reference. Raw precompute descriptors may remain under a
  `precompute_*.yaml` name and are not additional SFT launch configs.
- Keep `scripts/train/` to exactly `sft_multinode.fish`,
  `grpo_multinode.fish`, and `grpo_vlm_eval_multinode.fish`. They serve local
  and multi-node runs; express topology- or experiment-specific choices in
  reviewed configs or explicit overrides instead of adding wrapper scripts.
- `third_party/VBVR-EvalKit` is intentionally absent. The supported public
  rule-evaluation path is `scripts/eval/vbvr_pro/` plus helpers in `src/eval/`.
  Do not restore a vendored or implicit evaluator fallback.
- Keep the public VBVR-Pro launcher surface parameterized: `run.fish` owns one
  UniPC/Euler/Flow-CPS cell, `sweep.fish` expands sampler cells,
  `reproduce.fish` owns the pinned two-model/six-sampler published matrix, and
  `summarize.fish` verifies provenance. Do not add further checkpoint-, model-,
  cluster-, resolution-, or sampler-specific wrappers or checked-in viewers.

## Environment

- The locked environment uses Python 3.12 and official PyTorch 2.11 CUDA 12.6
  wheels. uv is the sole supported Python environment, dependency, and
  lockfile manager. Use direct `.venv/bin/python`, `.venv/bin/torchrun`, and
  `.venv/bin/ruff` commands in operator instructions.
- Reproduce the default environment with `uv sync --frozen` and verify
  manifest/lock consistency with `uv lock --check`. `decord2` still imports
  as `decord`; the media stack uses headless OpenCV and a bundled
  FFmpeg/ffprobe fallback.
- `opencv-python` and `opencv-python-headless` own the same `cv2` files. If a
  headless environment loads the GUI payload and fails on `libGL.so.1`,
  reinstall the exact headless wheel last with `uv pip install --python
  .venv/bin/python --reinstall --no-deps
  opencv-python-headless==4.13.0.92`, then rerun
  `.venv/bin/python -m src.eval.vbvr_runtime`.
- Fresh Triton/Inductor caches require a host C compiler and matching Python
  headers. Prefer system Python 3.12 development headers;
  `scripts/dev/bootstrap_triton_python_headers.fish` can create an ignored uv
  toolchain fallback. Multi-machine GRPO preflights Triton's driver on every
  machine before loading the model.
- The optional Qwen judge uses an isolated uv-managed environment beneath
  `storage/host_vllm`, resolved by `requirements/vllm.lock`; do not mix its
  dependency stack into the training `.venv` or introduce another manager.

## Data Contracts

- Raw I2V data is configured by a JSON list of Parquet descriptors, not a flat
  sample list. Rows use `video` for one target, plus `prompt` and optional
  `image`. Older `videos` lists are accepted only as a compatibility input and
  only their final entry is used. Missing `image` falls back to the target's
  first frame.
- `I2VDataset` currently samples `num_frames` uniformly over the complete
  decoded frame range. Its top-level `fps` is metadata and does not perform
  physical-time resampling. Reward/evaluation FPS fields independently define
  generated media contracts.
- Latent WebDataset uses `shard-*.tar` files with `{key}.safetensors` and
  `{key}.json`. Required tensors are `prompt_embeds`, `condition`, and one
  target named `latents`. Numbered multi-target latent keys are rejected;
  rebuild old shards before use. Extra tensors are passed through to rewards.
- Always set the exact `dataset_size` for latent/iterable configs. It controls
  rank-local epoch lengths and scheduling; mistakes can cause uneven-rank
  endings and collective hangs.
- A14B and TI2V-5B use different condition/timestep formats. Recompute latents
  with the same base family and preprocessing contract used by training.
- When raw production data is unavailable,
  `scripts/dev/create_i2v_smoke_dataset.py` creates a deterministic ignored
  H.264/PNG/Parquet fixture. Derive the matching bounded single-GPU profile
  from `configs/train_rl_5b_cps.yaml` with
  `scripts/dev/validate_grpo_parameter_update.py --one-gpu-smoke`; do not add a
  separate checked-in smoke YAML.

## Public VBVR-Pro RL Snapshot

- The official public source is `Video-Reason/VBVR-Pro-RL`. Release commands
  pin revision `ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1`, which contains 50
  image and 50 video task archives for 50,000 RL samples.
- Each `VBVR-Pro-RL-Video` archive already contains `first_frame.png`,
  `metadata.json`, `video/final_frame.png`, `video/ground_truth.mp4`, and
  `video/prompt.txt`; downloading the image archives is unnecessary for I2V
  training. The archives are raw assets, not latent WebDataset tensors.
- Restore the video archives with `scripts/data/vbvr_pro_unpack_hf.py`. The
  resumable command validates safe member layout and complete five-field
  samples, then writes `materialized/dataset.json`, a split manifest, and
  source provenance. `--verify-existing` byte-compares restored files.
- The raw materialized manifest is task-grouped. Shared-prompt RL applies a
  deterministic sampler permutation; inspect both sampler shuffle and
  `shuffle_raw_indices` before assuming input order.

## Model and Trainer Structure

- `WanI2VForTraining` reads A14B `boundary_ratio` and scheduler `flow_shift`,
  builds the shifted timestep schedule, and routes high/low experts. TI2V-5B
  is a dense single-transformer model with expanded token timesteps.
- `src.cli.train_i2v` uses `SFTConfig` and `I2VTrainer`;
  `src.cli.train_grpo` uses `RLConfig` and `DanceGRPOTrainer`.
- Keep the DanceGRPO implementation separated by responsibility:
  `dancegrpo_trainer.py` owns the algorithm core and standard replay,
  `dancegrpo_shared_prompt.py` owns shared-prompt/delayed replay,
  `dancegrpo_split_runtime.py` owns split train/rollout orchestration, and
  `dancegrpo_common.py` owns their shared records and partition helpers. Avoid
  growing topology-specific orchestration back into the coordinator.
- SFT and RL have separate `BaseTrainer` and `BaseRLTrainer` stacks with some
  duplicated distributed, model, dataset, optimizer, and checkpoint logic.
  Shared fixes often need mirrored implementation and tests.
- With `transformer_load_dtype: auto`, full fine-tuning loads trainable
  transformers in fp32; LoRA keeps frozen bases in bf16. Bf16 checkpoint
  recompute uses ambient autocast around non-reentrant `checkpoint()`; do not
  restore a nested autocast `context_fn`, which is not compile-compatible.
- `disable_cudnn_sdp` defaults true because cuDNN SDPA backward can produce
  non-finite values for low-noise Wan training. Attention backend changes
  require target-shape forward/backward validation.
- Liger replacements are limited to compatible Wan Q/K RMSNorm sites. Do not
  substitute Wan FFN, explicit-fp32 LayerNorm, or custom 3D RoPE kernels
  without equivalence tests.

## Distributed Invariants

- FSDP2 ranks must execute compatible wrapped-module forward sequences. Keep
  timestep sampling and RL replay selection rank-synchronized; high/low routing
  divergence can hang collectives.
- Expert parallel requires FSDP, both A14B experts, an even world size, and
  synchronized control flow. `duplicate` gives expert groups the same data;
  `split` shards data across all ranks. DanceGRPO rejects expert parallel.
- HSDP shards within a machine and replicates across machines; on one machine
  it falls back to plain FSDP.
- RL tensor parallel is applied before FSDP, is currently A14B/one-machine
  only, and rejects LoRA, HSDP, expert parallel, split RL, and trainable text
  encoders. TP ranks share sampler/RNG/reward inputs; pixel rewards execute on
  TP rank zero and broadcast.
- Compile wrapped modules in place to preserve DCP keys. The collective-aware
  TP Q/K norm stays an eager Dynamo boundary because its backward collective
  must remain explicit.

## DanceGRPO and Flow-CPS

- DanceGRPO requires `grpo_group_size >= 2`,
  `grpo_num_sampling_steps >= 2`, and a valid stochastic coefficient. It does
  not support expert parallel.
- Flow-CPS may use a fixed `grpo_sde_noise_scale` or
  `grpo_cps_noise_scale_range`. A random coefficient is sampled once per
  prompt group/step, shared by all `G` rollouts, and stored for replay.
- Shared-prompt mode makes `batch_size` the global prompt count. Without waves
  it must divide the data-parallel world. A prompt-wave size must divide both
  `batch_size` and the data-parallel world; ranks per prompt must divide `G`.
- Prompt waves prepare every wave before replay, allowing reward/replay overlap
  without making trajectories stale. `grpo_delayed_replay` is a separate
  experimental one-update-stale pipeline and requires shared-prompt mode.
- Current standard/shared-prompt replay reuses rollout chunks;
  `grpo_train_sample_batch_size` does not always independently rechunk stored
  trajectories. Reduce `grpo_sample_batch_size` too when lowering replay
  memory.
- Full-finetune FSDP replay can retain full unsharded gradients while sync is
  disabled. `grpo_fsdp_sync_each_backward: true` bounds this memory at the cost
  of more collectives.
- Split RL uses the first configured ranks for training and the remainder for
  rollout/reward actors. LoRA sync requires LoRA; full sync requires async
  rollout; `none` is only for controlled stale-actor tests.
- For stochastic `neg_loss` smokes, evaluate multiple group members together.
  Reward calls preserve/restore RNG, so repeated one-sample calls can receive
  identical values and yield zero advantages.
- The release-derived TI2V-5B single-GPU smoke completed on an H800 with
  512x512x81, G=2, T=2, Flow-CPS, LoRA-r16, and `neg_loss`: reward
  `-0.1609 +/- 0.0060`, grad norm `0.0003`, 240 changed tensors, and 22.6/23.0
  GiB allocated/reserved peak. This proves the bounded update path, not the
  production distributed topology or external reward quality.

## Checkpoints

- New checkpoints always use `high/` and `low/` expert directories when those
  experts exist. Model/EMA state uses DCP; optimizer and dataloader state are
  rank-local sidecars; LoRA runs also write PEFT adapter folders.
- Auto-resume from the latest checkpoint in `output_dir` is a true resume.
  Explicit `resume_from` with default `reset_dataloader` is weight-only
  initialization that resets optimizer, counters, RNG progression, and data
  position. Set the mode deliberately.
- Portable evaluation converts DCP to a validated Diffusers tree and records
  base model, checkpoint, EMA, LoRA merge, dtype, converter, and output-tree
  provenance.

## Rule Reward and Evaluation

- `vbvr_rule` requires both `vbvr_reward_evalkit_dir` and a 64-hex
  `vbvr_reward_evalkit_source_sha256`. Offline scoring likewise requires
  `--evalkit_dir` and `--expected_evalkit_source_sha256`.
- The fingerprint covers evaluator entrypoint, evaluator Python files,
  annotations, and requirements. Any change defines a new reward/metric
  contract and output namespace.
- The recorded `main_v2` workflow uses a compatible fork/revision, not an
  assumed public-upstream default. External source/licensing must remain
  separate from this repository.
- Scorer workers change into the evaluator checkout. Resolve model, GT,
  metadata, and temporary paths before process submission; unresolved
  repository-relative paths can silently score as zero after `chdir`.
- Online and offline rule paths share `src.eval.vbvr_runtime`, which pins and
  probes scientific/media dependencies. Validate it before model loading and
  record its digest in score provenance.
- The supported end-to-end launcher is `scripts/eval/vbvr_pro/run.fish`; its
  internal pipeline lives under `scripts/eval/vbvr_pro/lib/`. It validates
  conversion, split manifest, generated paths/media, frame-preserving
  preparation, exact scorer source/runtime, sample errors, and stage
  provenance.
- Published Hugging Face reproduction uses `reproduce.fish` and immutable local
  snapshots. Keep both release commit IDs and the reviewed `pipeline.py` digest
  pinned together with the 500-sample, 512x512x81/16-FPS, 30-step, CFG-1.0,
  seed-zero, six-sampler contract; do not execute mutable remote pipeline code.
- Rule evaluation may reuse a complete score only when both the exact sample
  set and score provenance still validate. Partial or mismatched scoring state
  is discarded and recomputed; never merge arbitrary partial evaluator output.

## VLM Reward

- `vbvr_vlm` defaults to 100 task-specific evaluator-derived prompts in
  `src/trainer/rewards/vbvr_vlm_eval_prompts.py`, source digest
  `4d3159232590bd4b99266c9e82df445a3a54ada50a7af30051cf505057574202`.
- Task-specific mode sends the input first frame and one complete in-memory
  H.264 rollout, never the ground-truth final frame. vLLM uniformly selects the
  configured frame count once; the HF processor's second sampling pass is
  disabled.
- Every task has a separate rubric schema. Validate fields, score ranges,
  reasons, and the 100-point arithmetic sum; constrained decoding alone cannot
  enforce cross-field arithmetic.
- `vlm_reward_image_max_edge` is downscale-only. Fail-open scoring prevents one
  service/schema error from terminating a distributed job but must be
  monitored because repeated fallback zeros bias advantages.
- The optional Qwen service uses an ignored isolated vLLM environment. Keep
  model revision, vLLM version, prompt digest, frame sampling, retry/error
  policy, and raw per-sample results in provenance.
