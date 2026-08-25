# Training

VBVR-RL provides two training modes for Wan2.2 image-to-video models: standard
supervised flow matching and DanceGRPO-style reinforcement learning. They
share model, data, distributed, and checkpoint infrastructure while using
different objectives and rollout lifecycles.

Start with [Getting Started](getting_started.md), then review
[Configuration](configuration.md) and the selected YAML in full. Reference
configs preserve research choices; they do not discover available hardware or
external artifacts automatically.

## Training Modes

| Mode | Entry point | Config selector | Main use |
| --- | --- | --- | --- |
| SFT | `src.cli.train_i2v` | `trainer: i2v` | Standard flow-matching supervision |
| DanceGRPO | `src.cli.train_grpo` | `trainer: dancegrpo` | Grouped on-policy rollout and clipped replay |

All commands below run from the repository root through the locked uv
environment or a launcher that activates `.venv`.

## Model Families

`WanI2VForTraining` wraps two supported Diffusers families:

- Wan2.2 I2V A14B has high- and low-noise transformer experts. The wrapper
  reads `boundary_ratio` from `model_index.json` and routes timesteps to the
  matching expert.
- Wan2.2 TI2V-5B is a dense single-transformer model. It uses expanded
  per-token timesteps and a different condition layout.

Latents and first-frame conditions are model-family-specific. Do not reuse
A14B precompute outputs with TI2V-5B or the reverse. A converted or fine-tuned
model must remain a valid Diffusers directory with the scheduler, VAE, text
encoder, tokenizer, and transformer metadata expected by the wrapper.

## Data Paths

### Raw Parquet input

`dataset_json` identifies one or more Parquet sources. A row contains a
`prompt`, an optional `image`, and one target `video`. Older manifests may use
a `videos` list; the loader uses only its final entry. Raw training loads
media, then runs T5 and VAE encoding online.

Use raw input when source media is convenient, augmentations or dimensions may
change, or a pixel-domain reward needs source files and metadata. It has higher
CPU, filesystem, and GPU inference cost than precomputed input.

### Latent WebDataset input

`latent_webdataset_dir` points to `shard-*.tar` files containing
`prompt_embeds`, `condition`, and target latents. This removes repeated T5/VAE
encoding from the training step. Set `dataset_size` exactly so all ranks agree
on epoch length and scheduling.

Pixel-domain rewards still load the VAE to decode generated rollouts. See
[Data and Precompute](data.md) for schemas and conversion commands.

## Common Training Lifecycle

Every trainer follows the same high-level sequence:

1. validate the Pydantic config and initialize distributed process groups;
2. load the Diffusers pipeline and select trainable modules;
3. install LoRA, tensor parallelism, and/or FSDP as configured;
4. create the dataset, sampler, and stateful dataloader;
5. load an initialization or resume checkpoint;
6. run optimizer steps, logging, EMA, and periodic checkpoint saves;
7. flush pending work and optionally save a final checkpoint.

All distributed ranks must enter compatible collective operations. A Python
exception on one rank can present as a timeout or NCCL failure elsewhere, so
inspect the first failing rank rather than only the final aggregate error.

## Supervised Flow Matching

Standard SFT samples a noise level, interpolates between Gaussian noise and
the target latent, predicts the flow velocity, and minimizes mean-squared
error against the target velocity.

A typical latent config includes:

```yaml
trainer: i2v
model_path: storage/models/Wan2.2-TI2V-5B-Diffusers
latent_webdataset_dir: storage/latents/example/webdataset/sft
dataset_size: 50000
output_dir: storage/checkpoints/example-sft
batch_size: 1
learning_rate: 1.0e-5
gradient_checkpointing: true
fsdp: true
lora_rank: 0
```

Launch on one machine:

```fish
fish scripts/train/sft_multinode.fish --nproc 8 -- \
  --config configs/train_sft_vbvr_5e-6.yaml
```

This release keeps one SFT reference. It targets A14B, expects 800,000
precomputed latent samples at `data/vbvr/latents/sft`, and uses FSDP expert
parallelism, so launch it with an even distributed world size. The latent
dataset is external and is not generated from the public raw RL archives by
the release setup steps.

`lora_rank: 0` performs full fine-tuning. A positive rank inserts adapters and
keeps base transformer weights frozen. When changing between full tuning and
LoRA, review `transformer_load_dtype`, optimizer memory, checkpoint loading,
and learning rate rather than changing only `lora_rank`.

## DanceGRPO Overview

DanceGRPO generates `G` rollouts for each prompt, scores them, normalizes
rewards within the prompt group, and replays selected denoising transitions
through a clipped policy objective. It can optionally add a reference-policy
penalty.

Minimum validity requirements include:

- `grpo_group_size >= 2`;
- `grpo_num_sampling_steps >= 2`;
- a valid stochastic coefficient for the chosen SDE/CPS formula;
- synchronized timestep selection across FSDP ranks;
- a reward that can distinguish at least some members of a group.

The core rollout dimensions are:

- `B`: prompts per optimizer step;
- `G`: rollouts per prompt;
- `T`: denoising steps per rollout;
- `R`: replayed timesteps, derived from
  `dancegrpo_timestep_selection_ratio`.

Compute, media decoding, and reward traffic grow with `B * G`; rollout
transformer work grows again with `T`. Validate a smaller shape before scaling
these dimensions together.

## Sampling Formulas

`grpo_sde_formula` selects `flowgrpo`, `dancegrpo`, or `flowcps`.

For fixed Flow-CPS exploration:

```yaml
grpo_sde_formula: flowcps
grpo_sde_noise_scale: 0.7
```

For randomized Flow-CPS exploration:

```yaml
grpo_sde_formula: flowcps
grpo_cps_noise_scale_range: [0.1, 0.9]
```

One randomized coefficient is sampled per prompt group and optimizer step,
shared by all `G` rollouts, and stored with the trajectory for exact replay.
The deterministic `noise_level=0` boundary uses first-order Euler behavior
when the sigma grid and CFG are otherwise identical.

Flow-CPS replay treats the surrogate log probability as dimension-mean
negative squared error. The reference term uses mean squared error. Avoid
changing tensor reductions without updating both implementation tests and
the documented objective.

## Shared-Prompt Distribution

With `grpo_shared_prompt_batch: false`, ranks receive ordinary sampler shards
and each local prompt produces its own group.

With `grpo_shared_prompt_batch: true`, all data-parallel ranks read the same
global prompt batch and split each prompt's group rollouts across ranks.
`batch_size` then means global prompts per optimizer step.

Prompt waves reduce the number of simultaneously active prompts:

```yaml
batch_size: 32
grpo_shared_prompt_batch: true
grpo_shared_prompt_microbatch_size: 16
```

The wave size must divide `batch_size` and the data-parallel world. The ranks
assigned to one prompt must divide `G`. The trainer prepares every wave before
replay, which allows later CPU reward work to overlap earlier replay without
changing the current-step policy.

## Rollout and Replay Chunking

`grpo_sample_batch_size` caps the number of generated group members evaluated
together. The saved trajectory chunks are reused by the current standard and
shared-prompt replay paths. As a result, lowering only
`grpo_train_sample_batch_size` may not reduce replay memory; lower the rollout
chunk too.

For full fine-tuning under FSDP2, leaving gradient synchronization disabled
across replay chunks can retain full unsharded gradients. Use:

```yaml
grpo_fsdp_sync_each_backward: true
```

when bounded gradient memory is more important than reducing collective
frequency. Validate memory and gradient equivalence at the intended model
shape.

`grpo_offload_inference_models` moves the frozen text encoder and VAE to CPU
after they are used, freeing memory for replay and optimizer state. This saves
GPU memory at the cost of transfers between phases.

## Delayed and Async Replay

`grpo_delayed_replay` is an experimental shared-prompt pipeline that prepares
the next trajectory slot and replays the previous slot. It is intentionally
one optimizer update stale. Checkpoint and shutdown boundaries drain the slot
before saving.

Split async rollout is different: trainer ranks consume complete prompt steps
from actor ranks through a bounded queue. `rl_async_rollout_prefetch_steps: 0`
chooses a queue depth automatically. Actor weights are synchronized according
to `rl_actor_weight_sync` and its interval. See
[Async Rollout Design](dancegrpo_async_rollout_design.md) for invariants and
failure handling.

## Distributed Modes

### FSDP2 and HSDP

FSDP2 shards trainable modules. HSDP shards within a machine and replicates
across machines. All ranks in a wrapped group must follow the same high/low
expert routing and replay timestep sequence.

### Expert parallel SFT

A14B expert parallel assigns high- and low-noise experts to separate groups.
It requires `train_experts: both`, FSDP, and an even world size.
`expert_parallel_data_mode: duplicate` gives both expert groups the same data
stream; `split` shards data across all ranks for global throughput.

DanceGRPO rejects expert parallelism.

### Tensor parallel A14B RL

`tensor_parallel_size: 2` applies tensor parallelism before FSDP2, shards Wan
attention and feed-forward projections, and preserves global-across-head Q/K
RMSNorm through an autograd-aware collective. This path is RL-only and
currently requires one machine, FSDP, full fine-tuning, no HSDP, no expert
parallel, no split RL, and no trainable text encoder.

Compile wrapped modules in place so DCP keys remain stable. A fresh Inductor
or Triton cache requires a working host compiler and matching Python headers.

## Reward Functions

Rewards are registered in
[`src/trainer/rewards/__init__.py`](../src/trainer/rewards/__init__.py).

| Name | Input | Intended use |
| --- | --- | --- |
| `neg_loss` | Policy/model tensors | Bounded plumbing tests and differentiable proxy experiments |
| `maze` | Decoded frames plus maze tensors | Synthetic ball-trajectory score |
| `maze_line` | Decoded colored-path frames | Growing-line mask and goal score |
| `maze_tracker` | Decoded sequence plus maze tensors | Tracker-aligned trajectory score |
| `vbvr_rule` | Prepared MP4 plus GT metadata | External pinned VBVR-Pro rule evaluator |
| `vbvr_vlm` | First frame plus generated MP4 | External task-specific multimodal judge |

Rule reward requires an external evaluator checkout and source digest; see
[External EvalKit](external_evalkit.md). VLM reward requires an
OpenAI-compatible multimodal service; see
[Qwen VLM Reward](vlm_judge_reward.md).

Rewards that decode pixels force VAE availability even for latent input.
Fail-open reward modes protect a distributed job from one scorer failure but
can flatten or bias group advantages. Monitor failure counts and sample-level
errors, not only mean reward.

For stochastic `neg_loss` tests, score at least two members of a group in one
reward call. Reward evaluation preserves and restores the torch RNG; repeated
one-sample calls can otherwise receive identical random values.

## Launch Commands

Run an RL preflight without loading weights:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<reviewed-rl-config>.yaml
```

One machine:

```fish
fish scripts/train/grpo_multinode.fish --nproc 8 \
  --config configs/<reviewed-rl-config>.yaml
```

Multiple machines, on every machine:

```bash
MASTER_ADDR=<rank-zero-host> \
MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> \
RANK=<machine-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/<reviewed-rl-config>.yaml
```

The same launcher serves both cases. When `MASTER_ADDR`, `WORLD_SIZE`, and
`RANK` are all absent it defaults to one local machine; for a multi-machine
run, set all three. `WORLD_SIZE` is the machine count, and the total process
count is `WORLD_SIZE * --nproc`. The launcher validates the reward/attention
runtime and a Triton driver build on every machine before starting `torchrun`.

Use `WAN_TRAINER_TRITON_PREFLIGHT_ONLY=1` to perform only the per-machine
compiler preflight. Set `WAN_TRAINER_SKIP_TRITON_PREFLIGHT=1` only when the
runtime image is already proven and startup policy requires skipping it.

## Monitoring and Completion

A healthy run should expose:

- periodic step, loss, learning rate, and gradient norm logs;
- reward mean/std and, for grouped RL, non-flat advantages when expected;
- bounded rollout/replay queue sizes;
- checkpoint completion on all required expert directories;
- no pending reward workers or replay slots at shutdown;
- a zero process exit status on every rank.

For a bounded validation, use `max_steps`, keep `save_final_checkpoint: true`,
and verify changed tensors with
[`scripts/dev/validate_grpo_parameter_update.py`](../scripts/dev/validate_grpo_parameter_update.py).
Zero learning rate during warmup or equal group rewards can legitimately
produce no update, so configure a smoke that can demonstrate the property you
intend to test.

## Common Failure Modes

### Collective hang

Find the first rank that diverged. Common causes are unsynchronized A14B expert
routing, per-rank replay timestep selection, uneven iterable-dataset lengths,
or one reward worker raising before a barrier.

### CUDA out of memory during replay

Reduce `grpo_sample_batch_size`, enable synchronized FSDP backward, offload
frozen inference models, reduce `B`, `G`, `T`, resolution, or frame count, or
switch to LoRA. Changing `grpo_train_sample_batch_size` alone may not rechunk
stored trajectories.

### All-zero advantage

Inspect every member's raw reward. Small hard-rule groups can all fail the same
criterion. Also check scorer fallbacks, unsupported tasks, path resolution,
and stochastic reward chunking.

### Raw DataLoader timeout

Confirm source paths and codec readability. For slow remote or shared storage,
use bounded prefetch settings and `dataloader_item_trace_seconds` to dump
worker stacks before the timeout. Stale remote cache lock directories can look
like random item timeouts.

### Resume starts from step zero

An explicit `resume_from` with default reset behavior initializes weights but
resets optimizer, dataloader, counters, and RNG. Use a true resume only when
the checkpoint and topology are compatible. See [Checkpoints](checkpoints.md).

## Reproducibility Record

For every result intended for comparison or publication, retain:

- repository commit and uncommitted-diff status;
- complete resolved YAML and launcher environment overrides;
- base/fine-tuned model identity and conversion provenance;
- dataset descriptor and split-manifest hashes;
- world topology and random seed;
- sampler, sigma grid, CFG, frame, resolution, and FPS settings;
- reward implementation, evaluator source digest, and runtime digest;
- checkpoint identity and whether EMA or LoRA merge was used;
- exit status and final checkpoint/provenance validation.
