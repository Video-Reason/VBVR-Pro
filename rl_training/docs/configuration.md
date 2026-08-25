# Configuration Reference

Training configuration is defined by Pydantic models in
[`src/trainer/config.py`](../src/trainer/config.py). Checked-in YAML files are
reference experiments, not automatic hardware profiles. Copy a close config,
give it a new output namespace, and review every path and topology field.

Some launcher environment variables retain the historical `WAN_TRAINER_*`
prefix for compatibility. They are public runtime controls, not references to
a separate checkout or machine profile.

## Precedence and CLI Overrides

The training entrypoints build configuration in this order:

1. model defaults;
2. values from `--config` YAML or JSON;
3. non-empty CLI overrides.

For example:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=1 \
  -m src.cli.train_grpo \
  --config configs/example.yaml \
  --max_steps 1 \
  --no-auto_resume
```

Booleans use `--field` and `--no-field`. Simple scalar overrides are the most
portable. Put lists, tuples, and structured values in YAML because the generic
CLI does not parse every composite type.

## Config Classes

| Entry point | Config class | Purpose |
| --- | --- | --- |
| `src.cli.train_i2v` | `SFTConfig` | Standard flow-matching SFT |
| `src.cli.train_grpo` | `RLConfig` | DanceGRPO rollout and replay |

`SFTConfig.trainer` is `i2v`; `RLConfig.trainer` is `dancegrpo`.

## Paths and Artifact Isolation

Use repository-relative paths under `storage/` for external and generated
artifacts:

```yaml
model_path: storage/models/Wan2.2-TI2V-5B-Diffusers
dataset_json: storage/datasets/example/dataset.json
output_dir: storage/checkpoints/my-experiment
vbvr_reward_tmp_dir: storage/tmp/my-experiment/vbvr_rule
```

A new experiment should have a new `output_dir`, W&B run name, temporary
directory, and evaluation output root. Reusing an output directory can trigger
`auto_resume` or mix artifacts from incompatible configurations.

## Data Selection

### Raw media

Set `dataset_json` to a descriptor containing one or more Parquet sources.
Rows contain `prompt`, optional `image`, and either `video` or ordered
`videos`. Training-time `height`, `width`, `max_area`, `num_frames`, and `fps`
override descriptor values when set.

The current raw video loader samples `num_frames` uniformly across the decoded
frame range. Its `fps` value is metadata; it does not resample the source by
physical time. Reward and evaluation FPS fields are separate media contracts.

`shuffle_raw_indices` applies a deterministic one-time permutation to the raw
index. Shared-prompt RL has its own deterministic sampler shuffle, so inspect
both settings before assuming input order.

### Latent WebDataset

Set `latent_webdataset_dir` to a directory of `shard-*.tar` files. Each sample
must contain `{key}.safetensors` and `{key}.json`; required tensors are
`prompt_embeds`, `condition`, and one target tensor named `latents`.

Always set the exact `dataset_size`. It determines rank-local epoch lengths and
the learning-rate schedule. Latents and conditions must be created with the
same Wan model family and preprocessing contract used by training.

Do not point `latent_webdataset_dir` at the published VBVR-Pro raw tar
snapshot. Materialize it and use `dataset_json`, or build a compatible latent
dataset first.

## Optimizer and Precision

Important base fields include:

| Field | Meaning |
| --- | --- |
| `batch_size` | Per-rank batch for standard SFT/RL; global prompts in shared-prompt RL |
| `gradient_accumulation_steps` | Micro-steps per optimizer update outside specialized RL replay |
| `learning_rate`, `warmup_steps` | Optimizer schedule |
| `optimizer` | `adamw` or `muon` |
| `lora_rank` | Positive enables LoRA; zero performs full fine-tuning |
| `param_dtype`, `reduce_dtype` | Parameter compute and collective dtypes |
| `transformer_load_dtype` | `auto`, `bfloat16`, or `float32` base load |
| `gradient_checkpointing` | Activation recomputation |
| `torch_compile` | Compile trainable transformer modules |
| `attention_backend` | Explicit Diffusers attention implementation |

With `transformer_load_dtype: auto`, full fine-tuning loads trainable
transformers in fp32 while LoRA keeps frozen base weights in bf16. Review
optimizer memory before changing this behavior.

## Distributed Topology

### FSDP and HSDP

`fsdp: true` enables FSDP2 sharding. `hsdp: true` shards within each machine
and replicates across machines; on one machine it falls back to plain FSDP.
All ranks must execute compatible wrapped-module forward sequences.

`expert_parallel: true` splits A14B high- and low-noise experts into separate
groups. It requires FSDP, both experts, an even world size, and synchronized
routing. It is supported by SFT, but not DanceGRPO.

### Tensor parallel RL

`tensor_parallel_size: 2` shards A14B attention and feed-forward projections
inside each pair, then composes those pairs with FSDP data replicas. This path
is RL-only and currently rejects LoRA, HSDP, expert parallel, split RL, and a
trainable text encoder. The global world size must be divisible by the tensor
parallel size.

### Split rollout actors

`rl_train_node_count` or `rl_train_rank_count` assigns the first ranks to
training and the remaining ranks to rollout/reward work. Actor synchronization
choices are:

- `lora`: transmit LoRA tensors and require `lora_rank > 0`;
- `full`: transmit trainable policy weights and require async rollout;
- `none`: keep actor initialization fixed, useful only for controlled tests.

`rl_async_rollout` adds a bounded future-step queue. It does not change which
prompts belong to an optimizer step. Read
[Async Rollout Design](dancegrpo_async_rollout_design.md) before changing its
queue or synchronization interval.

## DanceGRPO Semantics

Core fields:

```yaml
grpo_group_size: 32
grpo_num_sampling_steps: 30
grpo_sample_batch_size: 8
grpo_train_sample_batch_size: 8
grpo_clip_range: 0.001
grpo_kl_coeff: 0.004
dancegrpo_share_group_init_noise: true
dancegrpo_timestep_selection_ratio: 0.6
```

`grpo_group_size` is the number of rollouts, `G`, for one prompt.
`grpo_num_sampling_steps` is the rollout step count, `T`. The trainer selects a
subset of replay timesteps according to `dancegrpo_timestep_selection_ratio`.

`grpo_sample_batch_size` controls rollout chunking. On the current standard
and shared-prompt paths, stored trajectories preserve those chunks during
replay; `grpo_train_sample_batch_size` does not always independently split
them. Lower the rollout chunk too when reducing replay memory.

Full fine-tuning may need `grpo_fsdp_sync_each_backward: true` so each replay
backward reduce-scatters gradients instead of retaining full unsharded
gradients across chunks.

## Shared-Prompt Batches and Waves

With `grpo_shared_prompt_batch: true`, every data-parallel rank reads the same
global prompt batch and ranks divide each prompt's `G` rollouts. In this mode:

- `batch_size` is the global number of prompts per optimizer step;
- `batch_size` must divide the data-parallel world unless prompt waves are
  enabled;
- each prompt's participating rank count must divide `grpo_group_size`.

`grpo_shared_prompt_microbatch_size` divides the optimizer step into prompt
waves. It must be positive, no larger than `batch_size`, and divide both the
global prompt batch and the data-parallel world. All waves are prepared before
replay begins.

`grpo_delayed_replay` is an experimental one-update-stale pipeline and requires
shared-prompt mode. Checkpoint and final boundaries flush its pending slot.

## Flow-CPS

Select Flow-CPS with:

```yaml
grpo_sde_formula: flowcps
grpo_sde_noise_scale: 0.7
```

For randomized coefficients:

```yaml
grpo_sde_formula: flowcps
grpo_cps_noise_scale_range: [0.1, 0.9]
```

The random value is sampled once per prompt group and optimizer step. All `G`
rollouts share it, and replay uses the coefficient stored with the trajectory.
A fixed Flow-CPS coefficient must be in `(0, 1]`; a range must satisfy
`0 <= min < max <= 1`.

## Rewards

Reward implementations are registered in
[`src/trainer/rewards`](../src/trainer/rewards). Public names include
`neg_loss`, `maze`, `maze_line`, `maze_tracker`, `vbvr_rule`, and `vbvr_vlm`.

`vbvr_rule` requires both an external checkout and an exact source digest:

```yaml
grpo_reward_fn: vbvr_rule
vbvr_reward_evalkit_dir: storage/evalkits/<checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-digest>
vbvr_reward_fps: 16
vbvr_reward_cpu_workers: 1
vbvr_reward_cpu_threads_per_worker: 1
```

The reward decodes videos, applies the final-evaluation media preparation
contract, and invokes CPU scorer workers. Use a unique
`vbvr_reward_tmp_dir`. `vbvr_reward_fail_on_error: false` converts an affected
sample to `vbvr_reward_unsupported_score`; monitor warnings because repeated
fallbacks can flatten or bias group advantages.

`vbvr_vlm` uses an OpenAI-compatible multimodal endpoint. Its task-specific
mode requires the input first frame, 32 uniformly selected generated-video
frames by default, a 100-point rubric, and a validated per-task output schema.
See [Qwen VLM Reward](vlm_judge_reward.md).

Rewards that declare `requires_vae = True` force VAE loading even when training
from precomputed latents.

## Checkpoint and Resume Fields

```yaml
output_dir: storage/checkpoints/my-experiment
save_steps: 100
auto_resume: true
resume_from: null
reset_dataloader: null
```

Auto-resume from the latest checkpoint inside `output_dir` restores optimizer,
dataloader, counters, and RNG. An explicit `resume_from` with
`reset_dataloader: null` defaults to weight-only initialization. Set
`reset_dataloader: false` for a true explicit resume and `true` for intentional
initialization. Never rely on the implicit default when the distinction
matters.

## Reference Config Selection

The release intentionally ships one SFT config and four RL configs:

| Config | Intended use |
| --- | --- |
| `train_sft_vbvr_5e-6.yaml` | A14B latent SFT with expert parallelism |
| `train_rl_5b_cps.yaml` | TI2V-5B, 512x512x81, HSDP, rule reward, Flow-CPS eta 0.7 |
| `train_rl_5b_sde.yaml` | Paired TI2V-5B rule-reward run using DanceGRPO RF-SDE eta 0.3 |
| `train_rl_5b_vlm.yaml` | TI2V-5B, 512x512x81, HSDP, co-hosted Qwen VLM reward |
| `train_rl_a14b_rule.yaml` | A14B, 256x256x161, TP2 plus FSDP4, rule reward |

`train_sft_vbvr_5e-6.yaml` expects an external 800,000-sample latent
WebDataset at `data/vbvr/latents/sft`; it cannot consume the public raw RL
archives directly. Its expert-parallel topology requires FSDP, both A14B
experts, and an even world size.

The single-GPU update check derives temporary LoRA/`neg_loss` overrides from
`train_rl_5b_cps.yaml`; it is not a fifth training config. None of these
files guarantees that the external paths referenced by `model_path`,
`dataset_json`, `resume_from`, or evaluator settings exist in a fresh checkout.

## Preflight Checklist

Before a long run:

1. validate YAML by constructing its matching config class;
2. run `src.cli.validate_grpo_runtime` for RL;
3. run a one-step update at the target model family and media shape;
4. confirm reward values, gradients, and changed tensors are nonzero when the
   chosen objective should produce an update;
5. verify save and resume in the intended topology;
6. record the Git revision, config, model source, dataset manifest, evaluator
   fingerprint, and runtime lockfile.
