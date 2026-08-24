# Architecture

VBVR-RL is a distributed Wan2.2 training and VBVR-Pro evaluation stack. Its
primary control flow is:

```text
Fish launcher or Python CLI
  -> Pydantic configuration
  -> trainer and distributed topology
  -> WanI2VForTraining
  -> raw media or latent WebDataset
  -> objective and optional reward
  -> DCP checkpoint runtime
  -> provenance-checked evaluation
```

## Source Map

```text
src/cli/          user-facing training, generation, conversion, and evaluation CLIs
src/models/       Wan model wrapper and LoRA integration
src/data/         raw Parquet, latent WebDataset, and remote-I/O loaders
src/trainer/      configs, trainers, rewards, sharding, optimizers, checkpoints
src/precompute/   video/prompt latent builders and synthetic data generation
src/eval/         VBVR-Pro scorer adapters, VLM protocol, provenance, reports
configs/          release training and precompute configurations
scripts/          Fish launchers and bounded operator utilities
tests/            unit, consistency, CLI, and distributed-contract tests
```

Reusable logic belongs under `src/`. Launchers should configure and compose
that logic rather than duplicate algorithm or scoring implementations.

## Entry Points

| Entry point | Purpose |
| --- | --- |
| `src.cli.train_i2v` | Run supervised flow-matching training |
| `src.cli.train_grpo` | Run DanceGRPO rollout, reward, and replay |
| `src.cli.eval_i2v` | Batch UniPC generation and exact output validation |
| `src.cli.eval_i2v_euler` | Deterministic Euler generation |
| `src.cli.eval_i2v_cps` | Flow-CPS generation |
| `src.cli.convert_dcp_to_diffusers` | Convert a DCP checkpoint for portable inference |
| `src.eval.vbvr_run_evaluation_parallel` | Score prepared videos through external EvalKit |
| `src.cli.eval_vbvr_vlm_outputs` | Score existing evaluation cells with the VLM judge |

Fish wrappers source [`scripts/lib/env.fish`](../scripts/lib/env.fish), enter the
repository root, activate the locked uv `.venv`, set `PYTHONPATH`, and construct
`torchrun` arguments. Python entrypoints use the matching `.venv/bin/python`.

## Configuration Layer

[`src/trainer/config.py`](../src/trainer/config.py) defines:

```text
TrainConfig
  SFTConfig
  RLConfig
```

`TrainConfig` owns model, data, optimizer, precision, distributed, checkpoint,
and logging fields. `SFTConfig` selects supervised training, while `RLConfig`
adds rollout, replay, and reward settings. Entry points merge defaults, YAML,
and CLI overrides before constructing a validated config.

Runtime topology validation also occurs in the trainers because some
constraints depend on the initialized world size and device mesh. See
[Configuration](configuration.md).

## Model Wrapper

[`WanI2VForTraining`](../src/models/wan_i2v.py) provides the training view of a
Diffusers pipeline. Depending on data and reward requirements, it loads:

- tokenizer and UMT5 text encoder;
- Wan VAE;
- high-noise `transformer`;
- low-noise `transformer_2` for A14B;
- optional LoRA adapters.

The wrapper reads the A14B expert boundary and scheduler flow shift from model
metadata, constructs the shifted sigma/timestep schedule, and routes forward
passes to the correct expert. TI2V-5B uses one dense transformer and expanded
per-token timesteps.

Model-family condition layouts differ, so dataset precompute artifacts are
coupled to the base model family and preprocessing contract.

## Trainer Hierarchy

Supervised modes share one base:

```text
BaseTrainer
  I2VTrainer
```

The RL side is separate because it owns rollout policy state, reference-policy
logic, reward workers, trajectory storage, and optional actor ranks:

```text
BaseRLTrainer
  BaseGRPOTrainer
    DanceGRPOTrainer
```

`DanceGRPOTrainer` is a thin algorithm coordinator rather than the owner of
every execution topology. Its implementation is split by responsibility:

```text
dancegrpo_trainer.py          policy objective, Flow-CPS transitions, standard replay
dancegrpo_shared_prompt.py    shared-prompt waves and delayed replay
dancegrpo_split_runtime.py    synchronous/asynchronous train-rollout rank orchestration
dancegrpo_common.py           partition helpers and rollout state records
```

The two runtime modules are cooperative mixins over `BaseGRPOTrainer`; operator
entrypoints and the public `DanceGRPOTrainer` import remain unchanged.

Both bases initialize distributed state, model modules, datasets,
`StatefulDataLoader`, optimizers, EMA, W&B, and DCP. Changes to shared concerns
such as checkpointing, FSDP, data loading, or resume semantics must be reviewed
in both stacks.

## Data Flow

Raw media:

```text
dataset JSON -> Parquet row -> image + target video + prompt
  -> UMT5 prompt embeddings
  -> VAE video latent and first-frame condition
  -> trainer or rollout
```

Precomputed input:

```text
shard-*.tar -> {key}.safetensors + {key}.json
  -> prompt_embeds + condition + latents
  -> optional reward tensors and metadata
  -> trainer or rollout
```

The raw loader uses one target video per row. The latent loader pads/truncates
text embeddings and passes non-reserved tensors through to rewards. Iterable
latent datasets require an exact `dataset_size` to keep rank-local epochs
synchronized.

## Distributed Execution

### FSDP2 and HSDP

FSDP2 uses `fully_shard`, mixed-precision policies, and device meshes. HSDP
adds a replicated mesh dimension across machines while sharding within each
machine. Non-FSDP mode uses manual gradient all-reduce and is mainly intended
for bounded LoRA or diagnostic runs.

### Expert parallelism

A14B expert parallel assigns the high- and low-noise transformers to separate
rank groups. Each rank loads only its expert, but checkpoint output still uses
the unified `high/` and `low/` layout. Expert groups must keep sampling and
control flow synchronized.

### Tensor parallel RL

The RL-only tensor-parallel path shards attention and feed-forward projections
before applying FSDP over data replicas. It preserves Wan's global-across-head
Q/K RMSNorm with an autograd-aware collective. Tensor-parallel ranks share one
data shard, rollout seed, reward, and replay input; expensive pixel rewards run
on TP rank zero and broadcast their results.

The current tensor-parallel path rejects LoRA, HSDP, expert parallel, split
actors, and trainable text encoders. It compiles modules in place so DCP keys
remain stable.

### Split rollout actors

DanceGRPO can reserve the first rank group for FSDP training and the remaining
ranks for rollout/reward actors. A bounded async queue carries complete prompt
steps. LoRA or full policy tensors are synchronized at a configured interval;
the full-model option requires async rollout.

## DanceGRPO Pipeline

```text
prompt batch
  -> G stochastic trajectories per prompt
  -> VAE decode when required
  -> asynchronous rule/VLM/model reward
  -> group-relative advantages
  -> selected replay timesteps
  -> clipped policy surrogate + optional reference penalty
  -> optimizer step
```

Shared-prompt mode distributes one prompt's group across ranks. Prompt waves
prepare several subsets before replay. Delayed replay and split async rollout
are distinct pipelines with different staleness semantics; see
[Training](training.md) and
[Async Rollout Design](dancegrpo_async_rollout_design.md).

## Reward Boundary

Rewards implement the interface in
[`src/trainer/rewards/base.py`](../src/trainer/rewards/base.py) and are resolved
through the registry. A reward declares whether it needs VAE decoding or a
policy forward.

`vbvr_rule` resolves all model/GT paths, prepares media, and starts CPU scorer
workers against an explicit external EvalKit checkout. There is no bundled or
implicit evaluator fallback. `vbvr_vlm` encodes the first frame and complete
rollout for an external OpenAI-compatible service and validates a task-specific
response schema.

## Checkpoint Runtime

PyTorch Distributed Checkpoint stores model and EMA state. Rank-local sidecars
store optimizer and `StatefulDataLoader` state. New checkpoints use stable
expert directories:

```text
checkpoint-N/
  high/
    .metadata and *.distcp
    optimizer_*_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer/
  low/
    .metadata and *.distcp
    optimizer_*_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer_2/
```

Flat and expert-parallel jobs can exchange weight state through this layout
when the required expert directories are present. Full resume additionally
requires compatible optimizer, dataloader, counters, RNG, and topology. See
[Checkpoints](checkpoints.md).

## Evaluation Architecture

The public VBVR-Pro rule path separates GPU and CPU stages:

```text
DCP -> validated Diffusers tree -> GPU generation
    -> FFmpeg media preparation -> CPU external evaluator
    -> score JSON and stage provenance
```

[`src/eval/evaluation_provenance.py`](../src/eval/evaluation_provenance.py)
fingerprints stage parameters, implementation files, inputs, and output trees.
Generation and preparation can resume matching individual media; scoring is
rewritten and promoted only after the complete result passes sample/error
validation.

The optional VLM evaluation path reads already-generated cells into an
independent append-only result root and partitions complete cells across
evaluation machines.

## Design Constraints

- FSDP-wrapped ranks must execute compatible module sequences; timestep or
  expert-routing divergence can deadlock collectives.
- Raw and latent data paths are not interchangeable, and latent artifacts are
  model-family-specific.
- Full-finetune RL replay can retain unsharded gradients unless each backward
  synchronizes.
- Evaluator source, scientific-media dependencies, manifest selection, and
  video preparation all affect reported scores and are provenance inputs.
- Weight-only initialization and true resume use the same checkpoint reader
  but intentionally restore different state.
- Training and RL base classes duplicate some runtime infrastructure; shared
  changes require mirrored review and tests.
