# DanceGRPO Split Async Rollout Design

## Problem

The current split DanceGRPO path assigns one optimizer step's `G` rollout
samples across all inference ranks:

```text
groups_for_actor = range(rollout_rank, G, rollout_world_size)
```

When inference GPUs exceed `G`, the extra ranks receive no group index and wait
at the gather. For example, 24 inference GPUs with `G=10` use only 10 GPUs per
optimizer step.

Increasing the train prompt batch size would keep more inference GPUs busy, but
it makes each optimizer step wait for many more rollouts. That is not the goal:
we want more optimizer steps per wall-clock hour, with each step still consuming
the configured `batch_size` and `G`.

## Design

Use a bounded async rollout queue across optimizer steps.

Each optimizer step remains semantically unchanged:

- prompt batch size stays `batch_size`
- per-prompt group size stays `grpo_group_size`
- GRPO advantages are computed within the same step's `G` samples
- the train update still consumes exactly one step's rollout group

The difference is scheduling. The train root dispatches rollout jobs for future
steps while training waits for the current step. Rollout actors continuously
receive jobs, generate their assigned group indices, send results back, and
immediately receive another job if queue depth allows.

For 24 inference GPUs and `G=10`, auto prefetch creates enough future steps to
offer at least 24 rollout tasks:

```text
prefetch_steps = ceil(rollout_world_size / G) + 1
               = ceil(24 / 10) + 1
               = 4
```

The `+1` absorbs stragglers. In steady state, every inference GPU has useful
work without increasing the number of rollouts required by any single optimizer
step.

## Weight Versions

Async rollout is bounded-stale by construction. A queued step records the policy
state version used when its jobs are created. All group samples for that step use
the same actor policy version, so the within-step GRPO comparison is coherent.

With LoRA actor sync, the train root sends LoRA trainable state to an actor only
when that actor switches policy version. With `rl_actor_weight_sync: full`, the
same protocol sends the full trainable policy state for non-LoRA full fine-tuning;
this is heavier but keeps split rollout semantically correct. With
`rl_actor_weight_sync: none`, actors keep their initialized weights; this is useful
as a plumbing smoke test, but not recommended for real split RL because rollouts
become increasingly stale.

## Communication

The first implementation stays inside `torchrun` instead of starting a Ray
cluster:

- the launcher and DCP/FSDP process groups stay unchanged
- each rollout rank is already one long-lived GPU actor
- train root and each actor get a dedicated two-rank Gloo process group
- root sends Python jobs with CPU tensors using `send_object_list`; full-model
  actor sync streams weights tensor-by-tensor after the job header
- actors return rollout tensors with `send_object_list`

Ray remains a reasonable future backend if we later want elastic actors,
autoscaling, or cross-job reuse outside a single `torchrun` job. It is not needed
to remove the current `G < rollout_world_size` idle time.

## Config

New fields:

```yaml
rl_async_rollout: true
rl_async_rollout_prefetch_steps: 0  # 0 means auto
rl_split_debug_logs: true           # per-node progress/timing logs
rl_train_rank_count: 0              # optional single-node/non-node-aligned override
grpo_train_sample_batch_size: 1     # train replay chunking only, not rollout batch
```

Auto prefetch is:

```text
max(1, ceil(rollout_world_size / grpo_group_size) + 1)
```

`rl_train_rank_count > 0` overrides node-based splitting and uses the first N
global ranks for training, with all remaining ranks as rollout actors. This is
intended for single-node smoke tests such as 4 train GPUs + 4 rollout GPUs.

`grpo_train_sample_batch_size` is a separate train-side replay knob. Rollout
actors may still generate one group at a time, but the train ranks can coalesce
already-generated rollout chunks before policy replay. This reduces many small
FSDP all-gather/backward calls without increasing prompt batch size, `G`, `T`,
or rollout latency for a single optimizer step.

`rl_split_debug_logs` enables logs from `local_rank=0` on every node. The train
root reports async step creation, actor dispatch, result waits, rollout packing,
train replay, optimizer, checkpoint, and actor policy collection timings. Rollout
nodes report job receive waits, policy load time, encode/generate/reward chunk
timings, and result send time. This keeps 32-card runs debuggable without
emitting logs from every GPU process.

## Failure Semantics

The train root owns scheduling. If training reaches `max_steps` or dataset end,
it drains in-flight jobs, sends `stop` to every actor, and then tears down the
process group. If any actor returns duplicate or missing group indices for a
step, packing fails before the optimizer update.
