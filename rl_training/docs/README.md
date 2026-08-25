# Documentation

These guides describe the released repository and its public interfaces. They
use repository-relative example paths and assume generated artifacts are kept
under the ignored `storage/` tree.

## Start Here

1. [Getting Started](getting_started.md) — install the locked environment,
   download external artifacts, and run a one-GPU smoke.
2. [Configuration](configuration.md) — understand YAML precedence, data paths,
   distributed topology, rewards, and resume behavior.
3. [Data](data.md) — prepare raw Parquet data, latent WebDataset shards, and the
   public VBVR-Pro RL snapshot.
4. [Training](training.md) — launch SFT or DanceGRPO and adapt the reference
   configs safely.
5. [Evaluation](evaluation.md) — generate, prepare, score, and audit VBVR-Pro
   outputs.

## Reference Guides

| Guide | Use it when you need to… |
| --- | --- |
| [Architecture](architecture.md) | Locate trainers, model wrappers, data loaders, distributed execution, and checkpoint code. |
| [Checkpoints](checkpoints.md) | Resume a run, initialize from weights, convert DCP to Diffusers, or extract LoRA. |
| [External EvalKit](external_evalkit.md) | Install and fingerprint a compatible scorer without vendoring it. |
| [VBVR-Pro Evaluation](vbvr_pro_eval.md) | Run the complete manifest-locked rule-scoring workflow or checkpoint sweeps. |
| [Qwen VLM Reward](vlm_judge_reward.md) | Host the optional Qwen judge, train with `vbvr_vlm`, or score existing videos. |
| [Async Rollout Design](dancegrpo_async_rollout_design.md) | Understand the split actor/trainer queue and weight synchronization design. |

## Documentation Scope

This directory contains durable public guides for released code paths. It does
not contain experiment diaries, presentation material, email drafts, generated
media, machine-specific runbooks, or private planning notes. Reusable findings
from development should be incorporated into the relevant guide as a stable
contract or troubleshooting note.

## Documentation Conventions

- Commands run from the repository root unless stated otherwise.
- `.venv/bin/python` and `.venv/bin/torchrun` execute inside the locked uv
  project environment created by `uv sync --frozen`.
- `WORLD_SIZE` in Fish training launchers means machine count; `--nproc` means
  local processes/GPUs. Omit all rendezvous variables for one local machine,
  or set `MASTER_ADDR`, `WORLD_SIZE`, and `RANK` together.
- Paths under `storage/` are local runtime artifacts and are not part of the
  Git repository.
- `<placeholder>` values must be supplied by the user. Do not copy angle
  brackets literally.

## Source of Truth

When text and code differ, use this order:

1. Pydantic config validation in `src/trainer/config.py`;
2. the selected CLI or launcher;
3. the runnable YAML config;
4. these guides.

Please report stale commands or broken links through the repository issue
tracker. See [Contributing](../CONTRIBUTING.md) before submitting a patch.
