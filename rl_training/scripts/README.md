# Scripts

The checked-in script surface is limited to training and evaluation. Reusable
logic lives under `src/`; generated artifacts belong under ignored paths such
as `storage/`.

## Layout

| Directory | Purpose |
| --- | --- |
| `train/` | Single- and multi-machine SFT and DanceGRPO launchers |
| `eval/` | Benchmark generation, scoring, and reporting launchers |
| `lib/` | Shared launcher environment setup |
| `serve/` | VLM runtime used directly by training and evaluation launchers |

## Training

Launch SFT:

```fish
fish scripts/train/sft_multinode.fish --nproc 8 -- \
  --config configs/<reviewed-sft-config>.yaml
```

Launch DanceGRPO:

```fish
fish scripts/train/grpo_multinode.fish --nproc 8 \
  --config configs/<reviewed-rl-config>.yaml
```

For multiple machines, run the selected launcher on every machine with
`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK`. `WORLD_SIZE` is the
machine count; `--nproc` is the local process count.

## Evaluation

Evaluate a VBVR-Pro checkpoint:

```fish
fish scripts/eval/vbvr_pro/run.fish \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --converted-model storage/models/converted/<run>-checkpoint-100 \
  --output-root storage/eval_out/<run>/checkpoint-100/unipc \
  --sampler unipc \
  --dry-run
```

Reproduce the published sampler matrices:

```fish
fish scripts/eval/vbvr_pro/reproduce.fish \
  --output-base storage/eval_out/published-hf \
  --dry-run
```

Most Fish launchers source `scripts/lib/env.fish`. It enters the repository
root, activates the locked uv `.venv`, sets `PYTHONPATH`, and exposes matching
Python headers to Triton when available.
