# Getting Started

This guide takes a fresh checkout through training and evaluation setup. Model
weights, datasets, evaluator source, and generated artifacts are external to
the repository and should be placed under the ignored `storage/` directory.

## 1. Install the Environment

Requirements:

- Linux;
- Python 3.12;
- [uv](https://docs.astral.sh/uv/) 0.11.33;
- Fish for the provided launchers;
- FFmpeg and ffprobe on `PATH` for media workflows;
- a CUDA-capable NVIDIA GPU for training and generation;
- a C/C++ compiler when Triton compiles locally.

Clone the repository and reproduce the lockfile exactly:

```bash
git clone https://github.com/pufanyi/vbvr-rl.git
cd vbvr-rl
uv sync --frozen
uv lock --check
uv sync --frozen --check
```

Run a cheap import and media-runtime check:

```bash
.venv/bin/python -m src.eval.vbvr_runtime
```

uv creates the locked project environment under `.venv`. Operator commands use
its Python, Torchrun, and Ruff binaries directly; Fish remains a host
prerequisite for the launchers.

## 2. Download a Base Model

The TI2V-5B workflows expect a compatible Diffusers pipeline beneath
`storage/models/Wan2.2-TI2V-5B-Diffusers`:

```bash
.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir storage/models/Wan2.2-TI2V-5B-Diffusers
```

The A14B reference config instead expects
`storage/models/Wan2.2-I2V-A14B-Diffusers`. Review the model license and access
requirements before downloading either artifact. A config may point at a
compatible converted or fine-tuned Diffusers directory through `model_path`.

## 3. Prepare the Public RL Dataset

Download the video half of the official public dataset at the pinned revision:

```bash
.venv/bin/hf download Video-Reason/VBVR-Pro-RL \
  --repo-type dataset \
  --revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --include 'VBVR-Pro-RL-Video/*.tar.gz' \
  --local-dir storage/datasets/VBVR-Pro-RL
```

Materialize the fields required by `I2VDataset` and `vbvr_rule`:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/VBVR-Pro-RL \
  --output-dir storage/datasets/VBVR-Pro-RL/materialized \
  --source-revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --expected-tasks 50 \
  --expected-samples 50000 \
  --workers 8
```

The unpacker validates archive layout and sample completeness, safely writes
only the five required fields, and emits `materialized/dataset.json`, the
split manifest, and `materialization.json`. Pass `--verify-existing` to
byte-compare previously restored files before reuse. The published archives
are raw assets and are not compatible with `latent_webdataset_dir`. See
[Data and Precompute](data.md) for the complete schemas.

## 4. Install the Rule Evaluator When Needed

`vbvr_rule` is optional and its evaluator is not bundled. A rule-reward config
must point to a separately obtained compatible checkout and pin its source
fingerprint:

```yaml
grpo_reward_fn: vbvr_rule
vbvr_reward_evalkit_dir: storage/evalkits/<checkout>
vbvr_reward_evalkit_source_sha256: <64-hex-digest>
```

Follow [External EvalKit](external_evalkit.md) to validate the checkout,
EasyOCR assets, and scorer runtime. Do not replace an evaluator revision in an
existing result namespace: evaluator source is part of the metric definition.

## 5. Review a Config Before Launch

At minimum, verify:

- `model_path` exists and matches the model family used to create any latents;
- exactly one intended data path is selected;
- raw dimensions and frame count match the experiment;
- `dataset_size` is correct for latent WebDataset input;
- `output_dir` is new or has the intended resume checkpoint;
- batch, group, prompt-wave, and topology constraints are satisfied;
- reward-specific paths and service endpoints are available from every rank;
- `max_steps`, save cadence, and W&B settings are intentional.

Configuration precedence is defaults, then YAML, then explicit CLI overrides.
See [Configuration](configuration.md) for field semantics.

## 6. Launch Training

Single-machine SFT:

```fish
fish scripts/train/sft_multinode.fish --nproc 8 -- \
  --config configs/train_sft_vbvr_5e-6.yaml
```

The retained SFT config expects the A14B base model and an external
800,000-sample latent WebDataset at `storage/datasets/vbvr_sft`. The public raw RL
archives prepared above are not a drop-in replacement for those latents.

Single-machine DanceGRPO:

```fish
fish scripts/train/grpo_multinode.fish --nproc 8 \
  --config configs/train_rl_a14b_rule.yaml
```

Multi-machine DanceGRPO uses the same command on every machine:

```bash
MASTER_ADDR=<rank-zero-host> \
MASTER_PORT=29500 \
WORLD_SIZE=<machine-count> \
RANK=<machine-rank> \
fish scripts/train/grpo_multinode.fish --nproc 8 -- \
  --config configs/train_rl_5b_cps.yaml
```

The training launchers default to local rendezvous values only when
`MASTER_ADDR`, `WORLD_SIZE`, and `RANK` are all absent. For multiple machines,
set all three. Here `WORLD_SIZE` is the machine count and `--nproc` is the
local process count. The global process count is their product. The GRPO
launcher performs cheap runtime checks before loading model weights.
Use `configs/train_rl_5b_cps.yaml` for the Flow-CPS eta-0.7 rule reference or
`configs/train_rl_5b_sde.yaml` for its paired DanceGRPO RF-SDE eta-0.3 run.

## 7. Validate the Checkout

Run tests from the explicit project test directory:

```bash
.venv/bin/python -m pytest tests
```

Run the repository-wide lint and formatting checks from the parent checkout
root (see the [root development instructions](../../README.md#development)):

```bash
uv sync --project .. --frozen --only-group dev --no-install-project --inexact
../.venv/bin/ruff check --output-format=github ..
../.venv/bin/ruff format --check ..
```

For a selected RL config, run the same preflight used by the launcher:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/train_rl_5b_cps.yaml
```

## Troubleshooting

### `cv2` or EasyOCR cannot load `libGL.so.1`

Both pinned OpenCV distributions expose the same `cv2` package. On a headless
host, reinstall the headless wheel last, then repeat the runtime check:

```bash
uv pip install --python .venv/bin/python --reinstall --no-deps \
  opencv-python-headless==4.13.0.92
.venv/bin/python -m src.eval.vbvr_runtime
```

### Triton reports a missing `Python.h`

Install the development headers matching Python 3.12, or point
`WAN_TRAINER_PYTHON_INCLUDE`/`CPATH` at compatible headers.

The distributed launcher reports this failure before `torchrun` starts.

### A rule config fails before model loading

Confirm both evaluator fields are present, the checkout exists on every
machine, and its computed fingerprint matches the YAML. Then run:

```bash
.venv/bin/python -m src.cli.validate_grpo_runtime \
  --config configs/<rule-reward-config>.yaml
```

### Rewards are all zero

Do not assume this is a model-quality result. Check scorer warnings, metadata
paths, unsupported-task counts, prepared videos, and per-sample errors. Input
paths passed to scorer workers must resolve before those workers change their
working directory. For stochastic rewards, score multiple members from
the same group together so group advantages are not accidentally flat.

### A restart repeats data unexpectedly

`auto_resume: true` resumes the latest checkpoint under `output_dir`.
Explicit `resume_from` with the default `reset_dataloader` behavior is
weight-only initialization and resets counters. Set the mode deliberately;
details are in [Checkpoints](checkpoints.md).
