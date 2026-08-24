# VBVR 384x384x81 Isolated-Machine Precompute

This guide is for precomputing VBVR latents on six machines that cannot
communicate with each other. Each machine runs local `torchrun` on its own
eight GPUs, and `--rank=1` through `--rank=6` split the VBVR tar files without
overlap.

Do not use the multi-node torchrun launchers for this setup. They require
network rendezvous across machines. Use:

```bash
scripts/precompute/vbvr_384_isolated_node.bash
```

## What It Produces

Default paths:

```text
data/vbvr/latents/vbvr_384x384x81/
  prompt_embeds/    # optional T5 prompt embeddings, rank*_batch*.safetensors
  vae_latents/      # VAE video latents + first-frame condition, one file per sample
  webdataset/
    sft/            # final train split, built after collecting all machines
    rl/             # final RL split, built after collecting all machines
```

The VAE stage writes one file per sample:

```text
{tar_stem}_{index_in_tar}.safetensors
```

The T5 stage writes batch files:

```text
rank{global_rank}_batch{batch_idx}.safetensors
```

The isolated launcher offsets these rank numbers per machine, so outputs from
six machines can be copied into the same directory without filename collisions.

## Quick Start

Run one command per machine. Ranks are one-indexed.

Machine 1:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

Machine 2:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=2
```

Continue through machine 6:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=6
```

By default this runs only the VAE stage at `384x384x81`, assuming prompt
embeddings already exist or will be produced separately.

## Dry Run First

Before launching GPUs, check the assigned tar subset:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1 --dry-run
```

This writes a manifest and tar list under `logs/`, for example:

```text
logs/vbvr_384_rank1_of_6_tars.txt
logs/vbvr_384_rank1_of_6_manifest.json
```

Current VBVR metadata has 100 tar files and 1,000,000 samples. The split is:

```text
rank 1: 17 tars, 170000 samples
rank 2: 17 tars, 170000 samples
rank 3: 17 tars, 170000 samples
rank 4: 17 tars, 170000 samples
rank 5: 16 tars, 160000 samples
rank 6: 16 tars, 160000 samples
```

## Run Both T5 and VAE

If prompt embeddings have not been computed, run both stages on every machine:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1 --stage=all
```

Use the matching rank on each machine. You can also run just T5:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1 --stage=t5
```

## Common Overrides

Use all eight local GPUs by default:

```bash
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

Change batch sizes:

```bash
T5_BATCH_SIZE=2048 VAE_BATCH_SIZE=22 \
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1 --stage=all
```

Use a different output root:

```bash
OUTPUT_ROOT=/data/vbvr_latents/vbvr_384x384x81 \
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

Use a different model or dataset location:

```bash
MODEL_PATH=/models/Wan2.2-I2V-A14B-Diffusers \
METADATA=/data/VBVR-Dataset/data/metadata.parquet \
TAR_DIR=/data/VBVR-Dataset/tars \
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

If local port `29680 + rank` is busy:

```bash
LOCAL_MASTER_PORT=29701 \
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

## Resume

The launcher passes `--skip_existing`.

For VAE, existing sample files in `vae_latents/` are skipped.

For T5, existing `rank*_batch*.safetensors` batch files are skipped. Keep the
same `NPROC`, `T5_BATCH_SIZE`, and `--rank` when resuming T5, because those
settings determine batch filenames.

Just rerun the same command:

```bash
bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

## Monitor Progress

Each run streams output to the terminal and also writes a log:

```text
logs/vbvr_384_isolated_rank{rank}_{stage}_YYYYmmdd_HHMMSS.log
```

Count VAE sample files on one machine:

```bash
find data/vbvr/latents/vbvr_384x384x81/vae_latents \
  -maxdepth 1 -type f -name '*.safetensors' | wc -l
```

Expected VAE counts per rank are the sample counts listed above. After
collecting all six machines into one directory, the expected total is
`1000000`.

## Collect Outputs

After all machines finish, copy outputs to one aggregation machine or shared
storage. The filenames are designed not to collide.

Example from the aggregation machine:

```bash
mkdir -p data/vbvr/latents/vbvr_384x384x81/prompt_embeds
mkdir -p data/vbvr/latents/vbvr_384x384x81/vae_latents

rsync -av machine1:/path/to/VBVR-RL/data/vbvr/latents/vbvr_384x384x81/prompt_embeds/ \
  data/vbvr/latents/vbvr_384x384x81/prompt_embeds/
rsync -av machine1:/path/to/VBVR-RL/data/vbvr/latents/vbvr_384x384x81/vae_latents/ \
  data/vbvr/latents/vbvr_384x384x81/vae_latents/
```

Repeat for `machine2` through `machine6`.

Do not use `--delete` during collection unless you are certain the destination
only contains files from that same source machine.

## Build Final WebDataset Splits

Run this once after all prompt embeddings and VAE latents are collected. The
wrapper checks the six rank tar assignments, verifies the expected sample
counts, and then builds globally shuffled 80/20 SFT/RL shards:

```bash
bash scripts/precompute/vbvr_384_isolated_build_webdataset.bash
```

To only check whether the collected files are ready:

```bash
bash scripts/precompute/vbvr_384_isolated_build_webdataset.bash --verify-only
```

The wrapper defaults to `BUILD_WORKERS=$(nproc)`. Override it if storage
contention is higher than CPU pressure:

```bash
BUILD_WORKERS=64 bash scripts/precompute/vbvr_384_isolated_build_webdataset.bash
```

The underlying command is:

```bash
.venv/bin/python -m src.precompute.build_webdataset_split \
  --prompt_embeds_dir data/vbvr/latents/vbvr_384x384x81/prompt_embeds \
  --vae_latents_dir data/vbvr/latents/vbvr_384x384x81/vae_latents \
  --sft_output_dir data/vbvr/latents/vbvr_384x384x81/webdataset/sft \
  --rl_output_dir data/vbvr/latents/vbvr_384x384x81/webdataset/rl \
  --sft_ratio 0.8 \
  --samples_per_shard 1000 \
  --num_workers 64 \
  --seed 1337
```

The script reports how many samples have both prompt embeddings and VAE
latents. For the full dataset, the complete count should be `1000000`.

Training configs should point to one split directory, for example:

```yaml
latent_webdataset_dir: data/vbvr/latents/vbvr_384x384x81/webdataset/sft
dataset_size: 800000
```

The RL split is:

```yaml
latent_webdataset_dir: data/vbvr/latents/vbvr_384x384x81/webdataset/rl
dataset_size: 200000
```

## Sanity Checks

Check total VAE files after collection:

```bash
find data/vbvr/latents/vbvr_384x384x81/vae_latents \
  -maxdepth 1 -type f -name '*.safetensors' | wc -l
```

Check final split manifest:

```bash
cat data/vbvr/latents/vbvr_384x384x81/webdataset/split_manifest.json
```

Expected fields for the default split:

```text
total_samples: 1000000
sft_samples: 800000
rl_samples: 200000
samples_per_shard: 1000
```

## Troubleshooting

If a machine fails, rerun the same command with the same rank. Existing VAE
sample files and T5 batch files are skipped.

If `torchrun` cannot bind the local port, set `LOCAL_MASTER_PORT` to a free
port.

If GPU memory is too high, reduce `VAE_BATCH_SIZE`, for example:

```bash
VAE_BATCH_SIZE=12 bash scripts/precompute/vbvr_384_isolated_node.bash --rank=1
```

If prompt embedding files collide during collection, confirm every machine used
the isolated launcher and the same `NPROC` value. With the default eight GPUs,
rank 1 writes prompt files using global ranks `0..7`, rank 2 uses `8..15`, and
so on through rank 6 using `40..47`.
