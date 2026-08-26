# Data and Precompute

VBVR-RL has two data paths: raw media through a Parquet-backed dataset, and precomputed latents through WebDataset tar shards.

## Raw I2V Dataset

The raw dataset is configured by a JSON file whose entries point to Parquet files:

```json
[
  {
    "data_path": "/path/to/train.parquet",
    "root": "/path/to/media/root",
    "num_frames": 81,
    "height": 256,
    "width": 256,
    "fps": 16
  }
]
```

The Parquet schema is intentionally small:

- `video: string`: target video.
- `videos: list<string>`: compatibility form accepted by older manifests; only
  its final entry is used as the target.
- `prompt: string`: text prompt.
- `image: string`: optional reference image; if absent, the first frame of the
  target video is used.[^i2v-dataset]

Paths are resolved against `root`, or against the Parquet parent if `root` is
absent. Resolution is either fixed by config (`height` + `width`) or computed
from `max_area` and the target video's aspect ratio. The result is rounded down
to a multiple of 16, matching Wan2.2 VAE spatial scale and transformer patching
assumptions.[^i2v-dataset]

## Raw Batch Contract

`I2VDataset.__getitem__` returns:

```python
{
    "index": idx,
    "videos": [target_video],  # target is (C, T, H, W), uint8
    "image": image,  # (C, H, W), uint8
    "prompt": prompt,
}
```

The one-element `videos` list is retained as a batch-interface compatibility
detail. The common `collate` stacks its target tensor along the batch
dimension.[^trainer-utils]

## Latent WebDataset

Latent training uses `VBVRLatentDataset`, an `IterableDataset` over `shard-*.tar` files.[^latent-dataset] Each tar sample contains:

```text
{key}.safetensors
{key}.json
```

The safetensors payload must contain:

- `prompt_embeds`: variable or fixed length text embeddings;
- `condition`: first-frame condition tensor;
- `latents`: one target video latent tensor.

Numbered multi-target keys such as `latents_0` are rejected. Rebuild older
shards with the single-target schema before using this release.

The loader pads/truncates prompt embeddings to 512 tokens and passes through non-reserved tensor keys. The pass-through path is used by MazeReward for tensors such as `maze_grid`, `maze_frame_positions_pix`, `maze_goal`, and `maze_ball_rgb`.[^latent-dataset][^maze-reward]

Because this is an iterable dataset, configs should set `dataset_size`. The trainer uses it to compute total optimizer steps and rank-local epoch lengths, preventing uneven-rank epoch endings that can deadlock FSDP/NCCL.[^base-trainer]

## Precompute From Existing I2V Data

`src.precompute.i2v_latent_webdataset` converts a raw Parquet-backed training config into the latent WebDataset contract.[^i2v-latent-precompute]

Example:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.precompute.i2v_latent_webdataset \
  --config configs/precompute_diffsynth_mix_260603_raw_384.yaml \
  --output_dir storage/latents/example/webdataset \
  --batch_size 4 \
  --samples_per_shard 1000
```

The precompute script writes `dataset_info.json` with recommended `latent_webdataset_dir` and `dataset_size` fields.[^i2v-latent-precompute]

## Synthetic Maze WebDataset

`src.precompute.maze_webdataset` generates synthetic mazes, renders ball-trajectory videos, encodes video latents and first-frame conditions with the Wan VAE, encodes prompts with UMT5, and writes WebDataset shards with extra `maze_*` reward tensors.[^maze-webdataset]

Launch the builder directly through the locked environment:

```bash
.venv/bin/torchrun --nproc_per_node=8 -m src.precompute.maze_webdataset \
  --output_dir storage/latents/maze/webdataset \
  --sft_output_dir storage/latents/maze/webdataset/sft \
  --rl_output_dir storage/latents/maze/webdataset/rl \
  --model_path storage/models/Wan2.2-I2V-A14B-Diffusers \
  --num_samples 20000
```

Important generated tensors:

- `maze_grid`: wall/passage grid.
- `maze_frame_positions_pix`: expected ball positions per frame.
- `maze_goal`: goal cell.
- `maze_ball_rgb`: sample-specific ball color.
- `maze_cell_px`, `maze_image_hw`: geometry needed by MazeReward.[^maze-webdataset][^maze-reward]

## VBVR Precompute

The VBVR precompute code has separate VAE and text paths:

- `src.precompute.vbvr_vae_latents` writes one safetensors file per sample with `latents` and `condition`.[^vbvr-vae]
- `src.precompute.vbvr_prompt_embeds` writes one safetensors file per sample with `prompt_embeds`.[^vbvr-t5]
- `src.precompute.build_webdataset` assembles final tar shards.

## Published VBVR-Pro RL Snapshot

The official public source is the Hugging Face Dataset
[`Video-Reason/VBVR-Pro-RL`](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-RL).
Release commands pin revision
`ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1` instead of relying on a mutable
branch name.

That revision contains 50 task archives under `VBVR-Pro-RL-Image` and the
matching 50 under `VBVR-Pro-RL-Video`, about 11.7 GB in total. The video half
is about 10.4 GB and already contains all five raw-training fields for 50,000
samples:

```text
<task-data-generator>/<task-directory>/<sample-id>/
  first_frame.png
  metadata.json
  video/
    final_frame.png
    ground_truth.mp4
    prompt.txt
```

The separate image archives are therefore unnecessary for the current I2V
loader. Download only the video archives:

```bash
.venv/bin/hf download Video-Reason/VBVR-Pro-RL \
  --repo-type dataset \
  --revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --include 'VBVR-Pro-RL-Video/*.tar.gz' \
  --local-dir storage/datasets/VBVR-Pro-RL
```

Restore a flat VBVR-Pro tree and generate the descriptor consumed by the
checked-in RL configs:

```bash
.venv/bin/python -m scripts.data.vbvr_pro_unpack_hf \
  --dataset-root storage/datasets/VBVR-Pro-RL \
  --output-dir storage/datasets/VBVR-Pro-RL/materialized \
  --source-revision ca0aaffea93b07d269c6fe2fbfe533f1fdab9aa1 \
  --expected-tasks 50 \
  --expected-samples 50000 \
  --workers 8
```

The Hugging Face downloader validates the downloaded archive objects. The
materializer independently rejects unsafe member paths, duplicate fields,
incomplete samples, wrong task counts, and wrong sample counts. It never calls
whole-archive extraction. Existing same-size outputs are reused; add
`--verify-existing` to byte-compare them with the archive members first.

The output contains `dataset.json`, `split_manifest_rl.json`, and
`materialization.json`, which records the source repository, revision,
archive list, sizes, tasks, and sample counts. The official archives are raw
publication assets, not the tensor schema accepted by
`latent_webdataset_dir`; do not point the latent loader at them.

## Dataset Design Tradeoffs

The current design makes GPU training fast by moving expensive VAE/T5 work offline. The cost is a stricter data contract:

- latent shards must be regenerated when video resolution, number of frames, prompt cleaning, VAE normalization, or condition construction changes;
- rewards that need pixels still need the VAE loaded at training time;
- WebDataset exact epoch sizing must be managed explicitly with `dataset_size`.

[^i2v-dataset]: [`src/data/i2v_dataset.py`](../src/data/i2v_dataset.py)
[^trainer-utils]: [`src/trainer/utils.py`](../src/trainer/utils.py)
[^latent-dataset]: [`src/data/vbvr_latent_dataset.py`](../src/data/vbvr_latent_dataset.py)
[^maze-reward]: [`src/trainer/rewards/maze.py`](../src/trainer/rewards/maze.py)
[^base-trainer]: [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
[^i2v-latent-precompute]: [`src/precompute/i2v_latent_webdataset.py`](../src/precompute/i2v_latent_webdataset.py)
[^maze-webdataset]: [`src/precompute/maze_webdataset.py`](../src/precompute/maze_webdataset.py)
[^vbvr-vae]: [`src/precompute/vbvr_vae_latents.py`](../src/precompute/vbvr_vae_latents.py)
[^vbvr-t5]: [`src/precompute/vbvr_prompt_embeds.py`](../src/precompute/vbvr_prompt_embeds.py)
