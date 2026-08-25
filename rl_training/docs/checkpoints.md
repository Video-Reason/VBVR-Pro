# Checkpoints

VBVR-RL uses PyTorch Distributed Checkpoint (DCP) for model and EMA state, plus rank-local `.pt` files for optimizer and dataloader state. The runtime is implemented in `CheckpointRuntimeMixin` and low-level helpers in `checkpoint.py`.[^checkpoint-runtime][^checkpoint]

## Why DCP

DCP is designed for distributed stateful objects: it calls `state_dict` and `load_state_dict` on objects that implement that protocol, and it can save/load sharded distributed tensors without materializing a single global file on every rank.[^dcp] That matches Wan2.2 A14B training, where FSDP2 shards large transformer experts across ranks.

## Current Layout

All new checkpoints use the same layout:

```text
checkpoint-N/
  high/
    .metadata
    __*.distcp
    optimizer_transformer_rank{R}.pt
    optimizer_text_encoder_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer/adapter_model.safetensors
    lora/transformer/adapter_config.json
  low/
    .metadata
    __*.distcp
    optimizer_transformer_2_rank{R}.pt
    optimizer_text_encoder_rank{R}.pt
    dataloader_rank{R}.pt
    lora/transformer_2/adapter_model.safetensors
    lora/transformer_2/adapter_config.json
```

Flat trainers write both subdirectories. Expert-parallel trainers write only the local expert's subdirectory from each group, so the combined checkpoint still has `high/` and `low/`.[^checkpoint-runtime]

Shared scalar state (`step`, `epoch`, `batch_idx`, RNG state) and the text encoder, if trained, are duplicated across subdirectories. This duplication is small relative to transformer weights and makes flat-to-EP or EP-to-flat weight loading practical.[^checkpoint-runtime]

## TrainState

`TrainState` stores:

- model weights for `text_encoder`, `transformer`, and `transformer_2` when present;
- `step`, `epoch`, and `batch_idx`;
- CPU and CUDA RNG state.[^checkpoint]

The runtime applies a save filter before writing `high/` or `low/`, so each subdirectory receives the matching transformer and optional shared text encoder.[^checkpoint-runtime]

## EMA

`EMA` stores local shadow copies of trainable parameter shards. Because it
tracks the current sharded parameter tensors, it can be saved with DCP and
resharded on load.[^ema]

When loading as weight-only initialization, EMA is preferred if present. For evaluation, `eval_i2v --use_ema` explicitly asks `load_dcp_into_pipeline` to prefer EMA shadows.[^checkpoint]

## Optimizers And Dataloader State

Optimizers are saved outside DCP as one file per shard rank:

```text
optimizer_transformer_rank{R}.pt
optimizer_transformer_2_rank{R}.pt
optimizer_text_encoder_rank{R}.pt
```

The dataloader state is saved as `dataloader_rank{R}.pt`. The trainer uses `torchdata.stateful_dataloader.StatefulDataLoader`, whose purpose is to expose `state_dict` / `load_state_dict` for mid-epoch checkpointing.[^torchdata][^base-trainer]

Under HSDP, only the first replica writes optimizer shards to avoid duplicate optimizer files from replicated groups.[^checkpoint-runtime]

## Resume vs Initialization

The same `_load_checkpoint` path supports two modes:

- **Resume**: restore model, EMA, optimizer shards, dataloader state, counters, and RNG.
- **Initialize**: load weights only, reset optimizer/dataloader/counters, and reinitialize EMA from the loaded model weights.[^checkpoint-runtime]

`reset_dataloader` controls which mode is used. If it is left as `None`, explicit `resume_from` defaults to reset/init behavior, while auto-resume from `output_dir` defaults to true resume behavior.[^base-trainer]

## LoRA Handling

When `lora_rank > 0`, LoRA adapters are added to the trainable transformers before sharding.[^wan-wrapper] During checkpoint save, rank 0 gathers the full state dict for each expert and writes PEFT-compatible LoRA sidecars under `lora/<transformer_name>/`.[^checkpoint-runtime][^checkpoint]

For inference loading, `load_dcp_into_pipeline` detects these sidecars, wraps the pipeline module with matching `LoraConfig`, then loads the DCP state into that wrapped module.[^checkpoint]

The remapper supports:

- plain source -> plain model;
- plain source -> LoRA-wrapped model, by inserting `.base_layer` for target module weights;
- LoRA source -> matching LoRA model;
- LoRA source -> plain model is rejected with an explicit merge recipe.[^checkpoint]

## Conversion Utilities

`src.cli.convert_dcp_to_diffusers` loads a base pipeline, applies a DCP checkpoint, optionally merges LoRA, and saves a Diffusers-style model directory.[^convert-diffusers]

`src.cli.convert_dcp_to_lora` extracts LoRA tensors and writes adapter folders from a DCP checkpoint.[^convert-lora]

Example portable conversion:

```bash
.venv/bin/python -m src.cli.convert_dcp_to_diffusers \
  --checkpoint storage/checkpoints/<run>/checkpoint-100 \
  --base_model storage/models/Wan2.2-TI2V-5B-Diffusers \
  --output storage/models/converted/<run>-checkpoint-100 \
  --merge_lora
```

For a published evaluation, retain conversion provenance identifying the base
model, checkpoint tree, EMA choice, LoRA merge choice, dtype, converter source,
and complete converted output tree.

## Operational Limits

- Optimizer fallback state for Muon non-2D parameters is stepped but not checkpointed by design.[^optimizer]
- Weight-only init reads a full DCP subdirectory into CPU memory on rank 0. This is convenient but can be a startup memory spike for full 14B checkpoints.[^checkpoint-runtime]
- If a checkpoint contains only one expert subdirectory, a flat trainer will load the available expert and warn/skip the missing one.

[^checkpoint-runtime]: [`src/trainer/checkpoint_runtime.py`](../src/trainer/checkpoint_runtime.py)
[^checkpoint]: [`src/trainer/checkpoint.py`](../src/trainer/checkpoint.py)
[^dcp]: PyTorch, "Distributed Checkpoint", https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
[^ema]: [`src/trainer/ema.py`](../src/trainer/ema.py)
[^torchdata]: PyTorch/TorchData, "Stateful DataLoader", https://docs.pytorch.org/data/beta/torchdata.stateful_dataloader.html
[^base-trainer]: [`src/trainer/base_trainer.py`](../src/trainer/base_trainer.py)
[^wan-wrapper]: [`src/models/wan_i2v.py`](../src/models/wan_i2v.py)
[^convert-diffusers]: [`src/cli/convert_dcp_to_diffusers.py`](../src/cli/convert_dcp_to_diffusers.py)
[^convert-lora]: [`src/cli/convert_dcp_to_lora.py`](../src/cli/convert_dcp_to_lora.py)
[^optimizer]: [`src/trainer/optimizer.py`](../src/trainer/optimizer.py)
