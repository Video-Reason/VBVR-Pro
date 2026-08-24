#!/usr/bin/env fish
# Precompute T5 prompt embeddings for VBVR-Dataset
# Launch: 2 nodes x 8 GPUs (16 GPUs total)
#
# Usage:
#   MASTER_ADDR=<ip> RANK=<0|1> fish scripts/precompute/vbvr_t5.fish

set -gx NNODES 2
set -gx NPROC_PER_NODE 8

# Defaults — override via environment
set -q MASTER_ADDR; or begin
    echo "Error: MASTER_ADDR not set"; exit 1
end
set -q RANK; or begin
    echo "Error: RANK not set (0 or 1)"; exit 1
end

set -q METADATA;   or set -gx METADATA   data/vbvr/VBVR-Dataset/data/metadata.parquet
set -q TAR_DIR;    or set -gx TAR_DIR    data/vbvr/VBVR-Dataset/tars
set -q MODEL_PATH; or set -gx MODEL_PATH storage/models/Wan2.2-I2V-A14B-Diffusers
set -q OUTPUT_DIR;  or set -gx OUTPUT_DIR  data/vbvr/latents/prompt_embeds
set -q BATCH_SIZE;  or set -gx BATCH_SIZE  2048

source (dirname (status filename))/../lib/env.fish

torchrun \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m src.precompute.vbvr_prompt_embeds \
    --metadata $METADATA \
    --tar_dir $TAR_DIR \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --skip_existing \
    --compile
