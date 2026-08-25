#!/usr/bin/env fish
# One-shot VBVR-Dataset precompute for one machine with 8 GPUs.
#
# Pipeline:
#   1. T5 prompt embeddings
#   2. Wan VAE video latents + first-frame condition at 384x384x81
#   3. Shuffled WebDataset tar shards
#
# Usage:
#   fish scripts/precompute/vbvr_384_webdataset_single_node.fish
#
# Common overrides:
#   env OUTPUT_ROOT=/shared/vbvr384 VAE_BATCH_SIZE=2 \
#       fish scripts/precompute/vbvr_384_webdataset_single_node.fish
#
# Resume / partial reruns:
#   set SKIP_T5=1, SKIP_VAE=1, or SKIP_WEBDATASET=1 to skip a stage.
#   WebDataset output must be empty unless ALLOW_EXISTING_WEBDATASET=1 is set.

source (dirname (status filename))/../lib/env.fish

set -x PYTHONPATH (pwd) $PYTHONPATH

function _require_path
    set -l kind $argv[1]
    set -l path $argv[2]
    if not test -e "$path"
        echo "[error] missing $kind: $path" >&2
        exit 1
    end
end

function _stage
    echo
    echo "==> $argv"
end

set -q METADATA; or set -gx METADATA data/vbvr/VBVR-Dataset/data/metadata.parquet
set -q TAR_DIR; or set -gx TAR_DIR data/vbvr/VBVR-Dataset/tars
set -q MODEL_PATH; or set -gx MODEL_PATH storage/models/Wan2.2-I2V-A14B-Diffusers

set -q OUTPUT_ROOT; or set -gx OUTPUT_ROOT data/vbvr/latents/vbvr_384x384x81
set -q PROMPT_EMBEDS_DIR; or set -gx PROMPT_EMBEDS_DIR $OUTPUT_ROOT/prompt_embeds
set -q VAE_LATENTS_DIR; or set -gx VAE_LATENTS_DIR $OUTPUT_ROOT/vae_latents
set -q WEBDATASET_DIR; or set -gx WEBDATASET_DIR $OUTPUT_ROOT/webdataset

set -q NPROC; or set -gx NPROC 4
set -q MASTER_ADDR; or set -gx MASTER_ADDR 127.0.0.1
set -q MASTER_PORT; or set -gx MASTER_PORT 29500

set -q HEIGHT; or set -gx HEIGHT 384
set -q WIDTH; or set -gx WIDTH 384
set -q NUM_FRAMES; or set -gx NUM_FRAMES 81

set -q T5_BATCH_SIZE; or set -gx T5_BATCH_SIZE 2048
set -q VAE_BATCH_SIZE; or set -gx VAE_BATCH_SIZE 4
set -q SAMPLES_PER_SHARD; or set -gx SAMPLES_PER_SHARD 1000
set -q SEED; or set -gx SEED 1337

set -l cpu_count 16
if type -q nproc
    set cpu_count (command nproc)
end
if test $cpu_count -gt 64
    set cpu_count 64
end
set -q BUILD_WORKERS; or set -gx BUILD_WORKERS $cpu_count

set -q COMPILE; or set -gx COMPILE 1

_require_path metadata $METADATA
_require_path tar_dir $TAR_DIR
_require_path model_path $MODEL_PATH

mkdir -p "$PROMPT_EMBEDS_DIR" "$VAE_LATENTS_DIR" "$WEBDATASET_DIR"

echo "VBVR 384x384x81 precompute"
echo "  metadata:          $METADATA"
echo "  tar_dir:           $TAR_DIR"
echo "  model_path:        $MODEL_PATH"
echo "  output_root:       $OUTPUT_ROOT"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR"
echo "  webdataset_dir:    $WEBDATASET_DIR"
echo "  gpus:              $NPROC"
echo "  resolution:        "$HEIGHT"x"$WIDTH"x"$NUM_FRAMES
echo "  batches:           t5=$T5_BATCH_SIZE vae=$VAE_BATCH_SIZE"
echo "  shard/write:       samples_per_shard=$SAMPLES_PER_SHARD workers=$BUILD_WORKERS seed=$SEED"
echo "  compile:           $COMPILE"

set -l compile_args
if test "$COMPILE" != "0"
    set compile_args --compile
end

if set -q SKIP_T5
    _stage "skip T5 prompt embeddings"
else
    _stage "precompute T5 prompt embeddings"
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NPROC \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m src.precompute.vbvr_prompt_embeds \
        --metadata $METADATA \
        --tar_dir $TAR_DIR \
        --model_path $MODEL_PATH \
        --output_dir $PROMPT_EMBEDS_DIR \
        --batch_size $T5_BATCH_SIZE \
        $compile_args
    or exit 1
end

if set -q SKIP_VAE
    _stage "skip VAE latents"
else
    _stage "precompute VAE latents and first-frame conditions"
    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NPROC \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m src.precompute.vbvr_vae_latents \
        --metadata $METADATA \
        --tar_dir $TAR_DIR \
        --model_path $MODEL_PATH \
        --output_dir $VAE_LATENTS_DIR \
        --batch_size $VAE_BATCH_SIZE \
        --num_frames $NUM_FRAMES \
        --height $HEIGHT \
        --width $WIDTH \
        --skip_existing \
        $compile_args
    or exit 1
end

if set -q SKIP_WEBDATASET
    _stage "skip WebDataset build"
else
    set -l existing_shards (command find "$WEBDATASET_DIR" -maxdepth 1 -type f -name 'shard-*.tar' | wc -l | string trim)
    if test $existing_shards -gt 0; and not set -q ALLOW_EXISTING_WEBDATASET
        echo "[error] $WEBDATASET_DIR already contains $existing_shards shard-*.tar files" >&2
        echo "        move/remove them, or set ALLOW_EXISTING_WEBDATASET=1 to overwrite in place" >&2
        exit 1
    end

    _stage "build shuffled WebDataset"
    python -m src.precompute.build_webdataset \
        --prompt_embeds_dir $PROMPT_EMBEDS_DIR \
        --vae_latents_dir $VAE_LATENTS_DIR \
        --output_dir $WEBDATASET_DIR \
        --samples_per_shard $SAMPLES_PER_SHARD \
        --num_workers $BUILD_WORKERS \
        --seed $SEED
    or exit 1
end

echo
echo "Done."
echo "Use this in training configs:"
echo "  latent_webdataset_dir: $WEBDATASET_DIR"
