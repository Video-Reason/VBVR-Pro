#!/usr/bin/env bash
# Resume VBVR 384x384x81 precompute on one 8-GPU node.
#
# This wrapper:
#   1. Uses GPUs 0..7 by default.
#   2. Skips T5 prompt embeddings, assuming prompt_embeds already exist.
#   3. Resumes VAE latent writing with --skip_existing.
#   4. Builds globally shuffled 80/20 SFT/RL WebDataset splits at the end.
#
# Typical usage on the 8-GPU node:
#   bash scripts/precompute/vbvr_384_supervise_precompute_8gpu.bash
#
# Useful overrides:
#   GPUS=0,1,2,3,4,5,6,7 VAE_BATCH_SIZE=20 \
#     bash scripts/precompute/vbvr_384_supervise_precompute_8gpu.bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-22}"
PROTECTED_GPUS="${PROTECTED_GPUS-$GPUS}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/vbvr/latents/vbvr_384x384x81}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-$OUTPUT_ROOT/prompt_embeds}"
VAE_LATENTS_DIR="${VAE_LATENTS_DIR:-$OUTPUT_ROOT/vae_latents}"
WEBDATASET_DIR="${WEBDATASET_DIR:-$OUTPUT_ROOT/webdataset}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$WEBDATASET_DIR/sft}"
RL_WEBDATASET_DIR="${RL_WEBDATASET_DIR:-$WEBDATASET_DIR/rl}"

EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1000000}"
SFT_RATIO="${SFT_RATIO:-0.8}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
SEED="${SEED:-1337}"
BUILD_WORKERS="${BUILD_WORKERS:-64}"
MONITOR_SECONDS="${MONITOR_SECONDS:-120}"
STALL_SECONDS="${STALL_SECONDS:-1800}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29630}"
LOG_DIR="${LOG_DIR:-logs}"

count_files() {
    local dir="$1"
    find "$dir" -maxdepth 1 -type f -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' '
}

echo "VBVR 384 8-GPU supervised resume"
echo "  gpus:              $GPUS"
echo "  protected_gpus:    ${PROTECTED_GPUS:-<none>}"
echo "  vae_batch_size:    $VAE_BATCH_SIZE"
echo "  output_root:       $OUTPUT_ROOT"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR ($(count_files "$PROMPT_EMBEDS_DIR") files)"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR ($(count_files "$VAE_LATENTS_DIR") / $EXPECTED_SAMPLES samples)"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  split:             sft=$SFT_RATIO rl=$(awk -v r="$SFT_RATIO" 'BEGIN { printf "%.3f", 1-r }') seed=$SEED"
echo

if [[ ! -d "$PROMPT_EMBEDS_DIR" ]]; then
    echo "[error] missing prompt embeddings: $PROMPT_EMBEDS_DIR" >&2
    echo "        copy them from the current run, or run T5 first." >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: configuration looks OK; not launching."
    exit 0
fi

exec env \
    GPU0_HOLD_SECONDS=0 \
    INITIAL_GPUS="$GPUS" \
    FULL_GPUS="$GPUS" \
    PROTECTED_GPUS="$PROTECTED_GPUS" \
    VAE_BATCH_SIZE="$VAE_BATCH_SIZE" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    PROMPT_EMBEDS_DIR="$PROMPT_EMBEDS_DIR" \
    VAE_LATENTS_DIR="$VAE_LATENTS_DIR" \
    WEBDATASET_DIR="$WEBDATASET_DIR" \
    SFT_WEBDATASET_DIR="$SFT_WEBDATASET_DIR" \
    RL_WEBDATASET_DIR="$RL_WEBDATASET_DIR" \
    EXPECTED_SAMPLES="$EXPECTED_SAMPLES" \
    SFT_RATIO="$SFT_RATIO" \
    SAMPLES_PER_SHARD="$SAMPLES_PER_SHARD" \
    SEED="$SEED" \
    BUILD_WORKERS="$BUILD_WORKERS" \
    MONITOR_SECONDS="$MONITOR_SECONDS" \
    STALL_SECONDS="$STALL_SECONDS" \
    MASTER_PORT_BASE="$MASTER_PORT_BASE" \
    LOG_DIR="$LOG_DIR" \
    bash scripts/precompute/vbvr_384_supervise_precompute.bash
