#!/usr/bin/env bash
# Build VBVR 256x256x161 latent WebDataset splits on one 8-GPU node.
#
# Pipeline:
#   1. Precompute/reuse T5 prompt embeddings.
#   2. Resume VAE latent + first-frame condition encoding at 256x256x161.
#   3. Build globally shuffled 80/20 SFT/RL WebDataset shards.
#
# Typical usage:
#   bash scripts/precompute/vbvr_256x256x161_supervise_precompute_8gpu.bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<<"$GPUS")}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29661}"

METADATA="${METADATA:-data/vbvr/VBVR-Dataset/data/metadata.parquet}"
TAR_DIR="${TAR_DIR:-data/vbvr/VBVR-Dataset/tars}"
MODEL_PATH="${MODEL_PATH:-storage/models/Wan2.2-I2V-A14B-Diffusers}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/vbvr/latents/vbvr_256x256x161}"
PROMPT_EMBEDS_DIR="${PROMPT_EMBEDS_DIR:-$OUTPUT_ROOT/prompt_embeds}"
VAE_LATENTS_DIR="${VAE_LATENTS_DIR:-$OUTPUT_ROOT/vae_latents}"
WEBDATASET_DIR="${WEBDATASET_DIR:-$OUTPUT_ROOT/webdataset}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$WEBDATASET_DIR/sft}"
RL_WEBDATASET_DIR="${RL_WEBDATASET_DIR:-$WEBDATASET_DIR/rl}"

HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
NUM_FRAMES="${NUM_FRAMES:-161}"

EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-1000000}"
SFT_RATIO="${SFT_RATIO:-0.8}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
SEED="${SEED:-1337}"
T5_BATCH_SIZE="${T5_BATCH_SIZE:-2048}"
T5_MICRO_BATCH_SIZE="${T5_MICRO_BATCH_SIZE:-256}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-22}"
BUILD_WORKERS="${BUILD_WORKERS:-64}"
LOG_DIR="${LOG_DIR:-logs/vbvr_256x256x161}"

TORCHRUN_BIN="${TORCHRUN_BIN:-$ROOT_DIR/.venv/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "[error] uv environment torchrun is missing: $TORCHRUN_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[error] uv environment Python is missing: $PYTHON_BIN; run 'uv sync --frozen'" >&2
    exit 1
fi

require_path() {
    local kind="$1"
    local path="$2"
    if [[ ! -e "$path" ]]; then
        echo "[error] missing $kind: $path" >&2
        exit 1
    fi
}

count_prompt_samples() {
    "$PYTHON_BIN" - "$PROMPT_EMBEDS_DIR" <<'PY'
import json
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
total = 0
for path in sorted(root.glob("*.safetensors")):
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            continue
        header_size = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_size))
    total += sum(1 for key in header if key != "__metadata__")
print(total)
PY
}

require_path metadata "$METADATA"
require_path tar_dir "$TAR_DIR"
require_path model_path "$MODEL_PATH"

mkdir -p "$PROMPT_EMBEDS_DIR" "$VAE_LATENTS_DIR" "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR" "$LOG_DIR"

prompt_count="$(count_prompt_samples)"

echo "VBVR 256x256x161 supervised precompute"
echo "  metadata:          $METADATA"
echo "  tar_dir:           $TAR_DIR"
echo "  model_path:        $MODEL_PATH"
echo "  output_root:       $OUTPUT_ROOT"
echo "  prompt_embeds_dir: $PROMPT_EMBEDS_DIR ($prompt_count / $EXPECTED_SAMPLES samples)"
echo "  vae_latents_dir:   $VAE_LATENTS_DIR"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  gpus:              $GPUS"
echo "  resolution:        ${HEIGHT}x${WIDTH}x${NUM_FRAMES}"
echo "  batches:           t5=$T5_BATCH_SIZE t5_micro=$T5_MICRO_BATCH_SIZE vae=$VAE_BATCH_SIZE"
echo "  split:             sft=$SFT_RATIO seed=$SEED"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: configuration looks OK; not launching."
    exit 0
fi

if [[ "${SKIP_T5:-0}" != "1" && "$prompt_count" -lt "$EXPECTED_SAMPLES" ]]; then
    echo "==> precompute T5 prompt embeddings"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        "$TORCHRUN_BIN" \
            --nnodes=1 \
            --nproc_per_node="$NPROC" \
            --node_rank=0 \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            -m src.precompute.vbvr_prompt_embeds \
            --metadata "$METADATA" \
            --tar_dir "$TAR_DIR" \
            --model_path "$MODEL_PATH" \
            --output_dir "$PROMPT_EMBEDS_DIR" \
            --batch_size "$T5_BATCH_SIZE" \
            --encode_micro_batch_size "$T5_MICRO_BATCH_SIZE" \
            --skip_existing
else
    echo "==> skip T5 prompt embeddings"
fi

echo
echo "==> resume VAE latents and build SFT/RL WebDataset splits"
exec env \
    GPUS="$GPUS" \
    PROTECTED_GPUS="${PROTECTED_GPUS:-$GPUS}" \
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
    HEIGHT="$HEIGHT" \
    WIDTH="$WIDTH" \
    NUM_FRAMES="$NUM_FRAMES" \
    LOG_DIR="$LOG_DIR" \
    MONITOR_SECONDS="${MONITOR_SECONDS:-120}" \
    STALL_SECONDS="${STALL_SECONDS:-1800}" \
    MASTER_PORT_BASE="${MASTER_PORT_BASE:-29670}" \
    bash scripts/precompute/vbvr_384_supervise_precompute_8gpu.bash
