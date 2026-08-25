#!/usr/bin/env bash
# Generate random maze 384x384x81 latent WebDataset splits on one 8-GPU node.
#
# Outputs:
#   data/maze/latents/maze_384x384x81_perfect_v2/webdataset/sft
#   data/maze/latents/maze_384x384x81_perfect_v2/webdataset/rl
#   data/maze/latents/maze_384x384x81_perfect_v2/previews/*.mp4
#
# Typical usage:
#   bash scripts/precompute/maze_384_supervise_precompute_8gpu.bash
#
# Useful overrides:
#   GPUS=0,1,2,3,4,5,6,7 VAE_BATCH_SIZE=8 \
#     bash scripts/precompute/maze_384_supervise_precompute_8gpu.bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<<"$GPUS")}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29640}"
# Multi-node: the scheduler sets RANK=<0-indexed node id> and WORLD_SIZE=<num nodes>
# (same convention as scripts/train/grpo_multinode.fish), sharing one MASTER_ADDR/
# MASTER_PORT. torchrun then spans WORLD_SIZE*NPROC global ranks; maze_webdataset
# splits plan[rank::world_size] across them, all writing disjoint shards into the
# shared OUTPUT_ROOT (no merge needed).
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/maze/latents/maze_384x384x81_perfect_v2}"
WEBDATASET_DIR="${WEBDATASET_DIR:-$OUTPUT_ROOT/webdataset}"
SFT_WEBDATASET_DIR="${SFT_WEBDATASET_DIR:-$WEBDATASET_DIR/sft}"
RL_WEBDATASET_DIR="${RL_WEBDATASET_DIR:-$WEBDATASET_DIR/rl}"
PREVIEW_DIR="${PREVIEW_DIR:-$OUTPUT_ROOT/previews}"

MODEL_PATH="${MODEL_PATH:-storage/models/Wan2.2-I2V-A14B-Diffusers}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
SFT_RATIO="${SFT_RATIO:-0.8}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-1000}"
SHARD_WRITE_BATCH_SIZE="${SHARD_WRITE_BATCH_SIZE:-64}"
SEED="${SEED:-4242}"
SPLIT_SEED="${SPLIT_SEED:-$SEED}"

CELL_H="${CELL_H:-16}"
CELL_W="${CELL_W:-16}"
CELL_PX="${CELL_PX:-12}"
NUM_FRAMES="${NUM_FRAMES:-81}"
DIFFICULTY_NAMES="${DIFFICULTY_NAMES:-easy,mid,hard,xhard}"
DIFFICULTY_GEOMETRIES="${DIFFICULTY_GEOMETRIES:-}"
RENDER_MODE="${RENDER_MODE:-moving_ball}"
TAR_TAG="${TAR_TAG:-maze_384x384x81_perfect_v2}"

VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-8}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}"
NUM_PREVIEW_VIDEOS="${NUM_PREVIEW_VIDEOS:-100}"
PREVIEW_FPS="${PREVIEW_FPS:-16}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR" "$SFT_WEBDATASET_DIR" "$RL_WEBDATASET_DIR" "$PREVIEW_DIR"

RUN_LOG="${RUN_LOG:-$LOG_DIR/maze_384_precompute_$(date +%Y%m%d_%H%M%S).log}"

echo "Maze 384 random latent precompute"
echo "  gpus:              $GPUS"
echo "  nproc:             $NPROC"
echo "  model_path:        $MODEL_PATH"
echo "  output_root:       $OUTPUT_ROOT"
echo "  sft_output_dir:    $SFT_WEBDATASET_DIR"
echo "  rl_output_dir:     $RL_WEBDATASET_DIR"
echo "  preview_dir:       $PREVIEW_DIR"
echo "  samples:           $NUM_SAMPLES"
echo "  split:             sft=$SFT_RATIO rl=$(awk -v r="$SFT_RATIO" 'BEGIN { printf "%.3f", 1-r }') seed=$SPLIT_SEED"
echo "  shard batch:       samples_per_shard=$SAMPLES_PER_SHARD write_batch=$SHARD_WRITE_BATCH_SIZE"
echo "  geometry:          ${CELL_H}x${CELL_W} cells * ${CELL_PX}px, frames=$NUM_FRAMES"
echo "  difficulties:      $DIFFICULTY_NAMES"
if [[ -n "$DIFFICULTY_GEOMETRIES" ]]; then
    echo "  diff geometries:   $DIFFICULTY_GEOMETRIES"
fi
echo "  render mode:       $RENDER_MODE"
echo "  batch sizes:       vae=$VAE_BATCH_SIZE text=$TEXT_BATCH_SIZE"
echo "  log:               $RUN_LOG"
echo

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[error] missing model path: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -x "$TORCHRUN" ]]; then
    echo "[error] missing torchrun executable: $TORCHRUN" >&2
    exit 1
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN_ARGS+=(--dry_run)
fi

GEOMETRY_ARGS=()
if [[ -n "$DIFFICULTY_GEOMETRIES" ]]; then
    GEOMETRY_ARGS+=(--difficulty_geometries "$DIFFICULTY_GEOMETRIES")
fi

CUDA_VISIBLE_DEVICES="$GPUS" \
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
PYTHONUNBUFFERED=1 \
"$TORCHRUN" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m src.precompute.maze_webdataset \
    --output_dir "$WEBDATASET_DIR" \
    --sft_output_dir "$SFT_WEBDATASET_DIR" \
    --rl_output_dir "$RL_WEBDATASET_DIR" \
    --preview_dir "$PREVIEW_DIR" \
    --tar_tag "$TAR_TAG" \
    --model_path "$MODEL_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --samples_per_shard "$SAMPLES_PER_SHARD" \
    --shard_write_batch_size "$SHARD_WRITE_BATCH_SIZE" \
    --sft_ratio "$SFT_RATIO" \
    --seed "$SEED" \
    --split_seed "$SPLIT_SEED" \
    --cell_h "$CELL_H" \
    --cell_w "$CELL_W" \
    --cell_px "$CELL_PX" \
    --num_frames "$NUM_FRAMES" \
    --difficulty_names "$DIFFICULTY_NAMES" \
    "${GEOMETRY_ARGS[@]}" \
    --render_mode "$RENDER_MODE" \
    --vae_batch_size "$VAE_BATCH_SIZE" \
    --text_batch_size "$TEXT_BATCH_SIZE" \
    --num_preview_videos "$NUM_PREVIEW_VIDEOS" \
    --preview_fps "$PREVIEW_FPS" \
    --skip_existing \
    "${DRY_RUN_ARGS[@]}" \
    2>&1 | tee "$RUN_LOG"
