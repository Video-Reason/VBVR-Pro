#!/usr/bin/env bash

set -euo pipefail

TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VBVR_PRO_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"

MODELS_DIR="${MODELS_DIR:-${VBVR_PRO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${VBVR_PRO_ROOT}/data}"
DATA_PREP_DIR="${DATA_PREP_DIR:-${DATA_DIR}/prepared}"
OUTPUT_DIR="${OUTPUT_DIR:-${VBVR_PRO_ROOT}/outputs}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE="$((NNODES * NPROC_PER_NODE))"

DIFFSYNTH_REPO="${DIFFSYNTH_REPO:-${VBVR_PRO_ROOT}/vbvr_pro_models/DiffSynth-Studio}"
BAGEL_REPO="${BAGEL_REPO:-${VBVR_PRO_ROOT}/vbvr_pro_models/BAGEL}"
THINKMORPH_REPO="${THINKMORPH_REPO:-${VBVR_PRO_ROOT}/vbvr_pro_models/ThinkMorph}"

IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-${DATA_PREP_DIR}/extracted/image}"
VIDEO_BASE_PATH="${VIDEO_BASE_PATH:-${DATA_PREP_DIR}/extracted/video}"
IMAGE_METADATA_PATH="${IMAGE_METADATA_PATH:-${DATA_PREP_DIR}/metadata/diffsynth_image.jsonl}"
VIDEO_METADATA_PATH="${VIDEO_METADATA_PATH:-${DATA_PREP_DIR}/metadata/diffsynth_video.jsonl}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${MODELS_DIR}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-True}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

print_distributed_settings() {
  printf 'NNODES=%s NPROC_PER_NODE=%s NODE_RANK=%s MASTER_ADDR=%s MASTER_PORT=%s\n' \
    "${NNODES}" "${NPROC_PER_NODE}" "${NODE_RANK}" "${MASTER_ADDR}" "${MASTER_PORT}"
}

run_accelerate() {
  local launcher=(accelerate launch)
  if (( WORLD_SIZE > 1 )); then
    launcher+=(
      --multi_gpu
      --num_processes "${WORLD_SIZE}"
      --num_machines "${NNODES}"
      --machine_rank "${NODE_RANK}"
      --main_process_ip "${MASTER_ADDR}"
      --main_process_port "${MASTER_PORT}"
    )
  else
    launcher+=(--num_processes 1)
  fi
  "${launcher[@]}" "$@"
}
