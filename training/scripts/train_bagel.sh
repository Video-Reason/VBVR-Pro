#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-${MODELS_DIR}/ByteDance-Seed/BAGEL-7B-MoT}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/bagel}"
VBVR_PRO_DATA_DIR="${VBVR_PRO_DATA_DIR:-${DATA_PREP_DIR}/bagel/parquet}"
VBVR_PRO_PARQUET_INFO="${VBVR_PRO_PARQUET_INFO:-${DATA_PREP_DIR}/bagel/parquet_info.json}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRAINING_DIR}/configs/bagel_vbvr_pro.yaml}"

require_dir "${BAGEL_REPO}"
require_dir "${BASE_MODEL_DIR}"
require_dir "${VBVR_PRO_DATA_DIR}"
require_file "${VBVR_PRO_PARQUET_INFO}"
require_file "${DATASET_CONFIG}"
mkdir -p "${RUN_OUTPUT_DIR}/checkpoints"
export VBVR_PRO_DATA_DIR VBVR_PRO_PARQUET_INFO
export PYTHONPATH="${BAGEL_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

print_distributed_settings
cd "${BAGEL_REPO}"
torchrun \
  --nnodes "${NNODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  train/pretrain_unified_navit.py \
  --dataset_config_file "${DATASET_CONFIG}" \
  --model_path "${BASE_MODEL_DIR}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --resume_from "${BASE_MODEL_DIR}" \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --num_workers "${DATASET_NUM_WORKERS:-1}" \
  --prefetch_factor 1 \
  --log_every 1 \
  --lr "${LEARNING_RATE:-5e-6}" \
  --expected_num_tokens 40000 \
  --max_num_tokens 60864 \
  --max_num_tokens_per_sample 40000 \
  --freeze_llm False \
  --freeze_vit True \
  --visual_und False \
  --visual_gen True \
  --results_dir "${RUN_OUTPUT_DIR}" \
  --checkpoint_dir "${RUN_OUTPUT_DIR}/checkpoints" \
  --mse_weight 1 \
  --ce_weight 1 \
  --text_cond_dropout_prob 0 \
  --vit_cond_dropout_prob 0 \
  --total_steps "${TOTAL_STEPS:-100000}" \
  --save_every "${SAVE_EVERY:-1000}" \
  --wandb_offline True \
  --num_shard "${WORLD_SIZE}" \
  --num_replicate 1
