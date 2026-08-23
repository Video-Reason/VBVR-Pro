#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# The released ThinkMorph training recipe initializes from BAGEL, matching the
# upstream ThinkMorph documentation and the original VBVR-Pro experiment.
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${MODELS_DIR}/ByteDance-Seed/BAGEL-7B-MoT}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/thinkmorph}"
VBVR_PRO_DATA_DIR="${VBVR_PRO_DATA_DIR:-${DATA_PREP_DIR}/thinkmorph/parquet}"
VBVR_PRO_PARQUET_INFO="${VBVR_PRO_PARQUET_INFO:-${DATA_PREP_DIR}/thinkmorph/parquet_info.json}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRAINING_DIR}/configs/thinkmorph_vbvr_pro.yaml}"

require_dir "${THINKMORPH_REPO}"
require_dir "${BASE_MODEL_DIR}"
require_dir "${VBVR_PRO_DATA_DIR}"
require_file "${VBVR_PRO_PARQUET_INFO}"
require_file "${DATASET_CONFIG}"
mkdir -p "${RUN_OUTPUT_DIR}/checkpoints"
export VBVR_PRO_DATA_DIR VBVR_PRO_PARQUET_INFO
export PYTHONPATH="${THINKMORPH_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

print_distributed_settings
cd "${THINKMORPH_REPO}"
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
  --finetune_from_hf True \
  --resume_from "${BASE_MODEL_DIR}" \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --results_dir "${RUN_OUTPUT_DIR}" \
  --checkpoint_dir "${RUN_OUTPUT_DIR}/checkpoints" \
  --lr "${LEARNING_RATE:-1e-5}" \
  --num_workers "${DATASET_NUM_WORKERS:-4}" \
  --prefetch_factor 2 \
  --max_latent_size 64 \
  --expected_num_tokens 1 \
  --max_num_tokens 40000 \
  --max_num_tokens_per_sample 40000 \
  --vit_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps "${TOTAL_STEPS:-1000000}" \
  --save_every "${SAVE_EVERY:-1000}" \
  --wandb_offline True \
  --num_shard "${NPROC_PER_NODE}" \
  --num_replicate "${NNODES}"
