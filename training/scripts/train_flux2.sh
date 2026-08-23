#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MODEL_ROOT="${MODELS_DIR}/black-forest-labs/FLUX.2-dev"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/flux2-dev}"
MODEL_COMPONENTS="black-forest-labs/FLUX.2-dev:transformer/*.safetensors,black-forest-labs/FLUX.2-dev:text_encoder/*.safetensors,black-forest-labs/FLUX.2-dev:vae/diffusion_pytorch_model.safetensors"
LORA_TARGETS="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_qkv_mlp_proj,to_out.0,to_add_out,linear_in,linear_out,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out,single_transformer_blocks.20.attn.to_out,single_transformer_blocks.21.attn.to_out,single_transformer_blocks.22.attn.to_out,single_transformer_blocks.23.attn.to_out,single_transformer_blocks.24.attn.to_out,single_transformer_blocks.25.attn.to_out,single_transformer_blocks.26.attn.to_out,single_transformer_blocks.27.attn.to_out,single_transformer_blocks.28.attn.to_out,single_transformer_blocks.29.attn.to_out,single_transformer_blocks.30.attn.to_out,single_transformer_blocks.31.attn.to_out,single_transformer_blocks.32.attn.to_out,single_transformer_blocks.33.attn.to_out,single_transformer_blocks.34.attn.to_out,single_transformer_blocks.35.attn.to_out,single_transformer_blocks.36.attn.to_out,single_transformer_blocks.37.attn.to_out,single_transformer_blocks.38.attn.to_out,single_transformer_blocks.39.attn.to_out,single_transformer_blocks.40.attn.to_out,single_transformer_blocks.41.attn.to_out,single_transformer_blocks.42.attn.to_out,single_transformer_blocks.43.attn.to_out,single_transformer_blocks.44.attn.to_out,single_transformer_blocks.45.attn.to_out,single_transformer_blocks.46.attn.to_out,single_transformer_blocks.47.attn.to_out"

require_dir "${DIFFSYNTH_REPO}"
require_dir "${MODEL_ROOT}"
require_dir "${IMAGE_BASE_PATH}"
require_file "${IMAGE_METADATA_PATH}"
mkdir -p "${RUN_OUTPUT_DIR}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

print_distributed_settings
cd "${DIFFSYNTH_REPO}"
run_accelerate examples/flux2/model_training/train.py \
  --dataset_base_path "${IMAGE_BASE_PATH}" \
  --dataset_metadata_path "${IMAGE_METADATA_PATH}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --model_id_with_origin_paths "${MODEL_COMPONENTS}" \
  --tokenizer_path "${MODEL_ROOT}/tokenizer" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${RUN_OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "${LORA_TARGETS}" \
  --lora_rank "${LORA_RANK:-32}" \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers "${DATASET_NUM_WORKERS:-2}" \
  --save_steps "${SAVE_STEPS:-1000}" \
  --task "sft" \
  "${resume_args[@]}"
