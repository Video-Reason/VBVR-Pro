#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

EDIT_MODEL_ROOT="${MODELS_DIR}/Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_ROOT="${MODELS_DIR}/Qwen/Qwen-Image"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR}/qwen-image-edit-2511}"
MODEL_COMPONENTS="Qwen/Qwen-Image-Edit-2511:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"

require_dir "${DIFFSYNTH_REPO}"
require_dir "${EDIT_MODEL_ROOT}"
require_dir "${QWEN_IMAGE_ROOT}"
require_dir "${IMAGE_BASE_PATH}"
require_file "${IMAGE_METADATA_PATH}"
mkdir -p "${RUN_OUTPUT_DIR}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

print_distributed_settings
cd "${DIFFSYNTH_REPO}"
run_accelerate examples/qwen_image/model_training/train.py \
  --dataset_base_path "${IMAGE_BASE_PATH}" \
  --dataset_metadata_path "${IMAGE_METADATA_PATH}" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --dataset_repeat "${DATASET_REPEAT:-1}" \
  --model_id_with_origin_paths "${MODEL_COMPONENTS}" \
  --tokenizer_path "${EDIT_MODEL_ROOT}/tokenizer" \
  --processor_path "${EDIT_MODEL_ROOT}/processor" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${RUN_OUTPUT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank "${LORA_RANK:-32}" \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers "${DATASET_NUM_WORKERS:-8}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --find_unused_parameters \
  --zero_cond_t \
  "${resume_args[@]}"
